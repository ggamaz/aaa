# train.py
import os
import math
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.optimization import get_scheduler
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer
from qwen2_5_sd3.transformer_sd3_dynamic import SD3Transformer2DModel
from qwen2_5_sd3.qwen2_5_vl_sd3_hf_dynamic_fusion import Qwen2p5VLStableDiffusion3HF, guess_load_checkpoint, print_log
from _datasets.edit_datasets import MultiImageEditDataset
from tqdm import tqdm
SD3_PATH  = "/root/autodl-tmp/UniPic2-SD3.5M-Kontext-2B"
QWEN_PATH = "/root/autodl-tmp/Qwen2.5-VL-3B-Instruct"

PROMPT_TEMPLATE = dict(
    IMG_START_TOKEN='<|vision_start|>',
    IMG_END_TOKEN='<|vision_end|>',
    IMG_CONTEXT_TOKEN='<|image_pad|>',
    IMG_START_TOKEN_FOR_GENERATION=False,
    SYSTEM='<|im_start|>system\n{system}<|im_end|>\n',
    INSTRUCTION='<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n',
    SUFFIX='<|im_end|>',
    SUFFIX_AS_EOS=True,
    SEP='\n',
    STOP_WORDS=['<|im_end|>', '<|endoftext|>'],
    GENERATION='Generate an image: {input}',
    CFG='Generate an image.',
)

CONNECTOR_CFG = dict(
    hidden_size=2048,
    intermediate_size=11946,
    num_hidden_layers=6,
    _attn_implementation='flash_attention_2',
    num_attention_heads=32,
)


def parse_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--data_path',       type=str, default="/root/deepgen_lite/demo_dataset")
    parser.add_argument('--image_folder',    type=str, default="demo_dataset")
    parser.add_argument('--image_size',      type=int, default=512)
    parser.add_argument('--image_length',    type=int, default=256)
    parser.add_argument('--image_process',   type=str, default='fix_pixels',
                        choices=['dynamic', 'fix_pixels', 'resize2square'])
    # training
    parser.add_argument('--output_dir',      type=str, default='./output')
    parser.add_argument('--resume',          type=str, default=None)
    parser.add_argument('--num_epochs',      type=int, default=10)
    parser.add_argument('--batch_size',      type=int, default=1)
    parser.add_argument('--grad_accum',      type=int, default=4)
    parser.add_argument('--lr',              type=float, default=1e-4)
    parser.add_argument('--lr_scheduler',    type=str, default='cosine')
    parser.add_argument('--warmup_steps',    type=int, default=500)
    parser.add_argument('--max_grad_norm',   type=float, default=1.0)
    parser.add_argument('--save_every',      type=int, default=1000)
    parser.add_argument('--log_every',       type=int, default=1)
    parser.add_argument('--num_workers',     type=int, default=4)
    parser.add_argument('--seed',            type=int, default=42)
    # model
    parser.add_argument('--num_queries',     type=int, default=128)
    parser.add_argument('--max_length',      type=int, default=1024)
    parser.add_argument('--freeze_lmm',      action='store_true', default=True)
    parser.add_argument('--freeze_transformer', action='store_true', default=True)
    parser.add_argument('--use_activation_checkpointing', action='store_true', default=False)
    # distributed
    parser.add_argument('--local_rank',      type=int, default=-1)
    return parser.parse_args()


# ── Collate ───────────────────────────────────────────────────────────

def collate_fn(batch):
    """
    TwoImageEditDataset returns per-sample dicts with:
      pixel_values_src : list of tensors  (variable number of refs)
      pixel_values     : tensor  (C H W)
      text             : str
    We keep them as lists so the model can handle variable sizes.
    """
    return dict(
        pixel_values_src=[b['pixel_values_src'] for b in batch],  # list[list[Tensor]]
        pixel_values    =[b['pixel_values']      for b in batch],  # list[Tensor]
        texts           =[b['text']              for b in batch],  # list[str]
    )


# ── DDP helpers ───────────────────────────────────────────────────────

def setup_ddp(local_rank):
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)


def is_primary():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


# ── Main ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # ── Distributed setup ─────────────────────────────────────────────
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    is_ddp     = 'LOCAL_RANK' in os.environ
    if is_ddp:
        setup_ddp(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Tokenizer (needed for dataset + model) ────────────────────────
    print_log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_PATH, trust_remote_code=True, padding_side='left')

    # ── Dataset & DataLoader ──────────────────────────────────────────
    print_log("Building dataset...")
    dataset = MultiImageEditDataset(
        data_path=args.data_path,
        image_folder=args.image_folder,
        tokenizer=tokenizer,
        prompt_template=PROMPT_TEMPLATE,
        image_size=args.image_size,
        image_length=args.image_length,
        image_process=args.image_process,
        max_length=args.max_length,
    )

    sampler = DistributedSampler(dataset, shuffle=True) if is_ddp else None
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ── Model components ──────────────────────────────────────────────
    print_log("Loading LMM...")
    lmm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_PATH, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

    print_log("Loading transformer...")
    transformer = SD3Transformer2DModel.from_pretrained(
        SD3_PATH, subfolder="transformer", torch_dtype=torch.bfloat16)

    print_log("Loading scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        SD3_PATH, subfolder="scheduler")

    print_log("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        SD3_PATH, subfolder="vae", torch_dtype=torch.bfloat16)

    print_log("Building model...")
    model = Qwen2p5VLStableDiffusion3HF(
        transformer=transformer,
        train_scheduler=scheduler,
        test_scheduler=scheduler,
        vae=vae,
        lmm=lmm,
        tokenizer=tokenizer,
        prompt_template=PROMPT_TEMPLATE,
        connector=CONNECTOR_CFG,
        num_queries=args.num_queries,
        max_length=args.max_length,
        freeze_lmm=args.freeze_lmm,
        freeze_transformer=args.freeze_transformer,
        use_activation_checkpointing=args.use_activation_checkpointing,
        pretrained_pth=None,
    ).to(device=device, dtype=torch.bfloat16)

    if args.resume is not None:
        print_log(f"Resuming from {args.resume}")
        model.load_state_dict(guess_load_checkpoint(args.resume), strict=False)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    raw_model = model.module if is_ddp else model

    # ── Optimizer & LR scheduler ──────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    print_log(f"Trainable parameters: {sum(p.numel() for p in trainable) / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)

    total_steps = math.ceil(len(dataset) / (args.batch_size * max(1, dist.get_world_size() if is_ddp else 1))) \
                  * args.num_epochs // args.grad_accum

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Training loop ─────────────────────────────────────────────────
    global_step = 0
    optimizer.zero_grad()

    for epoch in tqdm(range(args.num_epochs)):
        if is_ddp:
            sampler.set_epoch(epoch)
        model.train()

        for step, batch in enumerate(loader):
            # forward
            losses = raw_model.compute_loss(batch)
            loss   = sum(losses.values()) / args.grad_accum

            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # logging
                if global_step % args.log_every == 0 and is_primary():
                    loss_str = '  '.join(f'{k}: {v.item():.4f}' for k, v in losses.items())
                    print_log(f"[epoch {epoch+1}  step {global_step}  lr {lr_scheduler.get_last_lr()[0]:.2e}]  {loss_str}")

                # checkpoint
                if global_step % args.save_every == 0 and is_primary():
                    ckpt_path = os.path.join(args.output_dir, f"step_{global_step}.pth")
                    torch.save(raw_model.state_dict(), ckpt_path)
                    print_log(f"Saved checkpoint → {ckpt_path}")

    # ── Final save ────────────────────────────────────────────────────
    if is_primary():
        final_path = os.path.join(args.output_dir, "final.pth")
        torch.save(raw_model.state_dict(), final_path)
        print_log(f"Training complete. Final model saved → {final_path}")


if __name__ == '__main__':
    main()