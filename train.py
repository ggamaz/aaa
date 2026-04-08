import os
import argparse

from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
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
from _datasets.edit_datasets import MultiImageEditDataset, TwoImageEditDataset
from tqdm import tqdm

from log_helper import log_training_images_threeline as log_training_images
from log_helper import print_log, init_logger

SD3_PATH  = "pretrain_ckpts/UniPic2-SD3.5M-Kontext-2B"
QWEN_PATH = "pretrain_ckpts/Qwen2.5-VL-3B-Instruct"

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
    parser.add_argument('--data_path',       type=str, default="/root/aaa/COCO/images/fusion_dataset_degrade.json")
    parser.add_argument('--image_folder',    type=str, default="/root/aaa/COCO/images")
    parser.add_argument('--image_size',      type=int, default=512)
    parser.add_argument('--image_length',    type=int, default=256)
    parser.add_argument('--image_process',   type=str, default='dynamic',
                        choices=['dynamic', 'fix_pixels', 'resize2square'])
    
    parser.add_argument('--output_dir',      type=str, default='./runs/output')
    parser.add_argument('--resume',          type=str, default="/root/autodl-tmp/model.pt")
    parser.add_argument('--max_steps',       type=int, default=10000, help="Total number of training steps")
    parser.add_argument('--batch_size',      type=int, default=1)
    parser.add_argument('--grad_accum',      type=int, default=8)
    parser.add_argument('--lr',              type=float, default=1e-4)
    parser.add_argument('--lr_scheduler',    type=str, default='cosine')
    parser.add_argument('--warmup_steps',    type=int, default=100)
    parser.add_argument('--max_grad_norm',   type=float, default=1.0)
    parser.add_argument('--save_every',      type=int, default=1000)
    parser.add_argument('--log_every',       type=int, default=1)
    parser.add_argument('--log_image_every', type=int, default=10, help="Log generated images every N steps")
    parser.add_argument('--num_workers',     type=int, default=8)
    parser.add_argument('--seed',            type=int, default=42)
    
    # model
    parser.add_argument('--num_queries',     type=int, default=128)
    parser.add_argument('--max_length',      type=int, default=1024)
    parser.add_argument('--freeze_lmm',      action='store_true', default=True)
    parser.add_argument('--freeze_transformer', action='store_true', default=False)
    parser.add_argument('--use_activation_checkpointing', action='store_true', default=False)
    
    # distributed
    parser.add_argument('--local_rank',      type=int, default=-1)
    return parser.parse_args()


# ── Collate ───────────────────────────────────────────────────────────
def collate_fn(batch):
    collated = dict(
        pixel_values_src=[b['pixel_values_src'] for b in batch],  # list[list[Tensor]]
        pixel_values    =[b['pixel_values']      for b in batch], # list[Tensor]
        texts           =[b['texts']              for b in batch], # list[str] (已经是在 Dataset 中随机选好的)
        prompt_types    =[b['prompt_types']       for b in batch], # list[str] (标记选的是 visual 还是 downstream)
    )
    return collated

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

    # Distributed setup
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    is_ddp     = 'LOCAL_RANK' in os.environ
    if is_ddp:
        setup_ddp(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    os.makedirs(args.output_dir, exist_ok=True)


    init_logger(args.output_dir)

    print_log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True, padding_side='left')

    print_log("Building dataset...")
    dataset = TwoImageEditDataset(
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

    print_log("Loading LMM...")
    lmm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_PATH, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

    print_log("Loading transformer...")
    transformer = SD3Transformer2DModel.from_pretrained(
        SD3_PATH, subfolder="transformer", torch_dtype=torch.bfloat16)

    print_log("Loading scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(SD3_PATH, subfolder="scheduler")
    # 【新增】深拷贝出一个独立的测试用 scheduler，防止 generate 污染训练状态
    test_scheduler = deepcopy(scheduler)
    
    print_log("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(SD3_PATH, subfolder="vae", torch_dtype=torch.bfloat16)

    print_log("Building model...")
    model = Qwen2p5VLStableDiffusion3HF(
        transformer=transformer,
        train_scheduler=scheduler,         # 训练用原来的
        test_scheduler=test_scheduler,     # <--- 测试用深拷贝出来的
        vae=vae,
        lmm=lmm, #Qwen2_5_VLForConditionalGeneration
        tokenizer=tokenizer,
        prompt_template=PROMPT_TEMPLATE,
        connector=CONNECTOR_CFG,
        num_queries=args.num_queries,
        max_length=args.max_length,
        freeze_lmm=args.freeze_lmm,
        freeze_transformer=args.freeze_transformer,
        max_steps=args.max_steps,  # [新增] 传入总训练步数以便 AdaLoRA 动态调整
        use_activation_checkpointing=args.use_activation_checkpointing,
        pretrained_pth=None,
    ).to(device=device, dtype=torch.bfloat16)

    if args.resume is not None:
        print_log(f"Resuming from {args.resume}")
        model.load_state_dict(guess_load_checkpoint(args.resume), strict=False)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    raw_model = model.module if is_ddp else model

    trainable = [p for p in model.parameters() if p.requires_grad]
    print_log(f"Trainable parameters: {sum(p.numel() for p in trainable) / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)

    # 按照实际更新步数计算总 scheduler steps
    total_steps = args.max_steps
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # 【新增】初始化 TensorBoard (仅在主卡创建)
    writer = None
    if is_primary():
        tb_dir = os.path.join(args.output_dir, "logs")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        print_log(f"TensorBoard logs will be saved to: {tb_dir}")
    
    # ── Training loop (Step based) ────────────────────────────────────
    global_step = 0
    inner_step = 0
    optimizer.zero_grad()
    
    loader_iter = iter(loader)
    epoch_counter = 0

    # 【新增】Checkpoint 清理与 Best Loss 追踪变量
    best_loss = float('inf')
    last_saved_ckpt = None  # 记录上一次保存的最新模型路径
    best_saved_ckpt = None  # 记录当前最好的模型路径
    running_loss_sum = 0.0  # 累计 Loss
    running_steps = 0       # 累计步数

    # 新增逻辑：在正式训练前，先记录一次 Step 0 的生图状态
    if is_primary():
        print_log("Logging initial image before training starts (step 0)...")
    try:
        first_batch = next(iter(loader))  # 单独创建一个 iter 不影响后续训练状态
        if is_primary():
            log_training_images(raw_model, first_batch, 0, args.output_dir, image_size=args.image_size)
    except Exception as e:
        if is_primary():
            print_log(f"Failed to log initial image: {e}")

    if is_ddp:
            dist.barrier()
    model.train()
    
    with tqdm(total=args.max_steps, disable=not is_primary()) as pbar:
        while global_step < args.max_steps:
            try:
                batch = next(loader_iter)
            except StopIteration:
                epoch_counter += 1
                if is_ddp:
                    sampler.set_epoch(epoch_counter)
                loader_iter = iter(loader)
                batch = next(loader_iter)

            # forward
            losses = raw_model.compute_loss(batch)
            loss   = sum(losses.values()) / args.grad_accum
            loss.backward()
            inner_step += 1
            
            # 累加未缩放的真实 Loss，用于计算平均值
            running_loss_sum += sum(losses.values()).item()
            running_steps += 1

            if inner_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                # [修改] 通过 freeze_transformer 判断是否需要调用 AdaLoRA 的裁剪逻辑
                if raw_model.freeze_transformer:
                    raw_model.transformer.base_model.update_and_allocate(global_step)
                    
                optimizer.zero_grad()
                
                global_step += 1
                pbar.update(1)

                # Logging Text & TensorBoard
                if global_step % args.log_every == 0 and is_primary():
                    # 原有的控制台打印
                    loss_str = '  '.join(f'{k}: {v.item():.4f}' for k, v in losses.items())
                    current_lr = lr_scheduler.get_last_lr()[0]
                    pbar.set_description(f"lr: {current_lr:.2e} | {loss_str}")
                    
                    if writer is not None:
                        writer.add_scalar('train/lr', current_lr, global_step)
                        writer.add_scalar('train/total_loss', loss.item() * args.grad_accum, global_step)
                        for k, v in losses.items():
                            writer.add_scalar(f'train/{k}', v.item(), global_step)
                            
                # Logging Image
                if global_step % args.log_image_every == 0 and is_primary():
                    log_training_images(raw_model, batch, global_step, args.output_dir, image_size=args.image_size)
                    if is_ddp:
                        dist.barrier()
                        
                # Checkpoint: 留最新，留最好，删其他
                if global_step % args.save_every == 0 and is_primary():
                    # --- 1. 保存最新的模型 ---
                    ckpt_path = os.path.join(args.output_dir, f"step_{global_step}.pth")
                    torch.save(raw_model.state_dict(), ckpt_path)
                    print_log(f"\nSaved latest checkpoint → {ckpt_path}")
                    
                    # 删除旧的最新模型
                    if last_saved_ckpt is not None and os.path.exists(last_saved_ckpt):
                        try:
                            os.remove(last_saved_ckpt)
                        except Exception as e:
                            print_log(f"Failed to delete old checkpoint: {e}")
                    last_saved_ckpt = ckpt_path

                    # --- 2. 评估并保存最好的模型 ---
                    avg_loss = running_loss_sum / max(1, running_steps)
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_path = os.path.join(args.output_dir, f"best_step_{global_step}.pth")
                        torch.save(raw_model.state_dict(), best_path)
                        print_log(f"New best model found (avg loss: {best_loss:.4f}) → {best_path}")
                        
                        # 删除旧的最好模型
                        if best_saved_ckpt is not None and os.path.exists(best_saved_ckpt) and best_saved_ckpt != best_path:
                            try:
                                os.remove(best_saved_ckpt)
                            except Exception as e:
                                print_log(f"Failed to delete old best checkpoint: {e}")
                        best_saved_ckpt = best_path
                    
                    running_loss_sum = 0.0
                    running_steps = 0

                if is_ddp:
                    dist.barrier()
                    

    # ── Final save ────────────────────────────────────────────────────
    if is_primary():
        final_path = os.path.join(args.output_dir, "final.pth")
        torch.save(raw_model.state_dict(), final_path)
        print_log(f"Training complete. Final model saved → {final_path}")
        
        if writer is not None:
            writer.close()
    
    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == '__main__':
    main()