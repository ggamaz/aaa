import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import re
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.optimization import get_scheduler
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

from qwen2_5_sd3.transformer_sd3_dynamic import SD3Transformer2DModel
from qwen2_5_sd3.qwen2_5_vl_sd3_hf_dynamic_fusion import Qwen2p5VLStableDiffusion3HF, guess_load_checkpoint
from _datasets.edit_datasets import MaskImageEditDataset as MyDataset
from tqdm import tqdm
import glob
import shutil


from log_helper import log_training_images_dynamic as log_training_images
from log_helper import print_log, init_logger, log_model_parameters, print_gpu_mem


# ── 纯 CPU 驱动的 EMA 类 (显存零占用) ──────────────────────────────────
class CPUOffloadedEMA:
    def __init__(self, model, momentum=0.99, update_interval=1):
        self.momentum = momentum
        self.update_interval = update_interval
        self.alpha = 1.0 - momentum
        self.shadow_params = {}
        self.tracked_names = []

        for name, p in model.named_parameters():
            if p.requires_grad:
                if any(key in name for key in ['projector', 'connector', 'meta_queries', 'transformer']):
                    self.shadow_params[name] = p.detach().cpu().clone()
                    self.tracked_names.append(name)
        
        print_log(f"CPU-EMA initialized. Tracking {len(self.tracked_names)} parameter tensors on RAM.", level="info")

    @torch.no_grad()
    def step(self, model, global_step):
        if global_step > 0 and global_step % self.update_interval == 0:
            for name, p in model.named_parameters():
                if name in self.tracked_names:
                    self.shadow_params[name].lerp_(p.detach().cpu(), self.alpha)

    def get_state_dict(self):
        return {k: v.clone() for k, v in self.shadow_params.items()}


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
    CFG="blurry, low quality, low resolution, distorted, deformed, broken content, missing parts, damaged details, artifacts, glitch, noise, pixelated, grainy, compression artifacts, bad composition, wrong proportion, incomplete editing, unfinished, unedited areas."
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
    parser.add_argument('--data_path',       type=str, default="COCO/fusion_and_seg_dataset_labeled.json")
    parser.add_argument('--image_folder',    type=str, default="COCO")
    parser.add_argument('--image_size',      type=int, default=512)
    parser.add_argument('--image_length',    type=int, default=256)
    parser.add_argument('--image_process',   type=str, default='fix_pixels', choices=['dynamic', 'fix_pixels', 'resize2square'])
    
    parser.add_argument('--output_dir',      type=str, default="./runs/output")
    parser.add_argument('--resume',          type=str, default=None, help="Path to a checkpoint to resume")
    parser.add_argument('--max_steps',       type=int, default=3000)
    parser.add_argument('--batch_size',      type=int, default=12)
    parser.add_argument('--grad_accum',      type=int, default=1)
    #train no_ema,no_log_img bs-accum:t-m  1-1: 3h-24G, 1-4: 15h-24G, 4-1: 6h-25G
    parser.add_argument('--lr',              type=float, default=2e-5)
    parser.add_argument('--lr_scheduler',    type=str, default='cosine')
    parser.add_argument('--warmup_steps',    type=int, default=250)
    parser.add_argument('--max_grad_norm',   type=float, default=1.0)
    parser.add_argument('--save_every',      type=int, default=100)
    parser.add_argument('--log_every',       type=int, default=1)
    parser.add_argument('--log_image_every', type=int, default=10)
    parser.add_argument('--num_workers',     type=int, default=16) 
    parser.add_argument('--seed',            type=int, default=42)
    
    # model
    parser.add_argument('--num_queries',     type=int, default=128)
    parser.add_argument('--max_length',      type=int, default=1024)
    parser.add_argument('--freeze_lmm',      action='store_true', default=True)
    parser.add_argument('--llm_lora_modules', type=str, default="auto")
    parser.add_argument('--freeze_transformer', action='store_true', default=True)
    parser.add_argument('--dit_lora_config', type=str, default=dict(r=64, lora_alpha=128))
    parser.add_argument('--freeze_meta_query', action='store_true', default=True)
    parser.add_argument('--use_activation_checkpointing', action='store_true', default=True)
    
    # args compatibility
    parser.add_argument('--local_rank',      type=int, default=-1)
    return parser.parse_args()

def collate_fn(batch):
    collated = dict(
        pixel_values_src=torch.stack([torch.stack(b['pixel_values_src']) for b in batch]).to(torch.bfloat16),
        pixel_values    =torch.stack([b['pixel_values'] for b in batch]).to(torch.bfloat16),
        texts           =[b['texts']            for b in batch],
        prompt_types    =[b['prompt_types']     for b in batch],
    )
    if "pixel_masks" in batch[0]:
        collated['pixel_masks'] = torch.stack([b['pixel_masks'] for b in batch]).to(torch.bfloat16)
    return collated

def main():
    args = parse_args()
    
    # ── 1. 初始化 Accelerate ───────────────────────────────────────────
    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs"))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_config=project_config
    )
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    init_logger(args.output_dir, is_main_process=accelerator.is_main_process)
    
    if accelerator.is_main_process:
        print_log("=" * 60)
        print_log("Training Arguments:")
        for k, v in vars(args).items():
            print_log(f"  {k:<25}: {v}")
        print_log("=" * 60)
        accelerator.init_trackers("dynamic_fusion_project")

    # ── 2. 加载模型与数据 ──────────────────────────────────────────────
    print_log("Loading tokenizer & dataset...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True, padding_side='left')

    dataset = MyDataset(
        data_path=args.data_path,
        image_folder=args.image_folder,
        tokenizer=tokenizer,
        prompt_template=PROMPT_TEMPLATE,
        image_size=args.image_size,
        image_length=args.image_length,
        image_process=args.image_process,
        max_length=args.max_length,
    )

    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    print_log("Loading Base Models (LMM, Transformer, VAE)...")
    lmm = Qwen2_5_VLForConditionalGeneration.from_pretrained(QWEN_PATH, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    transformer = SD3Transformer2DModel.from_pretrained(SD3_PATH, subfolder="transformer", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(SD3_PATH, subfolder="scheduler")
    test_scheduler = deepcopy(scheduler)
    vae = AutoencoderKL.from_pretrained(SD3_PATH, subfolder="vae", torch_dtype=torch.bfloat16)
    vae.enable_slicing()
    vae.enable_tiling()
    
    print_log("Building Fusion Model...")
    model = Qwen2p5VLStableDiffusion3HF(
        transformer=transformer,
        train_scheduler=scheduler,
        test_scheduler=test_scheduler,
        vae=vae,
        lmm=lmm,
        tokenizer=tokenizer,
        prompt_template=PROMPT_TEMPLATE,
        connector=CONNECTOR_CFG,
        num_queries=args.num_queries,
        max_length=args.max_length,
        freeze_lmm=args.freeze_lmm,
        lora_modules=args.llm_lora_modules,
        freeze_transformer=args.freeze_transformer,
        dit_lora_config=args.dit_lora_config,
        freeze_mq=args.freeze_meta_query,
        use_activation_checkpointing=args.use_activation_checkpointing,
        pretrained_pth="pretrain_ckpts/merged_model.pt",
        # pretrained_pth="pretrain_ckpts/model.pt",
        weighting_scheme='None',
    )
    # torch.save(model.state_dict(), "temp_merged.pth") # 临时保存一次合并后的权重，方便后续分析
    # raise ValueError("Debugging checkpoint loading - stop here to analyze temp_merged.pth")
    
    trainable_params = log_model_parameters(model, is_main_process=accelerator.is_main_process)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-2)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    # ── 3. Accelerate 接管 ─────────────────────────────────────────────
    model, optimizer, loader, lr_scheduler = accelerator.prepare(
        model, optimizer, loader, lr_scheduler
    )

    # ── 4. 初始化 CPU-EMA ──────────────────────────────────────────────
    unwrapped_model = accelerator.unwrap_model(model)
    # cpu_ema = CPUOffloadedEMA(unwrapped_model, momentum=0.99)

    # ── 5. 断点续训逻辑 ────────────────────────────────────────────────
    start_step = 0
    if args.resume is not None and os.path.exists(args.resume):
        print_log(f"Resuming from {args.resume}")
        try:
            accelerator.load_state(args.resume)
            match = re.search(r'step_(\d+)', args.resume)
            if match:
                start_step = int(match.group(1))
            print_log(f"✅ Successfully resumed via Accelerate state at step {start_step}")
        except Exception:
            print_log("Detected weights-only checkpoint. Applying fallback loading...")
            unwrapped_model.load_state_dict(guess_load_checkpoint(args.resume), strict=False)
            match = re.search(r'step_(\d+)', args.resume)
            if match:
                start_step = int(match.group(1))
    
    # ── 6. 训练主循环 ──────────────────────────────────────────────────
    global_step = start_step 
    loader_iter = iter(loader)
    best_loss = float('inf')
    second_best_loss = float('inf') # 新增：用于追踪次优的 loss
    running_loss_sum = 0.0  
    running_steps = 0       
    
    #sanity check
    if accelerator.is_main_process:
        print_log("Performing sanity check with a single batch...")
        try:
            sample_batch = next(iter(loader))
            with torch.no_grad():
                mini_batch = {}
                for k, v in sample_batch.items():
                    if isinstance(v, torch.Tensor):
                        mini_batch[k] = v[0:1].clone().detach() # 张量保留第 0 个的维度
                    elif isinstance(v, list):
                        mini_batch[k] = [v[0]] # 列表保留第 0 个
                log_training_images(unwrapped_model, mini_batch, step=start_step, output_dir=args.output_dir)
            print_log("Sanity check passed! Model forward works on a sample batch.")
        except Exception as e:
            print_log(f"Sanity check failed: {e}", level="error")
            accelerator.end_training()
            return
    
    model.train()
    with tqdm(total=args.max_steps, initial=start_step, disable=not accelerator.is_local_main_process) as pbar:
        while global_step < args.max_steps:
            try:
                batch = next(loader_iter)
            except StopIteration:
                print_log("DataLoader exhausted. Restarting epoch...")
                loader_iter = iter(loader)
                batch = next(loader_iter)

            with accelerator.accumulate(model):
                losses = model(batch, mode='loss', curr_step=global_step)
                loss = sum(losses.values()) 
                
                accelerator.backward(loss)
                running_loss_sum += loss.item()
                running_steps += 1
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    optimizer.step()
                    
                    unwrapped_model = accelerator.unwrap_model(model)
                    if unwrapped_model.freeze_transformer and hasattr(unwrapped_model.transformer.base_model, 'update_and_allocate'):
                        unwrapped_model.transformer.base_model.update_and_allocate(global_step)
                        
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    global_step += 1
                    
                    # cpu_ema.step(unwrapped_model, global_step)
                    
                    pbar.update(1)

                    if global_step % args.log_every == 0 and accelerator.is_main_process:
                        loss_str = '  '.join(f'{k}: {v.item():.4f}' for k, v in losses.items())
                        current_lr = lr_scheduler.get_last_lr()[0]
                        # 获取当前 GPU 的显存预留量 (转换为 GB)
                        if torch.cuda.is_available():
                            mem_gb = torch.cuda.memory_reserved(accelerator.device) / (1024 ** 3)
                            mem_str = f"Mem: {mem_gb:.1f}G"
                        else:
                            mem_str = "Mem: N/A"
                        pbar.set_description(f"lr: {current_lr:.2e} | {mem_str} | {loss_str}")
                        print_log(f"Step {global_step}: {loss_str} | LR: {current_lr:.2e} | {mem_str}", show_in_console=False)
                        
                        accelerator.log({
                            "train/lr": current_lr,
                            "train/total_loss": loss.item() * args.grad_accum,
                            **{f"train/{k}": v.item() for k, v in losses.items()}
                        }, step=global_step)

                    # 取消注释，打开log_image 
                    if global_step % args.log_image_every == 0 and accelerator.is_main_process:
                        print_log(f"Logging training images at step {global_step}...")
                        try:
                            mini_batch = {}
                            for k, v in batch.items():
                                if isinstance(v, torch.Tensor):
                                    mini_batch[k] = v[0:1].clone().detach() # 张量保留第 0 个的维度
                                elif isinstance(v, list):
                                    mini_batch[k] = [v[0]] # 列表保留第 0 个
                            log_training_images(unwrapped_model, mini_batch, global_step, args.output_dir)
                        except Exception as e:
                            print_log(f"Logging images failed: {e}", level="error")

                    if global_step % args.save_every == 0:
                        accelerator.wait_for_everyone()
                        
                        # 1. 保存最新的 Accelerate 训练状态
                        save_path = os.path.join(args.output_dir, f"state_step_{global_step}")
                        accelerator.save_state(save_path)
                        
                        if accelerator.is_main_process:
                            # 清理旧的 Accelerate 状态文件夹，仅保留当前的
                            for old_state in glob.glob(os.path.join(args.output_dir, "state_step_*")):
                                if old_state != save_path:
                                    try:
                                        shutil.rmtree(old_state)
                                    except Exception as e:
                                        print_log(f"Failed to delete old state {old_state}: {e}", level="warning")

                            # 2. 保存最新的模型权重 (固定名字，持续覆盖)
                            latest_path = os.path.join(args.output_dir, "latest.pth")
                            torch.save(unwrapped_model.state_dict(), latest_path)
                            
                            # 3. 计算当前的平均 loss
                            avg_loss = running_loss_sum / max(1, running_steps)
                            
                            # 定义文件名前缀
                            best_prefix = "best_model_step_"
                            second_best_prefix = "second_best_model_step_"

                            # 4. 判断并保存最好与次优的模型
                            if avg_loss < best_loss:
                                # 步骤 A: 删除旧的“次优模型”（因为旧的最好马上要变成次优了，老的次优该淘汰了）
                                for old_second in glob.glob(os.path.join(args.output_dir, f"{second_best_prefix}*.pth")):
                                    os.remove(old_second)
                                
                                # 步骤 B: 查找旧的“最好模型”，将其重命名为“次优模型”
                                for old_best in glob.glob(os.path.join(args.output_dir, f"{best_prefix}*.pth")):
                                    # 提取旧 best 模型的 step 数字
                                    old_step_str = old_best.split(best_prefix)[-1] 
                                    new_second_path = os.path.join(args.output_dir, f"{second_best_prefix}{old_step_str}")
                                    shutil.move(old_best, new_second_path)
                                    second_best_loss = best_loss  # 将记录的最好 loss 退居次位
                                
                                # 步骤 C: 保存当前最新的“最好模型”
                                best_loss = avg_loss
                                best_path = os.path.join(args.output_dir, f"{best_prefix}{global_step}.pth")
                                torch.save(unwrapped_model.state_dict(), best_path) 
                                print_log(f"New best model found (avg loss: {best_loss:.4f}) → {best_path}")
                                
                            elif avg_loss < second_best_loss:
                                # 如果不如最好的，但是比之前的次优好
                                # 步骤 A: 删除旧的“次优模型”
                                for old_second in glob.glob(os.path.join(args.output_dir, f"{second_best_prefix}*.pth")):
                                    os.remove(old_second)
                                
                                # 步骤 B: 保存当前最新的“次优模型”
                                second_best_loss = avg_loss
                                second_best_path = os.path.join(args.output_dir, f"{second_best_prefix}{global_step}.pth")
                                torch.save(unwrapped_model.state_dict(), second_best_path)
                                print_log(f"New second best model found (avg loss: {second_best_loss:.4f}) → {second_best_path}")
                                
                        # 重置 loss 统计
                        running_loss_sum = 0.0
                        running_steps = 0

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        final_path = os.path.join(args.output_dir, "final.pth")
        torch.save(unwrapped_model.state_dict(), final_path)
        
        # final_ema_path = os.path.join(args.output_dir, "final_ema.pth")
        # torch.save(cpu_ema.get_state_dict(), final_ema_path)
        
        print_log(f"Training complete. Final models saved.")
        accelerator.end_training()

if __name__ == '__main__':
    main()