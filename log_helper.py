import torch
from PIL import Image, ImageDraw, ImageFont

from loguru import logger
import sys
import os

def init_logger(output_dir, is_main_process=True, log_file_name="training.log"):
    # 1. 移除默认配置（Loguru 默认会向终端输出）
    logger.remove()

    # 2. 配置终端输出：通过 filter 实现“按需输出”
    # 我们定义一个特殊的信号叫做 "to_console"
    logger.add(
        sys.stderr, 
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:7}</level> | <level>{message}</level>",
        filter=lambda record: record["extra"].get("show_in_console", True),
        level="INFO" if is_main_process else "WARNING"
    )

    # 3. 配置输出到文件（仅主进程）
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, log_file_name)
        if os.path.exists(log_path):
            base, ext = os.path.splitext(log_file_name)
            idx = [f for f in os.listdir(output_dir) if f.startswith(base) and f.endswith(ext)]
            log_path = os.path.join(output_dir, f"{base}_{len(idx)}{ext}")
        logger.add(
            log_path, 
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:7} | {message}",
            level="INFO",
            enqueue=True  # 异步且多进程安全
        )

def print_log(msg, level="info", show_in_console=True):
    """
    使用 bind 动态传递变量给 filter
    """
    # .bind() 会把变量放入 record["extra"] 中，触发上面定义的 filter 逻辑
    log_method = getattr(logger.bind(show_in_console=show_in_console), level.lower(), logger.info)
    log_method(msg)

def print_gpu_mem(tag):
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    cached = torch.cuda.memory_reserved() / 1024**3
    print_log(f"[{tag}] Allocated: {allocated:.1f}G, Cached: {cached:.1f}G")

def log_model_parameters(raw_model, is_main_process=True):
    """统计模型参数，按模块/类别分类显示可训练参数数量。"""
    trainable_params = []
    module_param_counts = {}
    total_trainable = 0
    total_params = 0

    for name, param in raw_model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        if param.requires_grad:
            trainable_params.append(param)
            total_trainable += num_params
            
            top_level_module = name.split('.')[0] if '.' in name else name
            category = f"{top_level_module} (LoRA)" if 'lora' in name.lower() else top_level_module
            module_param_counts[category] = module_param_counts.get(category, 0) + num_params

    if is_main_process:
        print_log("=" * 60)
        print_log(f"{'Module / Category':<30} | {'Trainable Params (M)':<25}")
        print_log("-" * 60)
        for mod, count in sorted(module_param_counts.items(), key=lambda x: x[1], reverse=True):
            print_log(f"{mod:<30} | {count / 1e6:>20.4f} M")
        print_log("-" * 60)
        print_log(f"{'Total Trainable':<30} | {total_trainable / 1e6:>20.4f} M")
        print_log(f"{'Total Parameters':<30} | {total_params / 1e6:>20.4f} M")
        print_log(f"{'Trainable Ratio':<30} | {total_trainable / total_params * 100:>20.4f} %")
        print_log("=" * 60)
        
    return trainable_params


# ── Visualization Helper ──────────────────────────────────────────────

import os
import textwrap
import torch
from PIL import Image, ImageDraw, ImageFont

# --- 辅助函数 ---
def tensor_to_pil(tensor):
    """
    将 PyTorch Tensor 转换为 PIL Image。
    注意：输入 Tensor 的 shape 应为 [C, H, W]，如果原来是 [N, C, H, W]，需要在外部遍历。
    """
    np_img = torch.clamp(127.5 * tensor.cpu().float() + 128.0, 0, 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(np_img)


# --- 核心函数 1：排版与保存 ---
def save_visualization_grid(
    pil_refs, gt_visual_pil, gt_mask_pil, gen_imgs, 
    sample_texts, sample_types, ref_w, ref_h, 
    step, sample_idx, output_dir
):
    """
    负责将生成的图像、参考图、文本和 GT 图像进行网格排版，并保存到本地。
    """
    
    # 辅助绘图函数：文本转图像
    def draw_text_image(text_content, prefix="Prompt"):
        img = Image.new('RGB', (ref_w, ref_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        wrap_width = max(20, ref_w // 12) 
        wrapped_text = "\n".join(textwrap.wrap(f"[{prefix}]\n{text_content}", width=wrap_width))
        try:
            font = ImageFont.load_default(size=18)
        except TypeError:
            font = ImageFont.load_default()
        draw.text((20, 20), wrapped_text, fill=(0, 0, 0), font=font, spacing=8)
        return img

    # 辅助拼接函数：横向拼接一行
    def concat_row(images_list):
        total_width = sum(img.size[0] for img in images_list)
        max_height = max(img.size[1] for img in images_list)
        concat_img = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in images_list:
            concat_img.paste(img, (x_offset, 0))
            x_offset += img.size[0]
        return concat_img

    blank_img = Image.new('RGB', (ref_w, ref_h), color=(255, 255, 255))
    
    # 动态组装网格行
    # 第一行：通常放置参考图 (可见光, 红外等)，最后补一个空白占位对齐 GT 列
    first_row_imgs = pil_refs[:2] if len(pil_refs) >= 2 else pil_refs + [blank_img] * (2 - len(pil_refs))
    first_row_imgs.append(blank_img)
    rows = [concat_row(first_row_imgs)]

    # 后续行：遍历每个任务
    for p_type, p_text, g_img in zip(sample_types, sample_texts, gen_imgs):
        display_type = str(p_type).capitalize() if p_type else "Unknown Task"
        
        # 根据任务类型匹配 GT 图
        if p_type == 'visual':
            gt_img = gt_visual_pil
        elif p_type in ('segmentation', 'downstream'):
            gt_img = gt_mask_pil
        else:
            gt_img = blank_img
            
        # 每一行三列结构：文本描述 | 模型生成图 | GT 目标图
        row_imgs = [draw_text_image(p_text, display_type), g_img, gt_img]
        rows.append(concat_row(row_imgs))
        
    # 纵向拼接所有行
    total_height = sum(r.size[1] for r in rows)
    max_width = max(r.size[0] for r in rows)
    
    final_img = Image.new('RGB', (max_width, total_height))
    y_offset = 0
    for r in rows:
        final_img.paste(r, (0, y_offset))
        y_offset += r.size[1]
        
    # 执行保存
    sample_dir = os.path.join(output_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    if sample_idx is not None:
        save_path = os.path.join(sample_dir, f"step_{step:06d}_{sample_idx:04d}.jpg")
    else:
        save_path = os.path.join(sample_dir, f"step_{step:06d}.jpg")
    final_img.save(save_path)
    # print(f"Saved visualization: {save_path}") # 可选：打印保存路径

@torch.no_grad()
def log_training_images_dynamic(model, batch, step, output_dir):
    """
    接受原始 batch 数据，调用模型推理生成图像，然后将数据打包交给排版函数进行保存。
    """
    model.eval()
    # 将字典数据按样本打包，方便遍历
    batch_iterator = zip(
        batch['texts'], 
        batch['prompt_types'], 
        batch['pixel_values_src'], 
        batch['pixel_values'], 
        batch['pixel_masks']
    )
    BSZ = batch['texts'].shape[0] if isinstance(batch['texts'], torch.Tensor) else len(batch['texts'])
    for i, (texts, prompt_types, refs_tensor, gt_visual, gt_mask) in enumerate(batch_iterator):
        if not texts:
            print(f"Skipping visualization for sample {i} at step {step}: No prompts found.")
            continue

        # 提取空间维度
        _, _, ref_h, ref_w = refs_tensor.shape
        
        # 将参考图和 GT 转为 PIL
        pil_refs = [tensor_to_pil(ref) for ref in refs_tensor]
        gt_visual_pil = tensor_to_pil(gt_visual)
        gt_mask_pil = tensor_to_pil(gt_mask)

        # 准备生成参数
        num_prompts = len(texts)
        NEGATIVE_PROMPT = 'blurry, low quality, low resolution, distorted, deformed, broken content, missing parts, damaged details, artifacts, glitch, noise, pixelated, grainy, compression artifacts, bad composition, wrong proportion, incomplete editing, unfinished, unedited areas.'
        cfg_prompts = [NEGATIVE_PROMPT] * num_prompts
        refs_to_gen = refs_tensor.unsqueeze(0).expand(num_prompts, -1, -1, -1, -1)

        # 执行推理
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            gen_outs = model.generate(
                prompt=texts,
                cfg_prompt=cfg_prompts,
                pixel_values_src=refs_to_gen, 
                cfg_scale=4.5,
                num_steps=20,
                height=ref_h,
                width=ref_w,
                progress_bar=False
            )

        # 保存结果 (注意传入的文件名可能需要区分 batch 里的不同样本)
        save_visualization_grid(
            pil_refs=pil_refs,
            gt_visual_pil=gt_visual_pil,
            gt_mask_pil=gt_mask_pil,
            gen_imgs=[tensor_to_pil(img) for img in gen_outs],
            sample_texts=texts,
            sample_types=prompt_types,
            ref_w=ref_w,
            ref_h=ref_h,
            step=step,
            sample_idx=i if BSZ > 1 else None, # 建议传入索引，防止覆盖
            output_dir=output_dir
        )
        
    model.train()