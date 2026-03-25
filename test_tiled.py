import os
import math
import copy
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from einops import rearrange
import torch.nn.functional as F

from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer
from accelerate import Accelerator

# ── 导入您本地的模型组件 (请确保路径正确) ──
from qwen2_5_sd3.transformer_sd3_dynamic import SD3Transformer2DModel
from qwen2_5_sd3.qwen2_5_vl_sd3_hf_dynamic_fusion import Qwen2p5VLStableDiffusion3HF, guess_load_checkpoint

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

# ==========================================
# 🟢 1. 小波重构相关函数 (色彩与光照保护)
# ==========================================
def wavelet_blur(image: torch.Tensor, radius: int):
    # input shape: (B, C, H, W)
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125,  0.25,  0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    kernel = kernel[None, None].repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode='replicate')
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    return output

def wavelet_decomposition(image: torch.Tensor, levels=5):
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += (image - low_freq)
        image = low_freq
    return high_freq, low_freq

def wavelet_reconstruction(content_feat: torch.Tensor, style_feat: torch.Tensor):
    """用 style_feat 的低频(颜色/光照) 替换 content_feat 的低频"""
    content_high_freq, _ = wavelet_decomposition(content_feat)
    _, style_low_freq = wavelet_decomposition(style_feat)
    return content_high_freq + style_low_freq

# ==========================================
# 🟢 2. 滑动窗口与分块生成函数 (防御性对齐版本)
# ==========================================
def gaussian_weights(tile_height: int, tile_width: int) -> torch.Tensor:
    """生成高斯权重矩阵，用于平滑拼接缝隙"""
    var = 0.01
    midpoint_x = (tile_width - 1) / 2
    midpoint_y = (tile_height - 1) / 2
    x_probs = [np.exp(-(x - midpoint_x)**2 / (tile_width**2) / (2 * var)) / np.sqrt(2 * np.pi * var) for x in range(tile_width)]
    y_probs = [np.exp(-(y - midpoint_y)**2 / (tile_height**2) / (2 * var)) / np.sqrt(2 * np.pi * var) for y in range(tile_height)]
    weights = np.outer(y_probs, x_probs)
    return torch.from_numpy(weights).float()

def calculate_overlapping_windows(total_size: int, tile_size: int, tile_stride: int):
    """
    健壮的滑动窗口坐标计算逻辑：
    确保第一个窗口一定在 0 处，最后一个窗口一定覆盖 total_size 且对齐边缘。
    """
    if total_size <= tile_size:
        return [0]
    
    start_points = []
    # 按照步幅进行推进
    cur = 0
    while cur <= total_size - tile_size:
        start_points.append(cur)
        cur += tile_stride
    
    # 强制加上最后一个对齐边缘的窗口，防止右下角崩毁
    last_point = total_size - tile_size
    if len(start_points) > 0 and start_points[-1] != last_point:
        start_points.append(last_point)
    # 极小边情况处理：如果没有窗口，强行加一个0
    elif len(start_points) == 0:
        start_points = [0]
        
    return start_points

@torch.no_grad()
def tiled_generate(model, prompt, cfg_prompt, pixel_values_src, original_h, original_w, patch_size=512, stride=256, **kwargs):
    """
    分块生成并拼接主函数 (完全对齐保护版本)
    此函数会预将总画布垫高垫宽，直到能完整对齐步幅和 Patch 尺寸，并在最后将多余边缘裁掉。
    这是彻底防止右下角黑边的标准方案。
    """
    bsz = len(prompt)
    device = model.device
    
    # === [Modification Start: Defensive Padding for Complete Alignment] ===
    # 彻底解决黑边的核心：计算需要向上垫高多少，才能让 Patch 完整覆盖
    # 同时对齐到 VAE 需要的 32 像素整倍数
    comfort_zone_h = max(original_h, patch_size) # 垫高到至少能画一个 Patch
    comfort_zone_w = max(original_w, patch_size)
    
    # 预留一点步幅的余量，保证 calculate_overlapping_windows 不会生成负索引
    aligned_h = math.ceil((original_h - patch_size + stride) / stride) * stride + patch_size - stride
    # 防御性：确保对齐后的尺寸必须大于等于 Comfort Zone。
    # 既然你说输入图是 1024x768，它肯定是大于 Comfort Zone (512) 的。
    final_padded_h = max(comfort_zone_h, aligned_h)
    
    aligned_w = math.ceil((original_w - patch_size + stride) / stride) * stride + patch_size - stride
    final_padded_w = max(comfort_zone_w, aligned_w)
    
    # 对齐到 32 像素倍数，VAE 喜欢它
    final_padded_h = math.ceil(final_padded_h / 32) * 32
    final_padded_w = math.ceil(final_padded_w / 32) * 32
    
    pad_h = final_padded_h - original_h
    pad_w = final_padded_w - original_w
    
    accelerator.print(f"[Tiled Handling] Defensive standardization: {original_w}x{original_h} -> {final_padded_w}x{final_padded_h}")
    accelerator.print(f"[Tiled Handling] Padding bottom with {pad_h}px, right with {pad_w}px.")
    
    # 初始化空白对齐画布
    out_canvas = torch.zeros((bsz, 3, final_padded_h, final_padded_w), device=device, dtype=torch.float32)
    count_canvas = torch.zeros((bsz, 3, final_padded_h, final_padded_w), device=device, dtype=torch.float32)
    
    # 准备高斯权重 [1, 1, H, W]
    weights = gaussian_weights(patch_size, patch_size).to(device).unsqueeze(0).unsqueeze(0)

    # 计算窗口起始点列表 (使用防御性的健壮逻辑)
    h_starts = calculate_overlapping_windows(final_padded_h, patch_size, stride)
    w_starts = calculate_overlapping_windows(final_padded_w, patch_size, stride)
    windows = [(hi, hi + patch_size, wi, wi + patch_size) for hi in h_starts for wi in w_starts]
    # ==========================================

    print(f"\n[Tiled Processing] Standardization dimension: {final_padded_w}x{final_padded_h}")
    print(f"[Tiled Processing] 将对齐后的画布切分为 {len(windows)} 个区块进行生成...")

    # 对输入图像进行 Replication Padding，防止边缘生成伪影
    padded_src_batches = []
    for refs in pixel_values_src:
        padded_refs = [F.pad(ref, (0, pad_w, 0, pad_h), mode='replicate') for ref in refs]
        padded_src_batches.append(padded_refs)

    for hi, hi_end, wi, wi_end in tqdm(windows, desc="Tiled Generation"):
        # 1. 在完全垫高对齐的 padded_src_batches 上裁剪 Patch (坐标均为正)
        cropped_srcs = []
        for batch_idx in range(bsz):
            # ref_tensor shape: [C, PADDED_H, PADDED_W]
            batch_refs = [ref_tensor[:, hi:hi_end, wi:wi_end] for ref_tensor in padded_src_batches[batch_idx]]
            cropped_srcs.append(batch_refs)

        # 2. 调用模型生成当前 Patch (舒适区尺寸为 patch_size=512)
        tile_out = model.generate(
            prompt=prompt,
            cfg_prompt=cfg_prompt,
            pixel_values_src=cropped_srcs,
            height=patch_size,
            width=patch_size,
            progress_bar=False, 
            **kwargs
        ) # 返回形状: [b, 3, patch_size, patch_size]
        
        # 3. 将生成的图块乘以高斯权重，叠加回对齐画布
        out_canvas[:, :, hi:hi_end, wi:wi_end] += tile_out.float() * weights
        count_canvas[:, :, hi:hi_end, wi:wi_end] += weights

    # 4. 除以累加的权重，得到平滑拼接的标准化画布
    standardized_out = out_canvas / count_canvas
    
    # === [Modification End: Crop Standardization Canvas back to original] ===
    # 将预垫出来的边缘裁掉，彻底消除黑边
    final_out = standardized_out[:, :, :original_h, :original_w]
    # =========================================================================
    
    return final_out

# ==========================================
# 🟢 3. 图像预处理与主流程
# ==========================================
def resize_image(x, image_size, unit_image_size=32):
    w, h = x.size
    if w >= h and w >= image_size:
        target_w = image_size
        target_h = math.ceil(h * target_w / w / unit_image_size) * unit_image_size
    elif h > w and h >= image_size:
        target_h = image_size
        target_w = math.ceil(w * target_h / h / unit_image_size) * unit_image_size
    else:
        target_h = math.ceil(h / unit_image_size) * unit_image_size
        target_w = math.ceil(w / unit_image_size) * unit_image_size
    return x.resize((target_w, target_h))

def _process_image(image, image_size=512):
    image = resize_image(image, image_size=image_size)
    pixel_values = torch.from_numpy(np.array(image)).float()
    pixel_values = pixel_values / 255
    pixel_values = 2 * pixel_values - 1
    pixel_values = rearrange(pixel_values, 'h w c -> c h w')
    return pixel_values

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/root/aaa/runs/output/best_step_7000.pth")
    # parser.add_argument("--prompt", type=str, default='This is a natural, photorealistic fusion image based on visible light and infrared imagery. It aims to eliminate degradation artifacts in the input and restore realistic colors along with high-definition texture details. The image remains clean and natural, free of infrared artifacts, with all elements strictly aligned.')
    parser.add_argument("--prompt", type=str, default='A fusion image tailored for downstream tasks like object detection. It deeply integrates infrared saliency with visible light texture, significantly enhancing the contrast between objects and background, ensuring extremely clear object boundaries without prioritizing the naturalness of colors for human vision.')
    # 注意：确保这里传入的第二张图是可见光(VI)图，因为小波重构通常取它的色彩
    parser.add_argument("--src_imgs", type=str, nargs='+', default=['demo_dataset/input/ir/i00000.png', "demo_dataset/input/vi/v00000.png"])
    parser.add_argument("--cfg_prompt", type=str, default="blurry, low quality, low resolution, distorted, deformed, broken content, missing parts, damaged details, artifacts, glitch, noise, pixelated, grainy, compression artifacts, bad composition, wrong proportion, incomplete editing, unfinished, unedited areas.")
    parser.add_argument("--cfg_scale", type=float, default=2.0) # 建议长文本/分块生成使用较低的CFG
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--output', type=str, default='output')

    args = parser.parse_args()

    accelerator = Accelerator()
    accelerator.print(f"Loading models on GPU {accelerator.process_index}...")

    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True, padding_side='right')
    lmm = Qwen2_5_VLForConditionalGeneration.from_pretrained(QWEN_PATH, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    transformer = SD3Transformer2DModel.from_pretrained(SD3_PATH, subfolder="transformer", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(SD3_PATH, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(SD3_PATH, subfolder="vae", torch_dtype=torch.bfloat16)

    model = Qwen2p5VLStableDiffusion3HF(
        transformer=transformer,
        train_scheduler=scheduler,
        test_scheduler=copy.deepcopy(scheduler), 
        vae=vae,
        lmm=lmm,
        tokenizer=tokenizer,
        prompt_template=PROMPT_TEMPLATE,
        connector=CONNECTOR_CFG,
        num_queries=128,          
        max_length=1024,
        freeze_lmm=True,
        freeze_transformer=True,
        use_activation_checkpointing=False,
    )

    if args.checkpoint is not None:
        accelerator.print(f"Loading checkpoint from {args.checkpoint}...")
        state_dict = guess_load_checkpoint(args.checkpoint) 
        model.load_state_dict(state_dict, strict=False)
            
    model = model.to(device=accelerator.device, dtype=torch.bfloat16)
    model.eval()
    
    if accelerator.is_main_process:
        os.makedirs(args.output, exist_ok=True)

    generator = torch.Generator(device=model.device).manual_seed(args.seed)

    # ── 读取与处理图像 ──
    src_imgs = []
    for img_path in args.src_imgs:
        img = Image.open(img_path).convert('RGB')
        src_imgs.append(_process_image(img, image_size=max(args.height, args.width)))

    ref_h, ref_w = src_imgs[0].shape[1], src_imgs[0].shape[2]

    # 将可见光(VI)图作为颜色保护的参考 (默认args.src_imgs列表的第二个是VI)
    # 取值范围同样是 [-1, 1]
    color_ref_tensor = src_imgs[1].unsqueeze(0).to(accelerator.device) 

    batch_size = 2 # 显存受限时建议 batch_size 设为 1
    prompts = [args.prompt.strip()] * batch_size
    image_pixel_srcs = [src_imgs] * batch_size

    # ── 分块生成 (Tiled Generation) ──
    accelerator.print(f"Tiled Generating {batch_size} image(s) with size {ref_w}x{ref_h}...")
    with torch.no_grad():
        generated_images = tiled_generate(
            model=model,
            prompt=prompts, 
            cfg_prompt=[args.cfg_prompt] * batch_size, 
            pixel_values_src=image_pixel_srcs,
            original_h=ref_h,
            original_w=ref_w,
            patch_size=512,  # 强制模型在 512 的舒适区生成
            stride=256,      # 50% 的重叠率保证无缝拼接
            cfg_scale=args.cfg_scale, 
            num_steps=args.num_steps,
            generator=generator
        )

        # ── 小波重构 (Wavelet Reconstruction) ──
        accelerator.print("Applying Wavelet Reconstruction for color preservation...")
        # 让生成的图像 (content) 借用 可见光图像 (style/ref) 的低频颜色信息
        final_images = []
        for i in range(batch_size):
            # 取出单张生成的图像 [1, C, H, W]
            content_img = generated_images[i:i+1].to(torch.float32)
            ref_img = color_ref_tensor.to(torch.float32)
            
            # 小波融合
            fused_img = wavelet_reconstruction(content_img, ref_img)
            final_images.append(fused_img.squeeze(0))
            
        final_images = torch.stack(final_images)

    # ── 后处理与保存 ──
    final_images = rearrange(final_images, 'b c h w -> b h w c')
    final_images = torch.clamp(127.5 * final_images.cpu() + 128.0, 0, 255).to(torch.uint8).numpy()
    
    if accelerator.is_main_process:
        # 1. 将输入的参考图也转回图像矩阵 (H, W, C), 范围 0~255
        vis_src_imgs = []
        for src_tensor in src_imgs:
            # src_tensor 形状为 [C, H, W]，值域 [-1, 1]
            src_np = torch.clamp(127.5 * src_tensor.cpu() + 128.0, 0, 255).to(torch.uint8)
            src_np = rearrange(src_np, 'c h w -> h w c').numpy()
            vis_src_imgs.append(src_np)

        # 2. 遍历生成的 batch（当前设为 1）进行拼接和保存
        for i, gen_image in enumerate(final_images):
            # 使用 numpy 沿着宽度方向 (axis=1) 横向拼接
            gen_image = np.concatenate(vis_src_imgs + [gen_image], axis=1)
            
            # 保存拼接后的图像
            save_path = os.path.join(args.output, f"case_{i}.png")
            Image.fromarray(gen_image).save(save_path)
            accelerator.print(f"✅ 保存对比图成功 (IR | VI | Fused): {save_path}")