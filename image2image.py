import os
import math
import copy
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from einops import rearrange

# ── 替换掉 xtuner 和 mmengine，改用原生的 diffusers 和 transformers ──
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import gather_object

# ── 导入您本地的模型组件 ──
from qwen2_5_sd3.transformer_sd3_dynamic import SD3Transformer2DModel
from qwen2_5_sd3.qwen2_5_vl_sd3_hf_dynamic_fusion import Qwen2p5VLStableDiffusion3HF, guess_load_checkpoint


# ── 模型路径与配置常量 (请根据实际情况修改路径) ──
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

import numpy as np
import torch
from tqdm import tqdm

def gaussian_weights(tile_height: int, tile_width: int) -> torch.Tensor:
    """生成高斯权重矩阵，用于平滑拼接缝隙"""
    var = 0.01
    midpoint_x = (tile_width - 1) / 2
    midpoint_y = (tile_height - 1) / 2
    x_probs = [np.exp(-(x - midpoint_x)**2 / (tile_width**2) / (2 * var)) / np.sqrt(2 * np.pi * var) for x in range(tile_width)]
    y_probs = [np.exp(-(y - midpoint_y)**2 / (tile_height**2) / (2 * var)) / np.sqrt(2 * np.pi * var) for y in range(tile_height)]
    weights = np.outer(y_probs, x_probs)
    return torch.from_numpy(weights).float()

def sliding_windows(h: int, w: int, tile_size: int, tile_stride: int):
    """计算滑动窗口的坐标"""
    hi_list = list(range(0, h - tile_size + 1, tile_stride))
    if (h - tile_size) % tile_stride != 0: hi_list.append(h - tile_size)
    wi_list = list(range(0, w - tile_size + 1, tile_stride))
    if (w - tile_size) % tile_stride != 0: wi_list.append(w - tile_size)
    return [(hi, hi + tile_size, wi, wi + tile_size) for hi in hi_list for wi in wi_list]



@torch.no_grad()
def tiled_generate(model, prompt, cfg_prompt, pixel_values_src, original_h, original_w, patch_size=512, stride=256, **kwargs):
    """分块生成并拼接主函数"""
    bsz = len(prompt)
    device = model.device
    
    # 初始化空白画布和计数器
    out_canvas = torch.zeros((bsz, 3, original_h, original_w), device=device, dtype=torch.float32)
    count_canvas = torch.zeros((bsz, 3, original_h, original_w), device=device, dtype=torch.float32)
    
    # 准备高斯权重 [1, 1, H, W]
    weights = gaussian_weights(patch_size, patch_size).to(device).unsqueeze(0).unsqueeze(0)

    # 获取所有切割窗口
    windows = sliding_windows(original_h, original_w, patch_size, stride)
    print(f"\n[Tiled Processing] 将图像切分为 {len(windows)} 个区块进行生成...")

    for hi, hi_end, wi, wi_end in tqdm(windows, desc="Tiled Generation"):
        # 1. 裁剪输入图块 (针对你的双模态输入 List[List[Tensor]])
        cropped_srcs = []
        for batch_idx in range(bsz):
            batch_refs = []
            for ref_tensor in pixel_values_src[batch_idx]: # ref_tensor shape: [C, H, W]
                batch_refs.append(ref_tensor[:, hi:hi_end, wi:wi_end])
            cropped_srcs.append(batch_refs)

        # 2. 调用模型生成当前图块 (限定尺寸为 patch_size)
        tile_out = model.generate(
            prompt=prompt,
            cfg_prompt=cfg_prompt,
            pixel_values_src=cropped_srcs,
            height=patch_size,
            width=patch_size,
            progress_bar=False, # 关闭内部单步进度条防刷屏
            **kwargs
        ) # 返回形状: [b, 3, patch_size, patch_size]
        
        # 3. 将生成的图块乘以高斯权重，叠加回总画布
        out_canvas[:, :, hi:hi_end, wi:wi_end] += tile_out.float() * weights
        count_canvas[:, :, hi:hi_end, wi:wi_end] += weights

    # 4. 除以累加的权重，得到平滑拼接的最终图像
    final_out = out_canvas / count_canvas
    return final_out

# ── 动态等比例缩放逻辑 ──
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
    # 保持宽高比进行缩放
    image = resize_image(image, image_size=image_size)
    pixel_values = torch.from_numpy(np.array(image)).float()
    pixel_values = pixel_values / 255
    pixel_values = 2 * pixel_values - 1
    pixel_values = rearrange(pixel_values, 'h w c -> c h w')
    return pixel_values

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/root/aaa/runs/output/best_step_3000.pth")
    # parser.add_argument("--prompt", type=str, default='Fuse infrared and visible-light images to generate a naturally harmonious rainy street scene. Extract accurate positions and contours of thermal targets (pedestrians, vehicles) from the infrared image, but convert them to natural colors (warm yellow or orange tones) rather than directly overlaying highlights, avoiding overly abrupt thermal targets. Preserve clear details of building textures, shop sign texts, and road markings from the visible-light image while suppressing headlight scattering and rain-induced halos. Special attention: avoid blue-purple artifacts on the right-side vehicle, ensure license plate regions are not overexposed; keep pedestrian clothing colors natural without oversaturation due to thermal radiation. Adjust overall tone from cold blue to neutral-warm, maintaining rainy wetness while improving clarity, so the fusion result retains infrared detection advantages while possessing authentic visible-light visual experience.')
    parser.add_argument("--prompt", type=str, default='This is a natural, photorealistic fusion image based on visible light and infrared imagery. It aims to eliminate degradation artifacts in the input and restore realistic colors along with high-definition texture details. The image remains clean and natural, free of infrared artifacts, with all elements strictly aligned.')
    parser.add_argument("--src_imgs", type=str, nargs='+', default=['demo_dataset/input/ir/i00000.png',"demo_dataset/input/vi/v00000.png"], help="传入多张参考图的路径，用空格分隔")
    
    parser.add_argument("--cfg_prompt", type=str, default="blurry, low quality, low resolution, distorted, deformed, broken content, missing parts, damaged details, \
                        artifacts, glitch, noise, pixelated, grainy, compression artifacts, bad composition, wrong proportion, incomplete editing, unfinished, unedited areas.")
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=680)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--output', type=str, default='output')

    args = parser.parse_args()

    accelerator = Accelerator()
    message = [f"Hello this is GPU {accelerator.process_index}"]
    messages = gather_object(message)
    accelerator.print(f"Number of gpus: {accelerator.num_processes}")
    accelerator.print(messages)

    # ── 1. 手动构建模型结构 (替代 BUILDER.build 和 Config) ──
    accelerator.print("Loading tokenizer and models...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True, padding_side='right')
    lmm = Qwen2_5_VLForConditionalGeneration.from_pretrained(QWEN_PATH, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    transformer = SD3Transformer2DModel.from_pretrained(SD3_PATH, subfolder="transformer", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(SD3_PATH, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(SD3_PATH, subfolder="vae", torch_dtype=torch.bfloat16)

    model = Qwen2p5VLStableDiffusion3HF(
        transformer=transformer,
        train_scheduler=scheduler,
        test_scheduler=copy.deepcopy(scheduler),  # 使用深拷贝防止污染
        vae=vae,
        lmm=lmm,
        tokenizer=tokenizer,
        prompt_template=PROMPT_TEMPLATE,
        connector=CONNECTOR_CFG,
        num_queries=128,          # 请确保与你训练时的设置一致
        max_length=1024,
        freeze_lmm=True,
        freeze_transformer=True,
        use_activation_checkpointing=False,
    )

    # ── 2. 加载自定义 Checkpoint ──
    if args.checkpoint is not None:
        accelerator.print(f"Loading checkpoint from {args.checkpoint}...")
        # guess_load_checkpoint 已经从本地导入，不需要 xtuner
        state_dict = guess_load_checkpoint(args.checkpoint) 
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            accelerator.print(f"Unexpected parameters: {unexpected}")
            
    # 将模型放置到 accelerate 自动分配的设备上并转为 bfloat16
    model = model.to(device=accelerator.device, dtype=torch.bfloat16)
    model.eval()
    
    if accelerator.is_main_process:
        os.makedirs(args.output, exist_ok=True)

    generator = torch.Generator(device=model.device).manual_seed(args.seed)

    # ── 3. 处理多张输入图像 ──
    src_imgs = []
    for img_path in args.src_imgs:
        img = Image.open(img_path).convert('RGB')
        # 根据目标尺寸动态缩放，最大边至少达到 args.height/width
        src_imgs.append(_process_image(img, image_size=max(args.height, args.width)))
    
    # 获取真实处理后的图像高度和宽度
    ref_h, ref_w = src_imgs[0].shape[1], src_imgs[0].shape[2]

    # ── 4. 构建模型期望的数据结构 List[List[Tensor]] ──
    batch_size = 4
    prompts = [args.prompt.strip()] * batch_size
    image_pixel_srcs = [src_imgs] * batch_size

    # # ── 5. 执行生成 ──
    # accelerator.print(f"Generating {batch_size} images with size {ref_w}x{ref_h}...")
    # with torch.no_grad():
    #     images = model.generate(
    #         prompt=prompts, 
    #         cfg_prompt=[args.cfg_prompt] * batch_size, 
    #         pixel_values_src=image_pixel_srcs,
    #         cfg_scale=args.cfg_scale, 
    #         num_steps=args.num_steps,
    #         progress_bar=accelerator.is_main_process, # 仅主进程显示进度条
    #         generator=generator, 
    #         height=ref_h, # 使用动态高度
    #         width=ref_w   # 使用动态宽度
    #     )

# ── 5. 执行生成 (Tiled 分块处理) ──
    accelerator.print(f"Generating {batch_size} images with size {ref_w}x{ref_h} using Tiled Processing...")
    with torch.no_grad():
        images = tiled_generate(
            model=model,
            prompt=prompts, 
            cfg_prompt=[args.cfg_prompt] * batch_size, 
            pixel_values_src=image_pixel_srcs,
            original_h=ref_h,     # 传入原图的真实高度
            original_w=ref_w,     # 传入原图的真实宽度
            patch_size=512,       # 模型的舒适尺寸
            stride=256,           # 50% 的重叠率，保证拼接平滑
            cfg_scale=args.cfg_scale, 
            num_steps=args.num_steps,
            generator=generator
        )

    # ── 6. 后处理与保存 ──
    images = rearrange(images, 'b c h w -> b h w c')
    # 将模型输出的 [-1, 1] 区间转移到 CPU 并转为图像格式
    images = torch.clamp(127.5 * images.cpu() + 128.0, 0, 255).to(torch.uint8).numpy()
    
    if accelerator.is_main_process:
        for i, image in enumerate(images):
            save_path = os.path.join(args.output, f"case_{i}.png")
            Image.fromarray(image).save(save_path)
            accelerator.print(f"Saved: {save_path}")