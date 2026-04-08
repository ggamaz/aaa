import os
import argparse
import random
import math
from pathlib import Path
from tqdm import tqdm

import torch
from torch.nn import functional as F
import numpy as np
from PIL import Image
import torchvision.utils as vutils

# ================= 导入你的 Real-ESRGAN 依赖 =================
from RealESRGAN.utils import USMSharp, filter2D
from RealESRGAN.diffjpeg import DiffJPEG
from RealESRGAN.degradation import (
    circular_lowpass_kernel, 
    random_mixed_kernels,
    random_add_gaussian_noise_pt, 
    random_add_poisson_noise_pt
)

class RealESRGANDegradationPipeline:
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 1. 工具模块初始化
        self.jpeger = DiffJPEG(differentiable=False).to(self.device)
        self.usm_sharpener = USMSharp().to(self.device)
        self.use_sharpener = False # 默认不使用 USM 锐化，与原版一致可按需开启

        # 2. 模糊核生成基础参数 (严格对齐 realesrgan.py)
        self.kernel_range = [2 * v + 1 for v in range(3, 11)] # 7 到 21
        self.kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.blur_sigma = [0.2, 3.0]
        self.betag_range = [0.5, 4.0]
        self.betap_range = [1, 2]
        self.sinc_prob = 0.1

        self.kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.blur_sigma2 = [0.2, 1.5]
        self.betag_range2 = [0.5, 4.0]
        self.betap_range2 = [1, 2]
        self.sinc_prob2 = 0.1
        self.final_sinc_prob = 0.8
        
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1

        # 3. 物理退化参数 (严格对齐 batch_transform.py)
        self.resize_prob = [0.2, 0.7, 0.1] # up, down, keep
        self.resize_range = [0.15, 1.5]
        self.gray_noise_prob = 0.4
        self.gaussian_noise_prob = 0.5
        self.noise_range = [1, 30]
        self.poisson_scale_range = [0.05, 3]
        self.jpeg_range = [30, 95]

        self.second_blur_prob = 0.8
        self.stage2_scale = 1.2
        self.resize_prob2 = [0.3, 0.4, 0.3] # up, down, keep
        self.resize_range2 = [0.3, 1.2]
        self.gray_noise_prob2 = 0.4
        self.gaussian_noise_prob2 = 0.5
        self.noise_range2 = [1, 25]
        self.poisson_scale_range2 = [0.05, 2.5]
        self.jpeg_range2 = [30, 95]
        self.resize_back = True # 最后是否将图像插值回原图大小

    def _generate_kernels(self):
        """CPU: 生成第一阶段、第二阶段模糊核以及最终的 Sinc 振铃核"""
        # --- Kernel 1 ---
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            omega_c = np.random.uniform(np.pi / 3, np.pi) if kernel_size < 13 else np.random.uniform(np.pi / 5, np.pi)
            kernel1 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel1 = random_mixed_kernels(self.kernel_list, self.kernel_prob, kernel_size, 
                                           self.blur_sigma, self.blur_sigma, [-math.pi, math.pi], 
                                           self.betag_range, self.betap_range, None)
        pad_size = (21 - kernel_size) // 2
        kernel1 = np.pad(kernel1, ((pad_size, pad_size), (pad_size, pad_size)))

        # --- Kernel 2 ---
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            omega_c = np.random.uniform(np.pi / 3, np.pi) if kernel_size < 13 else np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(self.kernel_list2, self.kernel_prob2, kernel_size, 
                                           self.blur_sigma2, self.blur_sigma2, [-math.pi, math.pi], 
                                           self.betag_range2, self.betap_range2, None)
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # --- Sinc Kernel ---
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # 转换为 Tensor，扩充 Batch 维度并送到 GPU
        return (
            torch.FloatTensor(kernel1).unsqueeze(0).to(self.device), 
            torch.FloatTensor(kernel2).unsqueeze(0).to(self.device), 
            sinc_kernel.unsqueeze(0).to(self.device)
        )

    @torch.no_grad()
    def apply_degradation(self, hq_tensor):
        """
        执行完整的两阶段物理退化
        hq_tensor: [1, C, H, W] 范围在 [0, 1] 的原图张量
        """
        if self.use_sharpener:
            hq_tensor = self.usm_sharpener(hq_tensor)

        kernel1, kernel2, sinc_kernel = self._generate_kernels()
        ori_h, ori_w = hq_tensor.size()[2:4]
        out = hq_tensor.clone()

        # ===================================================================== #
        # 阶段 1：Real-ESRGAN 第一次退化 (Blur -> Resize -> Noise -> JPEG)
        # ===================================================================== #
        out = filter2D(out, kernel1)
        
        updown_type = random.choices(["up", "down", "keep"], self.resize_prob)[0]
        if updown_type == "up":
            scale = np.random.uniform(1, self.resize_range[1])
        elif updown_type == "down":
            scale = np.random.uniform(self.resize_range[0], 1)
        else:
            scale = 1
            
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        
        if np.random.uniform() < self.gaussian_noise_prob:
            out = random_add_gaussian_noise_pt(out, sigma_range=self.noise_range, clip=True, rounds=False, gray_prob=self.gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(out, scale_range=self.poisson_scale_range, gray_prob=self.gray_noise_prob, clip=True, rounds=False)
            
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range)
        out = torch.clamp(out, 0, 1)
        out = self.jpeger(out, quality=jpeg_p)

        # ===================================================================== #
        # 阶段 2：Real-ESRGAN 第二次退化 (Blur -> Resize -> Noise -> Sinc/JPEG)
        # ===================================================================== #
        if np.random.uniform() < self.second_blur_prob:
            out = filter2D(out, kernel2)

        stage2_scale = np.random.uniform(self.stage2_scale[0], self.stage2_scale[1]) if isinstance(self.stage2_scale, (list, tuple)) else self.stage2_scale
        stage2_h, stage2_w = int(ori_h / stage2_scale), int(ori_w / stage2_scale)

        updown_type2 = random.choices(["up", "down", "keep"], self.resize_prob2)[0]
        if updown_type2 == "up":
            scale = np.random.uniform(1, self.resize_range2[1])
        elif updown_type2 == "down":
            scale = np.random.uniform(self.resize_range2[0], 1)
        else:
            scale = 1
            
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, size=(int(stage2_h * scale), int(stage2_w * scale)), mode=mode)

        if np.random.uniform() < self.gaussian_noise_prob2:
            out = random_add_gaussian_noise_pt(out, sigma_range=self.noise_range2, clip=True, rounds=False, gray_prob=self.gray_noise_prob2)
        else:
            out = random_add_poisson_noise_pt(out, scale_range=self.poisson_scale_range2, gray_prob=self.gray_noise_prob2, clip=True, rounds=False)

        # Sinc 滤波器与最终 JPEG 压缩 (随机顺序以防出现扭曲伪影)
        if np.random.uniform() < 0.5:
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(out, size=(stage2_h, stage2_w), mode=mode)
            out = filter2D(out, sinc_kernel)
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(out, size=(stage2_h, stage2_w), mode=mode)
            out = filter2D(out, sinc_kernel)

        # 是否缩放回原尺寸
        if stage2_scale != 1 and self.resize_back:
            out = F.interpolate(out, size=(ori_h, ori_w), mode="bicubic")
            
        # 截断与量化 [0, 1]
        lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.0
        return lq


def process_directory(input_dir, output_dir, device="cuda"):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    images = [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in supported_exts]

    if not images:
        print(f"⚠️ 在 {input_dir} 下未找到受支持的图像文件！")
        return

    print(f"🚀 初始化 Real-ESRGAN 退化引擎 (设备: {device})...")
    pipeline = RealESRGANDegradationPipeline(device=device)

    print(f"📦 发现 {len(images)} 张图片，开始处理...")
    for img_file in tqdm(images):
        try:
            # 1. 使用 PIL 读取图像，严格遵循 HWC RGB uint8 -> BCHW float32 [0,1] 的处理流程
            img = Image.open(img_file).convert("RGB")
            # 将 [0, 255] 转换到 [0.0, 1.0]
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img_tensor = img_tensor.to(pipeline.device)

            # 2. 核心：执行 Real-ESRGAN 动态退化
            lq_tensor = pipeline.apply_degradation(img_tensor)

            # 3. 构造输出路径并保持原始的目录层级
            rel_path = img_file.relative_to(input_path)
            save_dest = output_path / rel_path
            save_dest.parent.mkdir(parents=True, exist_ok=True)
            
            # 使用 vutils 保存张量为图片
            vutils.save_image(lq_tensor, str(save_dest), normalize=False)

        except Exception as e:
            print(f"\n❌ 处理 {img_file.name} 时出错: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-ESRGAN 批量图像退化脚本")
    parser.add_argument("-i", "--input", type=str, required=True, help="输入的高清(HQ)图像目录")
    parser.add_argument("-o", "--output", type=str, required=True, help="输出的低清(LQ)图像目录")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备 (cuda 或 cpu)")
    
    args = parser.parse_args()
    
    process_directory(args.input, args.output, device=args.device)
    print("🎉 全部处理完成！")