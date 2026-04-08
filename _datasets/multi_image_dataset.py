import os
import random
import math
import torch
import numpy as np
from PIL import Image
from .edit_datasets import CaptionDataset
# ⚠️ 优化 3：极其关键的防死锁指令！禁止 OpenCV 在子进程中开多线程
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# 1. 导入你的 Flare 基础算子 (确保 flare_light.py 在同级或正确路径)
from .alg_degrade.flare_light import detect_lights, to_f, to_u8, screen, make_glow, make_bokeh, make_streaks, make_ghosts, make_flare_veil, local_contrast_loss, add_film_grain

# 2. 导入 Real-ESRGAN 生成模糊核的依赖
from .degradation import circular_lowpass_kernel, random_mixed_kernels

class RandomFlareTransform:
    """CPU 端的 Flare 渲染模块"""
    def __init__(self, p=0.3, flare_strength=(0.8, 1.2), overexpose=(0.1, 0.4)):
        self.p = p
        self.flare_strength = flare_strength
        self.overexpose = overexpose

    def __call__(self, img: Image.Image) -> Image.Image:
        # 概率控制：如果不触发，直接返回原图
        if random.random() > self.p:
            return img
            
        img_arr = np.array(img.convert("RGB"))
        H, W = img_arr.shape[:2]
        img_cx, img_cy = W // 2, H // 2
        base = to_f(img_arr)
        
        flare_s = random.uniform(*self.flare_strength)
        overexp = random.uniform(*self.overexpose)

        # CPU 串行提取高光区域
        lights = detect_lights(img_arr)
        if not lights:
            lights = [{"pos": (W//2, H//3), "size": 30, "brightness": 220}]

        composite = base.copy()
        short_side = min(H, W)
        _col, _row = np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32)
        XX, YY = np.meshgrid(_col, _row)

        for L in lights:
            lx, ly = L["pos"]
            bf = min(1.0, L["brightness"] / 240.0)
            fi = flare_s * bf
            xx, yy = XX - lx, YY - ly

            composite = screen(composite, make_glow(H, W, lx, ly, int(short_side*0.1*fi), 0.55*fi), 0.9)
            composite = screen(composite, make_bokeh(H, W, lx, ly, int(short_side*0.38*fi), 0.15*fi), 0.6)
            composite = screen(composite, make_streaks(H, W, xx, yy, 6, int(short_side*0.45*fi), 0.3*fi), 0.85)
            composite = screen(composite, make_ghosts(H, W, lx, ly, img_cx, img_cy, 0.15*fi), 0.65)
            composite = screen(composite, make_flare_veil(H, W, lx, ly, 0.12*fi), 0.75)
            composite = local_contrast_loss(composite, xx, yy, int(short_side*0.22*fi), 0.4*fi)

        gamma = 1.0 - overexp * 0.35
        composite = np.power(np.clip(composite, 1e-6, 1), gamma) + overexp * 0.04
        composite = add_film_grain(np.clip(composite, 0, 1), sigma=0.012)

        return Image.fromarray(to_u8(composite))


# --------------------------------------------------------------------------------

import os
import random
import math
import torch
import numpy as np
from PIL import Image
from .edit_datasets import CaptionDataset

# 仅导入 Real-ESRGAN 生成模糊核的依赖
from .degradation import circular_lowpass_kernel, random_mixed_kernels

class MultiImageEditDataset(CaptionDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 准备 Real-ESRGAN 模糊核生成参数 (这部分纯标量/极小矩阵计算，CPU 瞬间完成)
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        self.kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.blur_sigma = [0.2, 3.0]
        self.sinc_prob = 0.1
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1

    def _generate_kernels(self):
        """CPU 预先生成极小的矩阵随机核，开销可忽略不计"""
        # 第一阶段模糊核
        k_size = random.choice(self.kernel_range)
        if random.random() < self.sinc_prob:
            kernel1 = circular_lowpass_kernel(random.uniform(math.pi/3, math.pi), k_size, pad_to=False)
        else:
            kernel1 = random_mixed_kernels(self.kernel_list, self.kernel_prob, k_size, self.blur_sigma, self.blur_sigma, [-math.pi, math.pi], [0.5, 4], [1, 2], None)
        kernel1 = np.pad(kernel1, ((21-k_size)//2, (21-k_size)//2))

        # 第二阶段模糊核
        k_size2 = random.choice(self.kernel_range)
        if random.random() < self.sinc_prob:
            kernel2 = circular_lowpass_kernel(random.uniform(math.pi/3, math.pi), k_size2, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(self.kernel_list, self.kernel_prob, k_size2, self.blur_sigma, self.blur_sigma, [-math.pi, math.pi], [0.5, 4], [1, 2], None)
        kernel2 = np.pad(kernel2, ((21-k_size2)//2, (21-k_size2)//2))

        # Sinc 振铃伪影核
        if random.random() < 0.8:
            sinc_kernel = circular_lowpass_kernel(random.uniform(math.pi/3, math.pi), random.choice(self.kernel_range), pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        return torch.FloatTensor(kernel1), torch.FloatTensor(kernel2), sinc_kernel

    def __getitem__(self, idx):
        if self.debug: idx = 0
        try:
            data_sample = self.data_list[idx]
            
            # --- 1. 读取 Target 和 Source ---
            target_image = Image.open(os.path.join(self.image_folder, data_sample['output_image'])).convert('RGB')
            visible_image = Image.open(os.path.join(self.image_folder, data_sample['input_v_image'])).convert('RGB')
            infrared_image = Image.open(os.path.join(self.image_folder, data_sample['input_ir_image'])).convert('RGB')
            
            kernel1, kernel2, sinc_kernel = self._generate_kernels()

            visible_pixel_values = self._process_image(visible_image)['pixel_values']
            infrared_pixel_values = self._process_image(infrared_image)['pixel_values']
            target_pixel_values = self._process_image(target_image)['pixel_values']

            # --- 4. 文本 Prompt 随机选择 ---
            prompt_visual = data_sample.get('instruction', 'fuse for visual quality')
            prompt_downstream = data_sample.get('instruction_downstream', 'fuse for downstream detection task')
            if random.random() < 0.5:
                selected_prompt, prompt_type = prompt_visual, 'visual'
            else:
                selected_prompt, prompt_type = prompt_downstream, 'downstream'

            # --- 5. 组装发往 GPU 的 Batch ---
            return dict(
                target_pixel_values=target_pixel_values,
                visible_pixel_values=visible_pixel_values,
                infrared_pixel_values=infrared_pixel_values,
                text=selected_prompt,
                prompt_type=prompt_type,
                kernel1=kernel1,
                kernel2=kernel2,
                sinc_kernel=sinc_kernel
            )
        except Exception as e:
            print(f"Error when reading {self.data_list[idx]}: {e}", flush=True)
            return self._retry()