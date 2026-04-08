from typing import Sequence

import torch
from torch.nn import functional as F
import numpy as np
import random
import copy

from alg_degrade.RealESRGAN.utils import USMSharp, filter2D
from alg_degrade.RealESRGAN.diffjpeg import DiffJPEG
from alg_degrade.RealESRGAN.degradation import random_add_gaussian_noise_pt, random_add_poisson_noise_pt

class HybridRealESRGANBatchTransform:
    def __init__(
        self,
        hq_key="pixel_values",          # 输入/输出的 HQ 键名
        lq_key="pixel_values_src",      # 输出的 LQ 键名
        extra_keys=["text", "prompt_type", "image_dir"], # 其他需要同步打乱的键
        queue_size=8,
        realesrgan_prob=0.8, 
        
        # Real-ESRGAN 经典参数
        resize_prob=[0.2, 0.7, 0.1], resize_range=(0.15, 1.5),
        gray_noise_prob=0.4, gaussian_noise_prob=0.5, noise_range=(1, 30),
        poisson_scale_range=(0.05, 3), jpeg_range=(30, 95),
        second_blur_prob=0.8, stage2_scale=1.2,
        resize_prob2=[0.3, 0.4, 0.3], resize_range2=(0.3, 1.2),
        gray_noise_prob2=0.4, gaussian_noise_prob2=0.5, noise_range2=(1, 25),
        poisson_scale_range2=(0.05, 2.5), jpeg_range2=(30, 95),
        use_sharpener=False, resize_back=True
    ):
        self.hq_key = hq_key
        self.lq_key = lq_key
        self.extra_keys = extra_keys
        self.queue_size = queue_size
        self.realesrgan_prob = realesrgan_prob
        self.queue = {}
        self.queue_ptr = 0
        
        self.jpeger = DiffJPEG(differentiable=False)
        self.resize_back = resize_back
        
        self.use_sharpener = use_sharpener
        if self.use_sharpener:
            self.usm_sharpener = USMSharp()
        else:
            self.usm_sharpener = None

        self.resize_prob = resize_prob
        self.resize_range = resize_range
        self.gray_noise_prob = gray_noise_prob
        self.gaussian_noise_prob = gaussian_noise_prob
        self.noise_range = noise_range
        self.poisson_scale_range = poisson_scale_range
        self.jpeg_range = jpeg_range

        self.second_blur_prob = second_blur_prob
        self.stage2_scale = stage2_scale
        self.resize_prob2 = resize_prob2
        self.resize_range2 = resize_range2
        self.gray_noise_prob2 = gray_noise_prob2
        self.gaussian_noise_prob2 = gaussian_noise_prob2
        self.noise_range2 = noise_range2
        self.poisson_scale_range2 = poisson_scale_range2
        self.jpeg_range2 = jpeg_range2

    @torch.no_grad()
    def _dequeue_and_enqueue(self, values: dict) -> dict:
        """队列机制，用于增加 Batch 内样本的多样性"""
        if len(self.queue):
            if set(values.keys()) != set(self.queue.keys()):
                raise ValueError(f"Key mismatch, input keys: {values.keys()}, queue keys: {self.queue.keys()}")
        else:
            for k, v in values.items():
                if not isinstance(v, (torch.Tensor, list)):
                    raise TypeError(f"Queue of type {type(v)} is not supported")
                if isinstance(v, list) and not isinstance(v[0], str):
                    raise TypeError("Only support queue for list of string")
                if isinstance(v, torch.Tensor):
                    size = (self.queue_size, *v.shape[1:])
                    self.queue[k] = torch.zeros(size=size, dtype=v.dtype, device=v.device)
                elif isinstance(v, list):
                    self.queue[k] = [None] * self.queue_size
            self.queue_ptr = 0

        for k, v in values.items():
            if self.queue_size % len(v) != 0:
                raise ValueError(f"Queue size {self.queue_size} should be divisible by batch size {len(v)} for key {k}")

        results = {}
        if self.queue_ptr == self.queue_size:
            idx = torch.randperm(self.queue_size)
            for k, q in self.queue.items():
                v = values[k]
                b = len(v)
                if isinstance(q, torch.Tensor):
                    q_shuf = q[idx]
                    results[k] = q_shuf[0:b, ...].clone()
                    q_shuf[0:b, ...] = v.clone()
                    self.queue[k] = q_shuf
                else:
                    q_shuf = [q[i] for i in idx]
                    results[k] = q_shuf[0:b]
                    for i in range(b):
                        q_shuf[i] = v[i]
                    self.queue[k] = q_shuf
        else:
            for k, q in self.queue.items():
                v = values[k]
                b = len(v)
                if isinstance(q, torch.Tensor):
                    q[self.queue_ptr : self.queue_ptr + b, ...] = v.clone()
                else:
                    for i in range(b):
                        q[self.queue_ptr + i] = v[i]
            results = copy.deepcopy(values)
            self.queue_ptr = self.queue_ptr + b

        return results

    @torch.no_grad()
    def __call__(self, batch):
        # 1. 提取 HQ 并自动完成 [-1, 1] 到 [0, 1] 的映射以满足物理光度学计算
        hq_original = batch[self.hq_key]
        hq = (hq_original + 1.0) / 2.0 
        
        if random.random() > self.realesrgan_prob:
            lq = hq.clone()
        else:
            # =========================================================
            # 执行 Real-ESRGAN 矩阵级物理退化
            # =========================================================
            self.jpeger.to(hq.device)
            kernel1 = batch["kernel1"].to(hq.device)
            kernel2 = batch["kernel2"].to(hq.device)
            sinc_kernel = batch["sinc_kernel"].to(hq.device)

            ori_h, ori_w = hq.size()[2:4]
            out = hq.clone()

            # ---------- Stage 1 ----------
            out = filter2D(out, kernel1)
            updown_type = random.choices(["up", "down", "keep"], self.resize_prob)[0]
            scale = np.random.uniform(1, self.resize_range[1]) if updown_type == "up" else (np.random.uniform(self.resize_range[0], 1) if updown_type == "down" else 1)
            out = F.interpolate(out, scale_factor=scale, mode=random.choice(["area", "bilinear", "bicubic"]))
            
            if np.random.uniform() < self.gaussian_noise_prob:
                out = random_add_gaussian_noise_pt(out, sigma_range=self.noise_range, clip=True, rounds=False, gray_prob=self.gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(out, scale_range=self.poisson_scale_range, gray_prob=self.gray_noise_prob, clip=True, rounds=False)
                
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range)
            out = self.jpeger(torch.clamp(out, 0, 1), quality=jpeg_p)

            # ---------- Stage 2 ----------
            if np.random.uniform() < self.second_blur_prob:
                out = filter2D(out, kernel2)

            stage2_scale = np.random.uniform(self.stage2_scale[0], self.stage2_scale[1]) if isinstance(self.stage2_scale, Sequence) else self.stage2_scale
            stage2_h, stage2_w = int(ori_h / stage2_scale), int(ori_w / stage2_scale)

            updown_type2 = random.choices(["up", "down", "keep"], self.resize_prob2)[0]
            scale2 = np.random.uniform(1, self.resize_range2[1]) if updown_type2 == "up" else (np.random.uniform(self.resize_range2[0], 1) if updown_type2 == "down" else 1)
            out = F.interpolate(out, size=(int(stage2_h * scale2), int(stage2_w * scale2)), mode=random.choice(["area", "bilinear", "bicubic"]))

            if np.random.uniform() < self.gaussian_noise_prob2:
                out = random_add_gaussian_noise_pt(out, sigma_range=self.noise_range2, clip=True, rounds=False, gray_prob=self.gray_noise_prob2)
            else:
                out = random_add_poisson_noise_pt(out, scale_range=self.poisson_scale_range2, gray_prob=self.gray_noise_prob2, clip=True, rounds=False)

            if np.random.uniform() < 0.5:
                out = F.interpolate(out, size=(stage2_h, stage2_w), mode=random.choice(["area", "bilinear", "bicubic"]))
                out = filter2D(out, sinc_kernel)
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
                out = self.jpeger(torch.clamp(out, 0, 1), quality=jpeg_p)
            else:
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
                out = self.jpeger(torch.clamp(out, 0, 1), quality=jpeg_p)
                out = F.interpolate(out, size=(stage2_h, stage2_w), mode=random.choice(["area", "bilinear", "bicubic"]))
                out = filter2D(out, sinc_kernel)

            if stage2_scale != 1 and self.resize_back:
                out = F.interpolate(out, size=(ori_h, ori_w), mode="bicubic")
                
            lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

        # =========================================================
        # 2. 队列处理 (注意：队列机制不允许存入 List[Tensor])
        # =========================================================
        # 将张量先平铺在一个临时字典中过队列
        temp_dict = {
            "hq_tensor": hq,
            "lq_tensor": lq
        }
        for k in self.extra_keys:
            if k in batch: 
                temp_dict[k] = batch[k]
                
        if hasattr(self, '_dequeue_and_enqueue') and self.queue_size > 0:
            temp_dict = self._dequeue_and_enqueue(temp_dict)
            
        # =========================================================
        # 3. 组装格式与还原 [-1, 1] 值域，完美匹配 Dataset 输出！
        # =========================================================
        # 将过完队列的数据转回 [-1, 1] 供 Diffusion 模型使用
        final_hq = (temp_dict["hq_tensor"] * 2.0) - 1.0
        final_lq = (temp_dict["lq_tensor"] * 2.0) - 1.0
        
        batch_out = {
            self.hq_key: final_hq,
            self.lq_key: [final_lq],  # 🚀 将 lq 重新包裹为 list，结构与 MultiImageEditDataset 一致
        }
        
        for k in self.extra_keys:
            if k in temp_dict: 
                batch_out[k] = temp_dict[k]
                
        return batch_out