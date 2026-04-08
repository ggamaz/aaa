import os
from click import prompt
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils

# 导入你的 Dataset 类 (根据你实际使用的类名调整)
from _datasets.multi_image_dataset import MultiImageEditDataset
# 导入我们刚刚写的基于 Tensor 的 Transform
from _datasets.dynamic_transforms import HybridRealESRGANBatchTransform

def run_test():
    print("🚀 开始测试数据集加载与 GPU 动态退化流水线...")
    
    # ==========================================
    # 1. 配置并实例化 Dataset
    # ==========================================
    dummy_data_path = "/root/aaa/COCO/images/fusion_dataset_unified.json" 
    dummy_image_folder = "/root/aaa/COCO/images"
    
    dummy_prompt_template = {
        'IMG_START_TOKEN': '<img_start>',
        'IMG_CONTEXT_TOKEN': '<img_ctx>',
        'IMG_END_TOKEN': '<img_end>',
        'INSTRUCTION': '{input}'
    }

    def collate_fn(batch):
        collated = dict(
            pixel_values_src=[b['pixel_values_src'] for b in batch],  
            pixel_values    =[b['pixel_values']      for b in batch], 
            texts           =[b['text']              for b in batch], 
            prompt_types    =[b['prompt_type']       for b in batch], 
            # ⬇️ 增加打包 Kernel 的逻辑 (因为是单张 Tensor，可以直接 torch.stack)
            kernel1         =torch.stack([b['kernel1'] for b in batch]),
            kernel2         =torch.stack([b['kernel2'] for b in batch]),
            sinc_kernel     =torch.stack([b['sinc_kernel'] for b in batch])
        )
        return collated

    dataset = MultiImageEditDataset(
        data_path=dummy_data_path,
        image_folder=dummy_image_folder,
        image_size=512,               
        unit_image_size=32,
        image_process='dynamic',
        prompt_template=dummy_prompt_template,
        debug=False
    )
    print(f"✅ 数据集加载成功，共 {len(dataset)} 个样本。")

    # ==========================================
    # 2. 初始化 DataLoader
    # ==========================================
    batch_size = 1
    # 🚀 提速核心优化 2：榨干 DataLoader 并发性能
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8,               # 建议设置为 4 到 8（根据你的 CPU 核心数）
        pin_memory=True,             # 🌟 必须开启：锁页内存，CPU 到 GPU 的传输速度翻倍
        prefetch_factor=3,           # 🌟 必须开启：让每个 worker 提前准备好 3 个 Batch 等待
        drop_last=True
    )

    # ==========================================
    # 3. 初始化 BatchTransform (送入 GPU)
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  当前使用设备: {device}")
    
    # ⚠️ 优化 1：统一 extra_keys，与 Dataset 返回的键名严丝合缝对齐
    transform = HybridRealESRGANBatchTransform(
        hq_key="hq",
        extra_keys=["text", "prompt_type"], # 改为这两个
        queue_size=0, 
    )

    os.makedirs("test_outputs", exist_ok=True)
    
    for i in range(5):
        print(f"📦 正在处理第 {i+1} 个 Batch...")
        batch = next(iter(dataloader))
        
        src_list = batch['pixel_values_src']
        # 🚀 提速核心优化 3：异步拷贝数据到 GPU
        src_tensor = torch.cat(src_list, dim=0).to(device, non_blocking=True)
        hq_images = (src_tensor + 1.0) / 2.0  
        num_src_imgs = len(src_list)
        
        # 构造给 Transform 的输入字典
        trans_input = {
            "hq": hq_images,
            "kernel1": batch['kernel1'].repeat(num_src_imgs, 1, 1).to(device, non_blocking=True),
            "kernel2": batch['kernel2'].repeat(num_src_imgs, 1, 1).to(device, non_blocking=True),
            "sinc_kernel": batch['sinc_kernel'].repeat(num_src_imgs, 1, 1).to(device, non_blocking=True),
            "text": batch['text'],
            "prompt_type": batch['prompt_type']
        }
        
        # ⚡️ 在 GPU 上执行并行退化操作
        out_batch = transform(trans_input)
        
        # 提取 Ground Truth (未退化的源图) 和 Low Quality (退化后的源图)
        gt_tensor = out_batch["hq"]
        
        # 2. 退化图现在的 key 是 "pixel_values_src"
        # 并且为了完美模拟你的 Dataset，它被包在了一个列表里，所以我们要加 [0] 把它取出来
        lq_tensor = out_batch["pixel_values_src"][0]
        
        # ==========================================
        # 5. 可视化并保存结果
        # ==========================================
        # 拼接：上面一排是原始的 source_images，下面一排是对齐的退化后图像
        comparison = torch.cat([gt_tensor, lq_tensor], dim=0)
        
        save_path = f"test_outputs/batch_test_result_{i}.jpg"
        # nrow 设为源图的数量，这样排版刚好是上下一一对应
        vutils.save_image(comparison, save_path, nrow=len(src_list), normalize=False)
        print(f"✅ 可视化结果已保存至: {save_path}")

    print("🎉 测试流程结束！请查看 test_outputs 目录下的图片以确认效果。")

if __name__ == "__main__":
    run_test()