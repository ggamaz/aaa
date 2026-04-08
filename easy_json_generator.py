import os
import json
import random

def generate_unified_dataset_json(root, vi_dir, ir_dir, target_dir, output_json_path):
    dataset_list = []
    
    # 获取输入文件名（假设以 vi 文件夹下的文件为准，且所有文件夹命名对齐）
    image_filenames = sorted(os.listdir(os.path.join(root, vi_dir)))
    
    # 预设的 Prompt 模板库 (实际应用中可以接入 COCO 标签进行动态生成)
    prompts_visual = [
        "This is a natural, photorealistic fusion image based on visible light and infrared imagery. It aims to eliminate degradation artifacts in the input and restore realistic colors along with high-definition texture details. The image remains clean and natural, free of infrared artifacts, with all elements strictly aligned.",
        "By fusing visible and infrared images, it produces a visually appealing result. It preserves true environmental colors, leverages infrared features to enhance edge sharpness, suppresses glare and noise, and presents a well-lit, natural visual appearance.",
        # "这是一张基于可见光和红外图像的自然写实风融合图像。旨在消除输入中的退化干扰，恢复逼真的色彩和高清纹理细节。画面保持干净、自然，无红外伪影，所有元素严格对齐。"
        # "融合可见光与红外图像，生成视觉观感极佳的图像。保留真实的环境色彩，利用红外特征强化边缘清晰度，抑制眩光与噪声，呈现光线充足的自然视觉效果。"
    ]
    
    prompts_downstream = [
        "This is a fusion image designed for downstream machine vision tasks. It prioritizes highlighting salient targets (such as pedestrians and vehicles), enhancing their edge sharpness and high-frequency features, while suppressing irrelevant background interference. It maximizes the preservation of semantic information beneficial for machine understanding.",
        "A fusion image tailored for downstream tasks like object detection. It deeply integrates infrared saliency with visible light texture, significantly enhancing the contrast between objects and background, ensuring extremely clear object boundaries without prioritizing the naturalness of colors for human vision.",
        # "这是一张面向机器视觉下游任务的融合图像。优先凸显显著目标（如行人、车辆），强化其边缘锐度与高频特征，抑制无关背景干扰。最大化保留有利于机器理解的语义信息。",
        # "为目标检测等下游任务定制的融合图。深度融合红外显著性与可见光纹理，大幅提升目标与背景的对比度，确保物体边界极其分明，不以人类视觉色彩的自然性为首要目标。"
    ]

    for filename in image_filenames:
        if not filename.endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        vi_path = f"{vi_dir}/{filename}"
        ir_path = f"{ir_dir}/{filename}"
        
        # 记录高质量 GT 的路径 (如果你只有一种 GT 图，将这里改为统一的 output_path 即可)
        target_visual_path = f"{target_dir}/{filename}"

        # 构建统一的字典结构
        item = {
            "input_v_image": vi_path,
            "input_ir_image": ir_path,
            "output_image": target_visual_path,
            "instruction": random.choice(prompts_visual),
            "instruction_downstream": random.choice(prompts_downstream)
        }
        
        dataset_list.append(item)

    # 导出为 JSON
    output_json_path = os.path.join(root, output_json_path)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_list, f, ensure_ascii=False, indent=4)
        
    print(f"成功生成 JSON 数据集！共整合了 {len(dataset_list)} 对图像数据。")

# 运行配置
if __name__ == "__main__":
    generate_unified_dataset_json(
        root = "/root/aaa/COCO/images",  # 这个参数在当前函数中未使用，可以根据需要调整
        vi_dir="val2017_degrade",
        ir_dir="val2017_ir_degrade",
        target_dir="val2017",
        output_json_path="fusion_dataset_degrade.json"
    )