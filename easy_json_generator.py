# import os
# import json
# import random
# import numpy as np
# import cv2
# from pycocotools.coco import COCO

# def generate_coco_unified_dataset(root, vi_dir, ir_dir, target_dir, mask_save_dir, coco_anno_path, output_json_path):
#     # 1. 初始化 COCO API
#     print("正在加载 COCO 标注文件，这可能需要几秒钟...")
#     coco = COCO(os.path.join(root, coco_anno_path))
    
#     # 创建保存 Mask 图像的文件夹
#     mask_save_full_path = os.path.join(root, mask_save_dir)
#     os.makedirs(mask_save_full_path, exist_ok=True)
    
#     dataset_list = []
#     image_filenames = sorted(os.listdir(os.path.join(root, vi_dir)))
    
#     # 融合任务 Prompt
#     prompts_visual = [
#         # --- 强调去噪与修复 (Degradation & Restoration) ---
#         "The input visible and infrared images are heavily degraded with noise. Please fuse them to restore a clean, high-fidelity image, recovering lost textures and suppressing all artifacts.",
#         "Remove the noise and interference from these multi-modal inputs. Generate a photorealistic fused image that presents clear details and natural illumination.",
#         "Act as an image restoration model. Fuse the noisy RGB and thermal inputs to synthesize a visually pleasing, noise-free image with vibrant colors and sharp structural edges.",
        
#         # --- 强调多模态特征互补 (Feature Complementarity) ---
#         "Combine the thermal salient features from the noisy infrared image with the rich color information from the visible light image. Output a naturally fused result without any degradation artifacts.",
#         "Extract the high-frequency structural details from the infrared spectrum and blend them seamlessly with the visible spectrum. Ensure the final fused image is highly natural and completely clean.",
        
#         # --- 简洁指令式 (Concise & Direct) ---
#         "Fuse these noisy visible and IR images into a clean, photorealistic RGB image.",
#         "Denoise and fuse the given multimodal image pair. Output a high-definition natural image.",
        
#         # --- 口语化/交互式提问 (Conversational/Interactive) ---
#         "Can you clean up the noise in these visible and infrared images and merge them into a single, high-quality photograph?",
#         "These input images have a lot of degradation. Please fix them and provide a naturally fused image that looks like a high-res photo."
#     ]
    
#     # 分割任务 Prompt 模板 (留出 {} 用于填入真实的 label)
#     prompts_seg_templates = [
#         # --- 强调在噪声中提取目标 (Robust Segmentation in Noise) ---
#         "Despite the heavy noise in the visible and infrared inputs, accurately locate and segment every {label}. Output a clean binary mask.",
#         "The inputs suffer from severe degradation. Rely on the complementary thermal and visual cues to precisely segment the {label} from the cluttered background. Generate the mask.",
#         "Overcome the noise interference in the multimodal pair and extract a highly accurate segmentation mask for the '{label}' category.",

#         # --- 强调多模态特征融合分割 (Multimodal Guided Segmentation) ---
#         "Utilize the thermal radiation patterns from the IR image and the structural details from the visible image to segment the {label}. Provide the resulting binary mask.",
#         "Based on the fused representations of the noisy visible and infrared images, perform dense prediction to mask out the {label}.",
        
#         # --- 简洁指令式 (Concise & Direct) ---
#         "Segment the {label} from these noisy RGB-IR images and output the mask.",
#         "Generate a binary mask for all instances of '{label}' using the given degraded visible and thermal inputs.",
#         "Find and mask the {label} in this noisy multimodal image pair.",

#         # --- 口语化/交互式提问 (Conversational/Interactive) ---
#         "Can you ignore the noise and show me exactly where the {label} is in these images? Please output the mask.",
#         "I need a mask for the {label} in this scene. Please use both the visible and infrared noisy images to find it accurately."
#     ]

#     for filename in image_filenames:
#         if not filename.endswith(('.png', '.jpg', '.jpeg')):
#             continue
            
#         # COCO 图片名通常是 000000123456.jpg，我们需要提取数字 ID 去匹配标注
#         try:
#             img_id = int(os.path.splitext(filename)[0])
#         except ValueError:
#             print(f"警告: 文件名 {filename} 无法转换为 COCO image ID，已跳过。")
#             continue

#         # 获取该图片的基本信息和所有标注
#         img_info = coco.loadImgs(img_id)
#         if not img_info:
#             continue
#         img_info = img_info[0]
        
#         ann_ids = coco.getAnnIds(imgIds=img_id)
#         anns = coco.loadAnns(ann_ids)
        
#         if not anns:
#             continue # 如果这张图没有任何 COCO 目标标注，跳过分割任务
            
#         # 获取该图片中存在的所有类别 ID
#         cat_ids = list(set([ann['category_id'] for ann in anns]))
#         cats = coco.loadCats(cat_ids)
        
#         # 基础路径定义
#         vi_path = f"{vi_dir}/{filename}"
#         ir_path = f"{ir_dir}/{filename}"
#         target_visual_path = f"{target_dir}/{filename}"

#         # 2. 按类别（Label）拆分任务并生成对应的 Mask
#         for cat in cats:
#             cat_id = cat['id']
#             cat_name = cat['name'] # 真实的 Label，例如 'person', 'car'
            
#             # 获取当前图片中，属于当前类别的所有标注
#             cat_ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[cat_id])
#             cat_anns = coco.loadAnns(cat_ann_ids)
            
#             # 渲染当前类别的二值化 Mask
#             mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
#             for ann in cat_anns:
#                 # coco.annToMask 返回的是 0 和 1 的矩阵
#                 mask = np.maximum(mask, coco.annToMask(ann))
            
#             # 保存 Mask 图像 (保存为 0 和 255 的灰度图方便查看和训练)
#             mask_filename = f"{os.path.splitext(filename)[0]}_{cat_name}.png"
#             mask_full_path = os.path.join(mask_save_full_path, mask_filename)
#             cv2.imwrite(mask_full_path, mask * 255)

#             # 动态生成带有真实 Label 的指令
#             seg_instruction = random.choice(prompts_seg_templates).format(label=cat_name)

#             # 3. 构建字典结构
#             item = {
#                 "input_v_image": vi_path,
#                 "input_ir_image": ir_path,
#                 "output_image": target_visual_path,
#                 "output_mask": f"{mask_save_dir}/{mask_filename}",
#                 "target_label": cat_name, # 显式记录类别名称，方便后续 debug 或分析
#                 "instruction_fusion": random.choice(prompts_visual),
#                 "instruction_segmentation": seg_instruction
#             }
#             dataset_list.append(item)

#     # 4. 导出 JSON
#     output_json_path = os.path.join(root, output_json_path)
#     with open(output_json_path, 'w', encoding='utf-8') as f:
#         json.dump(dataset_list, f, ensure_ascii=False, indent=4)
        
#     print(f"\n✅ 成功生成 JSON 数据集及 Masks！")
#     print(f"处理完成：共生成了 {len(dataset_list)} 条训练数据（包含按类别拆分的样本）。")
#     print(f"Mask 图像已保存在: {mask_save_full_path}")

# # 运行配置
# if __name__ == "__main__":
#     generate_coco_unified_dataset(
#         root="/home/user/data/ssm/COCO",  # 建议将 root 设为 COCO 的根目录
#         vi_dir="images/val2017_degrade",
#         ir_dir="images/val2017_ir_degrade",
#         target_dir="images/val2017",
#         mask_save_dir="images/val2017_generated_masks", # 代码会自动创建这个文件夹并把渲染好的 Mask 放进去
#         coco_anno_path="annotations/instances_val2017.json", # COCO 原生的标注文件路径
#         output_json_path="fusion_and_seg_dataset_labeled.json"
#     )


import os
import json
import random
import numpy as np
import cv2
from pycocotools.coco import COCO

def generate_coco_unified_dataset(root, vi_dir, ir_dir, target_dir, mask_save_dir, coco_anno_path, output_json_path):
    # 1. 初始化 COCO API
    print("正在加载 COCO 标注文件，这可能需要几秒钟...")
    coco = COCO(os.path.join(root, coco_anno_path))
    
    # 创建保存 Mask 图像的文件夹
    mask_save_full_path = os.path.join(root, mask_save_dir)
    os.makedirs(mask_save_full_path, exist_ok=True)
    
    dataset_list = []
    image_filenames = sorted(os.listdir(os.path.join(root, vi_dir)))
    
    # 融合任务 Prompt
    prompts_visual = [
        # --- 强调去噪与修复 (Degradation & Restoration) ---
        "The input visible and infrared images are heavily degraded with noise. Please fuse them to restore a clean, high-fidelity image, recovering lost textures and suppressing all artifacts.",
        "Remove the noise and interference from these multi-modal inputs. Generate a photorealistic fused image that presents clear details and natural illumination.",
        "Act as an image restoration model. Fuse the noisy RGB and thermal inputs to synthesize a visually pleasing, noise-free image with vibrant colors and sharp structural edges.",
        
        # --- 强调多模态特征互补 (Feature Complementarity) ---
        "Combine the thermal salient features from the noisy infrared image with the rich color information from the visible light image. Output a naturally fused result without any degradation artifacts.",
        "Extract the high-frequency structural details from the infrared spectrum and blend them seamlessly with the visible spectrum. Ensure the final fused image is highly natural and completely clean.",
        
        # --- 简洁指令式 (Concise & Direct) ---
        "Fuse these noisy visible and IR images into a clean, photorealistic RGB image.",
        "Denoise and fuse the given multimodal image pair. Output a high-definition natural image.",
        
        # --- 口语化/交互式提问 (Conversational/Interactive) ---
        "Can you clean up the noise in these visible and infrared images and merge them into a single, high-quality photograph?",
        "These input images have a lot of degradation. Please fix them and provide a naturally fused image that looks like a high-res photo."
    ]
    
    # 分割任务 Prompt 模板 (留出 {} 用于填入真实的 label)
    prompts_seg_templates = [
        # --- 强调在噪声中提取目标 (Robust Segmentation in Noise) ---
        "Despite the heavy noise in the visible and infrared inputs, accurately locate and segment every '{label}'. Output a clean binary mask.",
        "The inputs suffer from severe degradation. Rely on the complementary thermal and visual cues to precisely segment the '{label}' from the cluttered background. Generate the mask.",
        "Overcome the noise interference in the multimodal pair and extract a highly accurate segmentation mask for the '{label}' category.",

        # --- 强调多模态特征融合分割 (Multimodal Guided Segmentation) ---
        "Utilize the thermal radiation patterns from the IR image and the structural details from the visible image to segment the '{label}'. Provide the resulting binary mask.",
        "Based on the fused representations of the noisy visible and infrared images, perform dense prediction to mask out the '{label}'.",
        
        # --- 简洁指令式 (Concise & Direct) ---
        "Segment the '{label}' from these noisy RGB-IR images and output the mask.",
        "Generate a binary mask for all instances of '{label}' using the given degraded visible and thermal inputs.",
        "Find and mask the '{label}' in this noisy multimodal image pair.",

        # --- 口语化/交互式提问 (Conversational/Interactive) ---
        "Can you ignore the noise and show me exactly where the '{label}' is in these images? Please output the mask.",
        "I need a mask for the '{label}' in this scene. Please use both the visible and infrared noisy images to find it accurately."
    ]

    for filename in image_filenames:
        if not filename.endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        # COCO 图片名通常是 000000123456.jpg，我们需要提取数字 ID 去匹配标注
        try:
            img_id = int(os.path.splitext(filename)[0])
        except ValueError:
            print(f"警告: 文件名 {filename} 无法转换为 COCO image ID，已跳过。")
            continue

        # 获取该图片的基本信息
        img_info_list = coco.loadImgs(img_id)
        if not img_info_list:
            continue
        img_info = img_info_list[0]
        
        # 基础路径定义
        vi_path = f"{vi_dir}/{filename}"
        ir_path = f"{ir_dir}/{filename}"
        target_visual_path = f"{target_dir}/{filename}"

        # 2. 构建基础的图像字典结构
        item = {
            "input_vi_image": vi_path,
            "input_ir_image": ir_path,
            "task": []
        }

        # 3. 将【融合任务】作为一个 task 优先加入列表
        item["task"].append({
            "instruction": random.choice(prompts_visual),
            "output_image": target_visual_path,
            "type": "visual"
        })

        # 4. 获取该图片的所有标注，准备【分割任务】
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        if anns:
            # 获取该图片中存在的所有类别 ID
            cat_ids = list(set([ann['category_id'] for ann in anns]))
            cats = coco.loadCats(cat_ids)
            
            # 按类别拆分任务并生成对应的 Mask
            for cat in cats:
                cat_id = cat['id']
                cat_name = cat['name']
                
                # 获取当前图片中，属于当前类别的所有标注
                cat_ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[cat_id])
                cat_anns = coco.loadAnns(cat_ann_ids)
                
                # 渲染当前类别的二值化 Mask
                mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
                for ann in cat_anns:
                    mask = np.maximum(mask, coco.annToMask(ann))
                
                # 保存 Mask 图像
                mask_filename = f"{os.path.splitext(filename)[0]}_{cat_name}.png"
                mask_full_path = os.path.join(mask_save_full_path, mask_filename)
                cv2.imwrite(mask_full_path, mask * 255)

                # 生成带有真实 Label 的指令，并作为独立的 task 加入列表
                seg_instruction = random.choice(prompts_seg_templates).format(label=cat_name)
                item["task"].append({
                    "instruction": seg_instruction,
                    "output_image": f"{mask_save_dir}/{mask_filename}",
                    "type": "segmentation"
                })

        # 5. 将整合好所有任务的单个图像字典存入总列表
        dataset_list.append(item)

    # 6. 导出 JSON
    output_json_path = os.path.join(root, output_json_path)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_list, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ 成功生成 JSON 数据集及 Masks！")
    print(f"处理完成：共生成了 {len(dataset_list)} 对图像数据（每对包含1个融合任务及相关的分割子任务）。")
    print(f"Mask 图像已保存在: {mask_save_full_path}")

# 运行配置
if __name__ == "__main__":
    generate_coco_unified_dataset(
        root="/home/user/data/ssm/aaa/COCO",
        vi_dir="images/val2017_degrade",
        ir_dir="images/val2017_ir_degrade",
        target_dir="images/val2017",
        mask_save_dir="images/val2017_generated_masks", 
        coco_anno_path="annotations/instances_val2017.json", 
        output_json_path="val2017_unified_dataset.json"
    )