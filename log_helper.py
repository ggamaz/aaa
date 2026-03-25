from PIL import Image, ImageDraw, ImageFont
import torch
import textwrap
import os


def print_log(msg, logger=None):
    print(msg, flush=True)


# ── Visualization Helper ──────────────────────────────────────────────
@torch.no_grad()
def log_training_images_oneline(model, batch, step, output_dir, image_size=512):
    """
    提取当前 batch 的第一个样本，将其对应的文本渲染成图像，并与输入参考图、生成的图像横向拼接。
    保持输入图像的真实比例，避免强制拉伸。
    """
    model.eval()
    
    try:
        text = batch['texts'][0]
        refs = batch['pixel_values_src'][0] # 这是第一条数据的参考图列表 (元素为 [C, H, W] 的 Tensor)
        
        # 【修复核心 1】动态获取输入图像的真实高宽
        ref_h, ref_w = refs[0].shape[1], refs[0].shape[2]
        
        # 1. 绘制文本 Prompt 图像，高度与参考图一致，宽度固定为 image_size
        prompt_img = Image.new('RGB', (image_size, ref_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(prompt_img)
        wrapped_text = "\n".join(textwrap.wrap(text, width=45))
        
        try:
            font = ImageFont.load_default(size=18)
        except TypeError:
            font = ImageFont.load_default()
            
        draw.text((20, 20), wrapped_text, fill=(0, 0, 0), font=font, spacing=8)
        
        # 2. 获取两张输入图像
        pil_refs = []
        for ref_tensor in refs:
            # ref_tensor 是 [C, H, W] 范围 [-1, 1]，转回 PIL
            ref_np = torch.clamp(127.5 * ref_tensor.cpu() + 128.0, 0, 255).byte().permute(1, 2, 0).numpy()
            # 【修复核心 2】取消强行的 .resize((image_size, image_size))，保持真实形状
            img = Image.fromarray(ref_np)
            pil_refs.append(img)
            
        while len(pil_refs) < 2:
            # 补充的白底图也使用真实的参考图尺寸
            pil_refs.append(Image.new('RGB', (ref_w, ref_h), color=(255, 255, 255)))
            
        # 3. 生成输出图像
        gen_out = model.generate(
            prompt=[text],
            cfg_prompt=['blurry, low quality, low resolution, distorted, deformed, broken content, missing parts, damaged details, artifacts, glitch, noise, pixelated, grainy, compression artifacts, bad composition, wrong proportion, incomplete editing, unfinished, unedited areas.'],
            pixel_values_src=[refs], 
            cfg_scale=4.5,
            num_steps=20,
            # 【修复核心 3】生成时强制使用输入图像的真实宽高
            height=ref_h,
            width=ref_w,
            progress_bar=False
        )
        # gen_out 形状: [1, C, H, W]
        gen_tensor = gen_out[0].cpu()
        gen_np = torch.clamp(127.5 * gen_tensor + 128.0, 0, 255).byte().permute(1, 2, 0).numpy()
        gen_img = Image.fromarray(gen_np)
        
        # 4. 拼接图像: Prompt -> Ref 1 -> Ref 2 -> Generated
        images_to_concat = [prompt_img, pil_refs[0], pil_refs[1], gen_img]
        
        total_width = sum(img.size[0] for img in images_to_concat)
        max_height = max(img.size[1] for img in images_to_concat) # 正常情况下大家高度都是 ref_h
        
        concat_img = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in images_to_concat:
            concat_img.paste(img, (x_offset, 0))
            x_offset += img.size[0]
            
        # 保存图片
        sample_dir = os.path.join(output_dir, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        save_path = os.path.join(sample_dir, f"step_{step:06d}.jpg")
        concat_img.save(save_path)
        
    except Exception as e:
        print_log(f"Visualization failed at step {step}: {e}")
        
    finally:
        model.train() # 恢复到训练模式


@torch.no_grad()
def log_training_images_threeline(model, batch, step, output_dir, image_size=512):
    """
    提取当前 batch 的第一个样本，生成视觉融合与下游任务融合的图像。
    排版格式：
    Row 1: [Ref 1 (VI)]        [Ref 2 (IR)]
    Row 2: [Prompt Visual]     [Gen Visual]
    Row 3: [Prompt Downstream] [Gen Downstream]
    """
    model.eval()
    
    try:
        # 1. 获取两种 Prompt
        text_visual = batch['texts'][0]
        text_downstream = batch.get('texts_downstream', [None])[0]
        has_downstream = text_downstream is not None

        # 2. 获取输入参考图 (VI 和 IR)
        refs = batch['pixel_values_src'][0] 
        ref_h, ref_w = refs[0].shape[1], refs[0].shape[2]
        
        # 将 Tensor 转回 PIL 图像
        pil_refs = []
        for ref_tensor in refs:
            ref_np = torch.clamp(127.5 * ref_tensor.cpu() + 128.0, 0, 255).byte().permute(1, 2, 0).numpy()
            pil_refs.append(Image.fromarray(ref_np))
            
        while len(pil_refs) < 2:
            pil_refs.append(Image.new('RGB', (ref_w, ref_h), color=(255, 255, 255)))

        # 3. 构造批量生成的输入 (同时推断 Visual 和 Downstream)
        prompts_to_gen = [text_visual]
        refs_to_gen = [refs]
        
        if has_downstream:
            prompts_to_gen.append(text_downstream)
            refs_to_gen.append(refs) # 将同一组参考图喂给第二个 Prompt

        # 统一的负面提示词
        cfg_prompts = ['blurry, low quality, low resolution, distorted, deformed, artifacts, noise'] * len(prompts_to_gen)

        # 4. 生成图像 (Batch Size = 1 或 2)
        gen_outs = model.generate(
            prompt=prompts_to_gen,
            cfg_prompt=cfg_prompts,
            pixel_values_src=refs_to_gen, 
            cfg_scale=4.5,
            num_steps=20,
            height=ref_h,
            width=ref_w,
            progress_bar=False
        )

        gen_imgs = []
        for gen_tensor in gen_outs:
            gen_np = torch.clamp(127.5 * gen_tensor.cpu() + 128.0, 0, 255).byte().permute(1, 2, 0).numpy()
            gen_imgs.append(Image.fromarray(gen_np))

        # 5. 辅助函数：绘制包含文字的 PIL 图像 (宽度设为 ref_w 以对齐两列网格)
        def draw_text_image(text_content, prefix="Prompt"):
            img = Image.new('RGB', (ref_w, ref_h), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            # 根据图像宽度动态调整换行字符数，粗略按每个字符 8-10 像素计算
            wrap_width = max(20, ref_w // 12) 
            wrapped_text = "\n".join(textwrap.wrap(f"[{prefix}]\n{text_content}", width=wrap_width))
            try:
                font = ImageFont.load_default(size=18)
            except TypeError:
                font = ImageFont.load_default()
            draw.text((20, 20), wrapped_text, fill=(0, 0, 0), font=font, spacing=8)
            return img

        # 辅助函数：拼接单行图像
        def concat_row(images_list):
            total_width = sum(img.size[0] for img in images_list)
            max_height = max(img.size[1] for img in images_list)
            concat_img = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for img in images_list:
                concat_img.paste(img, (x_offset, 0))
                x_offset += img.size[0]
            return concat_img

        # 6. 组装网格行
        rows = []
        
        # 第一行：Ref 1 -> Ref 2
        row1_imgs = [pil_refs[0], pil_refs[1]]
        rows.append(concat_row(row1_imgs))

        # 第二行：Prompt Visual -> Gen Visual
        row2_imgs = [draw_text_image(text_visual, "Visual"), gen_imgs[0]]
        rows.append(concat_row(row2_imgs))

        # 第三行：Prompt Downstream -> Gen Downstream (如果存在)
        if has_downstream:
            row3_imgs = [draw_text_image(text_downstream, "Downstream"), gen_imgs[1]]
            rows.append(concat_row(row3_imgs))
            
        # 纵向拼接所有行
        total_height = sum(r.size[1] for r in rows)
        max_width = max(r.size[0] for r in rows)
        
        final_img = Image.new('RGB', (max_width, total_height))
        y_offset = 0
        for r in rows:
            final_img.paste(r, (0, y_offset))
            y_offset += r.size[1]
            
        # 7. 保存图片
        sample_dir = os.path.join(output_dir, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        save_path = os.path.join(sample_dir, f"step_{step:06d}.jpg")
        final_img.save(save_path)
        
    except Exception as e:
        print_log(f"Visualization failed at step {step}: {e}")
        
    finally:
        model.train()