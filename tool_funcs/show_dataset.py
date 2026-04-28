import os
from random import shuffle
from PIL import Image

def concat_images_horizontally(dir_list, output_dir, show_len=10, prefix="case", ext=".png"):
    """
    读取任意多个目录下同名图片，横向拼接并以固定名称保存。
    
    :param dir_list: 包含所有输入文件夹路径的列表
    :param output_dir: 输出文件夹路径
    :param show_len: 随机挑选处理的数量（如果<=0则处理全部）
    :param prefix: 保存文件名的前缀（默认 "case"）
    :param ext: 保存文件的扩展名（默认 ".png"）
    """
    if not dir_list:
        print("❌ 错误：输入目录列表为空！")
        return

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的常见图片格式
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

    # 以第一个目录为基准，获取里面所有的图片文件名
    base_dir = dir_list[0]
    if not os.path.exists(base_dir):
        print(f"❌ 错误：基础目录 {base_dir} 不存在！")
        return
        
    files_in_base = [f for f in os.listdir(base_dir) if f.lower().endswith(valid_extensions)]
    
    if not files_in_base:
        print(f"⚠️ 在 {base_dir} 中没有找到图片文件。")
        return

    processed_count = 0
    # 打乱文件列表，增加随机性
    shuffle(files_in_base)
    if show_len > 0:
        print(f"🔍 将随机选择 {show_len} 张图片进行拼接展示...")
        files_in_base = files_in_base[:show_len]
        
    for filename in files_in_base:
        # 构建当前文件在所有目录中的绝对路径
        all_paths = [os.path.join(d, filename) for d in dir_list]

        # 检查同名文件是否在【所有】提供的目录中都存在
        if all(os.path.exists(p) for p in all_paths):
            try:
                # 动态打开所有图片
                imgs = [Image.open(p) for p in all_paths]

                # 动态计算拼接后的总宽度和最大高度
                total_width = sum(img.size[0] for img in imgs)
                max_height = max(img.size[1] for img in imgs)

                # 创建一张新的空白图片 (RGB 模式，白色背景)
                new_img = Image.new('RGB', (total_width, max_height), (255, 255, 255))

                # 将图片依次粘贴到新图片上
                current_x = 0
                for img in imgs:
                    new_img.paste(img, (current_x, 0))
                    current_x += img.size[0]  # 更新 X 坐标偏移量

                # 保存拼接后的图片，使用固定命名 (例如: case_0.png)
                save_filename = f"{prefix}_{processed_count}{ext}"
                output_path = os.path.join(output_dir, save_filename)
                new_img.save(output_path)
                
                print(f"✅ 成功拼接: 原名 [{filename}] -> 保存为 [{save_filename}]")
                processed_count += 1
                
            except Exception as e:
                print(f"❌ 处理文件 {filename} 时出错: {e}")
        else:
            print(f"⏭️ 跳过 {filename}: 并非在所有给定的目录中都存在该文件")

    print(f"\n🎉 处理完成！共成功拼接了 {processed_count} 张图片。")

# ==========================================
# 在下方修改为您电脑上的实际文件夹路径
# ==========================================
if __name__ == "__main__":
    # 将您需要拼接的所有目录放在这个列表中（想拼几列就放几个路径）
    DIRECTORIES = [
        "/home/user/data/ssm/aaa/COCO/images/val2017",
        "/home/user/data/ssm/aaa/COCO/images/val2017_ir",
        "/home/user/data/ssm/aaa/COCO/images/val2017_degrade",
        "/home/user/data/ssm/aaa/COCO/images/val2017_ir_degrade",
        # "/path/to/another/folder", 
    ]
    
    OUTPUT_DIRECTORY = "./output_folder"

    concat_images_horizontally(
        dir_list=DIRECTORIES, 
        output_dir=OUTPUT_DIRECTORY, 
        show_len=10,         # 每次随机处理的数量，设为 -1 则处理全部
        prefix="case",       # 输出名称前缀
        ext=".png"           # 输出格式
    )