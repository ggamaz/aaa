import os
from PIL import Image

def convert_rgb_to_grayscale(input_dir, output_dir):
    """
    将指定目录下的所有RGB图像转换为灰度图像，并保存到输出目录。

    :param input_dir: 输入目录路径
    :param output_dir: 输出目录路径
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 支持的图像格式
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_dir, filename)
            try:
                # 打开图像并转换为灰度
                img = Image.open(input_path).convert('L')
                # 构造输出路径，保持原文件名
                output_path = os.path.join(output_dir, filename)
                # 保存灰度图像
                img.save(output_path)
                print(f"转换完成: {filename}")
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")

if __name__ == "__main__":
    # 示例用法：将当前目录下的图像转换为灰度，并保存到 'grayscale' 子目录
    input_directory = "output"  # 当前目录
    output_directory = "grayscale"
    convert_rgb_to_grayscale(input_directory, output_dir=output_directory)