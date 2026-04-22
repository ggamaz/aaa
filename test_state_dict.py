import torch

def save_pth_keys(pth_path, output_path):
    # 1. 加载 pth 文件 (map_location='cpu' 确保即使没有 GPU 也能加载)
    print(f"正在加载: {pth_path} ...")
    try:
        checkpoint = torch.load(pth_path, map_location='cpu')
    except Exception as e:
        print(f"加载失败: {e}")
        return

    # 2. 获取所有 Key
    if isinstance(checkpoint, dict):
        keys = list(checkpoint.keys())
        print(f"成功提取 {len(keys)} 个 Keys。")
    else:
        print(f"警告: 该文件内容不是字典格式，其类型为: {type(checkpoint)}")
        # 如果不是字典，尝试将其类型和内容转为字符串保存
        keys = [f"非字典结构，类型: {type(checkpoint)}", str(checkpoint)]

    # 3. 将 Key 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for key in keys:
            f.write(f"{key}\n")
    
    print(f"Keys 已保存至: {output_path}")

# ================= 配置区域 =================
# 请将下面的路径修改为你实际的文件路径
# input_pth_file = 'pretrain_ckpts/model.pt'    # 你的 .pth 文件路径
# output_txt_file = "origin_model_keys.txt"  
# input_pth_file = "pretrain_ckpts/merged_model.pt"
# output_txt_file = 'merged_model_keys.txt'     # 保存结果的文本文件路径
input_pth_file = "temp_merged.pth"
output_txt_file = 'temp_merged_model_keys.txt'     # 保存结果的文本文件路径
# ===========================================

if __name__ == '__main__':
    save_pth_keys(input_pth_file, output_txt_file)