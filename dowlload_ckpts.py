from modelscope.hub.snapshot_download import snapshot_download
import os

def batch_download(model_list):
    """
    批量下载模型
    :param model_list: 包含模型配置的列表
    """
    for cfg in model_list:
        m_id = cfg.get("model_id")
        save_path = cfg.get("save_dir", "./models")
        
        # 获取过滤规则，如果没有定义则默认为 None (下载全部)
        allow = cfg.get("allow")
        ignore = cfg.get("ignore")
        
        print(f"\n" + "="*50)
        print(f" 开始任务: {m_id}")
        print(f" 目标路径: {os.path.abspath(save_path)}")
        
        try:
            downloaded_path = snapshot_download(
                model_id=m_id,
                local_dir=save_path,           # 直接定位到该目录
                allow_file_pattern=allow,
                ignore_file_pattern=ignore,
            )
            print(f"✅ 下载成功: {downloaded_path}")
        except Exception as e:
            print(f"❌ 下载失败 [{m_id}]: {e}")
        print("="*50)

if __name__ == "__main__":
    # --- 在这里配置你的下载清单 ---
    tasks = [
        {
            "model_id": "Skywork/UniPic2-SD3.5M-Kontext-2B",
            "save_dir": "./pretrain_ckpts/UniPic2-SD3.5M-Kontext-2B",
            "allow": ["config.json",
                      "model_index.json",
                      "vae/*.json", "vae/*.safetensors",
                      "transformer/*.json", "transformer/*.safetensors",
                      "scheduler/*.json",
                      ], # 仅演示：只下几个索引文件
            "ignore": ["*.bin", "*.png"] # 忽略大文件
        },
        {
            "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
            "save_dir": "./pretrain_ckpts/Qwen2.5-VL-3B-Instruct",
            "allow": ["*.json", "*.safetensors"],
            "ignore": ["*.bin", "*.png"]
        },
        {
            "model_id": "deepgenteam/DeepGen-1.0",
            "save_dir": "./pretrain_ckpts",
            "allow": ["model.pt"]
        }
    ]

    batch_download(tasks)
