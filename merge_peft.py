from xml.parsers.expat import model

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL

# 导入你本地的模型组件
from qwen2_5_sd3.transformer_sd3_dynamic import SD3Transformer2DModel
from qwen2_5_sd3.qwen2_5_vl_sd3_hf_dynamic_fusion import Qwen2p5VLStableDiffusion3HF

SD3_PATH  = "pretrain_ckpts/UniPic2-SD3.5M-Kontext-2B"
QWEN_PATH = "pretrain_ckpts/Qwen2.5-VL-3B-Instruct"
CHECKPOINT_PATH = "pretrain_ckpts/model.pt" # 你的带有 LoRA 权重的 checkpoint
OUTPUT_PATH = "pretrain_ckpts/merged_model.pt"
CONNECTOR_CFG = dict(
    hidden_size=2048,
    intermediate_size=11946,
    num_hidden_layers=6,
    _attn_implementation='flash_attention_2',
    num_attention_heads=32,
)
PROMPT_TEMPLATE = dict(
    IMG_START_TOKEN='<|vision_start|>',
    IMG_END_TOKEN='<|vision_end|>',
    IMG_CONTEXT_TOKEN='<|image_pad|>',
    IMG_START_TOKEN_FOR_GENERATION=False,
    SYSTEM='<|im_start|>system\n{system}<|im_end|>\n',
    INSTRUCTION='<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n',
    SUFFIX='<|im_end|>',
    SUFFIX_AS_EOS=True,
    SEP='\n',
    STOP_WORDS=['<|im_end|>', '<|endoftext|>'],
    GENERATION='Generate an image: {input}',
    CFG='Generate an image.',
    # CFG="blurry, low quality, low resolution, distorted, deformed, broken content, missing parts, damaged details, artifacts, glitch, noise, pixelated, grainy, compression artifacts, bad composition, wrong proportion, incomplete editing, unfinished, unedited areas."
)
def main():
    print("1. 加载基础组件...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
    lmm = Qwen2_5_VLForConditionalGeneration.from_pretrained(QWEN_PATH, torch_dtype=torch.bfloat16)
    transformer = SD3Transformer2DModel.from_pretrained(SD3_PATH, subfolder="transformer", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(SD3_PATH, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(SD3_PATH, subfolder="vae", torch_dtype=torch.bfloat16)

    print("2. 初始化 Qwen2p5VLStableDiffusion3HF 并加载权重...")
    model = Qwen2p5VLStableDiffusion3HF(
        transformer=transformer,
        train_scheduler=scheduler,
        test_scheduler=scheduler,
        vae=vae,
        lmm=lmm,
        tokenizer=tokenizer,
        prompt_template=PROMPT_TEMPLATE, 
        connector=CONNECTOR_CFG,
        pretrained_pth=None, # 这里会自动把你的 lora 权重 load 进 peft 模块
        num_queries=128,          # 请确保与你训练时的设置一致
        max_length=1024,
        freeze_lmm=True,
        lora_modules="auto",       # 开启 LMM LoRA
        freeze_transformer=False,  
        dit_lora_config=None,      # 使用预训练时的配置
        freeze_mq=False, 
    )
    print("初始模型加载完成。现在尝试加载权重...")
    missing, unexpected = model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'), strict=False) # 先加载 LoRA 权重到模型中
    if unexpected:
        print(f"警告: 加载时遇到 {len(unexpected)} 个意外键值，这些键值将被忽略: {unexpected}")
    
    print("3. 执行 PEFT 合并 (merge_and_unload)...")
    if hasattr(model.transformer, 'merge_and_unload'):
        print(" -> 合并 Transformer DoRA...")
        model.transformer = model.transformer.merge_and_unload()
    
    if hasattr(model.lmm, 'merge_and_unload'):
        print(" -> 合并 LMM LoRA...")
        model.lmm = model.lmm.merge_and_unload()

    print("4. 提取并彻底清洗权重...")
    # 【核心修改 1】：调用原始 nn.Module 的 state_dict，绕过你源码中丢弃主模型权重的过滤机制
    raw_state_dict = torch.nn.Module.state_dict(model)

    clean_state_dict = {}
    for k, v in raw_state_dict.items():
        # 剔除残留的幽灵 LoRA 键值（如 magnitude vector 等）
        if 'lora' in k.lower():
            continue
        
        # 【核心修改 2】：将 PEFT 包装产生的 .base_layer 嵌套重命名回标准结构
        new_key = k.replace(".base_layer", "")
        clean_state_dict[new_key] = v

    print(f"5. 正在保存绝对纯净的合并权重至 {OUTPUT_PATH} ...")
    torch.save(clean_state_dict, OUTPUT_PATH)
    print("合并彻底完成！你可以直接在无 LoRA 初始化的基线上加载此文件。")

if __name__ == "__main__":
    main()