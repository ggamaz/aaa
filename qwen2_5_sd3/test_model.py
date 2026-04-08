import argparse
from glob import glob
from PIL import Image
import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer
from transformer_sd3_dynamic import SD3Transformer2DModel
import math
import torch
from einops import rearrange
import os
from qwen2_5_vl_sd3_hf_dynamic_fusion import Qwen2p5VLStableDiffusion3HF, guess_load_checkpoint

# ── Image resize utility ──────────────────────────────────────────────
def resize_image(x, image_size, unit_image_size=32):
    w, h = x.size
    if w >= h and w >= image_size:
        target_w = image_size
        target_h = math.ceil(h * target_w / w / unit_image_size) * unit_image_size
    elif h > w and h >= image_size:
        target_h = image_size
        target_w = math.ceil(w * target_h / h / unit_image_size) * unit_image_size
    else:
        target_h = math.ceil(h / unit_image_size) * unit_image_size
        target_w = math.ceil(w / unit_image_size) * unit_image_size
    return x.resize((target_w, target_h))


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',  type=str,   default="/root/autodl-tmp/model.pt")
    parser.add_argument('--image',       type=str,   default=None)
    parser.add_argument('--prompt',      type=str,   default='a dog on the left and a cat on the right')
    parser.add_argument('--cfg_prompt',  type=str,   default='')
    parser.add_argument('--cfg_scale',   type=float, default=4.0)
    parser.add_argument('--num_steps',   type=int,   default=50)
    parser.add_argument('--height',      type=int,   default=512)
    parser.add_argument('--width',       type=int,   default=512)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--grid_size',   type=int,   default=2)
    parser.add_argument('--output',      type=str,   default='output.jpg')
    args = parser.parse_args()

    SD3_PATH  = "Skywork/UniPic2-SD3.5M-Kontext-2B"
    QWEN_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"

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
    )

    CONNECTOR_CFG = dict(
        hidden_size=2048,
        intermediate_size=11946,
        num_hidden_layers=6,
        _attn_implementation='flash_attention_2',
        num_attention_heads=32,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_PATH, trust_remote_code=True, padding_side='right')

    print("Loading LMM...")
    lmm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_PATH, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

    print("Loading transformer...")
    transformer = SD3Transformer2DModel.from_pretrained(
        SD3_PATH, subfolder="transformer", torch_dtype=torch.bfloat16)

    print("Loading scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        SD3_PATH, subfolder="scheduler")

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        SD3_PATH, subfolder="vae", torch_dtype=torch.bfloat16)

    print("Building model...")
    model = Qwen2p5VLStableDiffusion3HF(
        transformer=transformer,
        train_scheduler=scheduler,
        test_scheduler=scheduler,
        vae=vae,
        lmm=lmm,
        tokenizer=tokenizer,
        prompt_template=PROMPT_TEMPLATE,
        connector=CONNECTOR_CFG,
        num_queries=128,
        freeze_lmm=True,
        freeze_transformer=True,
        use_activation_checkpointing=False,
        pretrained_pth=None,
    ).cuda().bfloat16().eval()

    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}", flush=True)
        model.load_state_dict(guess_load_checkpoint(args.checkpoint), strict=False)

    # ── Inference ─────────────────────────────────────────────────────
    generator = torch.Generator(device=model.device).manual_seed(args.seed)
    bsz       = args.grid_size ** 2
    prompt     = [args.prompt]     * bsz
    cfg_prompt = [args.cfg_prompt] * bsz

    if args.image is not None:
        paths      = sorted(glob(f"{args.image}/*")) if os.path.isdir(args.image) else [args.image]
        ref_images = [Image.open(p).convert('RGB') for p in paths]
        ref_images = [resize_image(img, max(args.width, args.height), 32) for img in ref_images]

        width, height = ref_images[0].size if len(ref_images) == 1 else (args.width, args.height)

        pixel_values_src = [
            2 * (torch.from_numpy(np.array(img)).float() / 255) - 1
            for img in ref_images]
        pixel_values_src = [rearrange(t, 'h w c -> c h w') for t in pixel_values_src]
        pixel_values_src = [pixel_values_src] * bsz
    else:
        width, height    = args.width, args.height
        pixel_values_src = None

    samples = model.generate(
        prompt=prompt, cfg_prompt=cfg_prompt,
        pixel_values_src=pixel_values_src,
        cfg_scale=args.cfg_scale, num_steps=args.num_steps,
        generator=generator, height=height, width=width)

    samples = rearrange(samples, '(m n) c h w -> (m h) (n w) c', m=args.grid_size, n=args.grid_size)
    samples = torch.clamp(127.5 * samples + 128.0, 0, 255).cpu().to(torch.uint8).numpy()
    Image.fromarray(samples).save(args.output)
    print(f"Saved → {args.output}")