import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T
from functools import partial
from einops import rearrange
from peft import LoraConfig, get_peft_model
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from qwen2_5_sd3.modeling_connector import ConnectorConfig, ConnectorEncoder
from qwen2_5_sd3.pipeline_stable_diffusion_3_dynamic import (
    StableDiffusion3Pipeline, calculate_shift,
)    
from contextlib import contextmanager
import lpips
import warnings
# ─────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────

IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD  = (0.26862954, 0.26130258, 0.27577711)

def guess_load_checkpoint(pth):
    checkpoint = torch.load(pth, map_location='cpu')
    if 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    elif 'model' in checkpoint:
        return checkpoint['model']
    return checkpoint


def print_log(msg, logger=None):
    print(msg, flush=True)


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def find_target_linear_names(model, num_lora_modules=-1,
                              lora_namespan_exclude=[], verbose=False):
    linear_cls    = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    names = []
    for name, module in model.named_modules():
        if any(ex in name for ex in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            names.append(name)
    if num_lora_modules > 0:
        names = names[-num_lora_modules:]
    if verbose:
        print(f"Found {len(names)} lora modules: {names}")
    return names


# ─────────────────────────────────────────────────────────────────────
#  Main Model
# ─────────────────────────────────────────────────────────────────────
class Qwen2p5VLStableDiffusion3HF(nn.Module):

    # Class-level constants
    _SKIP_PREFIXES  = frozenset(('vae.', 'ema.'))
    _LORA_KEYWORDS  = frozenset(('lora', 'ranknum'))

    def __init__(self,
                 transformer,
                 train_scheduler,
                 test_scheduler,
                 vae,
                 lmm,
                 tokenizer,
                 prompt_template,
                 connector,
                 num_queries=64,
                 vit_input_size=448,
                 max_length=1024,
                 freeze_lmm=True,
                 freeze_connector=False,
                 freeze_mq=False,
                 res_vit=False,
                 pretrained_pth=None,
                 use_activation_checkpointing=True,
                 lora_modules="auto",
                 freeze_transformer=True,
                 dit_lora_config=None,
                 unconditional=0.1,
                 weighting_scheme='none',
                 logit_mean=0.0,
                 logit_std=1.0,
                 mode="train"
                 ):
        super().__init__()

        # ── Device / dtype tracker ────────────────────────────────────
        self.register_buffer('_device_dtype_tracker', torch.zeros(1), persistent=False)

        # ── LMM ──────────────────────────────────────────────────────
        print_log("Setting up LMM ...")
        self.lmm = lmm
        self.freeze_lmm = freeze_lmm
        if freeze_lmm:
            self.lmm.requires_grad_(False)

        # ── Transformer (DiT) ─────────────────────────────────────────
        print_log("Setting up Transformer ...")
        self.transformer = transformer
        self.freeze_transformer = freeze_transformer
                    
        # ── VAE ───────────────────────────────────────────────────────
        print_log("Setting up VAE ...")
        self.vae = vae
        self.vae.requires_grad_(False)

        # ── LPIPS ─────────────────────────────────────────────────────
        if mode == 'train':
            print_log("Setting up LPIPS ...")
            # 推荐使用 VGG，感知效果最好。设为 eval 且无需梯度
            with warnings.catch_warnings():
                # Suppress warnings from lpips
                warnings.simplefilter("ignore")
                self.lpips_model = lpips.LPIPS(net='vgg', verbose=False).to(device=self.device)
                self.lpips_model.eval().requires_grad_(False)

        # ── Misc ──────────────────────────────────────────────────────
        self.res_vit               = res_vit
        self.weighting_scheme      = weighting_scheme
        self.logit_mean            = logit_mean
        self.logit_std             = logit_std
        self.use_activation_checkpointing = use_activation_checkpointing
        self.tokenizer             = tokenizer
        self.prompt_template       = prompt_template
        self.vit_input_size        = vit_input_size
        self.max_length            = max_length
        self.num_queries           = num_queries
        self.unconditional         = unconditional
        self.train_scheduler       = train_scheduler
        self.test_scheduler        = test_scheduler

        self.image_token_id = tokenizer.convert_tokens_to_ids(prompt_template['IMG_CONTEXT_TOKEN'])
        self.register_buffer('vit_mean', torch.tensor(IMAGE_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('vit_std',  torch.tensor(IMAGE_STD).view(1, 3, 1, 1),  persistent=False)
        
        # ── Connector + Projectors ────────────────────────────────────
        print_log("Setting up connector and projectors ...")
        self.connector = ConnectorEncoder(ConnectorConfig(**connector))

        hidden = self.llm_hidden_size
        conn_h = self.connector.config.hidden_size

        self.projector_1 = nn.Linear(hidden * 6, conn_h, dtype=self.dtype, device=self.device)
        self.projector_2 = nn.Linear(conn_h, self.transformer.config.pooled_projection_dim, dtype=self.dtype, device=self.device)
        self.projector_3 = nn.Linear(conn_h, self.transformer.config.joint_attention_dim, dtype=self.dtype, device=self.device)
        
        for proj in (self.projector_2, self.projector_3):
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

        self.meta_queries = nn.Parameter(torch.zeros(num_queries, hidden))
        nn.init.normal_(self.meta_queries, std=1 / math.sqrt(hidden))

        # ── Pretrained base weights ────────────────────────────────────────
        if pretrained_pth is not None:
            print_log(f'Loaded pretrained from {pretrained_pth}')
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            missing, unexpected = self.load_state_dict(pretrained_state_dict, strict=False)
            if unexpected:
                print_log(f"Unexpected keys: {unexpected}")

        # ── Freeze meta-queries and projectors if specified ─────────────────
        self.freeze_connector = freeze_connector
        if freeze_connector:
            print_log("freeze_connector")
            for m in (self.projector_1, self.projector_2, self.projector_3, self.connector):
                m.requires_grad_(False)
        
        self.freeze_mq = freeze_mq
        if freeze_mq:
            print_log("freeze_meta_query")
            self.meta_queries.requires_grad_(False)

        # ── Activation checkpointing ──────────────────────────────────
        if use_activation_checkpointing:
            self.gradient_checkpointing_enable()

        """ ── LoRA for LMM ───────────────────────────────────────────────"""
        self.lora_modules = lora_modules
        if lora_modules is not None:
            assert self.freeze_lmm, "LoRA requires freeze_lmm=True"
            print_log("Applying LoRA to LMM ...")
            self.llm.config.tie_word_embeddings = True
            if lora_modules == 'auto':
                lora_modules = find_target_linear_names(self.lmm)
            self.lmm.add_adapter(
                LoraConfig(
                    r=64, lora_alpha=128,
                    init_lora_weights="gaussian",
                    target_modules=lora_modules,
                    lora_dropout=0.05,
                )
            )

        """── LoRA for Transformer (DiT) ─────────────────────────────────────────"""
        if self.freeze_transformer:
            print_log("freeze_transformer=True: Applying LoRA to DiT ...")
            if torch.cuda.is_available():
                self.transformer.cuda()
            default_dora_cfg = dict(
                r=32, lora_alpha=64, lora_dropout=0.05,
                target_modules=[
                    "to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_q_proj", "add_v_proj", "linear",
                ],
                # use_dora=True,
            )
            if dit_lora_config is not None:
                default_dora_cfg.update(dit_lora_config)
            self.transformer = get_peft_model(self.transformer, LoraConfig(**default_dora_cfg))
        else:
            print_log("freeze_transformer=False: DiT full fine-tune.")

        self.truncat_warning = True
        self._infer_pipeline = None
    
    def _get_pipeline(self):
        if self._infer_pipeline is None:
            self._infer_pipeline = StableDiffusion3Pipeline(
                transformer=self.transformer,
                scheduler=self.test_scheduler,
                vae=self.vae,
                text_encoder=None, tokenizer=None,
                text_encoder_2=None, tokenizer_2=None,
                text_encoder_3=None, tokenizer_3=None,
            )
            self._infer_pipeline.set_progress_bar_config(disable=True)
        return self._infer_pipeline
    
    @contextmanager  
    def _inference_offload(self):
        """
        推理期间把 LMM 和 Connector offload 到 CPU。
        关键：move back 前先 empty_cache + synchronize，
        防止推理残留碎片导致 move back 时 OOM。
        """
        # 记录原始设备
        lmm_device = next(self.lmm.parameters()).device
        connector_device = next(self.connector.parameters()).device
        
        # offload
        self.lmm.cpu()
        self.connector.cpu()
        torch.cuda.empty_cache()
        
        try:
            yield
        finally:
            # ⚠️ 关键：先彻底清理推理残留，再 move back
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            # 移回 GPU
            self.lmm.to(lmm_device)
            self.connector.to(connector_device)
            torch.cuda.empty_cache()

    # ── Properties ───────────────────────────────────────────────────
    @property
    def llm(self):
        return self.lmm.model

    @property
    def llm_hidden_size(self) -> int:
        cfg = self.lmm.config
        if hasattr(cfg, 'hidden_size'):
            return cfg.hidden_size
        return cfg.text_config.hidden_size

    @property
    def device(self):
        return self._device_dtype_tracker.device

    @property
    def dtype(self):
        return self._device_dtype_tracker.dtype

    # ── Train / eval helpers ──────────────────────────────────────────
    def train(self: T, mode: bool = True) -> T:
        super().train(mode=mode)
        if self.vae is not None:
            self.vae.train(False)
        if not mode:
            self.gradient_checkpointing_disable()
        elif self.use_activation_checkpointing:   # ← 补上这一行
            self.gradient_checkpointing_enable()  # ← 对称地 re-enable
        return self

    def gradient_checkpointing_enable(self):
        if self.freeze_lmm or self.lora_modules is not None:
            self.llm.gradient_checkpointing_enable()
            if hasattr(self.llm, "enable_input_require_grads"):
                self.llm.enable_input_require_grads()
                
        self.transformer.enable_gradient_checkpointing()
        if hasattr(self.transformer, "enable_input_require_grads"):
            self.transformer.enable_input_require_grads()
        elif hasattr(self.transformer, "base_model") and hasattr(self.transformer.base_model, "enable_input_require_grads"):
            self.transformer.base_model.enable_input_require_grads()
            
        self.connector.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.transformer.disable_gradient_checkpointing()
        self.connector.gradient_checkpointing = False

    def state_dict(self, *args, **kwargs) -> dict:
        sd = super().state_dict(*args, **kwargs)
        freeze_lmm = self.freeze_lmm
        freeze_tf  = self.freeze_transformer

        def _keep(k):
            kl = k.lower()
            if any(kw in kl for kw in self._LORA_KEYWORDS):
                return True
            if any(k.startswith(p) for p in self._SKIP_PREFIXES):
                return False
            if k.startswith('lmm.') and freeze_lmm:
                return False
            if k.startswith('transformer.') and freeze_tf:
                return False
            return True

        return {k: v for k, v in sd.items() if _keep(k)}

    def load_state_dict(self, state_dict, strict=True):
        """
        与 state_dict() 的过滤逻辑对称：
        - vae.* / ema.*                          → 永远不保存，允许缺失
        - lmm.*(非LoRA) 且 freeze_lmm=True      → 不保存，允许缺失
        - transformer.*(非LoRA) 且 freeze_tf=True → 不保存，允许缺失
        其余 key（connector / projector / meta_queries / LoRA）必须存在。
        """
        missing, unexpected = super().load_state_dict(state_dict, strict=False)

        if strict:
            def _should_be_present(k):
                kl = k.lower()
                # LoRA 权重必须存在
                if any(kw in kl for kw in self._LORA_KEYWORDS):
                    return True
                # 跳过的模块本来就不保存
                if any(k.startswith(p) for p in self._SKIP_PREFIXES):
                    return False
                # 冻结的 lmm 基础权重不保存
                if k.startswith('lmm.') and self.freeze_lmm:
                    return False
                # 冻结的 transformer 基础权重不保存
                if k.startswith('transformer.') and self.freeze_transformer:
                    return False
                return True

            real_missing = [k for k in missing if _should_be_present(k)]
            if real_missing or unexpected:
                raise RuntimeError(
                    f"Error(s) in loading state_dict for {type(self).__name__}:\n"
                    f"  Missing key(s): {real_missing}\n"
                    f"  Unexpected key(s): {unexpected}"
                )

        return missing, unexpected

    # ── Core projection ───────────────────────────────────────────────

    def llm2dit(self, x):
        x          = self.connector(self.projector_1(x))
        pooled_out = self.projector_2(x.mean(1))
        seq_out    = self.projector_3(x)
        return pooled_out, seq_out

    # ── VAE helpers ───────────────────────────────────────────────────
    @torch.no_grad()
    def batch_pixels_to_latents(self, pixels):
        if isinstance(pixels, list):
            pixels = torch.stack(pixels).to(device=self.device, dtype=self.dtype)
        """Encode a list of pixel tensors to latents in one VAE forward."""
        z = self.vae.encode(pixels).latent_dist.sample()
        return (z - self.vae.config.shift_factor) * self.vae.config.scaling_factor

    def batch_latents_to_pixels(self, z: torch.Tensor):
        assert len(z.shape) == 4, "Expected latent tensor of shape (B, C, H', W')"
        z = z / self.vae.config.scaling_factor + self.vae.config.shift_factor
        return self.vae.decode(z).sample


    # ── Forward ───────────────────────────────────────────────────────

    def forward(self, data, mode='loss', curr_step=None):
        if mode == 'loss':
            return self.compute_loss(data_dict=data, curr_step=curr_step)
        raise NotImplementedError(f"mode={mode} is not supported")

    # ── Input preparation ─────────────────────────────────────────────

    def prepare_forward_input(self,
                              query_embeds,
                              input_ids=None,
                              image_embeds=None,
                              image_grid_thw=None,
                              attention_mask=None,
                              past_key_values=None):
        b, l, _ = query_embeds.shape
        assert l == self.num_queries

        attention_mask = attention_mask.to(device=self.device, dtype=torch.bool)

        mm_token_type_ids = torch.zeros_like(input_ids)
        if image_grid_thw is not None:
            mm_token_type_ids[input_ids == self.image_token_id] = 1

        position_ids, _ = self.lmm.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            second_per_grid_ts=None,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
        )

        last_pos  = position_ids[..., -1:]
        query_pos = last_pos + torch.arange(1, l + 1, device=self.device).view(1, 1, l)
        position_ids = torch.cat([position_ids, query_pos], dim=-1)

        input_ids      = torch.cat([input_ids, input_ids.new_zeros(b, l)], dim=1).contiguous()
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(b, l)], dim=1).contiguous()

        if past_key_values is not None:
            inputs_embeds = query_embeds
            position_ids  = position_ids[..., -l:]
        else:
            ctx_input_ids = input_ids[:, :-l]
            if image_embeds is None:
                inputs_embeds = self.llm.get_input_embeddings()(ctx_input_ids)
            else:
                inputs_embeds = torch.zeros(
                    *ctx_input_ids.shape, self.llm_hidden_size,
                    device=self.device, dtype=self.dtype)
                img_mask  = ctx_input_ids == self.image_token_id
                text_mask = ~img_mask
                inputs_embeds[img_mask] = image_embeds.contiguous().view(-1, self.llm_hidden_size)
                inputs_embeds[text_mask] = self.llm.get_input_embeddings()(ctx_input_ids[text_mask])
            inputs_embeds = torch.cat([inputs_embeds, query_embeds], dim=1)

        return dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

    def _truncate_inputs(self, inputs, max_len):
        out = {}
        if 'inputs_embeds' in inputs:
            actual_len = inputs['inputs_embeds'].shape[1]
            if actual_len >= max_len:
                # 记录截断情况
                print_log(f"⚠️ [警告] 发生截断！当前序列总长 {actual_len} > 允许的最大长度 {max_len}。")
                print_log(f"--- 丢失了最左侧的 {actual_len - max_len} 个 Token，可能会影响图像上下文或早期指令。")
            elif self.truncat_warning:
                print_log(f"✅ 当前序列总长 {actual_len} 在{max_len}内，无需截断。")
                self.truncat_warning = False  # 只打印一次警告
        for k, v in inputs.items():
            if k == 'inputs_embeds':
                out[k] = v[:, -max_len:, :]
            elif k in ('attention_mask', 'position_ids'):
                out[k] = v[..., -max_len:] if v is not None else v
            else:
                out[k] = v
        return out

    # ── Visual feature extraction ─────────────────────────────────────    
    @torch.no_grad()
    def get_semantic_features_dynamic(self, pixel_values):
        if isinstance(pixel_values, list):
            pixel_values = torch.stack(pixel_values).to(device=self.device, dtype=self.dtype)
        batch = F.interpolate(pixel_values, scale_factor=28/32, mode='bilinear')
        
        # 直接返回完整的 Tensor 特征，杜绝拆分成 list
        image_embeds, image_grid_thw = self.get_semantic_features(batch, resize=False)
        return image_embeds, image_grid_thw #[B*n, L, D], [B*n, 3]

    @torch.no_grad()
    def get_semantic_features(self, pixel_values, resize=True):
        vit_mean = self.vit_mean.to(device=self.device, dtype=self.dtype)
        vit_std  = self.vit_std.to(device=self.device, dtype=self.dtype)
        pixel_values = (pixel_values + 1.0) / 2
        pixel_values = (pixel_values - vit_mean) / vit_std

        if resize:
            pixel_values = F.interpolate(
                pixel_values,
                size=(self.vit_input_size, self.vit_input_size),
                mode='bilinear')

        B, C, H, W = pixel_values.shape
        patch_size          = self.lmm.config.vision_config.patch_size
        spatial_merge_size  = self.lmm.config.vision_config.spatial_merge_size
        temporal_patch_size = self.lmm.config.vision_config.temporal_patch_size

        grid_t = 1
        grid_h, grid_w = H // patch_size, W // patch_size

        pixel_values = pixel_values.view(B, 1, 1, C, H, W).expand(
            B, grid_t, temporal_patch_size, C, H, W)
        pixel_values = pixel_values.view(
            B, grid_t, temporal_patch_size, C,
            grid_h // spatial_merge_size, spatial_merge_size, patch_size,
            grid_w // spatial_merge_size, spatial_merge_size, patch_size)
        pixel_values = rearrange(
            pixel_values,
            'b t tp c h m p w n q -> (b t h w m n) (c tp p q)')

        image_grid_thw = torch.tensor(
            [[grid_t, grid_h, grid_w]], device=self.device, dtype=torch.long
        ).expand(B, -1)

        image_embeds = self.lmm.model.visual(pixel_values, grid_thw=image_grid_thw).last_hidden_state
        image_embeds = self.lmm.model.visual.merger(image_embeds)
        image_embeds = rearrange(image_embeds, '(b l) d -> b l d', b=B)

        return image_embeds, image_grid_thw

    # ── Prompt preparation ────────────────────────────────────────────

    @torch.no_grad()
    def prepare_text2image_prompts(self, texts):
        texts = [self.prompt_template['GENERATION'].format(input=t) for t in texts]
        texts = [self.prompt_template['INSTRUCTION'].format(input=t) for t in texts]
        return self.tokenizer(
            texts, add_special_tokens=True, return_tensors='pt',
            padding=True, padding_side='left').to(self.device)

    @torch.no_grad()
    def prepare_image2image_prompts(self, texts, num_refs=None, seq_len=None, ref_lens=None, **kwargs):
        prompts = []
        
        # 🚀 【CFG 终极兼容逻辑】：如果文本数量(包含负向)是图片记录的两倍，自动复制补齐
        if ref_lens is not None and len(texts) == 2 * len(ref_lens):
            ref_lens = ref_lens + ref_lens
        if num_refs is not None and len(texts) == 2 * len(num_refs):
            num_refs = num_refs + num_refs

        if ref_lens is not None:
            for text, lens in zip(texts, ref_lens):
                if isinstance(lens, int):
                    lens = [lens]
                tokens = ""
                for l_ in lens:
                    tokens += (self.prompt_template['IMG_START_TOKEN'] + 
                               self.prompt_template['IMG_CONTEXT_TOKEN'] * l_ + 
                               self.prompt_template['IMG_END_TOKEN'])
                prompts.append(self.prompt_template['INSTRUCTION'].format(input=f'{tokens}\n{text}'))
                
        # 【极速分支】：训练阶段纯 Tensor 化的高效逻辑
        else:
            for text, n_ref in zip(texts, num_refs):
                tokens = (self.prompt_template['IMG_START_TOKEN'] + 
                          self.prompt_template['IMG_CONTEXT_TOKEN'] * seq_len + 
                          self.prompt_template['IMG_END_TOKEN']) * n_ref
                prompts.append(self.prompt_template['INSTRUCTION'].format(input=f'{tokens}\n{text}'))
            
        return self.tokenizer(
            prompts, add_special_tokens=True, return_tensors='pt',
            padding=True, padding_side='left').to(self.device)
    
    # ── LLM forward helper ────────────────────────────────────────────

    def _llm_forward_and_merge(self, inputs):
        if self.use_activation_checkpointing and 'inputs_embeds' in inputs:
            inputs['inputs_embeds'].requires_grad_(True)
        output = self.llm(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = output.hidden_states
        num_layers    = len(hidden_states) - 1
        selected      = list(range(num_layers - 1, 0, -6))
        merged = torch.cat([hidden_states[i] for i in selected], dim=-1)
        return self.llm2dit(merged)

    # ── Loss functions ────────────────────────────────────────────────

    def text2image_loss(self, data_dict):
        # 强制转换为 Batch Tensor
        if 'image_latents' in data_dict:
            image_latents = data_dict['image_latents']
            if isinstance(image_latents, list):
                image_latents = torch.stack(image_latents)
            image_latents = image_latents.to(dtype=self.dtype, device=self.device)
        else:
            image_latents = self.batch_pixels_to_latents(data_dict['pixel_values'])
            
        b = image_latents.shape[0]
        texts = ['' if random.random() < self.unconditional else t for t in data_dict['texts']]
        text_inputs = self.prepare_text2image_prompts(texts)
        
        query_emb = self.meta_queries[None].expand(b, -1, -1)
        inputs = self.prepare_forward_input(query_embeds=query_emb, **text_inputs)

        max_len = self.max_length + self.num_queries
        inputs = self._truncate_inputs(inputs, max_len)

        pooled_out, seq_out = self._llm_forward_and_merge(inputs)
        
        batch_base_loss = self.compute_diff_loss(image_latents, pooled_out, seq_out)
        return batch_base_loss

    def image2image_loss(self, data_dict, soar_lambda=1.0, pixel_lambda=1.0, return_dict=False):
        pixel_values_src = data_dict['pixel_values_src']
        b, n, c, h, w = pixel_values_src.shape
        # target pixel2latent
        image_latents = self.batch_pixels_to_latents(data_dict['pixel_values'])
        flat_src_tensor = pixel_values_src.view(b * n, c, h, w)
        # condition pixel2latent
        flat_src_latents = self.batch_pixels_to_latents(flat_src_tensor)
        image_latents_src = flat_src_latents.view(b, n, *flat_src_latents.shape[1:])
        
        image_embeds, image_grid_thw = self.get_semantic_features_dynamic(flat_src_tensor)
        seq_len = image_embeds.shape[1]

        num_refs = [n] * b
        # 10% 的概率将任务控制文本替换为CFG prompt
        texts = [
            self.prompt_template['CFG'] if random.random() < self.unconditional else t 
            for t in data_dict['texts']
        ]
        text_inputs = self.prepare_image2image_prompts(texts, num_refs=num_refs, seq_len=seq_len)
        
        query_emb = self.meta_queries[None].expand(b, -1, -1)
        
        inputs = self.prepare_forward_input(
            query_embeds=query_emb,
            image_embeds=image_embeds,      
            image_grid_thw=image_grid_thw,  
            **text_inputs)

        # 截断
        max_len = self.max_length + n * seq_len + self.num_queries
        inputs = self._truncate_inputs(inputs, max_len)

        # LLM 与 DiT 计算 Loss
        pooled_out, seq_out = self._llm_forward_and_merge(inputs)
        
        # ── 调度中心：依次调用解耦模块 ───────────────────────────────────────
        
        # [A] 基础 Flow Matching 计算
        batch_base_loss, model_pred, noisy_input, sigmas, noise, timesteps = self.compute_diff_loss(
            image_latents, pooled_out, seq_out, cond_input=image_latents_src, return_internal=True
        )
        total_batch_loss = batch_base_loss.clone()

        # 提前初始化，防止对应开关关闭时引发 UnboundLocalError
        soar_loss = 0.0
        pixel_loss = 0.0

        # [B] SOAR 损失叠加 (按需触发)
        if soar_lambda > 0:
            soar_N = 1
            
            with torch.no_grad():
                # 构造全是无条件指令的 Batch
                uncond_texts = [self.prompt_template['CFG']] * b
                text_inputs_uncond = self.prepare_image2image_prompts(
                    uncond_texts, num_refs=num_refs, seq_len=seq_len)
                
                inputs_uncond = self.prepare_forward_input(
                    query_embeds=query_emb,
                    image_embeds=image_embeds,      # 源图像特征保持不变！
                    image_grid_thw=image_grid_thw,  
                    **text_inputs_uncond)
                inputs_uncond = self._truncate_inputs(inputs_uncond, max_len)
                
                # 获取合法的无条件投影特征
                uncond_pooled_out, uncond_seq_out = self._llm_forward_and_merge(inputs_uncond)
            # ----------------------------------------
            soar_loss = self.compute_soar_loss(
                image_latents, noisy_input, model_pred, sigmas, noise, timesteps,
                pooled_out, seq_out,
                uncond_pooled_embeds=uncond_pooled_out,
                uncond_prompt_embeds=uncond_seq_out,
                cond_input=image_latents_src, soar_N=soar_N
            )
            total_batch_loss = (total_batch_loss + soar_lambda * soar_loss) / (1.0 + soar_lambda * soar_N)

        # [C] 像素级结构损失叠加 (按需触发)
        if pixel_lambda > 0:
            pixel_values = data_dict.get('pixel_values')
            if isinstance(pixel_values, list):
                pixel_values = torch.stack(pixel_values)
            pixel_loss = self.compute_pixel_loss(noisy_input, model_pred, sigmas, pixel_values)
            total_batch_loss += pixel_lambda * pixel_loss

        # ---------------- 精准修改：返回标量并做 Detach 处理 ----------------
        if return_dict:
            # 辅助函数：统一转换为剥离计算图的均值标量，防止外层误算梯度
            def _to_scalar(v):
                return v.mean().detach() if isinstance(v, torch.Tensor) else v

            return {
                'loss': total_batch_loss.mean(), # 只有这个主 Loss 保留梯度计算图
                'loss_diff_log': _to_scalar(batch_base_loss),
                'loss_soar_log': _to_scalar(soar_loss),
                'loss_pixel_log': _to_scalar(pixel_loss)
            }
        return total_batch_loss.mean()
        

    @staticmethod
    def _select_texts_for_task(texts_batch, prompt_types_batch, task):
        selected = []
        for texts, p_types in zip(texts_batch, prompt_types_batch):
            for t, pt in zip(texts, p_types):
                if pt == task:
                    selected.append(t)
                    break
            else:
                selected.append(texts[0])
        return selected

    def compute_loss(self, data_dict, curr_step):
        if 'pixel_values_src' not in data_dict or 'prompt_types' not in data_dict:
            return {'loss_text2image': self.text2image_loss(data_dict)}
        if curr_step is not None:
            task = 'segmentation' if curr_step % 2 == 0 else 'visual'
        else:
            task  = random.choice(['visual', 'segmentation'])

        texts = self._select_texts_for_task(data_dict['texts'], data_dict['prompt_types'], task)
        task_dict = {
            'pixel_values_src': data_dict['pixel_values_src'],
            'texts': texts,
            'pixel_values': (data_dict['pixel_masks'] if task == 'segmentation' else data_dict['pixel_values']),
        }

        # ---------------- 精准修改：展开详细字典 ----------------
        base_key = 'loss_visual' if task == 'visual' else 'loss_segmentation'
        
        # 获取包含详细分类的字典
        losses = self.image2image_loss(task_dict, soar_lambda=1.0, pixel_lambda=1.0, return_dict=True)
        
        # 组装返回，带有 _log 后缀的键只做监控，不能做 backward
        loss_dict = {
            base_key: losses['loss'],                              # e.g., 'loss_visual' (主梯度)
            f"{base_key}_diff_log": losses['loss_diff_log'],       # e.g., 'loss_visual_diff_log'
            f"{base_key}_soar_log": losses['loss_soar_log'],       # e.g., 'loss_visual_soar_log'
            f"{base_key}_pixel_log": losses['loss_pixel_log']      # e.g., 'loss_visual_pixel_log'
        }
        return loss_dict
    

    # ── Diffusion loss ────────────────────────────────────────────────
    def get_sigmas(self, timesteps, n_dim=4):
        sigmas = self.train_scheduler.sigmas.to(device=self.device, dtype=self.dtype)
        schedule_timesteps = self.train_scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)

        matches      = (schedule_timesteps.unsqueeze(0) == timesteps.unsqueeze(1))
        step_indices = matches.long().argmax(dim=1)

        sigma = sigmas[step_indices].flatten()
        while sigma.ndim < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # ── 解耦模块 1: 基础 Flow Matching 损失 ───────────────────────────
    def compute_diff_loss(self, model_input, pooled_prompt_embeds, prompt_embeds, cond_input=None, return_internal=False):
        model_input = model_input.to(self.device, dtype=self.dtype)
        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]

        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme, batch_size=bsz,
            logit_mean=self.logit_mean, logit_std=self.logit_std)

        if self.train_scheduler.config.use_dynamic_shifting:
            seq_len = (model_input.shape[-2] * model_input.shape[-1]) // (self.transformer.config.patch_size ** 2)
            image_seq_lens = torch.full((bsz,), seq_len, dtype=self.dtype, device=self.device)
            mu = calculate_shift(
                image_seq_lens,
                self.train_scheduler.config.get("base_image_seq_len", 256),
                self.train_scheduler.config.get("max_image_seq_len", 4096),
                self.train_scheduler.config.get("base_shift", 0.5),
                self.train_scheduler.config.get("max_shift", 1.15))
            shift = torch.exp(mu) if self.train_scheduler.config.time_shift_type == "exponential" else mu
            sigmas = u.to(dtype=self.dtype, device=self.device)
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
            timesteps = sigmas * self.train_scheduler.config.num_train_timesteps
            sigmas = sigmas.view(-1, 1, 1, 1)
        else:
            indices = (u * self.train_scheduler.config.num_train_timesteps).long()
            indices = torch.clamp(indices, 0, self.train_scheduler.config.num_train_timesteps - 1)
            timesteps = self.train_scheduler.timesteps[indices].to(device=self.device)
            sigmas = self.train_scheduler.sigmas[indices].to(device=self.device, dtype=self.dtype).view(-1, 1, 1, 1)

        noisy_input = (1.0 - sigmas.float()) * model_input.float() + sigmas.float() * noise.float()
        noisy_input = noisy_input.to(dtype=self.dtype)

        model_pred = self.transformer(
            hidden_states=noisy_input, cond_hidden_states=cond_input, 
            encoder_hidden_states=prompt_embeds, pooled_projections=pooled_prompt_embeds,
            timestep=timesteps, return_dict=False)[0]
        
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.weighting_scheme, sigmas=sigmas)
        target = noise - model_input
        batch_loss_base = (weighting * (model_pred - target) ** 2).view(bsz, -1).mean(dim=1)

        if return_internal:
            # 返回基础loss，以及共享的计算图组件给其他loss使用
            return batch_loss_base, model_pred, noisy_input, sigmas, noise, timesteps
        
        return batch_loss_base

    # ── 解耦模块 2: SOAR 自我纠正损失 ───────────────────────────────
    def compute_soar_loss(self, model_input, noisy_input, model_pred, sigmas, noise, timesteps,
                          pooled_prompt_embeds, prompt_embeds, 
                          uncond_pooled_embeds, uncond_prompt_embeds, # 接收真实的无条件特征
                          cond_input=None, soar_cfg_scale=4.5, soar_N=1):
        bsz = model_input.shape[0]
        loss_corr_sum = 0.0
        
        with torch.no_grad():            
            v_uncond = self.transformer(
                hidden_states=noisy_input, 
                cond_hidden_states=cond_input,
                encoder_hidden_states=uncond_prompt_embeds, 
                pooled_projections=uncond_pooled_embeds,
                timestep=timesteps, 
                return_dict=False)[0]
            
            v_cfg = v_uncond + soar_cfg_scale * (model_pred.detach() - v_uncond)
            step_size = 1.0 / getattr(self.train_scheduler.config, 'num_train_timesteps', 1000)
            sigmas_t1 = torch.clamp(sigmas - step_size, min=0.0)
            z_hat_t1 = noisy_input + (sigmas_t1.float() - sigmas.float()) * v_cfg.float()
            z_hat_t1 = z_hat_t1.to(dtype=self.dtype)

        for _ in range(soar_N):
            alpha = torch.rand((bsz, 1, 1, 1), device=self.device, dtype=self.dtype)
            sigmas_t_prime = (1.0 - alpha) * sigmas_t1 + alpha * 1.0
            z_sigma_t_prime = (1.0 - alpha) * z_hat_t1 + alpha * noise
            timesteps_t_prime = sigmas_t_prime.view(-1) * self.train_scheduler.config.num_train_timesteps
            
            v_off = self.transformer(
                hidden_states=z_sigma_t_prime.to(dtype=self.dtype), cond_hidden_states=cond_input,
                encoder_hidden_states=prompt_embeds, pooled_projections=pooled_prompt_embeds,
                timestep=timesteps_t_prime.to(dtype=timesteps.dtype), return_dict=False)[0]
            
            v_corr = ((z_sigma_t_prime - model_input) / sigmas_t_prime).detach() 
            weighting_aux = compute_loss_weighting_for_sd3(weighting_scheme=self.weighting_scheme, sigmas=sigmas_t_prime)
            loss_corr_sum += (weighting_aux * (v_off - v_corr)**2).view(bsz, -1).mean(dim=1)
            
        return loss_corr_sum

    # ── 解耦模块 3: 像素空间结构损失 (单步重建) ───────────────────────
    def compute_pixel_loss(self, noisy_input, model_pred, sigmas, target_pixels, lpips_weight=1.0):
        bsz = noisy_input.shape[0]
        pixel_loss = torch.zeros(bsz, device=self.device, dtype=self.dtype)
        
        # 仅在低噪声阶段 (sigma < 0.5) 触发，防止高噪声时的随机性破坏训练
        valid_mask = sigmas.flatten() < 0.5 # 确保 mask 形状对齐 [B]
        
        if valid_mask.any():
            # 基于训练过程的单步重建 (Single-step Reconstruction) 反演 z_0
            pred_z0 = noisy_input[valid_mask] - sigmas[valid_mask].float() * model_pred[valid_mask]
            pred_z0 = pred_z0.to(dtype=self.dtype)
            pred_pixels = self.batch_latents_to_pixels(pred_z0)
            
            targets = target_pixels[valid_mask].to(self.device, dtype=self.dtype)
            
            l1_loss = F.l1_loss(pred_pixels, targets, reduction='none').mean(dim=[1, 2, 3])
            
            # 图像梯度计算 (提取边缘结构)
            grad_x_pred = pred_pixels[..., 1:] - pred_pixels[..., :-1]
            grad_y_pred = pred_pixels[..., 1:, :] - pred_pixels[..., :-1, :]
            grad_x_tgt = targets[..., 1:] - targets[..., :-1]
            grad_y_tgt = targets[..., 1:, :] - targets[..., :-1, :]
            
            grad_loss_x = F.l1_loss(grad_x_pred, grad_x_tgt, reduction='none').mean(dim=[1, 2, 3])
            grad_loss_y = F.l1_loss(grad_y_pred, grad_y_tgt, reduction='none').mean(dim=[1, 2, 3])
            
            # --- 新增 LPIPS 计算 ---
            if lpips_weight > 0:
                # lpips 返回 shape 为 [B_valid, 1, 1, 1]，通过 view(-1) 展平为 [B_valid]
                loss_lpips = self.lpips_model(pred_pixels, targets).view(-1)
            else:
                loss_lpips = 0.0
            
            # 融合重建损失与感知损失
            pixel_loss[valid_mask] = 0.5 * l1_loss + 0.5 * (grad_loss_x + grad_loss_y) + lpips_weight * loss_lpips

        return pixel_loss

    # ──── loss debug ───────────────────────────────────────────────
    def _debug_loss_check(self, mean_loss, weighting, model_pred, target, model_input, sigmas, batch_loss):
        """将原来巨大的 print 块抽离为一个辅助方法，保持代码整洁"""
        if mean_loss.item() > 3.0:
            print("\n" + "="*50)
            print(f"💥 [DEBUG] Loss 爆炸拦截: {mean_loss.item():.4f}")
            print(f"--- 权重加成 (weighting) Max: {weighting.max().item():.4f}, Min: {weighting.min().item():.4f}")
            print(f"--- 预测值 (model_pred) Max: {model_pred.max().item():.4f}, Min: {model_pred.min().item():.4f}")
            print(f"--- 真实值 (target) Max: {target.max().item():.4f}, Min: {target.min().item():.4f}")
            print(f"--- 模型输入 (model_input) Max: {model_input.max().item():.4f}, Min: {model_input.min().item():.4f}")
            print(f"--- 时间步 (sigmas) Max: {sigmas.max().item():.4f}, Min: {sigmas.min().item():.4f}")
            
            bad_idx = torch.argmax(batch_loss).item()
            print(f"--- 爆炸的样本索引: {bad_idx}")
            print("="*50 + "\n")


    # ── Generation ───────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self,
                 prompt,
                 cfg_prompt,
                 pixel_values_src=None,
                 cfg_scale=4.5,
                 c2fg_lambda=1.0,  # [精准修改] 新增 C2FG 的核心控制参数
                 num_steps=50,
                 generator=None,
                 height=512,
                 width=512,
                 progress_bar=True):
        assert len(prompt) == len(cfg_prompt)
        b = len(prompt)

        if pixel_values_src is not None:
            num_refs = [len(r) for r in pixel_values_src]
            pixel_values_src = [
                [img.to(device=self.device, dtype=self.dtype) for img in refs]
                for refs in pixel_values_src
            ]
            all_src = [img for refs in pixel_values_src for img in refs] #[B*n, C, H, W]
            image_embeds, image_grid_thw = self.get_semantic_features_dynamic(all_src)
            ref_lens = []
            ptr = 0
            for n in num_refs:
                ref_lens.append([image_embeds[ptr + i].shape[0] for i in range(n)])
                ptr += n
            
            text_inputs = self.prepare_image2image_prompts(
                prompt + cfg_prompt,
                num_refs=num_refs * 2,
                ref_lens=ref_lens * 2)
            text_inputs['image_embeds']   = torch.cat([image_embeds] * 2)
            text_inputs['image_grid_thw'] = torch.cat([image_grid_thw] * 2)

            cond_latents = [
                [self.batch_pixels_to_latents(img[None])[0] for img in refs]
                for refs in pixel_values_src] * 2
        else:
            text_inputs  = self.prepare_text2image_prompts(prompt + cfg_prompt)
            cond_latents = None

        query_emb = self.meta_queries[None].expand(2 * b, -1, -1) #[2*B, L, D]
        inputs    = self.prepare_forward_input(query_embeds=query_emb, **text_inputs)
        pooled_out, seq_out = self._llm_forward_and_merge(inputs)

        pipeline = self._get_pipeline()
        pipeline.set_progress_bar_config(disable=not progress_bar)
        
        # [精准修改] C2FG 动态回调逻辑
        callback_on_step_end = None
        if c2fg_lambda is not None and c2fg_lambda > 0:
            omega_0 = cfg_scale
            def dynamic_cfg_callback(pipe, step_index, timestep, callback_kwargs):
                # 计算进度 0.0 -> 1.0 (对应论文中的 1 - t/t_max)
                progress = step_index / max(1, (num_steps - 1))
                
                # C2FG 核心公式: w(t) = w0 * exp(lambda * progress)
                current_omega = omega_0 * math.exp(c2fg_lambda * progress)
                
                # 动态覆盖 Pipeline 内部的 guidance_scale
                pipe._guidance_scale = current_omega 
                return callback_kwargs
            
            callback_on_step_end = dynamic_cfg_callback

        samples = pipeline(
            height=height, width=width,
            guidance_scale=cfg_scale, # 作为初始 w0 传入，将在回调中被动态覆盖
            num_inference_steps=num_steps,
            prompt_embeds=seq_out[:b],
            pooled_prompt_embeds=pooled_out[:b],
            negative_prompt_embeds=seq_out[b:],
            negative_pooled_prompt_embeds=pooled_out[b:],
            generator=generator,
            output_type='latent',
            cond_latents=cond_latents,
            callback_on_step_end=callback_on_step_end, # [精准修改] 挂载回调
        ).images.to(self.dtype)
        
        # 返回前转换为像素
        with torch.no_grad():
            res = self.batch_latents_to_pixels(samples)
        return res
