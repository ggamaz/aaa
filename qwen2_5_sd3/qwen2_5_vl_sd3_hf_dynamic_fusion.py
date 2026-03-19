import os
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.modules.module import T
from functools import partial
from copy import deepcopy
from six.moves import map, zip
from einops import rearrange
from peft import LoraConfig
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from qwen2_5_sd3.modeling_connector import ConnectorConfig, ConnectorEncoder
from qwen2_5_sd3.pipeline_stable_diffusion_3_dynamic import StableDiffusion3Pipeline, calculate_shift


# ── 替换 xtuner / mmengine 依赖 ───────────────────────────────────────
def guess_load_checkpoint(pth):
    checkpoint = torch.load(pth, map_location='cpu')
    if 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    elif 'model' in checkpoint:
        return checkpoint['model']
    return checkpoint


def print_log(msg, logger=None):
    print(msg, flush=True)
# ─────────────────────────────────────────────────────────────────────


IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD  = (0.26862954, 0.26130258, 0.27577711)


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=False):
    linear_cls    = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []
    for name, module in model.named_modules():
        if any(ex in name for ex in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


class Qwen2p5VLStableDiffusion3HF(nn.Module):
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
                 freeze_mq=False,
                 res_vit=False,
                 pretrained_pth=None,
                 use_activation_checkpointing=False,
                 lora_modules='auto',
                 lora_rank=64,
                 lora_alpha=128,
                 freeze_transformer=True,
                 unconditional=0.1,
                 ema_cfg=None,
                 weighting_scheme='none',
                 logit_mean=0.0,
                 logit_std=1.0,
                 ):
        super().__init__()

        # ── LMM ──────────────────────────────────────────────────────
        self.lmm = lmm
        self.freeze_lmm = freeze_lmm
        if freeze_lmm:
            self.lmm.requires_grad_(False)

        # ── Transformer (DiT) ─────────────────────────────────────────
        self.transformer = transformer
        self.freeze_transformer = freeze_transformer
        if freeze_transformer:
            self.transformer.requires_grad_(False)

        # ── VAE ───────────────────────────────────────────────────────
        self.vae = vae
        self.vae.requires_grad_(False)

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

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(prompt_template['IMG_CONTEXT_TOKEN'])
        self.register_buffer('vit_mean', torch.tensor(IMAGE_MEAN), persistent=False)
        self.register_buffer('vit_std',  torch.tensor(IMAGE_STD),  persistent=False)

        # ── Connector + Projectors ────────────────────────────────────
        self.connector = ConnectorEncoder(ConnectorConfig(**connector))

        hidden = self.llm_hidden_size
        conn_h = self.connector.config.hidden_size

        self.projector_1 = nn.Linear(hidden * 6, conn_h)
        self.projector_2 = nn.Linear(conn_h, self.transformer.config.pooled_projection_dim)
        self.projector_3 = nn.Linear(conn_h, self.transformer.config.joint_attention_dim)

        # zero-init output projectors so training starts from identity
        for proj in (self.projector_2, self.projector_3):
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

        self.meta_queries = nn.Parameter(torch.zeros(num_queries, hidden))
        nn.init.normal_(self.meta_queries, std=1 / math.sqrt(hidden))

        self.freeze_mq = freeze_mq
        if freeze_mq:
            for m in (self.projector_1, self.projector_2, self.projector_3, self.connector):
                m.requires_grad_(False)
            self.meta_queries.requires_grad_(False)

        # ── Activation checkpointing ──────────────────────────────────
        if use_activation_checkpointing:
            self.gradient_checkpointing_enable()

        # ── LoRA ──────────────────────────────────────────────────────
        if lora_modules is not None:
            assert self.freeze_lmm, "LoRA requires freeze_lmm=True"
            self.llm.config.tie_word_embeddings = False
            if lora_modules == 'auto':
                lora_modules = find_target_linear_names(self.lmm)
            self.lmm.add_adapter(LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                init_lora_weights="gaussian",
                target_modules=lora_modules,
                lora_dropout=0.05,
            ))

        # ── Pretrained weights ────────────────────────────────────────
        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)
            print_log(f'Load pretrained weight from {pretrained_pth}')

        # ── EMA ───────────────────────────────────────────────────────
        self.ema_cfg = ema_cfg
        if ema_cfg is not None:
            self.ema = nn.ModuleDict()
            self.ema.steps = 0
            if not self.freeze_transformer:
                self.ema.update(dict(transformer=deepcopy(self.transformer)))
            if not self.freeze_mq:
                self.ema.update(dict(
                    projector_1=deepcopy(self.projector_1),
                    projector_2=deepcopy(self.projector_2),
                    projector_3=deepcopy(self.projector_3),
                    connector=deepcopy(self.connector),
                ))
                self.ema.register_buffer('meta_queries', deepcopy(self.meta_queries.data))
            self.ema.requires_grad_(False)

            if 'checkpoint' in ema_cfg:
                ema_state_dict = guess_load_checkpoint(ema_cfg['checkpoint'])
                self.ema.load_state_dict(ema_state_dict, strict=False)
                print_log(f"Load ema weight from {ema_cfg['checkpoint']}")

    # ── Properties ───────────────────────────────────────────────────

    @property
    def llm(self):
        """Qwen2.5-VL: language model body lives at lmm.model"""
        return self.lmm.model

    @property
    def llm_hidden_size(self) -> int:
        """Unified accessor: Qwen2.5-VL stores it under text_config."""
        cfg = self.lmm.config
        if hasattr(cfg, 'hidden_size'):
            return cfg.hidden_size
        return cfg.text_config.hidden_size

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    # ── Train / eval helpers ──────────────────────────────────────────

    def train(self: T, mode: bool = True) -> T:
        super().train(mode=mode)
        if self.vae is not None:
            self.vae.train(False)          # VAE always in eval
        if not mode:
            self.gradient_checkpointing_disable()
        return self

    def gradient_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        self.transformer.enable_gradient_checkpointing()
        self.connector.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.transformer.disable_gradient_checkpointing()
        self.connector.gradient_checkpointing = False

    def state_dict(self, *args, **kwargs) -> dict:
        sd = super().state_dict(*args, **kwargs)
        # exclude frozen / non-trainable blobs to keep checkpoints small
        return {k: v for k, v in sd.items()
                if not any(k.startswith(p) for p in ('vae.', 'lmm.', 'ema.'))}

    # ── EMA ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def ema_step(self):
        if self.ema_cfg is None:
            return
        steps           = self.ema.steps
        update_interval = self.ema_cfg.get('update_interval', 1)
        save_interval   = self.ema_cfg.get('save_interval', 1000)
        momentum        = self.ema_cfg.get('momentum', 0.99)
        alpha           = 1.0 - momentum          # lerp weight toward current params

        if steps % update_interval == 0 and steps > 0:
            if not self.freeze_mq:
                pairs = [
                    (self.ema.projector_1, self.projector_1),
                    (self.ema.projector_2, self.projector_2),
                    (self.ema.projector_3, self.projector_3),
                    (self.ema.connector,   self.connector),
                ]
                for ema_mod, base_mod in pairs:
                    for ep, bp in zip(ema_mod.parameters(), base_mod.parameters()):
                        ep.data.lerp_(bp.data.detach(), alpha)
                self.ema.meta_queries.data.lerp_(self.meta_queries.data.detach(), alpha)

            if not self.freeze_transformer:
                for ep, bp in zip(self.ema.transformer.parameters(), self.transformer.parameters()):
                    ep.data.lerp_(bp.data.detach(), alpha)

        if steps % save_interval == 0 and steps > 0:
            is_ddp         = dist.is_available() and dist.is_initialized()
            is_primary     = (not is_ddp) or dist.get_rank() == 0
            if is_primary:
                save_path = self.ema_cfg.get('save_path')
                torch.save(self.ema.state_dict(), save_path)
                print(f"EMA saved to {save_path} at step {steps}", flush=True)
            if is_ddp:
                dist.barrier()

        self.ema.steps = steps + 1

    # ── Core projection ───────────────────────────────────────────────

    def llm2dit(self, x):
        x          = self.connector(self.projector_1(x))
        pooled_out = self.projector_2(x.mean(1))
        seq_out    = self.projector_3(x)
        return pooled_out, seq_out

    # ── VAE helpers ───────────────────────────────────────────────────

    @torch.no_grad()
    def pixels_to_latents(self, x):
        z = self.vae.encode(x).latent_dist.sample()
        return (z - self.vae.config.shift_factor) * self.vae.config.scaling_factor

    @torch.no_grad()
    def latents_to_pixels(self, z):
        z = z / self.vae.config.scaling_factor + self.vae.config.shift_factor
        return self.vae.decode(z).sample

    # ── Forward ───────────────────────────────────────────────────────

    def forward(self, data, data_samples=None, mode='loss'):
        if mode == 'loss':
            self.ema_step()
            return self.compute_loss(data_dict=data)
        raise NotImplementedError(f"mode={mode} is not supported")

    def compute_loss(self, data_dict):
        losses = {}
        for data_type in ('text2image', 'image2image'):
            if data_type in data_dict:
                losses[f'loss_{data_type}'] = getattr(self, f'{data_type}_loss')(data_dict[data_type])
        if not losses:
            if 'pixel_values_src' in data_dict:
                losses['loss_image2image'] = self.image2image_loss(data_dict)
            else:
                losses['loss_text2image'] = self.text2image_loss(data_dict)
        return losses
    
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

        # ── 用原始序列（不含 query）调 get_rope_index ─────────────────────
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
        # position_ids: [3, b, orig_seq_len]

        # ── 手动为 query token 续上位置 ───────────────────────────────────
        last_pos  = position_ids[..., -1:]                                     # [3, b, 1]
        query_pos = last_pos + torch.arange(1, l + 1, device=self.device).view(1, 1, l)
        position_ids = torch.cat([position_ids, query_pos], dim=-1)            # [3, b, orig+l]

        # ── 拼上 query 占位 ───────────────────────────────────────────────
        input_ids      = torch.cat([input_ids,      input_ids.new_zeros(b, l)],      dim=1)
        attention_mask = torch.cat([attention_mask,  attention_mask.new_ones(b, l)], dim=1)

        # ── 构造 inputs_embeds ────────────────────────────────────────────
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
                inputs_embeds[img_mask]  = image_embeds.contiguous().view(-1, self.llm_hidden_size)
                inputs_embeds[text_mask] = self.llm.get_input_embeddings()(ctx_input_ids[text_mask])
            inputs_embeds = torch.cat([inputs_embeds, query_embeds], dim=1)

        return dict(inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values)

    # ── Visual feature extraction ─────────────────────────────────────
    @torch.no_grad()
    def get_semantic_features_dynamic(self, pixel_values):
        pixel_values = [F.interpolate(p[None], scale_factor=28/32, mode='bilinear') for p in pixel_values]
        image_embeds, image_grid_thw = multi_apply(self.get_semantic_features, pixel_values, resize=False)
        image_embeds    = [x[0] for x in image_embeds]
        image_grid_thw  = torch.cat(image_grid_thw, dim=0)
        return image_embeds, image_grid_thw
    
    @torch.no_grad()
    def get_semantic_features(self, pixel_values, resize=True):
        # pixel_values: [-1, 1]
        pixel_values = (pixel_values + 1.0) / 2     # [0, 1]
        pixel_values = pixel_values - self.vit_mean.view(1, 3, 1, 1)
        pixel_values = pixel_values / self.vit_std.view(1, 3, 1, 1)

        if resize:
            pixel_values = F.interpolate(pixel_values,
                                        size=(self.vit_input_size, self.vit_input_size),
                                        mode='bilinear')
        b, c, h, w = pixel_values.shape

        patch_size          = self.lmm.config.vision_config.patch_size
        spatial_merge_size  = self.lmm.config.vision_config.spatial_merge_size
        temporal_patch_size = self.lmm.config.vision_config.temporal_patch_size

        grid_t         = 1
        grid_h, grid_w = h // patch_size, w // patch_size

        pixel_values = pixel_values[:, None].expand(b, temporal_patch_size, c, h, w)
        pixel_values = pixel_values.view(
            b, grid_t, temporal_patch_size, c,
            grid_h // spatial_merge_size, spatial_merge_size, patch_size,
            grid_w // spatial_merge_size, spatial_merge_size, patch_size,
        )
        pixel_values = rearrange(
            pixel_values, 'b t tp c h m p w n q -> (b t h w m n) (c tp p q)')

        image_grid_thw = torch.tensor(
            [(grid_t, grid_h, grid_w)] * b, device=self.device, dtype=torch.long)

        # ── 关键修复：先拿 unmerged hidden states，再显式调 merger ──────────
        # last_hidden_state shape: [b * grid_t * grid_h * grid_w, d]  (未合并)
        image_embeds = self.lmm.model.visual(
            pixel_values, grid_thw=image_grid_thw).last_hidden_state

        # merger shape 变换: [N*merge², d] → [N, d_model]
        # 其中 N = b * grid_t * (grid_h // merge) * (grid_w // merge)
        image_embeds = self.lmm.model.visual.merger(image_embeds)   # ← 这行是关键

        # 合并后每张图 token 数 = grid_t * (grid_h//merge) * (grid_w//merge)
        image_embeds = rearrange(image_embeds, '(b l) d -> b l d', b=b)

        return image_embeds, image_grid_thw

    # ── Prompt preparation ────────────────────────────────────────────
    @torch.no_grad()
    def prepare_text2image_prompts(self, texts):
        texts = [self.prompt_template['GENERATION'].format(input=t) for t in texts]
        texts = [self.prompt_template['INSTRUCTION'].format(input=t) for t in texts]
        return self.tokenizer(texts, add_special_tokens=True, return_tensors='pt',
                              padding=True, padding_side='left').to(self.device)

    @torch.no_grad()
    def prepare_image2image_prompts(self, texts, num_refs, ref_lens):
        prompts, cnt = [], 0
        for text, num_ref in zip(texts, num_refs):
            tokens = ''
            for _ in range(num_ref):
                tokens += (self.prompt_template['IMG_START_TOKEN'] +
                           self.prompt_template['IMG_CONTEXT_TOKEN'] * ref_lens[cnt] +
                           self.prompt_template['IMG_END_TOKEN'])
                cnt += 1
            prompts.append(self.prompt_template['INSTRUCTION'].format(input=f'{tokens}\n{text}'))
        return self.tokenizer(prompts, add_special_tokens=True, return_tensors='pt',
                              padding=True, padding_side='left').to(self.device)

    # ── LLM forward helper ────────────────────────────────────────────
    def _llm_forward_and_merge(self, inputs):
        output = self.llm(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states  = output.hidden_states
        num_layers     = len(hidden_states) - 1        # exclude embedding layer
        selected       = list(range(num_layers - 1, 0, -6))
        merged         = torch.cat([hidden_states[i] for i in selected], dim=-1)
        return self.llm2dit(merged)

    # ── Loss functions ────────────────────────────────────────────────

    def text2image_loss(self, data_dict):
        if 'image_latents' in data_dict:
            image_latents = [x.to(dtype=self.dtype, device=self.device)
                             for x in data_dict['image_latents']]
        else:
            image_latents = [self.pixels_to_latents(p.to(device=self.device, dtype=self.dtype)[None])[0]
                 for p in data_dict['pixel_values']]
        b = len(image_latents)

        texts       = ['' if random.random() < self.unconditional else t for t in data_dict['texts']]
        text_inputs = self.prepare_text2image_prompts(texts)
        query_emb   = self.meta_queries[None].expand(b, self.num_queries, -1)
        inputs      = self.prepare_forward_input(query_embeds=query_emb, **text_inputs)

        # max_len = self.max_length + self.num_queries
        # inputs  = {k: v[..., -max_len:] if v is not None else v for k, v in inputs.items()
        #            if k in ('inputs_embeds', 'attention_mask', 'position_ids')}
        # text2image_loss 里同样替换
        max_len = self.max_length + self.num_queries
        truncated_inputs = {}
        for k, v in inputs.items():
            if k not in ('inputs_embeds', 'attention_mask', 'position_ids'):
                continue
            if v is None:
                truncated_inputs[k] = v
            elif k == 'inputs_embeds':
                truncated_inputs[k] = v[:, -max_len:, :]   # ✅ 截 seq 维
            else:
                truncated_inputs[k] = v[..., -max_len:]
        inputs = truncated_inputs


        pooled_out, seq_out = self._llm_forward_and_merge(inputs)
        return self.diff_loss(image_latents, pooled_out, seq_out)

    def image2image_loss(self, data_dict):
        pixel_values_src = data_dict['pixel_values_src']
        num_refs = [len(r) for r in pixel_values_src]

        pixel_values_src = [[img.to(device=self.device, dtype=self.dtype) for img in refs]
                    for refs in pixel_values_src]

        image_latents_src = [[self.pixels_to_latents(img[None])[0] for img in refs] for refs in pixel_values_src]
        image_embeds, image_grid_thw = self.get_semantic_features_dynamic([img for refs in pixel_values_src for img in refs])
        ref_lens = [len(x) for x in image_embeds]

        image_latents = [self.pixels_to_latents(p.to(device=self.device, dtype=self.dtype)[None])[0]
                 for p in data_dict['pixel_values']]
        b = len(image_latents)

        text_inputs = self.prepare_image2image_prompts(data_dict['texts'], num_refs=num_refs, ref_lens=ref_lens)
        query_emb   = self.meta_queries[None].expand(b, self.num_queries, -1)
        inputs      = self.prepare_forward_input(query_embeds=query_emb,
                                                 image_embeds=torch.cat(image_embeds),
                                                 image_grid_thw=image_grid_thw,
                                                 **text_inputs)

        # max_len = self.max_length + max(num_refs) * max(ref_lens) + self.num_queries
        # inputs  = {k: v[..., -max_len:] for k, v in inputs.items()
        #            if k in ('inputs_embeds', 'attention_mask', 'position_ids')}

        max_len = self.max_length + max(num_refs) * max(ref_lens) + self.num_queries
        truncated_inputs = {}
        for k, v in inputs.items():
            if k not in ('inputs_embeds', 'attention_mask', 'position_ids'):
                continue
            if k == 'inputs_embeds':
                truncated_inputs[k] = v[:, -max_len:, :]   # ✅ 截 seq 维
            else:
                truncated_inputs[k] = v[..., -max_len:]     # attention_mask/position_ids
        inputs = truncated_inputs
        
        
        pooled_out, seq_out = self._llm_forward_and_merge(inputs)
        return self.diff_loss(image_latents, pooled_out, seq_out, cond_intput=image_latents_src)

    # ── Generation ───────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self,
                 prompt,
                 cfg_prompt,
                 pixel_values_src=None,
                 cfg_scale=4.5,
                 num_steps=50,
                 generator=None,
                 height=512,
                 width=512,
                 progress_bar=True):
        assert len(prompt) == len(cfg_prompt)
        b = len(prompt)

        if pixel_values_src is not None:
            num_refs = [len(r) for r in pixel_values_src]
            pixel_values_src = [[img.to(self.dtype, self.device) for img in refs] for refs in pixel_values_src]
            image_embeds, image_grid_thw = self.get_semantic_features_dynamic(
                [img for refs in pixel_values_src for img in refs])
            ref_lens = [len(x) for x in image_embeds]

            text_inputs = self.prepare_image2image_prompts(
                prompt + cfg_prompt, num_refs=num_refs * 2, ref_lens=ref_lens * 2)
            text_inputs['image_embeds']    = torch.cat(image_embeds * 2)
            text_inputs['image_grid_thw']  = torch.cat([image_grid_thw] * 2)

            cond_latents = [[self.pixels_to_latents(img[None])[0] for img in refs]
                            for refs in pixel_values_src] * 2
        else:
            text_inputs  = self.prepare_text2image_prompts(prompt + cfg_prompt)
            cond_latents = None

        query_emb = self.meta_queries[None].expand(2 * b, self.num_queries, -1)
        inputs    = self.prepare_forward_input(query_embeds=query_emb, **text_inputs)
        pooled_out, seq_out = self._llm_forward_and_merge(inputs)

        pipeline = StableDiffusion3Pipeline(
            transformer=self.transformer,
            scheduler=self.test_scheduler,
            vae=self.vae,
            text_encoder=None, tokenizer=None,
            text_encoder_2=None, tokenizer_2=None,
            text_encoder_3=None, tokenizer_3=None,
        )
        pipeline.set_progress_bar_config(disable=not progress_bar)

        samples = pipeline(
            height=height, width=width,
            guidance_scale=cfg_scale,
            num_inference_steps=num_steps,
            prompt_embeds=seq_out[:b],
            pooled_prompt_embeds=pooled_out[:b],
            negative_prompt_embeds=seq_out[b:],
            negative_pooled_prompt_embeds=pooled_out[b:],
            generator=generator,
            output_type='latent',
            cond_latents=cond_latents,
        ).images.to(self.dtype)

        return self.latents_to_pixels(samples)

    # ── Diffusion loss ────────────────────────────────────────────────

    def diff_loss(self, model_input, pooled_prompt_embeds, prompt_embeds, cond_intput=None):
        noise = [torch.randn_like(x) for x in model_input]
        bsz   = len(model_input)

        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme, batch_size=bsz,
            logit_mean=self.logit_mean, logit_std=self.logit_std)

        if self.train_scheduler.use_dynamic_shifting:
            assert self.weighting_scheme == 'logit_normal'
            image_seq_lens = [math.prod(x.shape[-2:]) // self.transformer.patch_size ** 2
                              for x in model_input]
            mu = calculate_shift(
                torch.tensor(image_seq_lens, dtype=self.dtype, device=self.device),
                self.train_scheduler.config.get("base_image_seq_len", 256),
                self.train_scheduler.config.get("max_image_seq_len", 4096),
                self.train_scheduler.config.get("base_shift", 0.5),
                self.train_scheduler.config.get("max_shift", 1.15))

            if self.train_scheduler.config.time_shift_type == "exponential":
                shift = torch.exp(mu)
            elif self.train_scheduler.config.time_shift_type == "linear":
                shift = mu
            else:
                raise NotImplementedError

            sigmas    = u.to(dtype=self.dtype, device=self.device)
            sigmas    = shift * sigmas / (1 + (shift - 1) * sigmas)
            timesteps = sigmas * self.train_scheduler.num_train_timesteps
            sigmas    = sigmas.view(-1, 1, 1, 1)
        else:
            indices   = (u * self.train_scheduler.config.num_train_timesteps).long()
            timesteps = self.train_scheduler.timesteps[indices].to(device=self.device)
            sigmas    = self.get_sigmas(timesteps, n_dim=model_input[0].ndim + 1)

        noisy_input = [(1.0 - s) * x + s * n for s, x, n in zip(sigmas, model_input, noise)]

        model_pred = self.transformer(
            hidden_states=noisy_input,
            cond_hidden_states=cond_intput,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            timestep=timesteps,
            return_dict=False,
        )[0]

        weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.weighting_scheme, sigmas=sigmas)
        target    = [n - x for n, x in zip(noise, model_input)]
        loss      = [(w.float() * (p.float() - t.float()) ** 2).mean()
                     for w, p, t in zip(weighting, model_pred, target)]
        return sum(loss) / len(loss)

    def get_sigmas(self, timesteps, n_dim=4):
        sigmas             = self.train_scheduler.sigmas.to(device=self.device, dtype=self.dtype)
        schedule_timesteps = self.train_scheduler.timesteps.to(self.device)
        timesteps          = timesteps.to(self.device)
        step_indices       = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while sigma.ndim < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma


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
    import argparse
    from glob import glob
    from PIL import Image
    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer
    from transformer_sd3_dynamic import SD3Transformer2DModel

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