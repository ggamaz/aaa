import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLaLoRALinear(nn.Module):
    """
    基于论文 "Bilevel Layer-Positioning LoRA" 的核心模块。
    实现了公式: W' = W_0 + \alpha * \gamma * \Delta W
    """
    def __init__(self, base_layer: nn.Linear, r: int = 8, lora_alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.scaling = lora_alpha / r
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # LoRA 参数 A 和 B
        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
        
        # 门控参数 alpha_logit (过 sigmoid 后变为论文中的 \alpha)
        # 初始值设为 0，sigmoid(0) = 0.5
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))
        
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        
        # 状态机：搜索阶段 (Stage 1) or 微调阶段 (Stage 2)
        self.is_searching = True
        self.is_topk_selected = False # 如果在 Stage 2 中被选中，则为 True
        
        self.reset_parameters()
        
        # 冻结基础模型权重
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 基础大模型前向传播
        base_out = self.base_layer(x)
        
        # LoRA 分支前向传播: \Delta W
        lora_out = F.linear(self.dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        
        if self.is_searching:
            # 阶段 1：使用平滑的 alpha 进行双层优化架构搜索
            alpha = torch.sigmoid(self.alpha_logit)
        else:
            # 阶段 2：如果被 Top-K 选中，alpha 为 1.0，否则彻底关闭为 0.0
            alpha = 1.0 if self.is_topk_selected else 0.0
            
        return base_out + alpha * self.scaling * lora_out


def inject_bilalora(model: nn.Module, target_modules: list, r: int = 8, lora_alpha: int = 16):
    """
    将 BiLaLoRALinear 注入到指定的目标线性层中。
    """
    injected_modules = []
    
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
            
        if any(target in name for target in target_modules):
            # 获取父模块以便替换
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[-1] if '.' in name else name
            
            parent = model.get_submodule(parent_name) if parent_name else model
            
            # 创建并替换
            new_module = BiLaLoRALinear(module, r=r, lora_alpha=lora_alpha)
            setattr(parent, child_name, new_module)
            injected_modules.append(new_module)
            
    return model, injected_modules


class BiLaLoRAManager:
    """
    管理 BiLaLoRA 的生命周期，包括计算超梯度 (Eq 7) 和进行 Top-K 截断。
    """
    def __init__(self, bilalora_modules: list):
        self.modules = bilalora_modules
        
    def get_alpha_parameters(self):
        return [m.alpha_logit for m in self.modules]
        
    def get_lora_parameters(self):
        params = []
        for m in self.modules:
            params.extend([m.lora_A, m.lora_B])
        return params

    def compute_hypergradient(self, loss_train: torch.Tensor, loss_val: torch.Tensor):
        """
        根据论文 Eq 7 计算 \alpha 的超梯度 (Hypergradient)。
        g_alpha \approx \nabla_\alpha \varphi - \frac{\nabla_\omega \varphi^T \nabla_\omega f}{||\nabla_\omega f||^2} \nabla_\alpha f
        """
        alphas = self.get_alpha_parameters()
        omegas = self.get_lora_parameters()
        
        # 计算 \nabla_\omega f (训练集对 LoRA 权重的梯度)
        grad_omega_f = torch.autograd.grad(loss_train, omegas, retain_graph=True, allow_unused=True)
        # 计算 \nabla_\alpha f (训练集对 Alpha 架构参数的梯度)
        grad_alpha_f = torch.autograd.grad(loss_train, alphas, retain_graph=True, allow_unused=True)
        
        # 计算 \nabla_\omega \varphi (验证集对 LoRA 权重的梯度)
        grad_omega_phi = torch.autograd.grad(loss_val, omegas, retain_graph=True, allow_unused=True)
        # 计算 \nabla_\alpha \varphi (验证集对 Alpha 架构参数的梯度)
        grad_alpha_phi = torch.autograd.grad(loss_val, alphas, retain_graph=True, allow_unused=True)
        
        # 将梯度展平，便于进行向量点乘
        flat_grad_omega_f = torch.cat([g.contiguous().view(-1) for g in grad_omega_f if g is not None])
        flat_grad_omega_phi = torch.cat([g.contiguous().view(-1) for g in grad_omega_phi if g is not None])
        
        # ||\nabla_\omega f||^2
        norm_grad_omega_f_sq = torch.dot(flat_grad_omega_f, flat_grad_omega_f) + 1e-8
        
        # \nabla_\omega \varphi^T \nabla_\omega f
        dot_product = torch.dot(flat_grad_omega_phi, flat_grad_omega_f)
        
        scalar_term = dot_product / norm_grad_omega_f_sq
        
        # 根据 Eq 7 将计算出的梯度手动赋值给 alpha_logit.grad
        for i, alpha_param in enumerate(alphas):
            g_alpha_phi = grad_alpha_phi[i] if grad_alpha_phi[i] is not None else torch.zeros_like(alpha_param)
            g_alpha_f   = grad_alpha_f[i]   if grad_alpha_f[i]   is not None else torch.zeros_like(alpha_param)
            
            hyper_grad = g_alpha_phi - scalar_term * g_alpha_f
            alpha_param.grad = hyper_grad

    def select_top_k(self, k: int):
        """
        执行 Algorithm 1 中的阶段 2：选取 Top-K 层并冻结 \alpha
        """
        print(f"--- 正在执行 BiLaLoRA Top-{k} 层筛选 ---")
        # 收集所有 alpha 值
        alpha_values = [(i, torch.sigmoid(m.alpha_logit).item()) for i, m in enumerate(self.modules)]
        # 根据 alpha 值降序排序
        alpha_values.sort(key=lambda x: x[1], reverse=True)
        
        # 找到 Top-K 的索引
        top_k_indices = {x[0] for x in alpha_values[:k]}
        
        for i, m in enumerate(self.modules):
            m.is_searching = False # 结束搜索阶段
            m.alpha_logit.requires_grad = False # 停止 \alpha 的梯度更新
            
            if i in top_k_indices:
                m.is_topk_selected = True
                print(f"Layer {i} 选中 (Score: {alpha_values[alpha_values.index((i, torch.sigmoid(m.alpha_logit).item()))][1]:.4f})")
            else:
                m.is_topk_selected = False
                m.lora_A.requires_grad = False # 未选中的层直接冻结 LoRA 权重
                m.lora_B.requires_grad = False