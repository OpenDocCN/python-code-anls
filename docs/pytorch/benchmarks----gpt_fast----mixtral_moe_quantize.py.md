# `.\pytorch\benchmarks\gpt_fast\mixtral_moe_quantize.py`

```
# flake8: noqa: E266, C417, B950
# 从 mixtral_moe_model 导入 ConditionalFeedForward 类
from mixtral_moe_model import ConditionalFeedForward

# 导入 PyTorch 库
import torch
import torch.nn as nn
import torch.nn.functional as F

##### 量化基元 ######

# 对输入张量 x 进行按通道动态量化
def dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
    # 假设对称量化
    # 假设轴向为 0
    # 假设密集内存格式
    # TODO（未来）：根据需要放宽上述假设

    # 默认设置：激活的仿射量化
    eps = torch.finfo(torch.float32).eps

    # 获取张量 x 的最小值和最大值
    min_val, max_val = torch.aminmax(x, dim=1)

    # 根据最小值和最大值计算量化的尺度（scales）和零点（zero_points）
    # 参考：https://fburl.com/code/srbiybme
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    device = min_val_neg.device

    # 参考：https://fburl.com/code/4wll53rk
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scales = max_val_pos / (float(quant_max - quant_min) / 2)
    # 确保尺度与原始张量相同的数据类型
    scales = torch.clamp(scales, min=eps).to(x.dtype)
    zero_points = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    # 根据 qmin/qmax/scales/zp 进行量化
    # 参考：https://www.internalfb.com/code/fbsource/[8edc275012b1]/fbcode/caffe2/torch/ao/quantization/fx/_decomposed.py?lines=63
    x_div = x / scales.unsqueeze(-1)
    x_round = torch.round(x_div)
    x_zp = x_round + zero_points.unsqueeze(-1)
    quant = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)

    return quant, scales, zero_points


##### 仅权重的 int8 按通道量化代码 ######

# 替换模块中的线性层权重为仅权重的 int8 按通道量化
def replace_linear_weight_only_int8_per_channel(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name != "gate":
            setattr(
                module,
                name,
                WeightOnlyInt8Linear(
                    child.in_features, child.out_features, target_dtype=torch.int8
                ),
            )
        elif isinstance(child, ConditionalFeedForward):
            num_experts, intermediate_size, dim = child.w1.shape
            setattr(
                module,
                name,
                ConditionalFeedForwardInt8(
                    num_experts, intermediate_size, dim, target_dtype=torch.int8
                ),
            )
        else:
            replace_linear_weight_only_int8_per_channel(child)


class WeightOnlyInt8QuantHandler:
    def __init__(self, mod):
        self.mod = mod

    @torch.no_grad()
    # 创建量化后的状态字典
    def create_quantized_state_dict(self):
        # 获取当前模型的状态字典
        cur_state_dict = self.mod.state_dict()
        
        # 遍历模型的所有模块和完全限定名
        for fqn, mod in self.mod.named_modules():
            # 如果模块是 torch.nn.Linear 并且不是以 ".gate" 结尾的模块
            if isinstance(mod, torch.nn.Linear) and not fqn.endswith(".gate"):
                # 对模块的权重进行通道级别的动态量化
                int8_weight, scales, _ = dynamically_quantize_per_channel(
                    mod.weight.float(), -128, 127, torch.int8
                )
                # 更新当前状态字典中的量化后的权重和缩放因子
                cur_state_dict[f"{fqn}.weight"] = int8_weight
                cur_state_dict[f"{fqn}.scales"] = scales.to(mod.weight.dtype)
            
            # 如果模块是 ConditionalFeedForward 类型
            elif isinstance(mod, ConditionalFeedForward):
                # 遍历模块中的三个权重
                for weight_idx in range(0, 3):
                    weight_name = f"w{weight_idx + 1}"
                    scales_name = f"scales{weight_idx + 1}"
                    weight = getattr(mod, weight_name)
                    num_experts, intermediate_size, dim = weight.shape

                    bit8_weight_list = []
                    scales_list = []
                    # 对每个专家的权重进行通道级别的动态量化
                    for expert_idx in range(num_experts):
                        bit8_weight, scales, _ = dynamically_quantize_per_channel(
                            weight[expert_idx].float(), -128, 127, torch.int8
                        )
                        bit8_weight_list.append(
                            bit8_weight.reshape(1, intermediate_size, dim)
                        )
                        scales_list.append(scales.reshape(1, intermediate_size))

                    # 更新当前状态字典中的量化后的权重和缩放因子
                    cur_state_dict[f"{fqn}.{weight_name}"] = torch.cat(
                        bit8_weight_list, dim=0
                    )
                    cur_state_dict[f"{fqn}.{scales_name}"] = torch.cat(
                        scales_list, dim=0
                    )

        # 返回更新后的当前状态字典
        return cur_state_dict

    # 将模型转换为运行时所需的格式（只替换线性层的权重为 int8 的通道级别量化）
    def convert_for_runtime(self):
        # 替换模型中所有线性层的权重为 int8 的通道级别量化
        replace_linear_weight_only_int8_per_channel(self.mod)
        # 返回更新后的模型
        return self.mod
class WeightOnlyInt8Linear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        target_dtype=None,
    ) -> None:
        assert target_dtype is not None
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features  # 设置输入特征数量
        self.out_features = out_features  # 设置输出特征数量
        # 注册权重张量，使用指定的目标数据类型
        self.register_buffer(
            "weight", torch.empty((out_features, in_features), dtype=target_dtype)
        )
        # 注册缩放因子张量，使用bfloat16数据类型
        self.register_buffer("scales", torch.ones(out_features, dtype=torch.bfloat16))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 执行线性变换，根据输入数据类型调整权重类型，并乘以缩放因子
        return F.linear(input, self.weight.to(dtype=input.dtype)) * self.scales


class ConditionalFeedForwardInt8(nn.Module):
    def __init__(self, num_experts, intermediate_size, dim, target_dtype):
        super().__init__()
        self.target_dtype = target_dtype  # 设置目标数据类型

        # 注册权重张量和缩放因子张量，使用指定的目标数据类型和bfloat16数据类型
        self.register_buffer(
            "w1", torch.empty(num_experts, intermediate_size, dim, dtype=target_dtype)
        )
        self.register_buffer(
            "w2", torch.empty(num_experts, dim, intermediate_size, dtype=target_dtype)
        )
        self.register_buffer(
            "w3", torch.empty(num_experts, intermediate_size, dim, dtype=target_dtype)
        )

        self.register_buffer(
            "scales1", torch.empty(num_experts, intermediate_size, dtype=torch.bfloat16)
        )
        self.register_buffer(
            "scales2", torch.empty(num_experts, dim, dtype=torch.bfloat16)
        )
        self.register_buffer(
            "scales3", torch.empty(num_experts, intermediate_size, dtype=torch.bfloat16)
        )

    def forward(self, x, expert_indices):
        # 根据专家索引获取对应的权重张量，并根据输入x的数据类型调整权重类型
        w1_weights = self.w1.to(x.dtype)[expert_indices]  # [T, A, D, D]
        w3_weights = self.w3.to(x.dtype)[expert_indices]  # [T, A, D, D]
        w2_weights = self.w2.to(x.dtype)[expert_indices]
        
        # 执行特定的张量运算，使用Einsum函数进行乘积计算，并乘以对应的缩放因子
        x1 = F.silu(
            torch.einsum("ti,taoi -> tao", x, w1_weights)
            * self.scales1[expert_indices].to(x.dtype)
        )
        x3 = torch.einsum("ti, taoi -> tao", x, w3_weights) * self.scales3[
            expert_indices
        ].to(x.dtype)
        # 执行最终的张量运算，计算专家输出，并乘以对应的缩放因子
        expert_outs = torch.einsum(
            "tao, taio -> tai", (x1 * x3), w2_weights
        ) * self.scales2[expert_indices].to(
            x.dtype
        )  # [T, A, D, D]
        return expert_outs
```