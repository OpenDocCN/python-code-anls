# `.\integrations\awq.py`

```py
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"AWQ (Activation aware Weight Quantization) integration file"
from ..activations import ACT2FN  # 导入激活函数映射
from ..modeling_utils import PreTrainedModel  # 导入预训练模型工具函数
from ..utils import is_auto_awq_available, is_torch_available  # 导入 AWQ 自动可用性检查和 Torch 可用性检查
from ..utils.quantization_config import (  # 导入量化配置
    AwqBackendPackingMethod,
    AwqConfig,
    AWQLinearVersion,
    ExllamaVersion,
)

if is_torch_available():  # 如果 Torch 可用
    import torch  # 导入 PyTorch
    import torch.nn as nn  # 导入 PyTorch 神经网络模块

# AWQ_FUSED_MAPPINGS 定义了不同模型类型的层映射字典，用于 AWQ 线性层替换
AWQ_FUSED_MAPPINGS = {
    "mistral": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "layernorm": ["input_layernorm", "post_attention_layernorm", "norm"],
        "use_alibi": False,
    },
    "mixtral": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["w1", "w3", "w2"],
        "layernorm": ["input_layernorm", "post_attention_layernorm", "norm"],
        "use_alibi": False,
        "rope_theta": 1000000.0,
    },
    "llama": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "layernorm": ["input_layernorm", "post_attention_layernorm", "norm"],
        "use_alibi": False,
    },
    "llava": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "layernorm": ["input_layernorm", "post_attention_layernorm", "norm"],
        "use_alibi": False,
    },
}

def replace_with_awq_linear(
    model,
    modules_to_not_convert=None,
    quantization_config=None,
    current_key_name=None,
    has_been_replaced=False,
) -> bool:
    """
    Public method that recursively replaces the Linear layers of the given model with AWQ quantized layers.
    `accelerate` is needed to use this method. Returns the converted model and a boolean that indicates if the
    conversion has been successful or not.

    During the module replacement, we also infer the backend to use through the `quantization_config` object.
    """
    Args:
        model (`torch.nn.Module`):
            要转换的模型，可以是任何 `torch.nn.Module` 实例。
        quantization_config (`AwqConfig`):
            包含量化参数的量化配置对象。
        modules_to_not_convert (`list`, *可选*):
            不需要转换的模块列表。如果模块名在列表中（例如 `lm_head`），则不会进行转换。
        current_key_name (`list`, *可选*):
            包含当前键名的列表。这用于递归，用户不应传递此参数。
        has_been_replaced (`bool`, *可选*):
            表示转换是否成功的布尔值。这用于递归，用户不应传递此参数。
    """
    # 如果未指定不转换的模块列表，则初始化为空列表
    if modules_to_not_convert is None:
        modules_to_not_convert = []

    # 获取量化配置中的后端信息
    backend = quantization_config.backend

    # 检查是否存在自动 AWQ 支持
    if not is_auto_awq_available():
        raise ValueError(
            "AWQ（`autoawq` 或 `llmawq`）不可用。请使用 `pip install autoawq` 安装或查看安装指南：https://github.com/mit-han-lab/llm-awq"
        )

    # 根据量化配置选择合适的量化线性层类
    if backend == AwqBackendPackingMethod.AUTOAWQ:
        if quantization_config.version == AWQLinearVersion.GEMM:
            # 导入 GEMM 版本的量化线性层类
            from awq.modules.linear.gemm import WQLinear_GEMM

            target_cls = WQLinear_GEMM
        elif quantization_config.version == AWQLinearVersion.GEMV:
            # 导入 GEMV 版本的量化线性层类
            from awq.modules.linear.gemv import WQLinear_GEMV

            target_cls = WQLinear_GEMV
        elif quantization_config.version == AWQLinearVersion.EXLLAMA:
            if quantization_config.exllama_config["version"] == ExllamaVersion.ONE:
                # 导入 Exllama 版本一的量化线性层类
                from awq.modules.linear.exllama import WQLinear_Exllama

                target_cls = WQLinear_Exllama
            elif quantization_config.exllama_config["version"] == ExllamaVersion.TWO:
                # 导入 Exllama 版本二的量化线性层类
                from awq.modules.linear.exllamav2 import WQLinear_ExllamaV2

                target_cls = WQLinear_ExllamaV2
            else:
                raise ValueError(f"未知的 Exllama 版本: {quantization_config.exllama_config['version']}")
        else:
            raise ValueError(f"未知的 AWQ 版本: {quantization_config.version}")
    else:
        # 若未选择 AUTOAWQ 后端，则使用默认的量化线性层类
        from awq.quantize.qmodule import WQLinear

        target_cls = WQLinear
    # 遍历模型的所有子模块，获取每个子模块的名称和实例
    for name, module in model.named_children():
        # 如果当前键名为 None，则初始化为一个空列表
        if current_key_name is None:
            current_key_name = []
        # 将当前子模块名称添加到当前键名列表中
        current_key_name.append(name)

        # 如果当前模块是 nn.Linear 类型，并且其名称不在不转换的模块列表中
        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            # 检查当前键名组合不在不转换模块列表中的任何键中
            if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
                # 获取线性层的输入和输出特征数
                in_features = module.in_features
                out_features = module.out_features

                # 将模型中的当前线性层替换为目标类的实例化对象
                model._modules[name] = target_cls(
                    w_bit=quantization_config.bits,
                    group_size=quantization_config.group_size,
                    in_features=in_features,
                    out_features=out_features,
                    bias=module.bias is not None,
                    dev=module.weight.device,
                )
                # 标记该模块已被替换
                has_been_replaced = True

                # 强制设置该模块的 requires_grad 为 False，以避免意外错误
                model._modules[name].requires_grad_(False)

        # 如果当前模块还有子模块，则递归调用此函数来替换子模块
        if len(list(module.children())) > 0:
            _, has_been_replaced = replace_with_awq_linear(
                module,
                modules_to_not_convert=modules_to_not_convert,
                current_key_name=current_key_name,
                quantization_config=quantization_config,
                has_been_replaced=has_been_replaced,
            )
        
        # 移除当前键名列表中的最后一个键名，为递归调用做准备
        current_key_name.pop(-1)
    
    # 返回替换后的模型和是否有模块被替换的标志
    return model, has_been_replaced
# 返回模型中需要融合的模块映射，根据给定的量化配置和模型
def get_modules_to_fuse(model, quantization_config):
    """
    Returns the fusing mapping given the quantization config and the model

    Args:
        model (`~PreTrainedModel`):
            The model to fuse - note this model should have been converted into AWQ format beforehand.
        quantization_config (`~transformers.quantization_config.AWQConfig`):
            The quantization configuration to use.
    """
    # 如果模型不是PreTrainedModel的实例，则抛出错误
    if not isinstance(model, PreTrainedModel):
        raise ValueError(f"The model should be an instance of `PreTrainedModel`, got {model.__class__.__name__}")

    # 总是默认使用 `quantization_config.modules_to_fuse`
    if quantization_config.modules_to_fuse is not None:
        current_fused_mapping = quantization_config.modules_to_fuse
        current_fused_mapping["max_seq_len"] = quantization_config.fuse_max_seq_len
    # 如果 `quantization_config.modules_to_fuse` 为None，则根据模型类型在 `AWQ_FUSED_MAPPINGS` 中查找对应的映射
    elif model.config.model_type in AWQ_FUSED_MAPPINGS:
        current_fused_mapping = AWQ_FUSED_MAPPINGS[model.config.model_type]

        # 处理多模态模型的情况（如Llava），区分 `model.config` 和 `model.config.text_config`
        if not hasattr(model.config, "text_config"):
            config = model.config
        else:
            config = model.config.text_config

        # 单独处理 `hidden_size`、`num_attention_heads` 和 `num_key_value_heads` 字段
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)

        # 填充 `current_fused_mapping` 中的预期值
        current_fused_mapping["hidden_size"] = hidden_size
        current_fused_mapping["num_attention_heads"] = num_attention_heads
        current_fused_mapping["num_key_value_heads"] = num_key_value_heads
        current_fused_mapping["max_seq_len"] = quantization_config.fuse_max_seq_len
    # 如果都没有找到合适的融合映射，则抛出错误
    else:
        raise ValueError(
            "Fusing mapping not found either on the quantization config or the supported `AWQ_FUSED_MAPPINGS`. Please pass a `fused_mapping` argument"
            " in the `quantization_config` or raise an issue on transformers https://github.com/huggingface/transformers to add its support."
        )
    return current_fused_mapping


# 可选地融合模型中的一些模块以加速推断
def fuse_awq_modules(model, quantization_config):
    """
    Optionally fuse some modules in the model to speedup inference.

    Args:
        model (`~PreTrainedModel`):
            The model to fuse - note this model should have been converted into AWQ format beforehand.
        quantization_config (`Union[AwqConfig, dict]`):
            The quantization configuration to use.
    """
    # 如果 `quantization_config` 是字典，则将其转换为 AwqConfig 对象以便获取 `backend` 等字段
    # 否则这些字段将不可用
    # https://github.com/huggingface/transformers/pull/27411#discussion_r1414044495
    if isinstance(quantization_config, dict):
        quantization_config = AwqConfig.from_dict(quantization_config)
    # 获取量化配置中的后端信息
    backend = quantization_config.backend

    # 获取需要融合的模块列表
    modules_to_fuse = get_modules_to_fuse(model, quantization_config)
    
    # 获取不需要转换的模块列表（如果有的话）
    modules_to_not_convert = getattr(quantization_config, "modules_to_not_convert", None)

    # 检查是否使用自动 AWQ 后端
    if backend == AwqBackendPackingMethod.AUTOAWQ:
        # 导入 AWQ 后端的融合模块
        from awq.modules.fused.attn import QuantAttentionFused
        from awq.modules.fused.mlp import QuantFusedMLP
        from awq.modules.fused.norm import FasterTransformerRMSNorm
    else:
        # 抛出数值错误，只支持 AutoAWQ 后端的融合
        raise ValueError("Fusing is only supported for the AutoAWQ backend")

    # 遍历模型的所有命名模块
    for name, module in model.named_modules():
        # 如果存在不需要转换的模块列表
        if modules_to_not_convert is not None:
            # 检查当前模块名是否在不需要转换的模块列表中的任意一个
            if any(module_name_to_not_convert in name for module_name_to_not_convert in modules_to_not_convert):
                # 如果是，则跳过当前模块的处理
                continue

        # 替换模型中的 LayerNorm 层
        _fuse_awq_layernorm(modules_to_fuse["layernorm"], module, FasterTransformerRMSNorm)

        # 替换模型中的 MLP 层
        _fuse_awq_mlp(model, name, modules_to_fuse["mlp"], module, QuantFusedMLP)

        # 替换模型中的 Attention 层
        _fuse_awq_attention_layers(model, module, modules_to_fuse, name, QuantAttentionFused)

    # 返回融合后的模型
    return model
# 融合 LayerNorm 层到目标类中，使用自动 AWQ（Automatic Weight Quantization）
def _fuse_awq_layernorm(fuse_module_names, module, target_cls):
    """
    Fuse the LayerNorm layers into a target class using autoawq

    Args:
        fuse_module_names (`List[str]`):
            The list of module names to fuse
        module (`nn.Module`):
            The pytorch parent module that has layernorm modules to fuse
        target_cls (`~autoawq.FasterTransformerRMSNorm`):
            The `FasterTransformerRMSNorm` class as it only supports that class
            for now.
    """
    # 遍历要融合的模块名列表
    for module_name in fuse_module_names:
        # 检查父模块是否具有指定的模块名
        if hasattr(module, module_name):
            # 获取旧的 LayerNorm 模块
            old_module = getattr(module, module_name)
            # 创建新的 target_cls 类实例来替换旧模块
            module._modules[module_name] = target_cls(
                old_module.weight,
                old_module.variance_epsilon,
            ).to(old_module.weight.device)
            # 删除旧模块，释放内存
            del old_module


# 融合 MLP 层到目标类中，使用自动 AWQ
def _fuse_awq_mlp(model, current_module_name, fuse_module_names, module, target_cls):
    """
    Fuse the MLP layers into a target class using autoawq

    Args:
        model (`~PreTrainedModel`):
            The input pretrained model
        current_module_name (`str`):
            The current submodule name
        fuse_module_names (`List[str]`):
            The list of module names to fuse. For the MLP layers it has to be an array
            of length 3 that consists of the 3 MLP layers in the order (gate (dense layer post-attention) / up / down layers)
        module (`nn.Module`):
            The pytorch parent module that has layernorm modules to fuse
        target_cls (`~autoawq.QuantFusedMLP`):
            The `QuantFusedMLP` class as it only supports that class
            for now.
    """
    # 如果没有要融合的模块名，直接返回
    if len(fuse_module_names) == 0:
        return

    # 检查父模块是否具有第一个要融合的模块名
    if hasattr(module, fuse_module_names[0]):
        # 获取三个 MLP 层的引用
        gate_proj = getattr(module, fuse_module_names[0])
        up_proj = getattr(module, fuse_module_names[1])
        down_proj = getattr(module, fuse_module_names[2])

        # 记录 gate_proj 的设备信息
        previous_device = gate_proj.qweight.device

        # 处理模型具有 `text_config` 属性的情况
        hidden_act = (
            model.config.hidden_act
            if not hasattr(model.config, "text_config")
            else model.config.text_config.hidden_act
        )
        # 根据 hidden_act 获取激活函数
        activation_fn = ACT2FN[hidden_act]
        # 创建新的 QuantFusedMLP 实例
        new_module = target_cls(gate_proj, down_proj, up_proj, activation_fn)

        # 分离当前模块的父子名称
        parent_name, child_name = current_module_name.rsplit(".", 1)
        # 获取父模块
        parent = model.get_submodule(parent_name)
        # 将新模块设置为子模块的属性，并转移到之前的设备上
        setattr(parent, child_name, new_module.to(previous_device))

        # 删除临时变量，释放内存
        del gate_proj, up_proj, down_proj


def _fuse_awq_attention_layers(model, module, modules_to_fuse, current_module_name, target_cls):
    """
    Fuse the Attention layers into a target class using autoawq
    """
    # 这部分代码还未提供，需要根据实际情况继续补充
    pass
    # 导入需要的模块：WQLinear_GEMM 和 WQLinear_GEMV，来自 awq.modules.linear
    from awq.modules.linear import WQLinear_GEMM, WQLinear_GEMV

    # 如果 modules_to_fuse 字典中的 attention 列表为空，直接返回
    if len(modules_to_fuse["attention"]) == 0:
        return
    # 检查模块是否具有指定名称的属性
    if hasattr(module, modules_to_fuse["attention"][0]):
        # 获取模块中指定注意力组件的引用
        q_proj = getattr(module, modules_to_fuse["attention"][0])

        # 根据不同的注意力组件类型选择相应的线性层类和连接维度
        if isinstance(q_proj, WQLinear_GEMV):
            linear_target_cls = WQLinear_GEMV
            cat_dim = 0
        elif isinstance(q_proj, WQLinear_GEMM):
            linear_target_cls = WQLinear_GEMM
            cat_dim = 1
        else:
            # 如果遇到不支持的 q_proj 类型，则抛出异常
            raise ValueError(f"Unsupported q_proj type: {type(q_proj)}")

        # 记录 q_proj 的设备信息
        previous_device = q_proj.qweight.device

        # 获取其他相关的注意力组件
        k_proj = getattr(module, modules_to_fuse["attention"][1])
        v_proj = getattr(module, modules_to_fuse["attention"][2])
        o_proj = getattr(module, modules_to_fuse["attention"][3])

        # 合并偏置项，如果存在的话
        bias = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0) if q_proj.bias is not None else None

        # 创建新的量化线性层，整合 QKV 权重、量化零点和缩放因子
        qkv_layer = linear_target_cls(
            q_proj.w_bit,
            q_proj.group_size,
            q_proj.in_features,
            q_proj.out_features + k_proj.out_features + v_proj.out_features,
            q_proj.bias is not None,
            next(iter(module.state_dict().values())).device,
        )

        # 合并 QKV 层的权重、量化零点和缩放因子
        qkv_layer.qweight = torch.cat([q_proj.qweight, k_proj.qweight, v_proj.qweight], dim=cat_dim)
        qkv_layer.qzeros = torch.cat([q_proj.qzeros, k_proj.qzeros, v_proj.qzeros], dim=cat_dim)
        qkv_layer.scales = torch.cat([q_proj.scales, k_proj.scales, v_proj.scales], dim=cat_dim)

        # 如果是 WQLinear_GEMV 类型的量化线性层，设置其特有的 split_k_iters 属性
        if isinstance(qkv_layer, WQLinear_GEMV):
            qkv_layer.split_k_iters = q_proj.split_k_iters

        # 设置合并后注意力层的偏置项
        qkv_layer.bias = bias

        # 创建融合后的注意力层，使用指定的参数初始化
        fused_attention_layer = target_cls(
            modules_to_fuse["hidden_size"],
            modules_to_fuse["num_attention_heads"],
            modules_to_fuse["num_key_value_heads"],
            qkv_layer,
            o_proj,
            previous_device,
            modules_to_fuse["max_seq_len"],
            use_alibi=modules_to_fuse["use_alibi"],
            rope_theta=modules_to_fuse.get("rope_theta", 10000.0),  # 设置默认的 rope_theta 值为 10000.0
        )

        # 标记融合后的注意力层是 HF Transformers 的一部分
        fused_attention_layer.is_hf_transformers = True

        # 将融合后的注意力层设置为模型中对应的子模块
        parent_name, child_name = current_module_name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, fused_attention_layer.to(previous_device))

        # 清理不再需要的变量引用，释放内存
        del q_proj, k_proj, v_proj, o_proj
def post_init_awq_exllama_modules(model, exllama_config):
    """
    Runs post init for Exllama layers which performs:
        - Weights unpacking, reordering and repacking
        - Devices scratch space allocation
    """

    # 检查配置中的 Exllama 版本是否为版本一
    if exllama_config["version"] == ExllamaVersion.ONE:
        # 如果是版本一，则导入版本一的初始化函数，并对模型进行处理
        from awq.modules.linear.exllama import exllama_post_init
        model = exllama_post_init(model)
    # 检查配置中的 Exllama 版本是否为版本二
    elif exllama_config["version"] == ExllamaVersion.TWO:
        # 如果是版本二，则导入版本二的初始化函数，并根据配置参数对模型进行处理
        from awq.modules.linear.exllamav2 import exllamav2_post_init
        model = exllamav2_post_init(
            model,
            max_input_len=exllama_config["max_input_len"],
            max_batch_size=exllama_config["max_batch_size"],
        )
    else:
        # 如果配置中的 Exllama 版本既不是一也不是二，则抛出异常
        raise ValueError(f"Unrecognized Exllama version: {exllama_config['version']}")

    # 返回经过 Exllama 模块处理后的模型
    return model
```