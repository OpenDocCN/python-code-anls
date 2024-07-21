# `.\pytorch\torch\ao\pruning\_experimental\data_sparsifier\quantization_utils.py`

```
# mypy: allow-untyped-defs
# 导入 PyTorch 库
import torch
# 导入神经网络模块定义
import torch.nn as nn
# 导入用于处理稀疏化的工具函数
from torch.ao.pruning.sparsifier.utils import module_to_fqn, fqn_to_module
# 导入类型提示
from typing import Dict, List, Optional

# 支持稀疏化的模块类型集合
SUPPORTED_MODULES = {
    nn.Embedding,
    nn.EmbeddingBag
}


def _fetch_all_embeddings(model):
    """Fetches Embedding and EmbeddingBag modules from the model
    """
    # 初始化嵌入模块列表
    embedding_modules = []
    # 使用栈来深度优先遍历模型中的所有子模块
    stack = [model]
    while stack:
        # 弹出栈顶模块
        module = stack.pop()
        # 遍历当前模块的所有子模块
        for _, child in module.named_children():
            # 获取子模块的全限定名
            fqn_name = module_to_fqn(model, child)
            # 如果子模块属于支持的嵌入模块类型，则加入到嵌入模块列表中
            if type(child) in SUPPORTED_MODULES:
                embedding_modules.append((fqn_name, child))
            else:
                # 否则将子模块压入栈中，继续深度优先遍历
                stack.append(child)
    # 返回找到的所有嵌入模块列表
    return embedding_modules


def post_training_sparse_quantize(model,
                                  data_sparsifier_class,
                                  sparsify_first=True,
                                  select_embeddings: Optional[List[nn.Module]] = None,
                                  **sparse_config):
    """Takes in a model and applies sparsification and quantization to only embeddings & embeddingbags.
    The quantization step can happen before or after sparsification depending on the `sparsify_first` argument.

    Args:
        - model (nn.Module)
            model whose embeddings needs to be sparsified
        - data_sparsifier_class (type of data sparsifier)
            Type of sparsification that needs to be applied to model
        - sparsify_first (bool)
            if true, sparsifies first and then quantizes
            otherwise, quantizes first and then sparsifies.
        - select_embeddings (List of Embedding modules)
            List of embedding modules to in the model to be sparsified & quantized.
            If None, all embedding modules with be sparsified
        - sparse_config (Dict)
            config that will be passed to the constructor of data sparsifier object.

    Note:
        1. When `sparsify_first=False`, quantization occurs first followed by sparsification.
            - before sparsifying, the embedding layers are dequantized.
            - scales and zero-points are saved
            - embedding layers are sparsified and `squash_mask` is applied
            - embedding weights are requantized using the saved scales and zero-points
        2. When `sparsify_first=True`, sparsification occurs first followed by quantization.
            - embeddings are sparsified first
            - quantization is applied on the sparsified embeddings
    """
    # 根据 sparse_config 创建数据稀疏化对象
    data_sparsifier = data_sparsifier_class(**sparse_config)

    # 如果 select_embeddings 为 None，则获取模型中的所有嵌入模块
    if select_embeddings is None:
        embedding_modules = _fetch_all_embeddings(model)
    else:
        embedding_modules = []
        assert isinstance(select_embeddings, List), "the embedding_modules must be a list of embedding modules"
        for emb in select_embeddings:
            assert type(emb) in SUPPORTED_MODULES, "the embedding_modules list must be an embedding or embedding bags"
            fqn_name = module_to_fqn(model, emb)
            assert fqn_name is not None, "the embedding modules must be part of input model"
            embedding_modules.append((fqn_name, emb))


        # 如果不是稀疏化优先模式，则准备选择的嵌入模块列表
        embedding_modules = []
        # 断言选择的嵌入模块列表是一个列表类型
        assert isinstance(select_embeddings, List), "the embedding_modules must be a list of embedding modules"
        # 遍历选择的嵌入模块，确保每个模块类型在支持的模块列表中
        for emb in select_embeddings:
            assert type(emb) in SUPPORTED_MODULES, "the embedding_modules list must be an embedding or embedding bags"
            # 获取嵌入模块的全限定名
            fqn_name = module_to_fqn(model, emb)
            # 断言全限定名不为 None，即嵌入模块必须是输入模型的一部分
            assert fqn_name is not None, "the embedding modules must be part of input model"
            # 将全限定名和模块本身添加到 embedding_modules 列表中
            embedding_modules.append((fqn_name, emb))


    if sparsify_first:
        # sparsify
        for name, emb_module in embedding_modules:
            valid_name = name.replace('.', '_')
            data_sparsifier.add_data(name=valid_name, data=emb_module)

        # 执行稀疏化数据处理的步骤
        data_sparsifier.step()
        # 压缩稀疏化掩码
        data_sparsifier.squash_mask()

        # quantize
        for _, emb_module in embedding_modules:
            # 设置量化配置为仅对权重进行浮点参数量化
            emb_module.qconfig = torch.ao.quantization.float_qparams_weight_only_qconfig

        # 在原地准备模型以进行量化
        torch.ao.quantization.prepare(model, inplace=True)
        # 在原地执行模型量化
        torch.ao.quantization.convert(model, inplace=True)


        # 如果设置了 sparsify_first 标志，则执行以下操作：
        # 对每个嵌入模块执行稀疏化处理，将名称中的点替换为下划线后添加到数据稀疏化器中
        for name, emb_module in embedding_modules:
            valid_name = name.replace('.', '_')
            data_sparsifier.add_data(name=valid_name, data=emb_module)

        # 执行数据稀疏化器的处理步骤
        data_sparsifier.step()
        # 压缩数据稀疏化器的掩码
        data_sparsifier.squash_mask()

        # 对每个嵌入模块执行量化准备，将量化配置设置为仅对权重进行浮点参数量化
        for _, emb_module in embedding_modules:
            emb_module.qconfig = torch.ao.quantization.float_qparams_weight_only_qconfig

        # 在原地准备模型以进行量化
        torch.ao.quantization.prepare(model, inplace=True)
        # 在原地执行模型量化
        torch.ao.quantization.convert(model, inplace=True)
    else:
        # quantize
        # 对嵌入模块进行量化
        for _, emb_module in embedding_modules:
            # 设置嵌入模块的量化配置为仅基于权重的浮点参数量化配置
            emb_module.qconfig = torch.ao.quantization.float_qparams_weight_only_qconfig

        # 准备模型以进行量化（就地操作）
        torch.ao.quantization.prepare(model, inplace=True)
        # 将模型转换为量化版本（就地操作）
        torch.ao.quantization.convert(model, inplace=True)

        # 检索量化参数：尺度（scale）、零点（zero_points）、去量化后的权重（dequant_weights）、轴（axis）、数据类型（dtype）
        quantize_params: Dict[str, Dict] = {'scales': {}, 'zero_points': {},
                                            'dequant_weights': {}, 'axis': {},
                                            'dtype': {}}

        for name, _ in embedding_modules:
            # 获取模型中量化后的嵌入模块
            quantized_emb = fqn_to_module(model, name)
            assert quantized_emb is not None  # 确保量化后的嵌入模块不为空，满足类型检查

            # 获取量化后的权重
            quantized_weight = quantized_emb.weight()  # type: ignore[operator]
            # 存储量化参数到字典中
            quantize_params['scales'][name] = quantized_weight.q_per_channel_scales()
            quantize_params['zero_points'][name] = quantized_weight.q_per_channel_zero_points()
            quantize_params['dequant_weights'][name] = torch.dequantize(quantized_weight)
            quantize_params['axis'][name] = quantized_weight.q_per_channel_axis()
            quantize_params['dtype'][name] = quantized_weight.dtype

            # 将数据附加到数据稀疏化器（假设存在名字中的点符号，用下划线替换）
            data_sparsifier.add_data(name=name.replace('.', '_'), data=quantize_params['dequant_weights'][name])

        # 执行数据稀疏化器的步骤
        data_sparsifier.step()
        # 压缩数据稀疏化器的掩码
        data_sparsifier.squash_mask()

        for name, _ in embedding_modules:
            # 再次获取量化后的嵌入模块
            quantized_emb = fqn_to_module(model, name)
            assert quantized_emb is not None  # 确保量化后的嵌入模块不为空，满足类型检查
            # 重新量化权重向量
            requantized_vector = torch.quantize_per_channel(quantize_params['dequant_weights'][name],
                                                            scales=quantize_params['scales'][name],
                                                            zero_points=quantize_params['zero_points'][name],
                                                            dtype=quantize_params['dtype'][name],
                                                            axis=quantize_params['axis'][name])

            # 将重新量化后的向量设置为量化嵌入模块的权重（假设操作符忽略类型错误）
            quantized_emb.set_weight(requantized_vector)  # type: ignore[operator]
```