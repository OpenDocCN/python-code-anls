# `.\pytorch\torch\ao\quantization\fx\lstm_utils.py`

```
# 导入必要的模块和函数
import copy  # 导入深拷贝函数
import operator  # 导入运算符模块
import torch  # 导入PyTorch库
from typing import Any, Callable, Optional, Tuple  # 引入类型提示模块

# 从PyTorch的量化模块中导入所需的类和函数
from torch.ao.quantization import (
    default_weight_observer,  # 默认权重观察器
    default_weight_fake_quant,  # 默认权重伪量化器
    FakeQuantizeBase,  # 伪量化基类
    QConfig,  # 量化配置类
    QConfigMapping,  # 量化配置映射类
)
from torch.ao.quantization.backend_config import BackendConfig  # 后端配置类
from torch.ao.quantization.observer import _PartialWrapper  # 偏函数包装器
from torch.ao.quantization.quantize_fx import (
    convert_to_reference_fx,  # 转换为参考FX
    prepare_fx,  # 准备FX
)

# TODO: 将所有来自fx/utils.py的LSTM实用函数移动到此文件中
def _get_lstm_with_individually_observed_parts(
    float_lstm: torch.nn.LSTM,  # 输入浮点型LSTM模型
    example_inputs: Tuple[Any, ...],  # LSTM模块前向函数的示例输入
    backend_config: Optional[BackendConfig] = None,  # 可选的后端配置参数，默认为None
    linear_output_obs_ctr: Optional[_PartialWrapper] = None,  # 线性输出的观察器或伪量化器
    sigmoid_obs_ctr: Optional[_PartialWrapper] = None,  # sigmoid激活函数的观察器或伪量化器
    tanh_obs_ctr: Optional[_PartialWrapper] = None,  # tanh激活函数的观察器或伪量化器
    cell_state_obs_ctr: Optional[_PartialWrapper] = None,  # 细胞状态的观察器或伪量化器
    hidden_state_obs_ctr: Optional[_PartialWrapper] = None,  # 隐藏状态和输出的观察器或伪量化器
) -> torch.ao.nn.quantizable.LSTM:
    """
    返回一个带有特定观察器或伪量化器的`torch.ao.nn.quantizable.LSTM`，它由`torch.nn.LSTM`
    创建而来，其内部操作或子模块分配了指定的观察器或伪量化器。

    在急切模式和FX图模式量化中，`torch.ao.nn.quantizable.LSTM`作为一个观察的自定义模块使用，
    负责插入自己的观察器。默认情况下，所有内部操作继承父自定义模块的QConfig。
    希望覆盖此行为的用户可以扩展`torch.ao.nn.quantizable.LSTM`并使用此辅助函数来定制观察器插入逻辑。

    用于将浮点模块转换为在自定义模块流程中的观察模块。

    Args:
        `float_lstm`: 浮点LSTM模块
        `example_inputs`: LSTM模块前向函数的示例输入
        `backend_config`: 用于观察LSTM模块的BackendConfig
        `linear_output_obs_ctr`: 线性输出的观察器或伪量化器，其中W是权重矩阵，b是偏置，x是输入或上一层的隐藏状态（如果有的话）
        `sigmoid_obs_ctr`: sigmoid激活函数的观察器或伪量化器
        `tanh_obs_ctr`: tanh激活函数的观察器或伪量化器
        `cell_state_obs_ctr`: 细胞状态的观察器或伪量化器
        `hidden_state_obs_ctr`: 隐藏状态和输出的观察器或伪量化器

    Return:
        分配了指定观察器或伪量化器给内部操作的`torch.ao.nn.quantizable.LSTM`
    """
    def make_qconfig(obs_ctr: _PartialWrapper) -> QConfig:
        """
        Make a QConfig with fixed qparams observers or fake quantizes.
        创建一个带有固定量化参数观察器或伪量化器的 QConfig。
        """
        if isinstance(obs_ctr(), FakeQuantizeBase):
            weight = default_weight_fake_quant
        else:
            weight = default_weight_observer
        return QConfig(activation=obs_ctr, weight=weight)
        # 根据观察器类型选择默认的权重参数，并返回相应的 QConfig 对象

    quantizable_lstm = torch.ao.nn.quantizable.LSTM(
        float_lstm.input_size, float_lstm.hidden_size, float_lstm.num_layers, float_lstm.bias,
        float_lstm.batch_first, float_lstm.dropout, float_lstm.bidirectional)
    quantizable_lstm.qconfig = float_lstm.qconfig
    # 创建一个支持量化的 LSTM 模型，设置其量化配置为与浮点模型相同的配置

    for idx in range(float_lstm.num_layers):
        quantizable_lstm.layers[idx] = torch.ao.nn.quantizable.modules.rnn._LSTMLayer.from_float(float_lstm,
                                                                                                 idx,
                                                                                                 float_lstm.qconfig,
                                                                                                 batch_first=False)
        # 逐层将浮点 LSTM 模型转换为支持量化的 LSTM 模型

    # 为 LSTM 单元构建 QConfigMapping
    # 注意：FloatFunctional 的量化配置将在下面单独配置
    cell_qm = QConfigMapping().set_global(float_lstm.qconfig)  # type: ignore[arg-type]
    # 创建一个全局的 QConfigMapping，使用浮点 LSTM 的量化配置
    if sigmoid_obs_ctr is not None:
        cell_qm.set_module_name("input_gate", make_qconfig(sigmoid_obs_ctr))
        cell_qm.set_module_name("forget_gate", make_qconfig(sigmoid_obs_ctr))
        cell_qm.set_module_name("output_gate", make_qconfig(sigmoid_obs_ctr))
        # 如果存在 sigmoid 观察器，为输入门、遗忘门和输出门设置相应的量化配置
    if tanh_obs_ctr is not None:
        cell_qm.set_module_name("cell_gate", make_qconfig(tanh_obs_ctr))
        # 如果存在 tanh 观察器，为细胞状态门设置相应的量化配置

    # 将观察器插入每个 LSTM 单元中
    # TODO: 可能需要使其在 layer_bw 上工作
    for layer in quantizable_lstm.layers:
        # 获取当前层的前向层的单元（cell）
        cell = layer.layer_fw.cell
        # 使用 prepare_fx 函数对单元进行准备，包括量化模型 (cell_qm) 和示例输入 (example_inputs)，同时传递后端配置 (backend_config)
        cell = prepare_fx(cell, cell_qm, example_inputs, backend_config=backend_config)
        # HACK: 手动替换激活后处理函数，用于以下操作。
        # 这对于 FloatFunctional 操作是必要的，因为当前在 FX 图模式量化中无法配置这些操作。
        # 这是因为在追踪后，FloatFunctional 模块在图中会消失。
        # 将来，我们应该重新编写可量化的 LSTM，而不使用 FloatFunctionals。
        op_index_to_activation_post_process_ctr = {
            (torch.add, 0): linear_output_obs_ctr,  # gates.add
            (torch.mul, 0): cell_state_obs_ctr,  # fgate_cx.mul
            (torch.mul, 1): cell_state_obs_ctr,  # igate_cgate.mul
            (torch.add, 1): cell_state_obs_ctr,  # fgate_cx_igate_cgate.add
            (torch.mul, 2): hidden_state_obs_ctr,  # ogate_cy.mul
        }
        add_count = 0
        mul_count = 0
        # 遍历单元的图中的节点
        for node in cell.graph.nodes:
            op_index: Optional[Tuple[Callable, int]] = None  # 例如 (torch.add, 1)
            # 如果节点目标是 torch.add
            if node.target == torch.add:
                op_index = (torch.add, add_count)
                add_count += 1
            # 如果节点目标是 torch.mul
            elif node.target == torch.mul:
                op_index = (torch.mul, mul_count)
                mul_count += 1
            else:
                # 如果既不是 torch.add 也不是 torch.mul，则跳过此节点
                continue
            # 如果 op_index 不在 op_index_to_activation_post_process_ctr 中，则跳过
            if op_index not in op_index_to_activation_post_process_ctr:
                continue
            # 确保节点的用户数为 1
            assert len(node.users) == 1
            # 获取下一个节点用户的名称作为激活后处理名称
            activation_post_process_name = next(iter(node.users.keys())).name
            # 获取对应的激活后处理计数器
            activation_post_process_ctr = op_index_to_activation_post_process_ctr[op_index]
            # 如果激活后处理计数器不为空，则设置单元的属性为此激活后处理计数器的实例
            if activation_post_process_ctr is not None:
                setattr(cell, activation_post_process_name, activation_post_process_ctr())
        # 更新当前层的前向层的单元为处理后的单元
        layer.layer_fw.cell = cell
    # 返回更新后的可量化 LSTM
    return quantizable_lstm
# 根据观察到的 LSTM 模型创建一个量化后的 `torch.ao.nn.quantized.LSTM` 模块
def _get_reference_quantized_lstm_module(
    observed_lstm: torch.ao.nn.quantizable.LSTM,
    backend_config: Optional[BackendConfig] = None,
) -> torch.ao.nn.quantized.LSTM:
    """
    从经过 `prepare_fx` 插入观察器或伪量化的 `torch.ao.nn.quantizable.LSTM` 创建一个
    `torch.ao.nn.quantized.LSTM` 模块，例如从 `_get_lstm_with_individually_observed_parts` 获得。

    这个函数用于在自定义模块流程中将观察到的模块转换为量化模块。

    Args:
        `observed_lstm`: 通过 `prepare_fx` 观察到的 `torch.ao.nn.quantizable.LSTM` 模块
        `backend_config`: 用于生成参考量化模型的 BackendConfig

    Return:
        一个参考的 `torch.ao.nn.quantized.LSTM` 模块。
    """
    # 创建一个 `torch.ao.nn.quantized.LSTM` 对象，使用与 `observed_lstm` 相同的参数
    quantized_lstm = torch.ao.nn.quantized.LSTM(
        observed_lstm.input_size, observed_lstm.hidden_size, observed_lstm.num_layers,
        observed_lstm.bias, observed_lstm.batch_first, observed_lstm.dropout,
        observed_lstm.bidirectional)

    # 遍历每一层量化 LSTM 的层
    for i, layer in enumerate(quantized_lstm.layers):
        # 深拷贝观察到的 LSTM 模块的第 `i` 层的 `cell`
        cell = copy.deepcopy(observed_lstm.layers.get_submodule(str(i)).layer_fw.cell)  # type: ignore[union-attr]
        # 将 `cell` 转换为参考效果图形式，使用指定的 `backend_config`
        cell = convert_to_reference_fx(cell, backend_config=backend_config)  # type: ignore[arg-type]
        assert isinstance(cell, torch.fx.GraphModule)

        # HACK: 手动移除输入量化节点和输出去量化节点，
        # 因为自定义模块目前要求 quint8 输入和输出。注意，这个功能理论上通过 `PrepareCustomConfig`
        # 的 `set_input_quantized_indexes` 和 `set_output_quantized_indexes` 处理，
        # 但是目前该 API 不处理元组输入和输出，所以我们目前必须手动处理。
        for node in cell.graph.nodes:
            if node.target == torch.quantize_per_tensor:
                arg = node.args[0]
                # 移除 quantize(x)，quantize(hidden[0]) 和 quantize(hidden[1])
                if arg.target == "x" or (arg.target == operator.getitem and arg.args[0].target == "hidden"):
                    with cell.graph.inserting_before(node):
                        node.replace_all_uses_with(arg)
                        cell.graph.erase_node(node)
            if node.target == "output":
                # 移除输出元组中的所有去量化节点
                for arg in node.args[0]:
                    with cell.graph.inserting_before(node):
                        node.replace_input_with(arg, arg.args[0])
        # 消除无效代码
        cell.graph.eliminate_dead_code()
        # 重新编译处理后的效果图
        cell.recompile()
        # 将处理后的 `cell` 设置回量化 LSTM 的相应层
        layer.layer_fw.cell = cell

    # 返回量化后的 LSTM 模块
    return quantized_lstm
```