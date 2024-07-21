# `.\pytorch\benchmarks\fastrnns\factory.py`

```
# 导入命名元组类型和类型提示
from collections import namedtuple
from typing import List, Tuple

# 导入PyTorch相关模块
import torch
from torch import Tensor

# 导入自定义的单元模块
from .cells import flat_lstm_cell, lstm_cell, premul_lstm_cell, premul_lstm_cell_no_bias


# list[list[T]] -> list[T]
# 将嵌套的列表展开为一维列表
def flatten_list(lst):
    result = []
    for inner in lst:
        result.extend(inner)
    return result


"""
定义一个创建器函数:
(options) -> (inputs, params, forward, backward_setup, backward)
inputs: 传递给 'forward' 的输入。可以直接调用 forward(*inputs)。
params: 所有 requires_grad=True 的参数的列表。
forward: 函数 / 图执行器 / 模块
    可以使用创建器的输出调用 rnn(rnn_inputs)。
backward_setup: backward_inputs = backward_setup(*outputs)
    然后，将 backward_inputs 传递给 backward。如果为 None，则假定为标识函数。
backward: 给定 `output = backward_setup(*forward(*inputs))`，执行反向传播。如果为 None，则不执行任何操作。

fastrnns.bench 用于前向和后向调用的计时。
"""


# 定义命名元组 ModelDef，包含输入、参数、前向函数、反向设置和反向函数
ModelDef = namedtuple(
    "ModelDef", ["inputs", "params", "forward", "backward_setup", "backward"]
)


# 根据 LSTM 输出设置简单的反向传播设置
def lstm_backward_setup(lstm_outputs, seed=None):
    hx, _ = lstm_outputs
    return simple_backward_setup(hx, seed)


# 简单的反向传播设置函数，生成随机梯度用于反向传播
def simple_backward_setup(output, seed=None):
    assert isinstance(output, torch.Tensor)
    if seed:
        torch.manual_seed(seed)
    grad_output = torch.randn_like(output)
    return output, grad_output


# 简单的反向传播函数，执行梯度传播
def simple_backward(output, grad_output, **kwargs):
    return output.backward(grad_output, **kwargs)


# 创建 PyTorch LSTM 模型的创建器函数
def pytorch_lstm_creator(**kwargs):
    # 调用 lstm_inputs 函数获取输入和隐藏状态，以及模块对象
    input, hidden, _, module = lstm_inputs(return_module=True, **kwargs)
    return ModelDef(
        inputs=[input, hidden],
        params=flatten_list(module.all_weights),
        forward=module,
        backward_setup=lstm_backward_setup,
        backward=simple_backward,
    )


# 创建 LSTM 模型的创建器函数
def lstm_creator(script=True, **kwargs):
    # 调用 lstm_inputs 函数获取输入、隐藏状态、参数和模块对象
    input, hidden, params, _ = lstm_inputs(return_module=False, **kwargs)
    inputs = [input, hidden] + params[0]
    return ModelDef(
        inputs=inputs,
        params=flatten_list(params),
        forward=lstm_factory(lstm_cell, script),
        backward_setup=lstm_backward_setup,
        backward=simple_backward,
    )


# 创建带有 LayerNorm 的 LSTM 模型的创建器函数
def lnlstm_creator(script=True, decompose_layernorm=False, **kwargs):
    assert script is True
    # 从自定义 LSTM 模块中导入脚本化的 LayerNorm LSTM
    from .custom_lstms import script_lnlstm

    input_size = kwargs["inputSize"]
    hidden_size = kwargs["hiddenSize"]
    seq_len = kwargs["seqLength"]
    batch_size = kwargs["miniBatch"]
    
    # 创建一个脚本化的 LayerNorm LSTM 模型对象，移到 GPU 上
    ge = script_lnlstm(
        input_size, hidden_size, 1, decompose_layernorm=decompose_layernorm
    ).cuda()

    # 创建输入张量和初始状态张量
    input = torch.randn(seq_len, batch_size, input_size, device="cuda")
    states = [
        (
            torch.randn(batch_size, hidden_size, device="cuda"),
            torch.randn(batch_size, hidden_size, device="cuda"),
        )
    ]
    # 返回一个 ModelDef 对象，其中包含输入、参数、前向传播函数、反向传播设置和反向传播函数
    return ModelDef(
        # 指定输入，通常包括输入数据和状态
        inputs=[input, states],
        # 获取模型参数并指定为参数列表
        params=ge.parameters(),
        # 指定前向传播函数，通常是模型的主要计算逻辑
        forward=ge,
        # 指定反向传播设置，通常是一个函数或方法，用于设置反向传播的计算
        backward_setup=lstm_backward_setup,
        # 指定反向传播函数，用于计算梯度并更新模型参数
        backward=simple_backward,
    )
# 创建一个函数来生成包含 dropout 的 LSTM 模型定义
def dropoutlstm_creator(script=True, **kwargs):
    # 确保参数 script 被设置为 True
    assert script is True
    # 导入自定义 LSTM 状态和 LSTM 实现模块
    from .custom_lstms import LSTMState, script_lstm
    
    # 从参数中获取输入大小、隐藏层大小、序列长度、批大小和层数
    input_size = kwargs["inputSize"]
    hidden_size = kwargs["hiddenSize"]
    seq_len = kwargs["seqLength"]
    batch_size = kwargs["miniBatch"]
    num_layers = kwargs["numLayers"]
    
    # 创建一个带有 dropout 功能的 LSTM 模型并移至 GPU（cuda）
    ge = script_lstm(input_size, hidden_size, num_layers, dropout=True).cuda()

    # 生成随机输入张量，尺寸为 (序列长度, 批大小, 输入大小)，在 GPU 上
    input = torch.randn(seq_len, batch_size, input_size, device="cuda")
    
    # 初始化 LSTM 的状态列表，每层都有一个状态对象，都在 GPU 上
    states = [
        LSTMState(
            torch.randn(batch_size, hidden_size, device="cuda"),  # 初始隐藏状态张量
            torch.randn(batch_size, hidden_size, device="cuda"),  # 初始记忆单元张量
        )
        for _ in range(num_layers)
    ]
    
    # 返回 LSTM 模型的定义，包括输入、参数、前向函数、反向设置和反向函数
    return ModelDef(
        inputs=[input, states],
        params=ge.parameters(),
        forward=ge,
        backward_setup=lstm_backward_setup,
        backward=simple_backward,
    )


# 创建一个函数来生成带预乘的 LSTM 模型定义
def lstm_premul_creator(script=True, **kwargs):
    # 获取 LSTM 输入、隐藏状态、参数和模型定义的元组
    input, hidden, params, _ = lstm_inputs(return_module=False, **kwargs)
    
    # 构建输入列表，包括输入、隐藏状态和参数
    inputs = [input, hidden] + params[0]
    
    # 返回 LSTM 模型的定义，包括输入、参数、前向函数、反向设置和反向函数
    return ModelDef(
        inputs=inputs,
        params=flatten_list(params),
        forward=lstm_factory_premul(premul_lstm_cell, script),
        backward_setup=lstm_backward_setup,
        backward=simple_backward,
    )


# 创建一个函数来生成带预乘和偏置的 LSTM 模型定义
def lstm_premul_bias_creator(script=True, **kwargs):
    # 获取 LSTM 输入、隐藏状态、参数和模型定义的元组
    input, hidden, params, _ = lstm_inputs(return_module=False, **kwargs)
    
    # 构建输入列表，包括输入、隐藏状态和参数
    inputs = [input, hidden] + params[0]
    
    # 返回 LSTM 模型的定义，包括输入、参数、前向函数、反向设置和反向函数
    return ModelDef(
        inputs=inputs,
        params=flatten_list(params),
        forward=lstm_factory_premul_bias(premul_lstm_cell_no_bias, script),
        backward_setup=lstm_backward_setup,
        backward=simple_backward,
    )


# 创建一个函数来生成简单 LSTM 模型定义
def lstm_simple_creator(script=True, **kwargs):
    # 获取 LSTM 输入、隐藏状态、参数和模型定义的元组
    input, hidden, params, _ = lstm_inputs(return_module=False, **kwargs)
    
    # 构建输入列表，包括输入、第一层隐藏状态和参数
    inputs = [input] + [h[0] for h in hidden] + params[0]
    
    # 返回 LSTM 模型的定义，包括输入、参数、前向函数、反向设置和反向函数
    return ModelDef(
        inputs=inputs,
        params=flatten_list(params),
        forward=lstm_factory_simple(flat_lstm_cell, script),
        backward_setup=lstm_backward_setup,
        backward=simple_backward,
    )


# 创建一个函数来生成多层 LSTM 模型定义
def lstm_multilayer_creator(script=True, **kwargs):
    # 获取 LSTM 输入、隐藏状态、参数和模型定义的元组
    input, hidden, params, _ = lstm_inputs(return_module=False, **kwargs)
    
    # 构建输入列表，包括输入、所有隐藏状态和参数（已展开）
    inputs = [input, hidden, flatten_list(params)]
    
    # 返回 LSTM 模型的定义，包括输入、参数、前向函数、反向设置和反向函数
    return ModelDef(
        inputs=inputs,
        params=flatten_list(params),
        forward=lstm_factory_multilayer(lstm_cell, script),
        backward_setup=lstm_backward_setup,
        backward=simple_backward,
    )


# 创建一个函数来生成 ImageNet CNN 模型定义
def imagenet_cnn_creator(arch, jit=True):
    # 内部函数，用于在指定设备上创建模型定义
    def creator(device="cuda", **kwargs):
        # 创建指定架构的模型并将其移至指定设备（cuda）
        model = arch().to(device)
        
        # 生成一个随机输入张量，形状为 (32, 3, 224, 224)，在指定设备上
        x = torch.randn(32, 3, 224, 224, device=device)
        
        # 如果启用 JIT 编译，则对模型进行跟踪
        if jit:
            model = torch.jit.trace(model, x)
        
        # 返回模型定义，包括输入、参数、前向函数、反向设置和反向函数
        return ModelDef(
            inputs=(x,),
            params=list(model.parameters()),
            forward=model,
            backward_setup=simple_backward_setup,
            backward=simple_backward,
        )

    # 返回内部函数 creator 作为结果
    return creator


# 定义用于可变长度 LSTM 输入的函数，后续代码未提供，暂无法继续注释
def varlen_lstm_inputs(
    minlen=30,                   # 设置最小长度为30
    maxlen=100,                  # 设置最大长度为100
    numLayers=1,                 # 设置层数为1
    inputSize=512,               # 设置输入大小为512
    hiddenSize=512,              # 设置隐藏层大小为512
    miniBatch=64,                # 设置小批量大小为64
    return_module=False,         # 设置返回模块为False
    device="cuda",               # 设置设备为cuda（GPU加速）
    seed=None,                   # 设置随机种子为None
    **kwargs,                    # 其他可选关键字参数，以字典形式传递
    ):
        # 如果指定了种子值，则设置随机数种子
        if seed is not None:
            torch.manual_seed(seed)
        # 生成长度随机的整数序列，表示每个输入序列的长度
        lengths = torch.randint(
            low=minlen, high=maxlen, size=[miniBatch], dtype=torch.long, device=device
        )
        # 根据长度随机生成输入序列 x，每个序列长度不同
        x = [torch.randn(length, inputSize, device=device) for length in lengths]
        # 随机生成初始的隐藏状态 hx 和细胞状态 cx
        hx = torch.randn(numLayers, miniBatch, hiddenSize, device=device)
        cx = torch.randn(numLayers, miniBatch, hiddenSize, device=device)
        # 创建 LSTM 模型对象 lstm
        lstm = torch.nn.LSTM(inputSize, hiddenSize, numLayers).to(device)

        if return_module:
            # 如果需要返回 LSTM 模型对象及其权重参数，则返回相应信息
            return x, lengths, (hx, cx), lstm.all_weights, lstm
        else:
            # 否则，返回不包含模型对象的相关信息
            # 注意：lstm.all_weights 的格式是 (wih, whh, bih, bhh) 的列表，每层一个元组
            return x, lengths, (hx, cx), lstm.all_weights, None


def varlen_lstm_backward_setup(forward_output, seed=None):
    # 如果指定了种子值，则设置随机数种子
    if seed:
        torch.manual_seed(seed)
    rnn_utils = torch.nn.utils.rnn
    # 获取前向传播输出中的序列 sequences
    sequences = forward_output[0]
    # 对序列进行填充操作，使其长度一致
    padded = rnn_utils.pad_sequence(sequences)
    # 随机生成一个与 padded 形状相同的梯度张量 grad
    grad = torch.randn_like(padded)
    # 返回填充后的序列和随机梯度
    return padded, grad


def varlen_pytorch_lstm_creator(**kwargs):
    rnn_utils = torch.nn.utils.rnn
    # 调用 varlen_lstm_inputs 函数获取输入序列 sequences、隐藏状态 hidden、LSTM 模型 module
    sequences, _, hidden, _, module = varlen_lstm_inputs(return_module=True, **kwargs)

    def forward(sequences, hidden):
        # 将序列列表 sequences 打包成 PackedSequence 格式
        packed = rnn_utils.pack_sequence(sequences, enforce_sorted=False)
        # 使用 LSTM 模型进行前向传播，得到输出 out 和新的隐藏状态 new_hidden
        out, new_hidden = module(packed, hidden)
        # 将 PackedSequence 格式的输出解包成普通张量 padded 和长度列表 lengths
        padded, lengths = rnn_utils.pad_packed_sequence(out)
        # 注意：存储输出结果时最好保留其填充形式，但这可能不利于损失计算。
        # 解包输出可以使反向传播过程慢 2 倍。
        # 返回解包后的输出和新的隐藏状态
        return padded, new_hidden

    # 返回一个 ModelDef 对象，包含输入、参数、前向传播函数、反向传播设置函数和简单反向传播函数
    return ModelDef(
        inputs=[sequences, hidden],
        params=flatten_list(module.all_weights),
        forward=forward,
        backward_setup=lstm_backward_setup,  # 使用了未定义的函数 lstm_backward_setup，可能是笔误，应该是 varlen_lstm_backward_setup
        backward=simple_backward,
    )


def varlen_lstm_factory(cell, script):
    def dynamic_rnn(
        sequences: List[Tensor],
        hiddens: Tuple[Tensor, Tensor],
        wih: Tensor,
        whh: Tensor,
        bih: Tensor,
        bhh: Tensor,
    ) -> Tuple[List[Tensor], Tuple[List[Tensor], List[Tensor]]]:
        # 解包隐藏状态 hiddens
        hx, cx = hiddens
        # 将隐藏状态 hx 按照第一维度（batch 维度）解绑为列表 hxs
        hxs = hx.unbind(1)
        # 将细胞状态 cx 按照第一维度解绑为列表 cxs
        cxs = cx.unbind(1)
        # 存储每个序列的输出和最终的隐藏状态列表
        outputs = []
        hx_outs = []
        cx_outs = []

        # 遍历每个 batch 中的序列
        for batch in range(len(sequences)):
            output = []
            # 获取当前 batch 的初始隐藏状态 hy 和细胞状态 cy
            hy, cy = hxs[batch], cxs[batch]
            # 解包当前序列的输入 inputs
            inputs = sequences[batch].unbind(0)

            # 遍历当前序列的每个时间步
            for seq_idx in range(len(inputs)):
                # 对序列的当前时间步输入进行 RNN 单元运算，更新隐藏状态和细胞状态
                hy, cy = cell(
                    inputs[seq_idx].unsqueeze(0), (hy, cy), wih, whh, bih, bhh
                )
                # 将当前时间步的隐藏状态输出加入 output 列表
                output += [hy]
            # 将当前序列的输出列表转换为张量并加入 outputs 列表
            outputs += [torch.stack(output)]
            # 将当前 batch 的最终隐藏状态 hy 和细胞状态 cy 加入对应的列表
            hx_outs += [hy.unsqueeze(0)]
            cx_outs += [cy.unsqueeze(0)]

        # 返回所有序列的输出列表和最终的隐藏状态元组
        return outputs, (hx_outs, cx_outs)
    # 如果 script 参数为真，将 cell 和 dynamic_rnn 脚本化（即用 Torch 的 JIT 编译成脚本）
    if script:
        # 将 cell 用 Torch 的 JIT 编译成脚本
        cell = torch.jit.script(cell)
        # 将 dynamic_rnn 用 Torch 的 JIT 编译成脚本
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    # 返回经过可能脚本化处理的 dynamic_rnn 对象
    return dynamic_rnn
# 创建一个变长 LSTM 模型的构造函数
def varlen_lstm_creator(script=False, **kwargs):
    # 调用 varlen_lstm_inputs 函数获取序列、隐藏状态、参数等
    sequences, _, hidden, params, _ = varlen_lstm_inputs(return_module=False, **kwargs)
    # 组装输入列表，包括序列、隐藏状态以及参数的组成部分
    inputs = [sequences, hidden] + params[0]
    # 返回一个 ModelDef 对象，包括输入、参数、前向传播函数、反向传播设置和反向传播函数
    return ModelDef(
        inputs=inputs,
        params=flatten_list(params),
        forward=varlen_lstm_factory(lstm_cell, script),  # 使用 varlen_lstm_factory 创建前向传播函数
        backward_setup=varlen_lstm_backward_setup,  # 指定反向传播设置函数
        backward=simple_backward,  # 指定简单的反向传播函数
    )


# layernorm_pytorch_lstm_creator 函数用于创建基于 PyTorch 的 Layernorm LSTM 模型
def layernorm_pytorch_lstm_creator(**kwargs):
    # 调用 lstm_inputs 函数获取输入、隐藏状态、模块等信息，并返回模块对象
    input, hidden, _, module = lstm_inputs(return_module=True, **kwargs)
    # 从 kwargs 中获取批量大小和隐藏状态大小
    batch_size = kwargs["miniBatch"]
    hidden_size = kwargs["hiddenSize"]
    # 创建三个 Layernorm 层对象，并将其放在 GPU 上
    ln_i = torch.nn.LayerNorm(4 * hidden_size).cuda()
    ln_h = torch.nn.LayerNorm(4 * hidden_size).cuda()
    ln_c = torch.nn.LayerNorm(hidden_size).cuda()
    # 生成一个随机张量作为输入，放在 GPU 上
    ln_input1 = torch.randn(batch_size, 4 * hidden_size, device="cuda")

    # 定义前向传播函数
    def forward(input, hidden):
        # 调用模块对象的前向传播函数，获取输出和新的隐藏状态
        out, new_hidden = module(input, hidden)
        # 模拟 Layernorm cudnn LSTM 前向传播的下界，加上 (序列长度 * 三个 Layernorm 单元计算)
        seq_len = len(input.unbind(0))  # 获取序列长度
        hy, cy = new_hidden  # 解包得到隐藏状态

        # 对每个序列长度进行循环，模拟 Layernorm 计算
        for i in range(seq_len):
            ln_i_output = ln_i(ln_input1)  # Layernorm 计算
            ln_h_output = ln_h(ln_input1)  # Layernorm 计算
            cy = ln_c(cy)  # Layernorm 计算

        return out, (hy, cy)  # 返回输出和新的隐藏状态

    # 返回一个 ModelDef 对象，包括输入、参数、前向传播函数、反向传播设置和反向传播函数为空
    return ModelDef(
        inputs=[input, hidden],
        params=flatten_list(module.all_weights),  # 展平模块的所有权重参数
        forward=forward,  # 指定定义的前向传播函数
        backward_setup=lstm_backward_setup,  # 指定 LSTM 反向传播设置函数
        backward=None,  # 指定反向传播函数为空
    )


# stack_weights 函数用于将 LSTM 权重堆叠为指定格式的 packed_weights
def stack_weights(weights):
    # 定义内部函数 unzip_columns，用于解压权重矩阵
    def unzip_columns(mat):
        assert isinstance(mat, list)  # 断言输入矩阵为列表
        assert isinstance(mat[0], list)  # 断言输入矩阵的第一行为列表
        layers = len(mat)  # 获取层数
        columns = len(mat[0])  # 获取列数
        # 返回按列解压的矩阵
        return [[mat[layer][col] for layer in range(layers)] for col in range(columns)]

    # 将输入的权重列表 all_weights 分解为多个列，并将它们堆叠为 packed_weights
    all_weights = weights
    packed_weights = [torch.stack(param) for param in unzip_columns(all_weights)]
    return packed_weights  # 返回堆叠后的权重列表
# returns: x, (hx, cx), all_weights, lstm module with all_weights as params
def lstm_inputs(
    seqLength=100,
    numLayers=1,
    inputSize=512,
    hiddenSize=512,
    miniBatch=64,
    dropout=0.0,
    return_module=False,
    device="cuda",
    seed=None,
):
    # 如果设置了种子，则使用该种子设置随机数生成的种子
    if seed is not None:
        torch.manual_seed(seed)
    # 生成一个随机的输入张量 x，形状为 (seqLength, miniBatch, inputSize)，在指定设备上进行计算
    x = torch.randn(seqLength, miniBatch, inputSize, device=device)
    # 生成随机的初始隐藏状态 hx 和细胞状态 cx，形状为 (numLayers, miniBatch, hiddenSize)，在指定设备上进行计算
    hx = torch.randn(numLayers, miniBatch, hiddenSize, device=device)
    cx = torch.randn(numLayers, miniBatch, hiddenSize, device=device)
    # 创建一个 LSTM 模型，输入大小为 inputSize，隐藏状态大小为 hiddenSize，层数为 numLayers，dropout 为 dropout
    lstm = torch.nn.LSTM(inputSize, hiddenSize, numLayers, dropout=dropout)
    # 如果设备是 cuda，则将 LSTM 模型移动到 GPU 上
    if "cuda" in device:
        lstm = lstm.cuda()

    # 如果 return_module 为 True，则返回 x, (hx, cx), lstm 的所有权重 all_weights 和 lstm 模型；否则返回最后一个参数为 None
    if return_module:
        return x, (hx, cx), lstm.all_weights, lstm
    else:
        # 注意：lstm.all_weights 的格式如下：
        # wih, whh, bih, bhh = lstm.all_weights[layer]
        return x, (hx, cx), lstm.all_weights, None


def lstm_factory(cell, script):
    # 定义一个动态 RNN 函数，接受输入 input，隐藏状态 hidden 和权重 wih, whh, bih, bhh，返回输出和更新后的隐藏状态
    def dynamic_rnn(
        input: Tensor,
        hidden: Tuple[Tensor, Tensor],
        wih: Tensor,
        whh: Tensor,
        bih: Tensor,
        bhh: Tensor,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = hidden
        outputs = []
        # 将输入 input 与权重 wih 的转置相乘，然后按序列维度解绑
        inputs = torch.matmul(input, wih.t()).unbind(0)
        hy, cy = hx[0], cx[0]
        # 遍历每个序列输入，使用给定的 cell 函数更新隐藏状态并记录输出
        for seq_idx in range(len(inputs)):
            hy, cy = cell(inputs[seq_idx], (hy, cy), whh, bih, bhh)
            outputs += [hy]
        # 返回输出序列和更新后的隐藏状态
        return torch.stack(outputs), (hy.unsqueeze(0), cy.unsqueeze(0))

    # 如果 script 为 True，则对 cell 函数和 dynamic_rnn 函数进行 Torch 脚本编译
    if script:
        cell = torch.jit.script(cell)
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    return dynamic_rnn


# premul: we're going to premultiply the inputs & weights
def lstm_factory_premul(premul_cell, script):
    # 定义一个动态 RNN 函数，对输入 input 进行预乘处理，接受隐藏状态 hidden 和权重 wih, whh, bih, bhh，返回输出和更新后的隐藏状态
    def dynamic_rnn(
        input: Tensor,
        hidden: Tuple[Tensor, Tensor],
        wih: Tensor,
        whh: Tensor,
        bih: Tensor,
        bhh: Tensor,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = hidden
        outputs = []
        # 对输入 input 和权重 wih 进行矩阵乘法，然后按序列维度解绑
        inputs = torch.matmul(input, wih.t()).unbind(0)
        hy, cy = hx[0], cx[0]
        # 遍历每个序列输入，使用给定的 premul_cell 函数更新隐藏状态并记录输出
        for seq_idx in range(len(inputs)):
            hy, cy = premul_cell(inputs[seq_idx], (hy, cy), whh, bih, bhh)
            outputs += [hy]
        # 返回输出序列和更新后的隐藏状态
        return torch.stack(outputs), (hy.unsqueeze(0), cy.unsqueeze(0))

    # 如果 script 为 True，则对 premul_cell 函数和 dynamic_rnn 函数进行 Torch 脚本编译
    if script:
        premul_cell = torch.jit.script(premul_cell)
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    return dynamic_rnn


# premul: we're going to premultiply the inputs & weights, and add bias
def lstm_factory_premul_bias(premul_cell, script):
    # 定义一个动态 RNN 函数，对输入 input 进行预乘和加偏置处理，接受隐藏状态 hidden 和权重 wih, whh, bih, bhh，返回输出和更新后的隐藏状态
    def dynamic_rnn(
        input: Tensor,
        hidden: Tuple[Tensor, Tensor],
        wih: Tensor,
        whh: Tensor,
        bih: Tensor,
        bhh: Tensor,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = hidden
        outputs = []
        # 对输入 input 和权重 wih 进行矩阵乘法，然后加上偏置 bih
        inputs = torch.matmul(input, wih.t()) + bih
        inputs = inputs.unbind(0)
        hy, cy = hx[0], cx[0]
        # 遍历每个序列输入，使用给定的 premul_cell 函数更新隐藏状态并记录输出
        for seq_idx in range(len(inputs)):
            hy, cy = premul_cell(inputs[seq_idx], (hy, cy), whh, bhh)
            outputs += [hy]
        # 返回输出序列和更新后的隐藏状态
        return torch.stack(outputs), (hy.unsqueeze(0), cy.unsqueeze(0))

    # 如果 script 为 True，则对 premul_cell 函数和 dynamic_rnn 函数进行 Torch 脚本编译
    if script:
        premul_cell = torch.jit.script(premul_cell)
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    return dynamic_rnn
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # 分解隐藏状态
        hx, cx = hidden
        # 初始化输出列表
        outputs = []
        # 获取输入张量的尺寸
        inpSize = input.size()
        # 为所有时间步骤添加偏置，而不是逐步进行，从而在反向传播中产生单一的降维核心
        # FIXME matmul(x,y) + bias 目前经过了 jit AD，而 AD 中的反向传播公式对此情况未进行优化。使用 mm 和 views 进行绕过处理。
        inpSize = input.size()
        # 计算输入与输入权重转置的矩阵乘积，加上输入隐藏层的偏置
        inputs = torch.mm(input.view(-1, inpSize[2]), wih.t()) + bih
        # 将结果重新调整形状为原始输入的三维形状，并按第一维度解绑
        inputs = inputs.view(inpSize[0], inpSize[1], -1).unbind(0)
        # 获取初始隐藏状态
        hy, cy = hx[0], cx[0]
        # 遍历所有输入序列
        for seq_idx in range(len(inputs)):
            # 应用预乘的单元格函数，更新隐藏状态
            hy, cy = premul_cell(inputs[seq_idx], (hy, cy), whh, bhh)
            # 将更新后的隐藏状态添加到输出列表中
            outputs += [hy]
        # 将所有输出堆叠成张量，并返回最终的隐藏状态元组
        return torch.stack(outputs), (hy.unsqueeze(0), cy.unsqueeze(0))

    # 如果启用了脚本模式，则对预乘单元格和动态RNN进行脚本化处理
    if script:
        premul_cell = torch.jit.script(premul_cell)
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    # 返回脚本化的动态RNN模型
    return dynamic_rnn
# 定义一个工厂函数，用于生成简单版本的动态循环神经网络（RNN）。
# 这个版本不支持元组形式的输入，也不会累积输出列表，主要用于旧版本的 JIT（即时编译器）性能测试。
def lstm_factory_simple(cell, script):
    # 定义一个动态 RNN 函数，接受输入、隐藏状态和细胞状态，以及权重和偏置作为参数
    def dynamic_rnn(input, hx, cx, wih, whh, bih, bhh):
        hy = hx  # 将初始隐藏状态赋值给 hy，用于作用域
        cy = cx  # 将初始细胞状态赋值给 cy，用于作用域
        # 将输入按照序列维度解绑，得到一个列表
        inputs = input.unbind(0)
        # 遍历每个输入序列
        for seq_idx in range(len(inputs)):
            # 调用给定的 cell 函数处理当前输入、隐藏状态和细胞状态，更新隐藏状态和细胞状态
            hy, cy = cell(inputs[seq_idx], hy, cy, wih, whh, bih, bhh)
        # 返回最终的隐藏状态和细胞状态
        return hy, cy

    # 如果需要进行脚本化（script），则将 cell 和 dynamic_rnn 函数转换为 Torch 脚本
    if script:
        cell = torch.jit.script(cell)
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    # 返回动态 RNN 函数
    return dynamic_rnn


# 定义一个工厂函数，用于生成多层版本的动态循环神经网络（RNN）。
# 这个版本支持元组形式的隐藏状态输入，同时适应多层的结构，并会将输出重新堆叠为张量。
def lstm_factory_multilayer(cell, script):
    # 定义一个动态 RNN 函数，接受输入、隐藏状态元组和参数列表作为输入，并返回输出和更新后的隐藏状态元组
    def dynamic_rnn(
        input: Tensor, hidden: Tuple[Tensor, Tensor], params: List[Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        params_stride = 4  # 注意：假设偏置项已存在
        hx, cx = hidden
        hy, cy = hidden  # 将初始隐藏状态和细胞状态赋值给 hy 和 cy，用于作用域
        # 将输入按照序列维度解绑，得到输入序列列表 inputs 和输出列表 outputs
        inputs, outputs = input.unbind(0), []
        # 遍历每一层
        for layer in range(hx.size(0)):
            # 获取当前层的隐藏状态和细胞状态
            hy = hx[layer]
            cy = cx[layer]
            base_idx = layer * params_stride
            # 获取当前层的权重和偏置
            wih = params[base_idx]
            whh = params[base_idx + 1]
            bih = params[base_idx + 2]
            bhh = params[base_idx + 3]
            # 遍历每个输入序列
            for seq_idx in range(len(inputs)):
                # 调用给定的 cell 函数处理当前输入、当前层的隐藏状态和细胞状态，更新隐藏状态和细胞状态
                hy, cy = cell(inputs[seq_idx], (hy, cy), wih, whh, bih, bhh)
                # 将更新后的隐藏状态添加到输出列表
                outputs += [hy]
            # 将当前层的输出列表作为下一层的输入序列
            inputs, outputs = outputs, []
        # 返回所有层处理完后的堆叠输出和最后一层的隐藏状态元组
        return torch.stack(inputs), (hy.unsqueeze(0), cy.unsqueeze(0))

    # 如果需要进行脚本化（script），则将 cell 和 dynamic_rnn 函数转换为 Torch 脚本
    if script:
        cell = torch.jit.script(cell)
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    # 返回动态 RNN 函数
    return dynamic_rnn
```