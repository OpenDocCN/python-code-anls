# `.\pytorch\benchmarks\fastrnns\cells.py`

```py
# 引入从 typing 模块导入的 Tuple 类型
from typing import Tuple

# 引入 PyTorch 库
import torch
from torch import Tensor


def milstm_cell(x, hx, cx, w_ih, w_hh, alpha, beta_i, beta_h, bias):
    # 计算输入乘以输入权重矩阵的转置
    Wx = x.mm(w_ih.t())
    # 计算隐藏状态乘以隐藏状态权重矩阵的转置
    Uz = hx.mm(w_hh.t())

    # 按照论文 https://arxiv.org/pdf/1606.06630.pdf 中的 Section 2.1 进行计算
    gates = alpha * Wx * Uz + beta_i * Wx + beta_h * Uz + bias

    # 将 gates 张量按照第一维度分成四块
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    # 对每个门进行激活函数处理
    ingate = ingate.sigmoid()
    forgetgate = forgetgate.sigmoid()
    cellgate = cellgate.tanh()
    outgate = outgate.sigmoid()

    # 更新细胞状态和隐藏状态
    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * cy.tanh()

    return hy, cy


def lstm_cell(
    input: Tensor,
    hidden: Tuple[Tensor, Tensor],
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
) -> Tuple[Tensor, Tensor]:
    hx, cx = hidden
    # 计算输入乘以输入权重矩阵的转置，加上隐藏状态乘以隐藏状态权重矩阵的转置，再加上偏置
    gates = torch.mm(input, w_ih.t()) + torch.mm(hx, w_hh.t()) + b_ih + b_hh

    # 将 gates 张量按照第一维度分成四块
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    # 对每个门进行激活函数处理
    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    # 更新细胞状态和隐藏状态
    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy


def flat_lstm_cell(
    input: Tensor,
    hx: Tensor,
    cx: Tensor,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
) -> Tuple[Tensor, Tensor]:
    # 计算输入乘以输入权重矩阵的转置，加上隐藏状态乘以隐藏状态权重矩阵的转置，再加上偏置
    gates = torch.mm(input, w_ih.t()) + torch.mm(hx, w_hh.t()) + b_ih + b_hh

    # 将 gates 张量按照第一维度分成四块
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    # 对每个门进行激活函数处理
    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    # 更新细胞状态和隐藏状态
    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy


def premul_lstm_cell(
    igates: Tensor,
    hidden: Tuple[Tensor, Tensor],
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
) -> Tuple[Tensor, Tensor]:
    hx, cx = hidden
    # 计算输入门直接乘以隐藏状态权重矩阵的转置，再加上偏置
    gates = igates + torch.mm(hx, w_hh.t()) + b_ih + b_hh

    # 将 gates 张量按照第一维度分成四块
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    # 对每个门进行激活函数处理
    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    # 更新细胞状态和隐藏状态
    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy


def premul_lstm_cell_no_bias(
    igates: Tensor, hidden: Tuple[Tensor, Tensor], w_hh: Tensor, b_hh: Tensor
) -> Tuple[Tensor, Tensor]:
    hx, cx = hidden
    # 计算输入门直接乘以隐藏状态权重矩阵的转置，再加上偏置
    gates = igates + torch.mm(hx, w_hh.t()) + b_hh

    # 将 gates 张量按照第一维度分成四块
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    # 对每个门进行激活函数处理
    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    # 更新细胞状态和隐藏状态
    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy


def gru_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    # 计算输入乘以输入权重矩阵的转置，加上偏置
    gi = torch.mm(input, w_ih.t()) + b_ih
    # 计算隐藏状态与隐藏层权重的转置矩阵的乘积，并加上隐藏层偏置向量，得到隐藏状态的线性变换结果
    gh = torch.mm(hidden, w_hh.t()) + b_hh
    # 将输入门、重置门和新门的权重进行分块，分别存储在 i_r, i_i, i_n 中
    i_r, i_i, i_n = gi.chunk(3, 1)
    # 将隐藏层门控的权重也进行分块，分别存储在 h_r, h_i, h_n 中
    h_r, h_i, h_n = gh.chunk(3, 1)

    # 计算重置门，使用 sigmoid 函数对输入门和隐藏门的加权和进行变换
    resetgate = torch.sigmoid(i_r + h_r)
    # 计算输入门，使用 sigmoid 函数对输入门和隐藏门的加权和进行变换
    inputgate = torch.sigmoid(i_i + h_i)
    # 计算新的候选值，使用 tanh 函数对输入门和（重置门与隐藏状态的点积）的加权和进行变换
    newgate = torch.tanh(i_n + resetgate * h_n)
    # 计算最终的隐藏状态，结合输入门和新的候选值对当前隐藏状态进行更新
    hy = newgate + inputgate * (hidden - newgate)

    # 返回更新后的隐藏状态
    return hy
# 定义一个使用 ReLU 激活函数的 RNN 单元
def rnn_relu_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    # 计算输入和权重之间的乘积并加上偏置，形成输入门
    igates = torch.mm(input, w_ih.t()) + b_ih
    # 计算隐藏状态和权重之间的乘积并加上偏置，形成隐藏门
    hgates = torch.mm(hidden, w_hh.t()) + b_hh
    # 返回 ReLU 激活函数应用后的输入门和隐藏门之和作为输出
    return torch.relu(igates + hgates)


# 定义一个使用 Tanh 激活函数的 RNN 单元
def rnn_tanh_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    # 计算输入和权重之间的乘积并加上偏置，形成输入门
    igates = torch.mm(input, w_ih.t()) + b_ih
    # 计算隐藏状态和权重之间的乘积并加上偏置，形成隐藏门
    hgates = torch.mm(hidden, w_hh.t()) + b_hh
    # 返回 Tanh 激活函数应用后的输入门和隐藏门之和作为输出
    return torch.tanh(igates + hgates)
```