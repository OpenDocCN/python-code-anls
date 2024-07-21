# `.\pytorch\benchmarks\functional_autograd_benchmark\audio_text_models.py`

```
import torchaudio_models as models  # 导入名为 models 的 torchaudio 模块

from utils import check_for_functorch, extract_weights, GetterReturnType, load_weights  # 从 utils 模块中导入函数和类型

import torch  # 导入 PyTorch 库
from torch import nn, Tensor  # 从 PyTorch 中导入 nn 模块和 Tensor 类型


has_functorch = check_for_functorch()  # 检查是否存在 functorch 模块


def get_wav2letter(device: torch.device) -> GetterReturnType:
    N = 10  # 批次大小为 10
    input_frames = 700  # 输入帧数为 700
    vocab_size = 28  # 词汇表大小为 28
    model = models.Wav2Letter(num_classes=vocab_size)  # 使用 Wav2Letter 模型，设置类别数为 vocab_size
    criterion = torch.nn.NLLLoss()  # 使用负对数似然损失函数
    model.to(device)  # 将模型移动到指定设备上
    params, names = extract_weights(model)  # 提取模型的权重参数和名称

    inputs = torch.rand([N, 1, input_frames], device=device)  # 生成随机输入张量，形状为 [N, 1, input_frames]
    labels = torch.rand(N, 3, device=device).mul(vocab_size).long()  # 生成随机标签张量，形状为 [N, 3]

    def forward(*new_params: Tensor) -> Tensor:
        load_weights(model, names, new_params)  # 加载新的权重参数到模型
        out = model(inputs)  # 对输入数据进行前向传播

        loss = criterion(out, labels)  # 计算模型输出与标签之间的损失
        return loss  # 返回损失张量

    return forward, params  # 返回前向传播函数和模型的权重参数


def get_deepspeech(device: torch.device) -> GetterReturnType:
    sample_rate = 16000  # 采样率为 16000 Hz
    window_size = 0.02  # 窗口大小为 0.02 秒
    window = "hamming"  # 窗口函数类型为汉明窗
    audio_conf = dict(
        sample_rate=sample_rate, window_size=window_size, window=window, noise_dir=None
    )  # 设置音频处理的配置参数

    N = 10  # 批次大小为 10
    num_classes = 10  # 类别数为 10
    spectrogram_size = 161  # 频谱图大小为 161
    seq_length = 500  # 每个输入序列的长度为 500（原始大小为 1343）
    target_length = 10  # 目标序列的长度为 10（原始大小为 50）
    labels = torch.rand(num_classes, device=device)  # 生成随机标签张量，形状为 [num_classes]
    inputs = torch.rand(N, 1, spectrogram_size, seq_length, device=device)  # 生成随机输入张量，形状为 [N, 1, spectrogram_size, seq_length]
    
    # 每个输入的序列长度
    inputs_sizes = (
        torch.rand(N, device=device).mul(seq_length * 0.1).add(seq_length * 0.8)
    )
    targets = torch.rand(N, target_length, device=device)  # 生成随机目标张量，形状为 [N, target_length]
    targets_sizes = torch.full((N,), target_length, dtype=torch.int, device=device)  # 生成全为目标长度的张量，形状为 [N]

    model = models.DeepSpeech(
        rnn_type=nn.LSTM,
        labels=labels,
        rnn_hidden_size=1024,
        nb_layers=5,
        audio_conf=audio_conf,
        bidirectional=True,
    )  # 使用 DeepSpeech 模型，配置各种参数

    if has_functorch:
        from functorch.experimental import replace_all_batch_norm_modules_

        replace_all_batch_norm_modules_(model)  # 替换模型中的所有批标准化模块

    model = model.to(device)  # 将模型移动到指定设备上
    criterion = nn.CTCLoss()  # 使用 CTC 损失函数
    params, names = extract_weights(model)  # 提取模型的权重参数和名称

    def forward(*new_params: Tensor) -> Tensor:
        load_weights(model, names, new_params)  # 加载新的权重参数到模型
        out, out_sizes = model(inputs, inputs_sizes)  # 对输入数据进行前向传播

        out = out.transpose(0, 1)  # 调整输出张量的维度顺序以适应 CTC 损失函数的要求

        loss = criterion(out, targets, out_sizes, targets_sizes)  # 计算模型输出与目标之间的损失
        return loss  # 返回损失张量

    return forward, params  # 返回前向传播函数和模型的权重参数


def get_transformer(device: torch.device) -> GetterReturnType:
    N = 64  # 批次大小为 64
    seq_length = 128  # 序列长度为 128
    ntoken = 50  # 词汇表大小为 50
    model = models.TransformerModel(
        ntoken=ntoken, ninp=720, nhead=12, nhid=2048, nlayers=2
    )  # 使用 Transformer 模型，配置各种参数
    model.to(device)  # 将模型移动到指定设备上

    if has_functorch:
        # 为了一致性检查，禁用 dropout
        model.eval()  # 将模型设置为评估模式，即禁用 dropout

    criterion = nn.NLLLoss()  # 使用负对数似然损失函数
    params, names = extract_weights(model)  # 提取模型的权重参数和名称
    # 使用 PyTorch 生成一个形状为 (N, seq_length + 1) 的张量，其中元素随机初始化，设备为指定的设备，然后乘以 ntoken 并转换为 long 类型
    data = torch.rand(N, seq_length + 1, device=device).mul(ntoken).long()
    # 从生成的数据张量中切出输入张量，其形状为 (N, seq_length)，包含第 0 列到第 seq_length 列
    inputs = data.narrow(1, 0, seq_length)
    # 从生成的数据张量中切出目标张量，其形状也为 (N, seq_length)，包含第 1 列到第 seq_length + 1 列
    targets = data.narrow(1, 1, seq_length)

    # 定义一个内部函数 forward，接受任意数量的张量参数 new_params，返回一个张量
    def forward(*new_params: Tensor) -> Tensor:
        # 调用 load_weights 函数加载模型权重，使用给定的 names 和 new_params 参数
        load_weights(model, names, new_params)
        # 将输入张量 inputs 传递给模型 model，计算输出
        out = model(inputs)

        # 计算损失，使用 criterion 计算模型输出 out 和目标张量 targets 的损失值，
        # 其中需要将 out 和 targets 重新形状为 (N * seq_length,) 和 (N * seq_length,)
        loss = criterion(
            out.reshape(N * seq_length, ntoken), targets.reshape(N * seq_length)
        )
        # 返回计算得到的损失张量
        return loss

    # 返回定义的内部函数 forward 和外部参数 params
    return forward, params
# 定义一个函数，返回多头注意力机制的前向传播函数和模型参数
def get_multiheadattn(device: torch.device) -> GetterReturnType:
    # 定义多头注意力机制的相关参数
    embed_dim, nhead, tgt_len, src_len, bsz = 10, 5, 6, 10, 64

    # 创建输入投影容器，包含三个线性层，用于投影查询、键和值
    in_proj = models.InProjContainer(
        torch.nn.Linear(embed_dim, embed_dim, bias=False),
        torch.nn.Linear(embed_dim, embed_dim, bias=False),
        torch.nn.Linear(embed_dim, embed_dim, bias=False),
    )

    # 创建多头注意力机制模型容器，包括头数、输入投影、注意力计算方法和输出投影
    model = models.MultiheadAttentionContainer(
        nhead,
        in_proj,
        models.ScaledDotProduct(),
        torch.nn.Linear(embed_dim, embed_dim, bias=False),
    )

    # 将模型移动到指定设备（GPU或CPU）
    model.to(device)

    # 提取模型的权重参数及其名称
    params, names = extract_weights(model)

    # 生成随机查询、键和值张量，用于前向传播计算
    query = torch.rand((tgt_len, bsz, embed_dim), device=device)
    key = value = torch.rand((src_len, bsz, embed_dim), device=device)

    # 创建注意力掩码，随机生成二进制张量并转换为布尔型
    attn_mask_2D = torch.randint(0, 2, (tgt_len, src_len), device=device).to(torch.bool)
    
    # 创建键和值的偏置张量
    bias_k = bias_v = torch.rand((1, 1, embed_dim), device=device)

    # 复制并重塑偏置张量，以匹配多头注意力机制的需求
    attn_mask = torch.stack([attn_mask_2D] * (bsz * nhead))
    bias_k = bias_k.repeat(1, bsz, 1).reshape(1, bsz * nhead, -1)
    bias_v = bias_v.repeat(1, bsz, 1).reshape(1, bsz * nhead, -1)

    # 定义前向传播函数，接受新的参数并加载到模型中，计算多头注意力机制的输出及注意力权重
    def forward(*new_params: Tensor) -> Tensor:
        load_weights(model, names, new_params)
        mha_output, attn_weights = model(
            query, key, value, attn_mask=attn_mask, bias_k=bias_k, bias_v=bias_v
        )

        # 计算损失，这里简单地将输出张量及注意力权重的所有元素求和作为损失
        loss = mha_output.sum() + attn_weights.sum()

        return loss

    # 返回前向传播函数和模型参数
    return forward, params
```