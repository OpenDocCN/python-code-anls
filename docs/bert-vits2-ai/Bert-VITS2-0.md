# BertVITS2源码解析 0

# `attentions.py`

这段代码定义了一个名为 "LayerNorm" 的类，继承自 PyTor 的 nn.Module 类。这个类的实现与 LSTM 有关，用于对输入数据进行预处理和扩充。

具体来说，这段代码的作用如下：

1. 引入了 math 和 torch 两个模块，以便于在训练和测试时使用。
2. 从 torch 模块中引入了 functional.此函数用于在训练和测试时执行各种前向和后向兼容操作，如矩阵乘法，卷积，激活函数等。
3. 从 torch.nn.functional 中导入 LayerNorm 的实现。
4. 定义了一个名为 "LayerNorm" 的类，其中 LayerNorm 是颗类，包含了三个成员变量：channels、gamma 和 beta。这些成员变量在类初始化时设置好，并在 forward 方法中进行了修改。
5. 在 LayerNorm 的 forward 方法中，实现了一个前向传播的函数，对输入数据进行预处理和扩充。首先将输入数据 transpose(1, -1)，然后执行 LSTM 操作，使用学到的 gamma 和 beta 进行前向计算。最后，对结果进行 transpose(1, -1) 操作，以便于后面计算。


```py
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import logging

logger = logging.getLogger(__name__)


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


```

This is a PyTorch implementation of a Multi-Head Attention module. It takes in an input sequence `x` and a mask `x_mask`, and outputs the attention output `x_attn`.

The implementation uses a multi-head attention mechanism, where each attention head computes a weighted sum of the input features and passes the result through a feed-forward neural network (FFN). The FFN has two linear layers with a dropout rate `p_dropout` between them.

The `MultiHeadAttention` class takes in two hidden channels `h_c` and passes the input through a linear layer with a self-parameterized growth rate `g`. This linear layer is then added to the attention score.

The `attn_layers`, `norm_layers_1`, `norm_layers_2`, `ffn_layers`, `norm_layers_1_linear`, `norm_layers_2_linear` classes are used to normalize the input features and apply the learned features to the input.

The `forward` method defines the forward pass of the class, which applies the multi-head attention mechanism and the learned features to the input.


```py
@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=1,
        p_dropout=0.0,
        window_size=4,
        isflow=True,
        **kwargs
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        # if isflow:
        #  cond_layer = torch.nn.Conv1d(256, 2*hidden_channels*n_layers, 1)
        #  self.cond_pre = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, 1)
        #  self.cond_layer = weight_norm(cond_layer, name='weight')
        #  self.gin_channels = 256
        self.cond_layer_idx = self.n_layers
        if "gin_channels" in kwargs:
            self.gin_channels = kwargs["gin_channels"]
            if self.gin_channels != 0:
                self.spk_emb_linear = nn.Linear(self.gin_channels, self.hidden_channels)
                # vits2 says 3rd block, so idx is 2 by default
                self.cond_layer_idx = (
                    kwargs["cond_layer_idx"] if "cond_layer_idx" in kwargs else 2
                )
                logging.debug(self.gin_channels, self.cond_layer_idx)
                assert (
                    self.cond_layer_idx < self.n_layers
                ), "cond_layer_idx should be less than n_layers"
        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask, g=None):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            if i == self.cond_layer_idx and g is not None:
                g = self.spk_emb_linear(g.transpose(1, 2))
                g = g.transpose(1, 2)
                x = x + g
                x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


```

This is a PyTorch implementation of an encoder and decoder architecture for a neural network. The encoder has two main components: the self-attention mechanism and the feed-forward neural network (FFN). The self-attention mechanism is a multi-head attention layer that computes the attention between the input features and the hidden representations of the encoder. The feed-forward neural network is a fully connected neural network that performs a simple linear transformation of the input features.

The decoder has two main components: the self-attention mechanism and the feed-forward neural network (FFN). The self-attention mechanism is a multi-head attention layer that computes the attention between the input features and the hidden representations of the decoder. The feed-forward neural network is a fully connected neural network that performs a simple linear transformation of the input features.


```py
class Decoder(nn.Module):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=1,
        p_dropout=0.0,
        proximal_bias=False,
        proximal_init=True,
        **kwargs
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init

        self.drop = nn.Dropout(p_dropout)
        self.self_attn_layers = nn.ModuleList()
        self.norm_layers_0 = nn.ModuleList()
        self.encdec_attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.self_attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    proximal_bias=proximal_bias,
                    proximal_init=proximal_init,
                )
            )
            self.norm_layers_0.append(LayerNorm(hidden_channels))
            self.encdec_attn_layers.append(
                MultiHeadAttention(
                    hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                    causal=True,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask, h, h_mask):
        """
        x: decoder input
        h: encoder output
        """
        self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(
            device=x.device, dtype=x.dtype
        )
        encdec_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.self_attn_layers[i](x, x, self_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_0[i](x + y)

            y = self.encdec_attn_layers[i](x, h, encdec_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


```

This is a custom implementation of a RelU activation function, which is a common activation in neural networks. It takes in an input sequence `x`, which has shape `(batch_size, batch_size, sequence_length, sequence_length)`. The output is a Tensor that has shape `(batch_size,)`:

The `relu` function applies the relu activation function to the input sequence `x`. This is achieved by first subtracting the input to a square root to help prevent exploding due to very large input values. The result is then passed through a modulo operation to ensure that the output is within the range `[0, 1)`.

The attention bias function is a custom implementation that adds a bias term to the attention weights for each position in the input sequence. This helps to encourage the model to pay more attention to positions that are closer to the end of the sequence. The function has shape `(1, 1, sequence_length, sequence_length)`, where `sequence_length` is the length of the input sequence, and is used to ensure that the attention weight has the same shape as the input.

The bias term is computed by first projecting the attention weights to a 1D feature space using a learnable parameter `length`. This is then combined with the absolute positional representation of the input sequence, which is computed using the parameter `rel_pos`. The final result is a Tensor that has shape `(1, 1, sequence_length, sequence_length)`.


```py
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        n_heads,
        p_dropout=0.0,
        window_size=None,
        heads_share=True,
        block_length=None,
        proximal_bias=False,
        proximal_init=False,
    ):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
        if self.window_size is not None:
            assert (
                t_s == t_t
            ), "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(
                query / math.sqrt(self.k_channels), key_relative_embeddings
            )
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(
                device=scores.device, dtype=scores.dtype
            )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                assert (
                    t_s == t_t
                ), "Local attention is only available for self-attention."
                block_mask = (
                    torch.ones_like(scores)
                    .triu(-self.block_length)
                    .tril(self.block_length)
                )
                scores = scores.masked_fill(block_mask == 0, -1e4)
        p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_v, t_s
            )
            output = output + self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings
            )
        output = (
            output.transpose(2, 3).contiguous().view(b, d, t_t)
        )  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        """
        x: [b, h, l, d]
        y: [h or 1, m, d]
        ret: [b, h, l, m]
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        2 * self.window_size + 1
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings,
                commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
            )
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[
            :, slice_start_position:slice_end_position
        ]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        batch, heads, length, _ = x.size()
        # Concat columns of pad to shift from relative to absolute indexing.
        x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(
            x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [0, length - 1]])
        )

        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[
            :, :, :length, length - 1 :
        ]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.size()
        # pad along column
        x = F.pad(
            x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]])
        )
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length):
        """Bias for self-attention to encourage attention to close positions.
        Args:
          length: an integer scalar.
        Returns:
          a Tensor with shape [1, 1, length, length]
        """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


```

This is a class definition for a neural network model. The model has a Conv1D layer with a filter channel of `filter_channels`, a kernel size of `kernel_size`, and a dropout rate of `p_dropout`.

It also has a ReLU activation function and a variable called `causal` which is a boolean value indicating whether the model is causal or not. If `causal` is `True`, it adds a causal padding to the input data.

The model has two Conv1D layers with a padding scheme of `self.padding`. The first convolutional layer has a ch downsample layer with a padding of `self.padding(x * x_mask)` and the second convolutional layer has a padding of `self.padding(x * x_mask)`.

The model also has a variable called `activation` which is a string indicating the activation function to use. The final layer is defined with an output of `x * x_mask` which is the output of the model.


```py
class FFN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        filter_channels,
        kernel_size,
        p_dropout=0.0,
        activation=None,
        causal=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal

        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding(x * x_mask))
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x

```

# `bert_gen.py`

这段代码的作用是进行文本转语音（TTS）的预处理，包括去除空白和特殊字符，分离出多个语言的语音信号，将每个语言的语音信号送入预训练的BERT模型中进行语音识别。

具体来说，这段代码包括以下步骤：

1. 读取输入文本，并解析出每句话中的语音信号，包括去除空白、特殊字符和重复字符，以及将每个语言的语音信号送入预训练的BERT模型中。
2. 对于每个输入文本，首先会读取对应的WAV文件，然后将其中的文本信息转换为序列。
3. 对于每个已经转好的文本序列，会使用commons库中的函数进行处理，包括去除空白、特殊字符和重复字符，以及将每个语言的语音信号送入预训练的BERT模型中。
4. 在处理过程中，如果当前的文本已经被预训练的BERT模型所支持，那么会将预训练的模型保存到磁盘上，以便后续使用。
5. 在所有处理完成后，可以将处理好的文本和对应的BERT模型存入一个文件夹中，并使用argparse库中的函数对输入的参数进行设置。


```py
import torch
from multiprocessing import Pool
import commons
import utils
from tqdm import tqdm
from text import cleaned_text_to_sequence, get_bert
import argparse
import torch.multiprocessing as mp


def process_line(line):
    rank = mp.current_process()._identity
    rank = rank[0] if len(rank) > 0 else 0
    if torch.cuda.is_available():
        gpu_id = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{gpu_id}")
    wav_path, _, language_str, text, phones, tone, word2ph = line.strip().split("|")
    phone = phones.split(" ")
    tone = [int(i) for i in tone.split(" ")]
    word2ph = [int(i) for i in word2ph.split(" ")]
    word2ph = [i for i in word2ph]
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    bert_path = wav_path.replace(".wav", ".bert.pt")

    try:
        bert = torch.load(bert_path)
        assert bert.shape[-1] == len(phone)
    except Exception:
        bert = get_bert(text, word2ph, language_str, device)
        assert bert.shape[-1] == len(phone)
        torch.save(bert, bert_path)


```

这段代码是一个Python程序，它的主要目的是从指定文件夹中读取并并行处理训练数据和验证数据。

具体来说，这个程序定义了两个可执行参数：

- `-c` 参数：指定了一个配置文件(config.json)的路径，这个文件中定义了训练和验证数据的文件夹结构、数据集的大小等参数。
- `--num_processes` 参数：指定了一个并行处理的核心数量(处理器数量)，这个数量决定了并行处理的任务分配给多少个线程进行处理。

程序首先定义了这两个参数，然后从`config.json`文件中读取了训练和验证数据的文件夹结构，并从每个文件中读取所有的行，使用`utils.get_hparams_from_file`函数从文件中读取了一些参数的配置信息。

最后，程序创建了一个线程池，使用了`Pool`类将`num_processes`个处理器并行处理输入的行，每个线程使用`process_line`函数对输入行进行处理，最终输出结果。程序使用了`tqdm`模块来展示了每个线程处理行时的百分比完成率。


```py
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/config.json")
    parser.add_argument("--num_processes", type=int, default=2)
    args = parser.parse_args()
    config_path = args.config
    hps = utils.get_hparams_from_file(config_path)
    lines = []
    with open(hps.data.training_files, encoding="utf-8") as f:
        lines.extend(f.readlines())

    with open(hps.data.validation_files, encoding="utf-8") as f:
        lines.extend(f.readlines())

    num_processes = args.num_processes
    with Pool(processes=num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(process_line, lines), total=len(lines)):
            pass

```

# `commons.py`

这段代码定义了一个称为`Padding2D`的类，用于在神经网络中增加额外的维度。

具体来说，代码中定义了两个函数：`init_weights`和`get_padding`。

`init_weights`函数的作用是在创建神经网络模型时初始化每个层的权重。它接收两个参数：`m`表示模型的类名，如果类名包含“Conv”关键字，则执行一些特殊的权重初始化。

`get_padding`函数的作用是在神经网络中计算每个卷积层的输入区域。它接收一个参数：`kernel_size`，表示卷积层的内核大小，以及一个参数`dilation`，表示卷积层输入区域的大小与 kernel_size 的比例因子。函数返回一个计算出的输入区域大小。

整段代码的主要目的是创建一个方便使用的工具，用于在神经网络中添加额外的维度。通过使用`Padding2D`类，可以在创建神经网络模型时自动执行一些初始化和权重分布的工作，从而简化开发者的神经网络模型。


```py
import math
import torch
from torch.nn import functional as F


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


```



这段代码定义了两个函数，一个用于将一个张量的大小变为指定大小的填充张量，另一个是将一个列表中的元素以指定的间隔进行插入。

第一个函数 `convert_pad_shape` 接受一个张量 `pad_shape`，并返回一个新的张量 `pad_shape`。函数首先将输入张量的维度变为相反数，然后将其展开并合并为一个新的张量。最后，函数返回新的填充张量。

第二个函数 `intersperse` 接受一个列表 `lst`，并返回一个新的列表 `result`。函数将在新列表中插入与原始列表相同的元素，并将其长度加倍。最后，函数返回新列表 `result`。

第三个函数 `kl_divergence` 接受两个张量 `m_p` 和 `logs_p`，以及两个张量 `m_q` 和 `logs_q`。函数计算两个张量之间的kl散度。函数首先将两个张量的log木斯映射到各自的均值和标准差上。然后，函数使用exponentials函数计算两个张量的高斯映射。最后，函数将两个高斯映射相减，并计算它们的乘积，得到kl散度。函数返回kl散度。


```py
def convert_pad_shape(pad_shape):
    layer = pad_shape[::-1]
    pad_shape = [item for sublist in layer for item in sublist]
    return pad_shape


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def kl_divergence(m_p, logs_p, m_q, logs_q):
    """KL(P||Q)"""
    kl = (logs_q - logs_p) - 0.5
    kl += (
        0.5 * (torch.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2.0 * logs_q)
    )
    return kl


```



这段代码定义了三个函数，其中第一个函数 `rand_gumbel` 从高斯分布中采样随机数，并返回负对数形式的随机数。第二个函数 `rand_gumbel_like` 类似于第一个函数，但是使用了 NumPy 函数来获取输入张量的形状，同时对输入张量进行了大小调整为张量大小为 1。第三个函数 `slice_segments` 用于对输入张量 `x` 根据给定的 IDS 字符串进行切片，并将切片结果返回。

具体来说，`rand_gumbel` 的实现思路是，对于高斯分布中的任意样本点 `u`，计算一个均匀分布的样本点 $u$ 对数形式的随机数 $z$。这里我们通过将高斯分布中的样本点 $u$ 乘以一个比例因子 $0.99998$，并加上一个偏移量 $0.00001$，来达到防止过采样导致样本点过于集中，从而失去分布的性质。具体地，我们可以得到：

```py
import torch
import numpy as np

def rand_gumbel(shape):
   """Sample from the Gumbel distribution, protect from overflows."""
   uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
   return -torch.log(-torch.log(uniform_samples))
```

`rand_gumbel_like` 的实现则较为简单，直接通过 NumPy 函数获取输入张量的形状，并返回一个形状与输入张量相同的张量。这里我们注意，由于 `rand_gumbel_like` 函数返回的结果是张量，因此我们需要通过 `to` 方法将其张量类型设置为与输入张量相同的类型，即 `x.dtype`。

最后，我们定义了 `slice_segments` 函数，用于对输入张量 `x` 根据给定的 IDS 字符串进行切片。具体地，我们首先使用循环遍历输入张量 `x` 的每一行，然后提取出该行对应 IDS 字符串的起始位置和结束位置，并将起始位置和结束位置之间的所有元素作为输出返回。切片的过程如下：

```py
def slice_segments(x, ids_str, segment_size=4):
   ret = torch.zeros_like(x[:, :, :segment_size])
   for i in range(x.size(0)):
       idx_str = ids_str[i]
       idx_end = idx_str + segment_size
       ret[i] = x[i, :, idx_str:idx_end]
   return ret
```

在实际应用中，我们可以根据需要设置 `segment_size` 参数，以控制切片的粒度大小。


```py
def rand_gumbel(shape):
    """Sample from the Gumbel distribution, protect from overflows."""
    uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
    return -torch.log(-torch.log(uniform_samples))


def rand_gumbel_like(x):
    g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
    return g


def slice_segments(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


```

这段代码定义了一个名为 `rand_slice_segments` 的函数，它接受一个二维张量 `x`，以及一个可选项 `x_lengths`，表示要从中切出的光注意程的长度。函数实现了一个随机选取子张量的算法，并返回两个参数：一个切分后的张量 `ret`，以及一个包含 ID 对象的元组 `ids_str`。

函数首先检查 `x_lengths` 是否为空，如果是，则将其设置为张量的大小 `t`。然后，通过从 `x` 中随机获取一个大小为 `ids_str_max` 的子张量，并将其转换为整数类型，将其赋值给 `ids_str`。接下来，使用 `slice_segments` 函数将张量 `x` 切分为指定长度的子张量，并将结果返回。最后，函数返回切分后的张量 `ret` 和一个包含 ID 对象的元组 `ids_str`。

函数 `get_timing_signal_1d` 接受一个长度为 `length` 的二维张量 `channels` 和每行所需的最低采样率 `min_samples`，以及一个可选项 `max_采样率`。函数返回一个采样率为 `max_采样率` 的 timing signal。

get_timing_signal_1d 函数会将 `channels` 中的每个元素拉伸到其右侧，然后将其复制到一个大小为 `max_采样率` 的新的张量中。通过从每个元素的时间轴上提取一个采样率，将其乘以 `max_采样率` 并将其复制到一个新的张量中，这样每个元素就是一个采样率为 `max_采样率` 的周期性信号。添加周期性信号之后，通过从 `max_采样率` 减去每个周期内 `min_采样率` 的倍数，来获取每个采样点的实际采样率。


```py
def rand_slice_segments(x, x_lengths=None, segment_size=4):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length, dtype=torch.float)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (
        num_timescales - 1
    )
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment
    )
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
    signal = F.pad(signal, [0, 0, 0, channels % 2])
    signal = signal.view(1, channels, length)
    return signal


```



这段代码定义了三个函数，它们用于对一个一维信号的时间戳进行加时差信号(TSC)和复制粘性进位(CA)。

`add_timing_signal_1d`函数接受一个一维信号 `x`，以及最小时间间隔 `min_timescale` 和最大时间间隔 `max_timescale`。它使用 `get_timing_signal_1d` 函数获取 TSC，并将信号加到输入信号上，然后将其转换为与输入信号相同的数据类型，并返回输入信号。

`cat_timing_signal_1d`函数与 `add_timing_signal_1d` 函数类似，但将其结果保存到指定的轴上。它同样使用 `get_timing_signal_1d` 函数获取 TSC，并将信号加到输入信号上，然后将其转换为与输入信号相同的数据类型，并将结果存储在指定的轴上。

`subsequent_mask`函数接受一个长度为 `length` 的信号 `x`。它返回一个长度为 `length`，与输入信号相同的数据类型，并包含从输入信号中提取的只包含符号位置的 mask。

总结起来，这三个函数用于对一个一维信号的时间戳进行加时差信号和复制粘性进位，以便于将其存储到指定的轴上。


```py
def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal.to(dtype=x.dtype, device=x.device)


def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)


def subsequent_mask(length):
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask


```

这段代码定义了一个名为 "fused_add_tanh_sigmoid_multiply" 的函数，接受两个输入张量 "input_a" 和 "input_b"，以及一个具有通道数量的输出张量 "output"。这个函数返回一个包含输出张量中所有元素的产品。

具体来说，这个函数的实现步骤如下：

1. 将输入张量 "input_a" 和 "input_b" 相加，得到一个具有相同通道数目的张量 "in_act"。
2. 对 "in_act" 中的每个通道，应用一个 tanh 函数，得到一个包含了 "in_act" 中所有通道的值的一个张量 "t_act"。
3. 对 "t_act" 中的每个通道，应用一个 sigmoid 函数，得到一个包含了 "t_act" 中所有通道的值的一个张量 "s_act"。
4. 对 "t_act" 和 "s_act" 中的所有通道，将它们相乘，得到一个包含了 "in_act" 中所有通道的值的一个张量 "acts"。
5. 返回 "acts" 中的所有元素，作为输出张量返回。

此外，函数 "convert_pad_shape" 接受一个带有通道数量和填充位置的张量，并返回一个新的张量，其中填充位置和通道数量已经适应输入张量的形状。这个函数可以用于在训练时对输入张量的形状进行修改，以适应模型的输入要求。


```py
@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def convert_pad_shape(pad_shape):
    layer = pad_shape[::-1]
    pad_shape = [item for sublist in layer for item in sublist]
    return pad_shape


```

这段代码定义了三个函数，分别是：

1. `shift_1d(x)`：该函数的作用是移除一个一维向量 `x` 的所有 padding，并在末尾插入新的填充元素，使得 `x` 的长度变为它本身的长度减一。
2. `sequence_mask(length, max_length=None)`：该函数的作用是在一个一维向量 `length` 上创建一个滑动窗口，并在窗口上执行一个循环移位操作。这个窗口的大小是由 `max_length` 参数指定，如果没有传递这个参数，则使用默认的最大长度为 `None`。
3. `generate_path(duration, mask)`：该函数的作用是在一个一维向量 `duration` 上生成一个路径，这个路径由 `mask` 创建的滑动窗口沿当前的 `t_y` 轴方向循环移位，并且最后一个元素不进行移动，即 `mask` 的最后一个元素会被滞留在路径的最后一个位置。


```py
def shift_1d(x):
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
    return x


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """

    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


```

这段代码定义了一个名为 `clip_grad_value_` 的函数，它接受一个参数 `parameters`，一个参数 `clip_value`，和一个参数 `norm_type`。

函数首先检查 `parameters` 是否是一个 PyTorch 的 `Tensor` 类型，如果不是，则将其转换为该类型。然后，函数使用列表过滤器遍历 `parameters` 中的所有元素，检查它们是否有梯度信息。如果 `clip_value` 是一个非空的引用，函数将其转换为浮点数，并将其设置为梯度的 `norm_type` 属性的值。

接下来，函数跟踪其总梯度平方根，并对每个参数应用 clip。最后，函数返回总梯度平方根的 `norm_type` 次方。


```py
def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm

```

# `data_utils.py`

This is a PyTorch implementation of a program that listens to a text file and generates audio files for certain sentences in the text. The text files contain sentences of a certain length and the audio files contain the text and the audio of the sentences. The audio files are in the form of BERT models and the text files are in the form of one-hot encoded text. The program uses the `transformers` library to pre-train the BERT models and then uses the `Turntask` library to generate the audio files for the text. The program can be run on a GPU to speed up the training process.


```py
import os
import random
import torch
import torch.utils.data
from tqdm import tqdm
from loguru import logger
import commons
from mel_processing import spectrogram_torch, mel_spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text
from text import cleaned_text_to_sequence, get_bert

"""Multi speaker version"""


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.spk_map = hparams.spk2id
        self.hparams = hparams

        self.use_mel_spec_posterior = getattr(
            hparams, "use_mel_posterior_encoder", False
        )
        if self.use_mel_spec_posterior:
            self.n_mel_channels = getattr(hparams, "n_mel_channels", 80)

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 300)

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        skipped = 0
        logger.info("Init dataset...")
        for _id, spk, language, text, phones, tone, word2ph in tqdm(
            self.audiopaths_sid_text
        ):
            audiopath = f"{_id}"
            if self.min_text_len <= len(phones) and len(phones) <= self.max_text_len:
                phones = phones.split(" ")
                tone = [int(i) for i in tone.split(" ")]
                word2ph = [int(i) for i in word2ph.split(" ")]
                audiopaths_sid_text_new.append(
                    [audiopath, spk, language, text, phones, tone, word2ph]
                )
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
            else:
                skipped += 1
        logger.info(
            "skipped: "
            + str(skipped)
            + ", total: "
            + str(len(self.audiopaths_sid_text))
        )
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, language, text, phones, tone, word2ph = audiopath_sid_text

        bert, ja_bert, phones, tone, language = self.get_text(
            text, word2ph, phones, tone, language, audiopath
        )

        spec, wav = self.get_audio(audiopath)
        sid = torch.LongTensor([int(self.spk_map[sid])])
        return (phones, spec, wav, sid, tone, language, bert, ja_bert)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "{} {} SR doesn't match target {} SR".format(
                    filename, sampling_rate, self.sampling_rate
                )
            )
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if self.use_mel_spec_posterior:
            spec_filename = spec_filename.replace(".spec.pt", ".mel.pt")
        try:
            spec = torch.load(spec_filename)
        except:
            if self.use_mel_spec_posterior:
                spec = mel_spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.n_mel_channels,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    self.hparams.mel_fmin,
                    self.hparams.mel_fmax,
                    center=False,
                )
            else:
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(self, text, word2ph, phone, tone, language_str, wav_path):
        phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)
        if self.add_blank:
            phone = commons.intersperse(phone, 0)
            tone = commons.intersperse(tone, 0)
            language = commons.intersperse(language, 0)
            for i in range(len(word2ph)):
                word2ph[i] = word2ph[i] * 2
            word2ph[0] += 1
        bert_path = wav_path.replace(".wav", ".bert.pt")
        try:
            bert = torch.load(bert_path)
            assert bert.shape[-1] == len(phone)
        except:
            bert = get_bert(text, word2ph, language_str)
            torch.save(bert, bert_path)
            assert bert.shape[-1] == len(phone), phone

        if language_str == "ZH":
            bert = bert
            ja_bert = torch.zeros(768, len(phone))
        elif language_str == "JP":
            ja_bert = bert
            bert = torch.zeros(1024, len(phone))
        else:
            bert = torch.zeros(1024, len(phone))
            ja_bert = torch.zeros(768, len(phone))
        assert bert.shape[-1] == len(phone), (
            bert.shape,
            len(phone),
            sum(word2ph),
            p1,
            p2,
            t1,
            t2,
            pold,
            pold2,
            word2ph,
            text,
            w2pho,
        )
        phone = torch.LongTensor(phone)
        tone = torch.LongTensor(tone)
        language = torch.LongTensor(language)
        return bert, ja_bert, phone, tone, language

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


```

This is a Python function that performs some text preprocessing and standardization on a given text data. Here is a brief description of what this function does:

1. It takes a text data as input.
2. It splits the input text into a list of texts, where each text is a sequence of integers representing the tokens in the text.
3. It extracts the first n-1 words (where n is the length of the input text) from each text in the input list.
4. It removes special characters (such as `/` and `) from each word.
5. It replaces all the words in each text with lowercase.
6. It normalizes the特别词汇 (such as numbers and days) in each text.
7. It removing stop words (such as "a", "an", etc) from each text.
8. It normalizes the specified JavaScript spell checker.
9. It removes punctuation from each text.
10. It returns the normalized input list.

It should be noted that this function assumes that the input text data only contains words and numbers and not any special characters, and it also assumes that the input text data follows the format of a specific data type. It is also important to note that this function is not production-ready and should be tested and modified to suit the specific requirements of the project it is used for.


```py
class TextAudioSpeakerCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True
        )

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        tone_padded = torch.LongTensor(len(batch), max_text_len)
        language_padded = torch.LongTensor(len(batch), max_text_len)
        bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)
        ja_bert_padded = torch.FloatTensor(len(batch), 768, max_text_len)

        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        tone_padded.zero_()
        language_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        bert_padded.zero_()
        ja_bert_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]

            tone = row[4]
            tone_padded[i, : tone.size(0)] = tone

            language = row[5]
            language_padded[i, : language.size(0)] = language

            bert = row[6]
            bert_padded[i, :, : bert.size(1)] = bert

            ja_bert = row[7]
            ja_bert_padded[i, :, : ja_bert.size(1)] = ja_bert

        return (
            text_padded,
            text_lengths,
            spec_padded,
            spec_lengths,
            wav_padded,
            wav_lengths,
            sid,
            tone_padded,
            language_padded,
            bert_padded,
            ja_bert_padded,
        )


```

This is a class that implements the batching of samples, which can be useful for parallel processing.

It contains the following methods:

* `__init__(self, num_samples, num_features, batch_size, shuffle=True)`: This parameterizes the class.
* `_ bisect(self, x, lo=0, hi=None):` This method is a implementation of the bisection algorithm for finding the index of the element that lies between two indices `lo` and `hi`.
* `__ batch_samples(self, samples):` This method returns a batch of samples.
* `__ shuffle(self):` This method is a simple implementation of the shuffle algorithm, which randomly shuffles an array of elements.
* `__ get_indices(self):` This method returns the indices of the samples in the batch.

The class takes in several parameters, including `num_samples`, `num_features`, `batch_size`, and `shuffle`, which specify the batch size, the number of features per sample, and whether the samples should be shuffled.

The class also contains a `__ bisect` method that is used for finding the index of the element that lies between two indices `lo` and `hi`. This method is implemented using the bisection algorithm, which is a divide-and-conquer algorithm that works well for searching for elements that are approximately equal to the midpoint of two intervals.

This class can be useful for implementing parallel batching of samples, which can improve the efficiency of processing large datasets that have many features.


```py
class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        try:
            for i in range(len(buckets) - 1, 0, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)
            assert all(len(bucket) > 0 for bucket in buckets)
        # When one bucket is not traversed
        except Exception as e:
            print("Bucket warning ", e)
            for i in range(len(buckets) - 1, -1, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            if len_bucket == 0:
                continue
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size

```

# `losses.py`

这段代码定义了两个函数：feature_loss 和 discriminator_loss。它们都是基于 deep learning 中常见的损失函数（如 L2 损失、L1 损失等）。

1. feature_loss 的作用是计算特征（fmap_r 和 fmap_g）的损失。具体来说，它通过遍历二进制掩码（dr 和 dg）中的元素，计算每个元素与目标元素（rl 和 gl）之间的差异，然后取这些差异的平方的平均值。最后，将平均值乘以一个因子 2，再将其作为损失函数的值返回。

2. discriminator_loss 的作用是计算生成器（disc_generated_outputs）和真实数据（disc_real_outputs）之间的差异。具体来说，它通过遍历生成器和真实数据中的元素，计算每个元素之间的差异的平方。然后将这些差异的平方值求和，再乘以一个因子 2，得到一个损失值。最后，将这个损失值同时作为真实数据和生成器的损失，再将其作为整个函数的输出返回。

这段代码的主要目的是实现一个用于训练深度学习模型的损失函数。通过使用这两个函数，可以确保模型在训练过程中得到合理的反馈，以提高模型的性能。


```py
import torch


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


```



这段代码定义了两个函数：`generator_loss` 和 `kl_loss`。它们的作用分别如下：

1. `generator_loss`函数：

这个函数的输入是 `disc_outputs`，也就是Disc上的输出。函数内部维护一个名为 `gen_losses` 的列表，用于存储生成器损失。在每个`disc_outputs` 上执行以下操作：

1. 把输入的 `dg` 从float类型转换为 PyTorch 的 tensor类型。
2. 计算生成器损失的均值。
3. 将计算得到的均值添加到 `gen_losses` 列表中。
4. 累加损失并返回。

1. `kl_loss`函数：

这个函数的输入是 `z_p`、`logs_q`、`m_p` 和 `logs_p`，也就是模型参数 `z_p` 的输出、模型参数 `logs_q` 的输出以及模型参数 `m_p` 的输出。函数内部执行以下操作：

1. 把输入的所有张量类型转换为 PyTorch 的 tensor类型。
2. 计算两个变量（`z_p` 和 `logs_q`）的差值对数的对值，也就是日志。
3. 把这两个日志数的对值平方。
4. 对这两个对值平方的结果进行求和，再除以模长的对数。
5. 最后，将这个结果作为损失函数的值返回。


```py
def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l

```

# `mel_processing.py`

这段代码使用了PyTorch库来实现动态范围压缩。其作用是将输入信号x压缩到一个新的范围内，使得其值在压缩因子C和 clip_val 之间。

具体来说，代码首先引入了两个必要的库：torch和torch.utils.data。然后，定义了一个名为 dynamic_range_compression_torch 的函数，该函数接受一个输入信号x，以及两个参数：压缩因子C和clip_val，其中clip_val用于指定压缩度的下限。

函数内部首先将输入信号x传递给 librosa_mel_fn 库，使用其中的 mel 函数将信号转换为梅尔频率信号，这是用于压缩数据的有效信息。然后，使用 dynamic_range_compression_torch 函数对信号进行压缩，具体实现为将输入信号乘以一个动态范围压缩因子 C，并且对结果进行归一化，使得最终的结果-1到1之间。

最终，函数返回压缩后的信号x，可用于进一步的处理和分析。


```py
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


```



This code defines a function `dynamic_range_decompression_torch` which takes a variable `x` and a compression factor `C`, and returns the decompressed version of the input `x` using the specified compression factor.

The other two functions `spectral_normalize_torch` and `spectral_de_normalize_torch` take input `magnitudes` (a tensor of magnitudes), and return the normalized and decompressed versions of the input `magnitudes` respectively. These functions are likely used to perform some kind of data normalization or de-normalization before being used in a neural network.

The `PARAMS` section of the code defines the input arguments `x` and `C` for the `dynamic_range_decompression_torch` function.

Note that the code does not include any examples of how to use these functions or provide any documentation on how to do so.


```py
def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


```

这段代码定义了两个变量mel_basis和hann_window，以及一个名为 spectrogram_torch的函数。

mel_basis是一个字典，用于存储Mel频率信息。每个键是一个音频信号的ID，对应于一个包含多个Mel频率的数组。

hann_window是一个字典，用于存储Hannwindow对象的实例。每个键是一个采样率，对应的Hannwindow对象是一个具有该采样率的Mel频率图。

Spectrogram_torch是一个类，用于从给定的音频信号中提取Mel频率信息。它使用Mel频率图来表示音频信号的时域信息。它还实现了将Mel频率图上的值开根号并增加1e-6以增强音频信号的方法。

函数 Spectrogram_torch 的输入参数包括：音频信号(y)、采样率(n_fft)、步长(hop_size)、窗口大小(win_size)和是否以中心为基准(center)。函数返回一个包含原始音频信号的Mel频率图。


```py
mel_basis = {}
hann_window = {}


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


```

This is a Python function that performs a Mel-Frequency Cepstral Coefficients (MFCC) operation on a given audio signal using the Mel-Frequency Cepstral Coefficients (MFCC) algorithm. The MFCC algorithm is a well-established frequency representation algorithm that is widely used in speech processing and other areas of acoustics.

The function takes in an audio signal (in seconds) as input and returns the Mel-Frequency Cepstral Coefficients as a NumPy array. The audio signal must be passed through an upsampling layer before being passed to the MFCC algorithm. This is because the MFCC algorithm requires a fixed-length input, and the output of the upsampling layer may not be monocha形态的。

The function uses the Mel-Frequency Cepstral Coefficients (MFCC) algorithm to decompose the input audio signal into its constituent frequencies and compute the Mel-Frequency Cepstral Coefficients from each frequency component. The MFCC algorithm is implemented using PyTorch and is based on the Fast Fourier Transform (FFT).


```py
def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=spec.dtype, device=spec.device
        )
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=y.dtype, device=y.device
        )
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)

    return spec

```