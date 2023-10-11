<!--yml
category: 未分类
date: 2023-10-10 23:20:12
-->

# Bert Pytorch 源码分析：三、Transformer块_绝不原创的飞龙的博客-CSDN博客

> 来源：[https://blog.csdn.net/wizardforcel/article/details/131397186](https://blog.csdn.net/wizardforcel/article/details/131397186)

```py
 class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Linear(d_model, d_ff)

        self.w_2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

        self.activation = GELU()

    def forward(self, x):

        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()

        self.norm = LayerNorm(size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):

        return x + self.dropout(sublayer(self.norm(x)))

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()

        self.a_2 = nn.Parameter(torch.ones(features))

        self.b_2 = nn.Parameter(torch.zeros(features))

        self.eps = eps

    def forward(self, x):

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()

        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)

        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)

        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)

        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):

        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))

        x = self.output_sublayer(x, self.feed_forward)

        return self.dropout(x) 
```