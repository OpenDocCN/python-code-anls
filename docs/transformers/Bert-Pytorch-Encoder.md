<!--yml
category: 未分类
date: 2023-10-10 23:20:05
-->

# Bert Pytorch 源码分析：四、编解码器_绝不原创的飞龙的博客-CSDN博客

> 来源：[https://blog.csdn.net/wizardforcel/article/details/131398331](https://blog.csdn.net/wizardforcel/article/details/131398331)

```py
 class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()

        self.hidden = hidden

        self.n_layers = n_layers

        self.attn_heads = attn_heads

        self.feed_forward_hidden = hidden * 4

        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, segment_info):

        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        x = self.embedding(x, segment_info)

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()

        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):

        return self.softmax(self.linear(x[:, 0]))

class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()

        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):

        return self.softmax(self.linear(x)) 
```