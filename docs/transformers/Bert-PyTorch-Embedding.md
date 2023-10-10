<!--yml
category: 未分类
date: 2023-10-10 23:20:29
-->

# Bert PyTorch 源码分析：一、嵌入层-CSDN博客

> 来源：[https://blog.csdn.net/wizardforcel/article/details/131382024](https://blog.csdn.net/wizardforcel/article/details/131382024)

```
 class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)

class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model).float()

        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)

        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):

        return self.pe[:, :x.size(1)]

class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1\. TokenEmbedding : normal embedding matrix
        2\. PositionalEmbedding : adding positional information using sin, cos
        2\. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()

        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label):

        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x) 
```