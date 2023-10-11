<!--yml
category: 未分类
date: 2023-10-10 23:20:21
-->

# Bert Pytorch 源码分析：二、注意力层_绝不原创的飞龙的博客-CSDN博客

> 来源：[https://blog.csdn.net/wizardforcel/article/details/131383685](https://blog.csdn.net/wizardforcel/article/details/131383685)

```py
 class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):

        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()

        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])

        self.output_linear = nn.Linear(d_model, d_model)

        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):

        batch_size = query.size(0)

		'''
        query, key, value = [
			l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
		    for l, x in zip(self.linear_layers, (query, key, value))
		]
		'''

		lq, lk, lv = self.linear_layers
		query, key, value = lq(query), lk(key), lv(value) 

		query, key, value = [
			x.view(batch_size, -1, self.h, self.d_k)
			for x in (query, key, value)
		]

		query, key, value = [
			x.transpose(1, 2)
			for x in (query, key, value)
		]

        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x) 
```

### 缩写表

*   BS：批量大小，即一批数据中样本大小，训练集和测试集可能不同，那就是TBS和VBS
*   ES：嵌入大小，嵌入向量空间的维数，也是注意力层的隐藏单元数量，GPT 中一般是 768
*   ML：输入序列最大长度，一般是512或者1024，不够需要用`<pad>`填充
*   HC：头部的数量，需要能够整除ES，因为每个头的输出拼接起来才是层的输出
*   HS：头部大小，等于`ES // HC`
*   VS：词汇表大小，也就是词的种类数量

### 尺寸备注

*   嵌入层的矩阵尺寸应该是`VS * ES`
*   注意力层的输入尺寸是`BS * ML * ES`
*   输出以及 Q K V 和输入形状相同
*   每个头的 QKV 尺寸为`BS * ML * HS`
*   权重矩阵尺寸为`ES * ES`
*   相关矩阵 S 尺寸为`BS * ML * ML`