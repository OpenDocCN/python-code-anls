# `.\encoder_transformer\encoder_exec.py`

```
import numpy as np 
import encoder_classes as ENCODER

sentence = "Today is sunday"

vocabulary = ['Today', 'is', 'sunday', 'saturday']

# Initial Embedding (one-hot encoding):

x_1 = [1,0,0,0] # Today 
x_2 = [0,1,0,0] # is 
x_3 = [0,0,1,0] # Sunday
x_4 = [0,0,0,1] # Saturday

X_vocab = [x_1, x_2, x_3, x_4]

init_embbeding = dict(zip(vocabulary, X_vocab))
# 创建初始嵌入字典，将词汇表中的词与对应的one-hot编码对应起来

X = np.stack([init_embbeding['Today'], init_embbeding['is'], init_embbeding['sunday']], axis=0)
# 从初始嵌入字典中选择几个词的向量，创建输入矩阵X，用于多头注意力机制

multi_head_attention = ENCODER.Multi_Head_Attention(2, X=X, d_k=4, d_v=4)
# 使用ENCODER模块中的Multi_Head_Attention类创建一个多头注意力机制的实例

multi_head_attention.print_W_matrices_each_head()
# 打印每个注意力头的权重矩阵W

multi_head_attention.print_QKV_each_head()
# 打印每个注意力头的查询、键、值矩阵Q、K、V

multi_head_attention.print_W_0()
# 打印权重矩阵W_0

V_updated_by_context = multi_head_attention.compute()
# 计算多头注意力机制的输出，更新后的值V

print(V_updated_by_context)
# 打印更新后的上下文信息
```