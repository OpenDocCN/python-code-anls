# `.\decoder_transformer\decoder_exec.py`

```
# 导入名为DECODER的模块作为别名DECODER
import decoder as DECODER
# 导入NumPy库并使用别名np
import numpy as np

# 设置ANSI转义序列以在终端中显示颜色
ENDCOLOR = '\033[0m'
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'

# encoder构造K和V向量。假设每个堆栈有3个五维向量。
K = np.random.uniform(low=-1, high=1, size=(3, 5))
V = np.random.uniform(low=-1, high=1, size=(3, 5))

# 词汇表，包含预定义的单词和对应的向量表示
vocabulary = {
    'Hoje': np.array([1, 0, 0, 0, 0]),
    'é': np.array([0, 1, 0, 0, 0]),
    'domingo': np.array([0, 0, 1, 0, 0]),
    'sábado': np.array([0, 0, 0, 1, 0]),
    'EOS': np.array([0, 0, 0, 0, 1]),
    'START': np.array([0.2, 0.2, 0.2, 0.2, 0.2])
}

# 从词汇表中获取START和EOS的向量表示
START = vocabulary['START']
EOS = vocabulary['EOS']
# 初始化输入和输出令牌为START
INPUT_TOKEN = START
OUTPUT_TOKENS = START
LAST_TOKEN = START
X = INPUT_TOKEN

# 创建位置编码对象
PE = DECODER.ENCODER.Positional_Encoding()
# 创建多头掩码注意力对象，设定头数为8，模型维度d_model为5，键和值的维度d_k为4，值的维度d_v为5
multi_head_masked_attention = DECODER.Multi_Head_Masked_Attention(n_heads=8, d_model=5, d_k=4, d_v=5)
# 创建编码器-解码器注意力对象，设定键的维度d_k为4
encoder_decoder_attention = DECODER.One_Head_Encoder_Decoder_Attention(d_k=4)
# 创建前馈神经网络对象，设定值的维度d_v为5，层的尺寸layer_sz为8，输出维度d_output为5
ffn = DECODER.ENCODER.FFN(d_v=5, layer_sz=8, d_output=5)

# 循环直到遇到终止标记EOS或循环次数达到10次
count = 0
while (not np.array_equal(LAST_TOKEN, EOS)) and (count < 10):
    # 计算输入令牌的位置编码
    X_PE = PE.compute(X)

    # 打印输入令牌的形状和值
    print(BLUE + 'shape of X:', X.shape, ENDCOLOR)
    print(BLUE + 'X:\n', X, ENDCOLOR)
    print(BLUE + 'X_PE:\n', X_PE, ENDCOLOR)

    # 计算多头掩码注意力
    output_masked_attention = multi_head_masked_attention.compute(X_PE)

    # 计算添加和归一化操作，用于残差连接
    Q_star = DECODER.ENCODER.add_and_norm(output_masked_attention, X_PE)  # 残差连接1

    # 计算编码器-解码器注意力
    output_encoder_decoder_attention = encoder_decoder_attention.compute_1_head_attention(Q=Q_star, K=K, V=V)

    # 计算添加和归一化操作，用于残差连接
    Rc2 = DECODER.ENCODER.add_and_norm(output_encoder_decoder_attention, Q_star)  # 残差连接2

    # 计算前馈神经网络的结果
    ffn_result = ffn.compute(Rc2)

    # 计算添加和归一化操作，用于残差连接
    OUTPUT_TOKENS_before_softmax = DECODER.ENCODER.add_and_norm(ffn_result, Rc2)  # 第3个残差连接

    # 如果输出令牌的维度为1，则进行最后的softmax处理
    if OUTPUT_TOKENS_before_softmax.ndim == 1:
        # 对单个令牌进行softmax处理
        OUTPUT_TOKENS = np.exp(OUTPUT_TOKENS_before_softmax) / np.sum(np.exp(OUTPUT_TOKENS_before_softmax))
        # 获取softmax后概率最大的位置
        position_of_max = np.argmax(OUTPUT_TOKENS)
        # 将概率最大位置的令牌设为1，其余为0
        OUTPUT_TOKENS = np.eye(OUTPUT_TOKENS.shape[0])[position_of_max]
        # 更新LAST_TOKEN为输出令牌
        LAST_TOKEN = OUTPUT_TOKENS
    else:
        # 对多个令牌进行softmax处理
        OUTPUT_TOKENS = np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x)), 1, OUTPUT_TOKENS_before_softmax)
        # 获取每个序列中概率最大的位置
        position_of_max = np.argmax(OUTPUT_TOKENS, axis=1)
        # 将每个序列中概率最大位置的令牌设为1，其余为0
        OUTPUT_TOKENS = np.eye(OUTPUT_TOKENS.shape[1])[position_of_max]
        # 更新LAST_TOKEN为最后一个输出令牌
        LAST_TOKEN = OUTPUT_TOKENS[-1, :]

    # 更新输入序列X，将最新生成的输出令牌添加到序列末尾
    X = np.vstack([X, LAST_TOKEN])

    # 打印输出令牌的形状
    print('shape of OUTPUT_TOKENS:', OUTPUT_TOKENS.shape)

    # 打印输出令牌的值和最后一个令牌的值
    print(RED + 'OUTPUT_TOKENS:\n', OUTPUT_TOKENS, ENDCOLOR)
    print(RED + 'LAST_TOKEN:\n', LAST_TOKEN, ENDCOLOR)
    print(RED + '=====================================' + ENDCOLOR)

    # 更新循环计数器
    count = count + 1

# 根据词汇表中的向量表示，识别序列中的令牌并构建输出句子
OUTPUT_SENTENCE = []
output_sentence_str = ''
for token_pos in range(len(X[:, 0])):
    token = X[token_pos, :]
    # 遍历vocabulary字典中的每个键值对，其中键是词汇名称，值是一个数组
    for name, array in vocabulary.items():
        # 检查当前数组（array）是否与给定的token数组（token）完全相等
        if np.array_equal(array, token):
            # 如果数组完全相等，则将对应的词汇名称（name）添加到OUTPUT_SENTENCE列表中
            OUTPUT_SENTENCE.append(name)
# 遍历 OUTPUT_SENTENCE 列表中的每个元素，元素被赋值给变量 token_name
for token_name in OUTPUT_SENTENCE:
    # 将当前 token_name 添加到 output_sentence_str 后面，加上一个空格
    output_sentence_str += token_name + ' '
# 打印输出合并后的字符串
print(output_sentence_str)
```