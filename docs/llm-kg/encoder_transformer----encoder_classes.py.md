# `.\encoder_transformer\encoder_classes.py`

```
import numpy as np  # 导入NumPy库，用于数值计算
from scipy.special import softmax  # 导入softmax函数，用于计算概率分布
import math  # 导入math库，用于数学运算


class W_matrices:
    def __init__(self, X_size, d_k, d_v):
        # 初始化W_matrices类，生成随机初始化的W_Q, W_K, W_V矩阵
        self.W_Q = np.random.uniform(low=-1, high=1, size=(X_size, d_k))
        self.W_K = np.random.uniform(low=-1, high=1, size=(X_size, d_k))
        self.W_V = np.random.uniform(low=-1, high=1, size=(X_size, d_v))

    def print_W_matrices(self):
        # 打印W_Q, W_K, W_V矩阵
        print('W_Q : \n', self.W_Q)
        print('W_K : \n', self.W_K)
        print('W_V : \n', self.W_V)

class One_Head_Attention:
    def __init__(self, X, d_k, d_v):
        # 初始化单头注意力机制，计算Q, K, V矩阵
        self.d_model = len(X)
        self.W_mat = W_matrices(self.d_model, d_k, d_v)

        self.Q = np.matmul(X, self.W_mat.W_Q)
        self.K = np.matmul(X, self.W_mat.W_K)
        self.V = np.matmul(X, self.W_mat.W_V)

    def print_QKV(self):
        # 打印Q, K, V矩阵
        print('Q : \n', self.Q)
        print('K : \n', self.K)
        print('V : \n', self.V)

    def compute_1_head_attention(self):
        # 计算单头注意力机制的注意力分数和softmax后的结果
        Attention_scores = np.matmul(self.Q, np.transpose(self.K))
        print('Attention_scores before normalization : \n', Attention_scores)
        Attention_scores = Attention_scores / np.sqrt(self.d_model)
        print('Attention scores after Renormalization: \n ', Attention_scores)
        Softmax_Attention_Matrix = np.apply_along_axis(softmax, 1, Attention_scores)
        print('result after softmax: \n', Softmax_Attention_Matrix)

        result = np.matmul(Softmax_Attention_Matrix, self.V)
        print('softmax result multiplied by V: \n', result)

        return result

    def backpropagate(self):
        # 后向传播函数，用于更新W_mat
        # 实际操作代码待添加
        pass

class Multi_Head_Attention:
    def __init__(self, n_heads, X, d_k, d_v):
        # 初始化多头注意力机制，生成多个单头注意力机制实例
        self.d_model = len(X)
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_concat = self.d_model * self.n_heads
        self.W_0 = np.random.uniform(-1, 1, size=(self.d_concat, self.d_model))
        self.heads = []
        self.heads_results = []
        i = 0
        while i < self.n_heads:
            self.heads.append(One_Head_Attention(X, d_k=d_k, d_v=d_v))
            i += 1

    def print_W_0(self):
        # 打印W_0矩阵
        print('W_0 : \n', self.W_0)

    def print_QKV_each_head(self):
        # 打印每个头部的Q, K, V矩阵
        i = 0
        while i < self.n_heads:
            print(f'Head {i}: \n')
            self.heads[i].print_QKV()
            i += 1

    def print_W_matrices_each_head(self):
        # 打印每个头部的W_matrices矩阵
        i = 0
        while i < self.n_heads:
            print(f'Head {i}: \n')
            self.heads[i].W_mat.print_W_matrices()
            i += 1

    def compute(self):
        # 计算多头注意力机制的结果
        for head in self.heads:
            self.heads_results.append(head.compute_1_head_attention())

        multi_head_results = np.concatenate(self.heads_results, axis=1)

        V_updated = np.matmul(multi_head_results, self.W_0)
        return V_updated

    def back_propagate(self):
        # 反向传播函数，用于更新W_0和调用每个头部的backpropagate函数
        # 实际操作代码待添加
        pass

class FFN:
    # 待实现的Feed Forward网络类
    pass
    # 初始化神经网络模型类，设置初始权重和层大小
    def __init__(self, V_updated, layer1_sz, layer2_sz):
        # 设置第一层和第二层的大小
        self.layer1_sz = layer1_sz
        self.layer2_sz = layer2_sz
        # 使用均匀分布随机初始化第一层权重矩阵，形状为 (输入特征数, 第一层大小)
        self.layer1_weights = np.random.uniform(low=-1, high=1, size=(V_updated.shape[1], layer1_sz))
        # 使用均匀分布随机初始化第二层权重矩阵，形状为 (第二层大小, 输入特征数)
        self.layer2_weights = np.random.uniform(low=-1, high=1, size=(layer2_sz, V_updated.shape[1]))
    
    # 计算前向传播的结果
    def compute(self, V_updated):
        # 计算第一层的输出结果，矩阵乘法运算
        result = np.matmul(V_updated, self.layer1_weights)
        # 计算第二层的输出结果，矩阵乘法运算
        result = np.matmul(result, self.layer2_weights)
        # 返回前向传播计算的结果
        return result
    
    # 反向传播的前馈神经网络部分暂未实现
    def backpropagate_ffn(self):
        # 暂时不做任何操作，保留作为占位符
        pass
class Positional_Encoding:
    # 初始化函数，初始化 Positional_Encoding 类
    def __init__(self, X):
        # 记录输入数据 X 的形状
        self.PE_shape = X.shape
        # 创建一个空数组 PE，形状与输入数据 X 相同
        self.PE = np.empty(self.PE_shape)
        # 获取模型的维度大小 d_model
        self.d_model = self.PE_shape[1]

    # 计算位置编码的方法
    def compute(self, X):
        # 遍历输入数据 X 的第一维度
        for i in range(self.PE_shape[0]):
            # 遍历输入数据 X 的第二维度
            for j in range(self.PE_shape[1]):
                # 计算位置编码的 sin 部分并赋值给 PE 数组的偶数索引位置
                self.PE[i, 2*j] = math.sin(i / (10000 ** (2*j / self.d_model)))
                # 计算位置编码的 cos 部分并赋值给 PE 数组的奇数索引位置
                self.PE[i, 2*j+1] = math.cos(i / (10000 ** (2*j / self.d_model)))
                # 注释说明，假设向量在 X 中是有序堆叠的方式

        # 返回原始数据 X 与位置编码 PE 相加后的结果
        return X + self.PE
```