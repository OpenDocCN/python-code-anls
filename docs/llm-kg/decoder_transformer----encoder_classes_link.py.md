# `.\decoder_transformer\encoder_classes_link.py`

```
import numpy as np  # 导入 NumPy 库，用于数组操作
from scipy.special import softmax  # 从 SciPy 库中导入 softmax 函数，用于计算 softmax
import math  # 导入 math 模块，用于一些数学操作

# 定义一个函数，将两个数组相加并进行归一化处理
def add_and_norm(array1, array2):
    # 如果两个数组的维度不同，抛出 ValueError 异常
    if array1.ndim != array2.ndim:
        raise ValueError("Incompatible shapes!")

    # 将两个数组相加得到结果
    result = array1 + array2

    # 如果数组的维度为 1
    if array1.ndim == 1:
        # 计算结果数组的均值和方差
        mean = result.mean()
        variance = result.var()
        # 对结果数组进行归一化处理
        result = (result - mean) / variance
    else:
        # 对于二维数组，逐行进行归一化处理
        for r in range(array1.shape[0]):
            mean = result[r, :].mean()
            variance = result[r, :].var()
            result[r, :] = (result[r, :] - mean) / variance

    # 返回处理后的结果数组
    return result

# 定义一个类，用于生成 W 矩阵
class W_matrices:
    def __init__(self, X_size, d_k, d_v):  # 初始化函数，参数为 X 的大小、d_k 和 d_v
        # 生成随机的 W_Q、W_K 和 W_V 矩阵
        self.W_Q = np.random.uniform(low=-1, high=1, size=(X_size, d_k))
        self.W_K = np.random.uniform(low=-1, high=1, size=(X_size, d_k))
        self.W_V = np.random.uniform(low=-1, high=1, size=(X_size, d_v))

    # 打印 W 矩阵
    def print_W_matrices(self):
        print('W_Q : \n', self.W_Q)
        print('W_K : \n', self.W_K)
        print('W_V : \n', self.W_V)

# 定义一个类，用于实现单头注意力机制
class One_Head_Attention:
    def __init__(self, X, d_k, d_v):  # 初始化函数，参数为输入矩阵 X、d_k 和 d_v
        self.d_model = len(X)  # 计算输入矩阵的大小
        self.W_mat = W_matrices(self.d_model, d_k, d_v)  # 生成 W 矩阵

        # 计算查询（Q）、键（K）和值（V）矩阵
        self.Q = np.matmul(X, self.W_mat.W_Q)
        self.K = np.matmul(X, self.W_mat.W_K)
        self.V = np.matmul(X, self.W_mat.W_V)

    # 打印查询（Q）、键（K）和值（V）矩阵
    def print_QKV(self):
        print('Q : \n', self.Q)
        print('K : \n', self.K)
        print('V : \n', self.V)

    # 计算单头注意力机制的输出
    def compute_1_head_attention(self):
        # 计算注意力分数
        Attention_scores = np.matmul(self.Q, np.transpose(self.K))
        # 对注意力分数进行缩放
        Attention_scores = Attention_scores / np.sqrt(self.d_model)
        # 对注意力分数进行 softmax 归一化
        Softmax_Attention_Matrix = np.apply_along_axis(softmax, 1, Attention_scores)
        # 计算注意力加权后的值
        result = np.matmul(Softmax_Attention_Matrix, self.V)

        # 返回注意力加权后的结果
        return result

    def backpropagate(self):
        # 这里应该是用于反向传播更新 W_mat 的参数，暂时留空
        pass

# 定义一个类，用于实现多头注意力机制
class Multi_Head_Attention:
    def __init__(self, n_heads, X, d_k, d_v):  # 初始化函数，参数为头数 n_heads、输入矩阵 X、d_k 和 d_v
        self.d_model = len(X)  # 计算输入矩阵的大小
        self.n_heads = n_heads  # 设置头数
        self.d_k = d_k  # 设置查询和键的维度
        self.d_v = d_v  # 设置值的维度
        self.d_concat = self.d_model * self.n_heads  # 计算多头注意力机制的输出维度
        self.W_0 = np.random.uniform(-1, 1, size=(self.d_concat, self.d_model))  # 生成连接多头输出的权重矩阵
        self.heads = []  # 初始化头列表
        i = 0
        # 循环生成每个头的注意力机制
        while i < self.n_heads:
            self.heads.append(One_Head_Attention(X, d_k=d_k, d_v=d_v))  # 生成单头注意力机制
            i += 1

    # 打印连接多头输出的权重矩阵
    def print_W_0(self):
        print('W_0 : \n', self.W_0)

    # 打印每个头的查询（Q）、键（K）和值（V）矩阵
    def print_QKV_each_head(self):
        i = 0
        while i < self.n_heads:
            print(f'Head {i}: \n')
            self.heads[i].print_QKV()
            i += 1
    # 打印每个注意力头的权重矩阵
    def print_W_matrices_each_head(self):
        i = 0
        while i < self.n_heads:
            # 打印当前注意力头的编号
            print(f'Head {i}: \n')
            # 调用该头部注意力对象的权重矩阵打印方法
            self.heads[i].W_mat.print_W_matrices()
            i += 1

    # 计算多头注意力机制的结果
    def compute(self):
        # 对每个注意力头进行计算，并将结果存入 heads_results 列表
        for head in self.heads:
            self.heads_results.append(head.compute_1_head_attention())

        # 将所有头部注意力计算结果在 axis=1 方向上进行连接
        multi_head_results = np.concatenate(self.heads_results, axis=1)

        # 使用权重矩阵 W_0 对多头注意力结果进行矩阵乘法运算，得到更新后的 V
        V_updated = np.matmul(multi_head_results, self.W_0)
        return V_updated

    # 反向传播方法，目前仅有注释信息，实际功能未实现
    def back_propagate(self):
        # 反向传播 W_0 权重矩阵的梯度
        # 调用每个头部注意力对象的反向传播方法 _backprop
        pass
# 定义一个前馈神经网络类
class FFN:
    # 初始化函数，接收输入向量的维度、第一层隐藏层的大小和输出向量的维度
    def __init__(self, d_v, layer_sz, d_output):
        # 设置第一隐藏层的大小
        self.layer1_sz = layer_sz
        # 随机初始化第一层权重矩阵，维度为(d_v, layer_sz)
        self.layer1_weights = np.random.uniform(low=-1, high=1, size=(d_v, layer_sz))
        # 随机初始化第二层权重矩阵，维度为(layer_sz, d_output)
        self.layer2_weights = np.random.uniform(low=-1, high=1, size=(layer_sz, d_output))

    # 前向传播计算函数，接收更新后的输入向量
    def compute(self, V_updated):
        # 计算第一层的结果
        result = np.matmul(V_updated, self.layer1_weights)
        # 计算第二层的结果
        result = np.matmul(result, self.layer2_weights)
        # 返回结果
        return result

    # 反向传播函数，暂时未实现
    def backpropagate_ffn(self):
        pass

# 定义一个位置编码类
class Positional_Encoding:
    # 初始化函数
    def __init__(self):
        pass

    # 计算位置编码，接收输入矩阵X
    def compute(self, X):
        # 记录位置编码的形状
        self.PE_shape = X.shape
        # 初始化位置编码矩阵PE
        self.PE = np.empty(self.PE_shape)
        # 判断输入矩阵的维度
        if X.ndim == 2:
            # 如果是二维矩阵，记录模型的维度
            self.d_model = self.PE_shape[1]
            # 计算位置编码
            for i in range(self.PE_shape[0]): 
                for j in range(int(self.PE_shape[1]/2)):
                    self.PE[i,2*j] = math.sin(i/(10000**(2*j/self.d_model)))
                    self.PE[i,2*j+1] = math.cos(i/(10000**(2*j/self.d_model)))
        else:
            # 如果是一维矩阵，记录模型的维度
            self.d_model = len(X)
            # 计算位置编码
            for j in range(int(len(X)/2)):
                self.PE[2*j] = math.sin(1/(10000**(2*j/self.d_model)))
                self.PE[2*j+1] = math.cos(1/(10000**(2*j/self.d_model)))

        # 返回位置编码加上输入矩阵后的结果
        return X + self.PE
```