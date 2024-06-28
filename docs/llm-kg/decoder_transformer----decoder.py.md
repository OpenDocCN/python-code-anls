# `.\decoder_transformer\decoder.py`

```
# 导入数学库和NumPy库
import math 
import numpy as np

# 导入自定义的编码器类模块
import encoder_classes_link as ENCODER

# 定义单头掩码注意力机制类
class One_Head_Masked_Attention:
    def __init__(self, d_model, d_k, d_v):
        # 初始化模型维度和权重矩阵
        self.d_model = d_model
        self.W_mat = ENCODER.W_matrices(self.d_model, d_k, d_v)

    # 计算查询(Q)、键(K)、值(V)矩阵
    def compute_QKV(self, X):
        self.Q = np.matmul(X, self.W_mat.W_Q)
        self.K = np.matmul(X, self.W_mat.W_K)
        self.V = np.matmul(X, self.W_mat.W_V)

    # 打印查询(Q)、键(K)、值(V)矩阵
    def print_QKV(self):
        print('Q : \n', self.Q)
        print('K : \n', self.K)
        print('V : \n', self.V)

    # 计算单头掩码注意力机制
    def compute_1_head_masked_attention(self):
        # 计算注意力分数
        Attention_scores = np.matmul(self.Q, np.transpose(self.K)) 
        
        # 对注意力分数进行掩码处理
        if Attention_scores.ndim > 1:
            M = np.zeros(Attention_scores.shape)
            for i in range(Attention_scores.shape[0]):
                for j in range(i+1, Attention_scores.shape[1]):
                    M[i,j] = -np.inf
        else:
            M = 0

        Attention_scores += M
        
        # 归一化注意力分数
        Attention_scores = Attention_scores / np.sqrt(self.d_model) 

        # 对注意力分数应用softmax函数
        if Attention_scores.ndim > 2:
            Softmax_Attention_Matrix = np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x)), 1, Attention_scores)
        else: 
            Softmax_Attention_Matrix = np.exp(Attention_scores) / np.sum(np.exp(Attention_scores))

        # 根据softmax后的注意力分数计算加权值
        if Attention_scores.ndim > 1:
            if Softmax_Attention_Matrix.shape[1] != self.V.shape[0]:
                raise ValueError("Incompatible shapes!")

            result = np.matmul(Softmax_Attention_Matrix, self.V)
        else: 
            result = Softmax_Attention_Matrix * self.V

        return result

    # 反向传播方法，用于更新权重矩阵W_mat
    def backpropagate(self):
        # 这里可以添加具体的反向传播逻辑
        pass


# 定义多头掩码注意力机制类
class Multi_Head_Masked_Attention:
    def __init__(self, n_heads, d_model, d_k, d_v):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_concat = self.d_v*self.n_heads
        self.W_0 = np.random.uniform(-1, 1, size=(self.d_concat, self.d_v))
        self.heads = []

        # 初始化多个单头注意力机制对象
        i = 0
        while i < self.n_heads:
            self.heads.append(One_Head_Masked_Attention(d_model=d_model, d_k=d_k , d_v=d_v ))
            i += 1

    # 打印权重矩阵W_0
    def print_W_0(self):
        print('W_0 : \n', self.W_0)

    # 打印每个单头注意力机制的查询(Q)、键(K)、值(V)矩阵
    def print_QKV_each_head(self):
        i = 0
        while i < self.n_heads:
            print(f'Head {i}: \n')
            self.heads[i].print_QKV()
            i += 1
    # 打印每个注意力头的权重矩阵
    def print_W_matrices_each_head(self):
        i = 0
        # 循环每个注意力头
        while i < self.n_heads:
            # 打印当前头的标识
            print(f'Head {i}: \n')
            # 调用当前头的权重矩阵打印方法
            self.heads[i].W_mat.print_W_matrices()
            i += 1

    # 计算多头注意力机制后的结果
    def compute(self, X):
        # 存储每个头的计算结果
        self.heads_results = []
        # 遍历每个注意力头
        for head in self.heads:
            # 计算当前头的查询、键、值
            head.compute_QKV(X)
            # 计算当前头的掩码注意力结果并存储
            self.heads_results.append(head.compute_1_head_masked_attention())

        # 如果输入X的维度大于1
        if X.ndim > 1:
            # 拼接所有头的结果，沿着列方向（水平方向）
            multi_head_results = np.concatenate(self.heads_results, axis=1)
            # 更新值V，使用多头结果与权重矩阵W_0的乘积
            V_updated = np.matmul(multi_head_results, self.W_0)
        else:
            # 拼接所有头的结果，沿着行方向（垂直方向）
            multi_head_results = np.concatenate(self.heads_results, axis=0)
            print('Dimension of multihead_results:', multi_head_results.shape)
            print('Dimension of W_0:', self.W_0.shape)
            # 更新值V，使用多头结果与权重矩阵W_0的乘积
            V_updated = np.matmul(multi_head_results, self.W_0)

        return V_updated

    # 反向传播函数
    def back_propagate(self):
        # 反向传播W_0参数
        # 对每个注意力头调用_backprop方法进行反向传播
        pass
# 定义一个名为 One_Head_Encoder_Decoder_Attention 的类，用于实现单头编码器-解码器注意力机制

def __init__(self, d_k):
    # 类的初始化方法，接收参数 d_k，表示注意力机制中的维度大小
    self.d_k = d_k

def print_QKV(self):
    # 打印 Q、K、V 三个属性的方法
    print('Q : \n', self.Q)
    print('K : \n', self.K)
    print('V : \n', self.V)

def compute_1_head_attention(self, Q, K, V):
    # 计算单头注意力机制的方法
    self.Q = Q  # 将参数 Q 赋值给对象的 Q 属性，来自解码器的遮蔽注意力
    self.K = K  # 将参数 K 赋值给对象的 K 属性，来自编码器
    self.V = V  # 将参数 V 赋值给对象的 V 属性，编码器的最终结果

    # 计算注意力分数
    Attention_scores = np.matmul(self.Q, np.transpose(self.K))
    # 对注意力分数进行归一化处理
    Attention_scores = Attention_scores / np.sqrt(self.d_k)
    Softmax_Attention_Matrix = np.exp(Attention_scores - np.max(Attention_scores, axis=-1, keepdims=True))
    Softmax_Attention_Matrix /= np.sum(Softmax_Attention_Matrix, axis=-1, keepdims=True)

    # 检查结果的维度是否匹配
    if Softmax_Attention_Matrix.ndim > 1:
        if Softmax_Attention_Matrix.shape[1] != self.V.shape[0]:
            raise ValueError("Incompatible shapes!")

    # 计算加权后的结果向量
    result = np.matmul(Softmax_Attention_Matrix, self.V)

    return result

# 编码器-解码器注意力中没有可学习的参数
```