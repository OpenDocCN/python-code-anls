# `.\pytorch\torch\_lazy\tensor_factory_functions.py`

```py
# 导入 torch 库
import torch

"""
tensor_factory_functions 定义了一组创建张量的 torch 函数列表。
这个列表是通过以下正则表达式在 native_functions.yaml 文件中搜索得到的：

  cat native_functions.yaml | grep 'func:' | grep -v "Tensor.*->" | grep "[-]>.*Tensor"

可能会有新的张量工厂函数被添加，使得这个列表变得不准确。
使用时请自行评估风险或重新生成列表。
"""
# 定义张量工厂函数列表
tensor_factory_functions = (
    torch._cudnn_init_dropout_state,   # 初始化 cuDNN dropout 状态
    torch.arange,                     # 创建等差序列张量
    torch.bartlett_window,            # 创建巴特利特窗口张量
    torch.blackman_window,            # 创建布莱克曼窗口张量
    torch._empty_affine_quantized,    # 创建空的仿射量化张量
    torch.empty_strided,              # 创建空的步长张量
    torch.eye,                        # 创建单位矩阵张量
    torch.full,                       # 创建填充张量
    torch.from_file,                  # 从文件中创建张量
    torch.hann_window,                # 创建汉宁窗口张量
    torch.hamming_window,             # 创建海明窗口张量
    torch.kaiser_window,              # 创建凯泽窗口张量
    torch.linspace,                   # 创建线性空间张量
    torch.logspace,                   # 创建对数空间张量
    torch.ones,                       # 创建全 1 张量
    torch.scalar_tensor,              # 创建标量张量
    torch.rand,                       # 创建随机张量
    torch.randint,                    # 创建随机整数张量
    torch.randn,                      # 创建随机正态分布张量
    torch.randperm,                   # 创建随机排列张量
    torch.range,                      # 创建范围张量
    torch._efficientzerotensor,       # 创建高效的零张量
    torch.zeros,                      # 创建全 0 张量
    torch.tril_indices,               # 返回下三角索引张量
    torch.triu_indices,               # 返回上三角索引张量
    # 注意：以下函数符合上述正则表达式搜索，但不在 torch 模块中。注释掉。
    # torch._sparse_coo_tensor_with_dims,
    # torch.fft_fftfreq,
    # torch.fft_rfftfreq,
) + (
    # torch.tensor 是特殊的，因为它不在 native_functions.yaml 中
    # 单独添加它
    torch.tensor,
)
```