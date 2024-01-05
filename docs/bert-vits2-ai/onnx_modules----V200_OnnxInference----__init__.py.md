# `d:/src/tocomm/Bert-VITS2\onnx_modules\V200_OnnxInference\__init__.py`

```
﻿import numpy as np  # 导入 numpy 库，用于处理数组和矩阵运算
import onnxruntime as ort  # 导入 onnxruntime 库，用于运行 ONNX 模型

# 定义一个函数，用于转换 pad_shape
def convert_pad_shape(pad_shape):
    layer = pad_shape[::-1]  # 将 pad_shape 列表倒序排列
    pad_shape = [item for sublist in layer for item in sublist]  # 将二维列表转换为一维列表
    return pad_shape  # 返回转换后的 pad_shape

# 定义一个函数，用于生成序列掩码
def sequence_mask(length, max_length=None):
    if max_length is None:  # 如果未提供 max_length 参数
        max_length = length.max()  # 则将 max_length 设置为 length 中的最大值
    x = np.arange(max_length, dtype=length.dtype)  # 生成一个长度为 max_length 的数组
    return np.expand_dims(x, 0) < np.expand_dims(length, 1)  # 返回一个布尔类型的数组，用于掩码

# 定义一个函数，用于生成路径
def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    """
    mask: [b, 1, t_y, t_x]
    """

    # 获取 mask 的形状信息，分别为 b, _, t_y, t_x
    b, _, t_y, t_x = mask.shape
    # 对 duration 沿着最后一个维度进行累积求和
    cum_duration = np.cumsum(duration, -1)

    # 将累积求和后的结果展开成一维数组
    cum_duration_flat = cum_duration.reshape(b * t_x)
    # 根据累积求和后的结果生成路径 mask
    path = sequence_mask(cum_duration_flat, t_y)
    # 将路径 mask 重新变形成 b, t_x, t_y 的形状
    path = path.reshape(b, t_x, t_y)
    # 对路径 mask 进行异或操作，并在最后一个维度上进行填充
    path = path ^ np.pad(path, ((0, 0), (1, 0), (0, 0)))[:, :-1]
    # 在第二个维度上扩展路径 mask，并进行维度转置
    path = np.expand_dims(path, 1).transpose(0, 1, 3, 2)
    # 返回生成的路径 mask
    return path


class OnnxInferenceSession:
    def __init__(self, path, Providers=["CPUExecutionProvider"]):
        # 初始化 InferenceSession 对象，分别加载指定路径下的模型
        self.enc = ort.InferenceSession(path["enc"], providers=Providers)
        self.emb_g = ort.InferenceSession(path["emb_g"], providers=Providers)
        self.dp = ort.InferenceSession(path["dp"], providers=Providers)
        self.sdp = ort.InferenceSession(path["sdp"], providers=Providers)
        self.flow = ort.InferenceSession(path["flow"], providers=Providers)
        # 使用给定的路径和提供者创建一个用于推断的会话对象，存储在self.flow中
        self.dec = ort.InferenceSession(path["dec"], providers=Providers)
        # 使用给定的路径和提供者创建另一个用于推断的会话对象，存储在self.dec中

    def __call__(
        self,
        seq,
        tone,
        language,
        bert_zh,
        bert_jp,
        bert_en,
        sid,
        seed=114514,
        seq_noise_scale=0.8,
        sdp_noise_scale=0.6,
        length_scale=1.0,
        sdp_ratio=0.0,
    ):
        if seq.ndim == 1:
            seq = np.expand_dims(seq, 0)
        # 如果输入的seq的维度为1，则使用np.expand_dims将其扩展为2维
        # 如果 tone 的维度为 1，则在其前面添加一个维度，使其变为二维数组
        if tone.ndim == 1:
            tone = np.expand_dims(tone, 0)
        # 如果 language 的维度为 1，则在其前面添加一个维度，使其变为二维数组
        if language.ndim == 1:
            language = np.expand_dims(language, 0)
        # 使用断言确保 seq、tone 和 language 的维度都为 2
        assert (seq.ndim == 2, tone.ndim == 2, language.ndim == 2)
        # 使用 self.emb_g 对象的 run 方法，传入参数并获取返回值
        g = self.emb_g.run(
            None,
            {
                "sid": sid.astype(np.int64),
            },
        )[0]
        # 在 g 的最后一个维度上添加一个维度
        g = np.expand_dims(g, -1)
        # 使用 self.enc 对象的 run 方法，传入参数并获取返回值
        enc_rtn = self.enc.run(
            None,
            {
                "x": seq.astype(np.int64),
                "t": tone.astype(np.int64),
                "language": language.astype(np.int64),
                "bert_0": bert_zh.astype(np.float32),
                "bert_1": bert_jp.astype(np.float32),
                "bert_2": bert_en.astype(np.float32),  # 将变量 bert_en 转换为 np.float32 类型，并赋值给字典的键 "bert_2"
                "g": g.astype(np.float32),  # 将变量 g 转换为 np.float32 类型，并赋值给字典的键 "g"
            },
        )
        x, m_p, logs_p, x_mask = enc_rtn[0], enc_rtn[1], enc_rtn[2], enc_rtn[3]  # 从 enc_rtn 中获取四个值分别赋给变量 x, m_p, logs_p, x_mask
        np.random.seed(seed)  # 设置随机数种子为 seed
        zinput = np.random.randn(x.shape[0], 2, x.shape[2]) * sdp_noise_scale  # 生成服从标准正态分布的随机数矩阵，并乘以 sdp_noise_scale
        logw = self.sdp.run(  # 调用 self.sdp 的 run 方法
            None, {"x": x, "x_mask": x_mask, "zin": zinput.astype(np.float32), "g": g}  # 传入参数为 x, x_mask, zinput, g
        )[0] * (sdp_ratio) + self.dp.run(None, {"x": x, "x_mask": x_mask, "g": g})[  # 调用 self.dp 的 run 方法
            0
        ] * (
            1 - sdp_ratio
        )
        w = np.exp(logw) * x_mask * length_scale  # 计算 w 值
        w_ceil = np.ceil(w)  # 对 w 进行向上取整
        y_lengths = np.clip(np.sum(w_ceil, (1, 2)), a_min=1.0, a_max=100000).astype(  # 计算 y_lengths
            np.int64
        )
        y_mask = np.expand_dims(sequence_mask(y_lengths, None), 1)  # 生成 y_mask
        # 生成注意力掩码，将输入的 x_mask 和 y_mask 进行扩展维度后相乘得到
        attn_mask = np.expand_dims(x_mask, 2) * np.expand_dims(y_mask, -1)
        # 生成路径注意力，使用 w_ceil 和 attn_mask 作为参数
        attn = generate_path(w_ceil, attn_mask)
        # 计算 m_p，使用注意力矩阵和 m_p 的转置进行矩阵相乘，并进行维度转换
        m_p = np.matmul(attn.squeeze(1), m_p.transpose(0, 2, 1)).transpose(
            0, 2, 1
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        # 计算 logs_p，使用注意力矩阵和 logs_p 的转置进行矩阵相乘，并进行维度转换
        logs_p = np.matmul(attn.squeeze(1), logs_p.transpose(0, 2, 1)).transpose(
            0, 2, 1
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        # 计算 z_p，使用 m_p、logs_p 和随机数进行计算
        z_p = (
            m_p
            + np.random.randn(m_p.shape[0], m_p.shape[1], m_p.shape[2])
            * np.exp(logs_p)
            * seq_noise_scale
        )

        # 使用流程运行 z_p
        z = self.flow.run(
            None,
            {
                "z_p": z_p.astype(np.float32),
                "y_mask": y_mask.astype(np.float32),  # 将y_mask转换为32位浮点数类型
                "g": g,  # 设置参数g
            },
        )[0]  # 返回结果字典的第一个元素

        return self.dec.run(None, {"z_in": z.astype(np.float32), "g": g})[0]  # 运行self.dec对象的run方法，传入参数z和g，并返回结果字典的第一个元素
```