# `d:/src/tocomm/Bert-VITS2\onnx_modules\V220_OnnxInference\__init__.py`

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
        # 使用给定的路径和提供者创建一个推理会话对象，用于执行推理流程
        self.flow = ort.InferenceSession(path["flow"], providers=Providers)
        # 使用给定的路径和提供者创建一个推理会话对象，用于执行解码流程
        self.dec = ort.InferenceSession(path["dec"], providers=Providers)

    def __call__(
        self,
        seq,
        tone,
        language,
        bert_zh,
        bert_jp,
        bert_en,
        emo,
        sid,
        seed=114514,
        seq_noise_scale=0.8,
        sdp_noise_scale=0.6,
        length_scale=1.0,
        sdp_ratio=0.0,
    ):
        # 如果输入序列的维度为1
        if seq.ndim == 1:
        # 将seq数组扩展为二维数组
        seq = np.expand_dims(seq, 0)
        # 如果tone数组的维度为1，则将其扩展为二维数组
        if tone.ndim == 1:
            tone = np.expand_dims(tone, 0)
        # 如果language数组的维度为1，则将其扩展为二维数组
        if language.ndim == 1:
            language = np.expand_dims(language, 0)
        # 断言seq、tone和language数组的维度为2
        assert (seq.ndim == 2, tone.ndim == 2, language.ndim == 2)
        # 使用emb_g模型运行，传入sid数组作为输入
        g = self.emb_g.run(
            None,
            {
                "sid": sid.astype(np.int64),
            },
        )[0]
        # 将g数组扩展为一维数组
        g = np.expand_dims(g, -1)
        # 使用enc模型运行，传入seq、tone、language和bert_zh数组作为输入
        enc_rtn = self.enc.run(
            None,
            {
                "x": seq.astype(np.int64),
                "t": tone.astype(np.int64),
                "language": language.astype(np.int64),
                "bert_0": bert_zh.astype(np.float32),
                "bert_1": bert_jp.astype(np.float32),  # 将bert_jp转换为32位浮点数类型，并存储在字典中的键"bert_1"下
                "bert_2": bert_en.astype(np.float32),  # 将bert_en转换为32位浮点数类型，并存储在字典中的键"bert_2"下
                "emo": emo.astype(np.float32),  # 将emo转换为32位浮点数类型，并存储在字典中的键"emo"下
                "g": g.astype(np.float32),  # 将g转换为32位浮点数类型，并存储在字典中的键"g"下
            },
        )
        x, m_p, logs_p, x_mask = enc_rtn[0], enc_rtn[1], enc_rtn[2], enc_rtn[3]  # 从enc_rtn中获取x, m_p, logs_p, x_mask的值
        np.random.seed(seed)  # 设置随机数种子为seed
        zinput = np.random.randn(x.shape[0], 2, x.shape[2]) * sdp_noise_scale  # 生成服从标准正态分布的随机数，并乘以sdp_noise_scale
        logw = self.sdp.run(
            None, {"x": x, "x_mask": x_mask, "zin": zinput.astype(np.float32), "g": g}
        )[0] * (sdp_ratio) + self.dp.run(None, {"x": x, "x_mask": x_mask, "g": g})[
            0
        ] * (
            1 - sdp_ratio
        )  # 计算logw的值
        w = np.exp(logw) * x_mask * length_scale  # 计算w的值
        w_ceil = np.ceil(w)  # 对w向上取整
        y_lengths = np.clip(np.sum(w_ceil, (1, 2)), a_min=1.0, a_max=100000).astype(
            np.int64  # 对w_ceil在指定维度上求和，并限制在1.0和100000之间，转换为64位整数类型
        )
        # 创建一个二维的布尔掩码，用于指示每个序列的有效位置
        y_mask = np.expand_dims(sequence_mask(y_lengths, None), 1)
        # 创建一个三维的注意力掩码，用于指示哪些位置需要进行注意力计算
        attn_mask = np.expand_dims(x_mask, 2) * np.expand_dims(y_mask, -1)
        # 生成路径注意力，根据给定的权重和注意力掩码
        attn = generate_path(w_ceil, attn_mask)
        # 计算路径注意力和位置编码的矩阵乘积，得到新的路径注意力
        m_p = np.matmul(attn.squeeze(1), m_p.transpose(0, 2, 1)).transpose(
            0, 2, 1
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        # 计算路径注意力和位置编码的矩阵乘积，得到新的路径注意力
        logs_p = np.matmul(attn.squeeze(1), logs_p.transpose(0, 2, 1)).transpose(
            0, 2, 1
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        # 添加随机噪声到路径注意力中
        z_p = (
            m_p
            + np.random.randn(m_p.shape[0], m_p.shape[1], m_p.shape[2])
            * np.exp(logs_p)
            * seq_noise_scale
        )

        # 使用流程运行路径注意力
        z = self.flow.run(
            None,
# 将 z_p 转换为 np.float32 类型，并存储在字典中
"z_p": z_p.astype(np.float32),

# 将 y_mask 转换为 np.float32 类型，并存储在字典中
"y_mask": y_mask.astype(np.float32),

# 将 g 存储在字典中
"g": g,

# 调用 self.dec.run 方法，传入参数 z 和 g，并取返回结果的第一个元素
)[0]

# 返回 self.dec.run 方法的结果，传入参数为 None 和一个包含 z 和 g 的字典，并取返回结果的第一个元素
return self.dec.run(None, {"z_in": z.astype(np.float32), "g": g})[0]
```