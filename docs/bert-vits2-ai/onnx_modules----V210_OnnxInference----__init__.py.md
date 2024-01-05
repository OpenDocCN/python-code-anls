# `d:/src/tocomm/Bert-VITS2\onnx_modules\V210_OnnxInference\__init__.py`

```
﻿import numpy as np  # 导入 numpy 库，用于处理数组和矩阵运算
import onnxruntime as ort  # 导入 onnxruntime 库，用于运行 ONNX 模型

# 定义函数 convert_pad_shape，用于转换 pad_shape
def convert_pad_shape(pad_shape):
    layer = pad_shape[::-1]  # 将 pad_shape 反转
    pad_shape = [item for sublist in layer for item in sublist]  # 将反转后的 pad_shape 展开成一维数组
    return pad_shape  # 返回转换后的 pad_shape

# 定义函数 sequence_mask，用于生成序列掩码
def sequence_mask(length, max_length=None):
    if max_length is None:  # 如果 max_length 未指定
        max_length = length.max()  # 则将 max_length 设置为 length 中的最大值
    x = np.arange(max_length, dtype=length.dtype)  # 生成一个长度为 max_length 的数组 x
    return np.expand_dims(x, 0) < np.expand_dims(length, 1)  # 返回一个布尔类型的数组，表示 x 是否小于 length

# 定义函数 generate_path，用于生成路径
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

    # 将累积求和后的结果展平成一维数组
    cum_duration_flat = cum_duration.reshape(b * t_x)
    # 根据展平后的累积求和结果生成路径
    path = sequence_mask(cum_duration_flat, t_y)
    # 将路径重新整形成 b, t_x, t_y 的形状
    path = path.reshape(b, t_x, t_y)
    # 对路径进行异或操作，并在最后一个维度上进行填充
    path = path ^ np.pad(path, ((0, 0), (1, 0), (0, 0)))[:, :-1]
    # 在第二个维度上扩展路径，并进行维度转置
    path = np.expand_dims(path, 1).transpose(0, 1, 3, 2)
    # 返回路径
    return path


class OnnxInferenceSession:
    def __init__(self, path, Providers=["CPUExecutionProvider"]):
        # 初始化推理会话，分别加载 enc, emb_g, dp, sdp 四个模型
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
        vqidx,
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
                "bert_1": bert_jp.astype(np.float32),  # 将bert_jp数组转换为32位浮点数类型，并存储在字典中的键"bert_1"下
                "bert_2": bert_en.astype(np.float32),  # 将bert_en数组转换为32位浮点数类型，并存储在字典中的键"bert_2"下
                "g": g.astype(np.float32),  # 将g数组转换为32位浮点数类型，并存储在字典中的键"g"下
                "vqidx": vqidx.astype(np.int64),  # 将vqidx数组转换为64位整数类型，并存储在字典中的键"vqidx"下
                "sid": sid.astype(np.int64),  # 将sid数组转换为64位整数类型，并存储在字典中的键"sid"下
            },
        )
        x, m_p, logs_p, x_mask = enc_rtn[0], enc_rtn[1], enc_rtn[2], enc_rtn[3]  # 从enc_rtn中获取四个值分别赋给x, m_p, logs_p, x_mask
        np.random.seed(seed)  # 设置随机数种子为seed
        zinput = np.random.randn(x.shape[0], 2, x.shape[2]) * sdp_noise_scale  # 生成服从标准正态分布的随机数数组，并乘以sdp_noise_scale
        logw = self.sdp.run(
            None, {"x": x, "x_mask": x_mask, "zin": zinput.astype(np.float32), "g": g}
        )[0] * (sdp_ratio) + self.dp.run(None, {"x": x, "x_mask": x_mask, "g": g})[
            0
        ] * (
            1 - sdp_ratio
        )  # 使用sdp和dp模型运行得到logw
        w = np.exp(logw) * x_mask * length_scale  # 计算w值
        w_ceil = np.ceil(w)  # 对w进行向上取整
        y_lengths = np.clip(np.sum(w_ceil, (1, 2)), a_min=1.0, a_max=100000).astype(  # 对w_ceil进行求和并进行裁剪和类型转换
        np.int64  # 将数组中的元素转换为64位整数
    )
    y_mask = np.expand_dims(sequence_mask(y_lengths, None), 1)  # 根据y_lengths生成一个掩码，然后在第一维度上扩展维度为1
    attn_mask = np.expand_dims(x_mask, 2) * np.expand_dims(y_mask, -1)  # 在第二维度上扩展x_mask的维度为2，然后与y_mask在最后一维度上扩展维度为1的结果相乘
    attn = generate_path(w_ceil, attn_mask)  # 使用w_ceil和attn_mask生成注意力矩阵
    m_p = np.matmul(attn.squeeze(1), m_p.transpose(0, 2, 1)).transpose(
        0, 2, 1
    )  # 对attn在第一维度上去掉维度为1后与m_p在第一维度上转置后的矩阵相乘，然后再次转置
    logs_p = np.matmul(attn.squeeze(1), logs_p.transpose(0, 2, 1)).transpose(
        0, 2, 1
    )  # 对attn在第一维度上去掉维度为1后与logs_p在第一维度上转置后的矩阵相乘，然后再次转置

    z_p = (
        m_p
        + np.random.randn(m_p.shape[0], m_p.shape[1], m_p.shape[2])
        * np.exp(logs_p)
        * seq_noise_scale
    )  # 生成z_p，包括m_p和随机数乘以exp(logs_p)再乘以seq_noise_scale

    z = self.flow.run(  # 运行self.flow并将结果赋给z
            None,  # 第一个参数为 None
            {  # 第二个参数为字典，包含以下键值对
                "z_p": z_p.astype(np.float32),  # 键为 "z_p"，值为 z_p 转换为 np.float32 类型
                "y_mask": y_mask.astype(np.float32),  # 键为 "y_mask"，值为 y_mask 转换为 np.float32 类型
                "g": g,  # 键为 "g"，值为 g
            },
        )[0]  # 返回元组的第一个元素

        return self.dec.run(None, {"z_in": z.astype(np.float32), "g": g})[0]  # 调用 self.dec.run 方法，传入参数为 None 和一个包含 "z_in" 和 "g" 键的字典，返回元组的第一个元素
```