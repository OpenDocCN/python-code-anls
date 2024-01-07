# `Bert-VITS2\onnx_modules\V200_OnnxInference\__init__.py`

```

# 导入 numpy 库并重命名为 np
import numpy as np
# 导入 onnxruntime 库并重命名为 ort
import onnxruntime as ort

# 定义函数 convert_pad_shape，用于转换 pad_shape
def convert_pad_shape(pad_shape):
    # 将 pad_shape 列表倒序
    layer = pad_shape[::-1]
    # 将倒序后的列表展开成一维列表
    pad_shape = [item for sublist in layer for item in sublist]
    return pad_shape

# 定义函数 sequence_mask，用于生成序列掩码
def sequence_mask(length, max_length=None):
    # 如果未提供最大长度，则取 length 中的最大值
    if max_length is None:
        max_length = length.max()
    # 生成 0 到 max_length-1 的数组
    x = np.arange(max_length, dtype=length.dtype)
    # 将 x 和 length 进行扩展维度后比较，生成掩码
    return np.expand_dims(x, 0) < np.expand_dims(length, 1)

# 定义函数 generate_path，用于生成路径
def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    # 获取 mask 的形状信息
    b, _, t_y, t_x = mask.shape
    # 对 duration 进行累积求和
    cum_duration = np.cumsum(duration, -1)
    # 将累积求和后的数组展平
    cum_duration_flat = cum_duration.reshape(b * t_x)
    # 生成路径
    path = sequence_mask(cum_duration_flat, t_y)
    path = path.reshape(b, t_x, t_y)
    path = path ^ np.pad(path, ((0, 0), (1, 0), (0, 0)))[:, :-1]
    path = np.expand_dims(path, 1).transpose(0, 1, 3, 2)
    return path

# 定义类 OnnxInferenceSession
class OnnxInferenceSession:
    # 初始化方法
    def __init__(self, path, Providers=["CPUExecutionProvider"]):
        # 创建推理会话对象
        self.enc = ort.InferenceSession(path["enc"], providers=Providers)
        self.emb_g = ort.InferenceSession(path["emb_g"], providers=Providers)
        self.dp = ort.InferenceSession(path["dp"], providers=Providers)
        self.sdp = ort.InferenceSession(path["sdp"], providers=Providers)
        self.flow = ort.InferenceSession(path["flow"], providers=Providers)
        self.dec = ort.InferenceSession(path["dec"], providers=Providers)

    # 调用方法
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
        # 对输入进行维度扩展
        if seq.ndim == 1:
            seq = np.expand_dims(seq, 0)
        if tone.ndim == 1:
            tone = np.expand_dims(tone, 0)
        if language.ndim == 1:
            language = np.expand_dims(language, 0)
        # 断言输入的维度
        assert (seq.ndim == 2, tone.ndim == 2, language.ndim == 2)
        # 运行推理会话
        g = self.emb_g.run(
            None,
            {
                "sid": sid.astype(np.int64),
            },
        )[0]
        g = np.expand_dims(g, -1)
        enc_rtn = self.enc.run(
            None,
            {
                "x": seq.astype(np.int64),
                "t": tone.astype(np.int64),
                "language": language.astype(np.int64),
                "bert_0": bert_zh.astype(np.float32),
                "bert_1": bert_jp.astype(np.float32),
                "bert_2": bert_en.astype(np.float32),
                "g": g.astype(np.float32),
            },
        )
        x, m_p, logs_p, x_mask = enc_rtn[0], enc_rtn[1], enc_rtn[2], enc_rtn[3]
        # 生成随机种子
        np.random.seed(seed)
        zinput = np.random.randn(x.shape[0], 2, x.shape[2]) * sdp_noise_scale
        logw = self.sdp.run(
            None, {"x": x, "x_mask": x_mask, "zin": zinput.astype(np.float32), "g": g}
        )[0] * (sdp_ratio) + self.dp.run(None, {"x": x, "x_mask": x_mask, "g": g})[0] * (1 - sdp_ratio)
        w = np.exp(logw) * x_mask * length_scale
        w_ceil = np.ceil(w)
        y_lengths = np.clip(np.sum(w_ceil, (1, 2)), a_min=1.0, a_max=100000).astype(np.int64)
        y_mask = np.expand_dims(sequence_mask(y_lengths, None), 1)
        attn_mask = np.expand_dims(x_mask, 2) * np.expand_dims(y_mask, -1)
        attn = generate_path(w_ceil, attn_mask)
        m_p = np.matmul(attn.squeeze(1), m_p.transpose(0, 2, 1)).transpose(0, 2, 1)
        logs_p = np.matmul(attn.squeeze(1), logs_p.transpose(0, 2, 1)).transpose(0, 2, 1)
        z_p = (
            m_p
            + np.random.randn(m_p.shape[0], m_p.shape[1], m_p.shape[2])
            * np.exp(logs_p)
            * seq_noise_scale
        )
        z = self.flow.run(
            None,
            {
                "z_p": z_p.astype(np.float32),
                "y_mask": y_mask.astype(np.float32),
                "g": g,
            },
        )[0]
        return self.dec.run(None, {"z_in": z.astype(np.float32), "g": g})[0]

```