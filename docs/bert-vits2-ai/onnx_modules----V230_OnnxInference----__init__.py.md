# `Bert-VITS2\onnx_modules\V230_OnnxInference\__init__.py`

```
# 导入 numpy 库，并重命名为 np
import numpy as np
# 导入 onnxruntime 库，并重命名为 ort
import onnxruntime as ort

# 定义函数，将输入的 pad_shape 列表进行转换
def convert_pad_shape(pad_shape):
    # 将 pad_shape 列表进行逆序
    layer = pad_shape[::-1]
    # 将逆序后的列表展开成一维列表
    pad_shape = [item for sublist in layer for item in sublist]
    # 返回转换后的 pad_shape 列表
    return pad_shape

# 定义函数，生成序列掩码
def sequence_mask(length, max_length=None):
    # 如果未提供 max_length，则取 length 中的最大值
    if max_length is None:
        max_length = length.max()
    # 生成 0 到 max_length-1 的数组
    x = np.arange(max_length, dtype=length.dtype)
    # 将 x 和 length 进行扩展维度后进行比较，生成序列掩码
    return np.expand_dims(x, 0) < np.expand_dims(length, 1)

# 定义函数，生成路径
def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    # 获取 mask 的形状信息
    b, _, t_y, t_x = mask.shape
    # 对 duration 进行累积求和
    cum_duration = np.cumsum(duration, -1)
    # 将累积求和后的数组展平成一维数组
    cum_duration_flat = cum_duration.reshape(b * t_x)
    # 生成路径
    path = sequence_mask(cum_duration_flat, t_y)
    path = path.reshape(b, t_x, t_y)
    path = path ^ np.pad(path, ((0, 0), (1, 0), (0, 0)))[:, :-1]
    path = np.expand_dims(path, 1).transpose(0, 1, 3, 2)
    # 返回生成的路径
    return path

# 定义类，用于进行 ONNX 推理会话
class OnnxInferenceSession:
    def __init__(self, path, Providers=["CPUExecutionProvider"]):
        # 创建编码器的 ONNX 推理会话
        self.enc = ort.InferenceSession(path["enc"], providers=Providers)
        # 创建嵌入层的 ONNX 推理会话
        self.emb_g = ort.InferenceSession(path["emb_g"], providers=Providers)
        # 创建 dp 的 ONNX 推理会话
        self.dp = ort.InferenceSession(path["dp"], providers=Providers)
        # 创建 sdp 的 ONNX 推理会话
        self.sdp = ort.InferenceSession(path["sdp"], providers=Providers)
        # 创建流的 ONNX 推理会话
        self.flow = ort.InferenceSession(path["flow"], providers=Providers)
        # 创建解码器的 ONNX 推理会话
        self.dec = ort.InferenceSession(path["dec"], providers=Providers)

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
```