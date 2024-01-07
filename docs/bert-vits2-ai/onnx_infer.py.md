# `Bert-VITS2\onnx_infer.py`

```

# 导入OnnxInferenceSession类和numpy库
from onnx_modules.V220_OnnxInference import OnnxInferenceSession
import numpy as np

# 创建OnnxInferenceSession对象，传入模型文件路径和执行提供者
Session = OnnxInferenceSession(
    {
        "enc": "onnx/BertVits2.2PT/BertVits2.2PT_enc_p.onnx",
        "emb_g": "onnx/BertVits2.2PT/BertVits2.2PT_emb.onnx",
        "dp": "onnx/BertVits2.2PT/BertVits2.2PT_dp.onnx",
        "sdp": "onnx/BertVits2.2PT/BertVits2.2PT_sdp.onnx",
        "flow": "onnx/BertVits2.2PT/BertVits2.2PT_flow.onnx",
        "dec": "onnx/BertVits2.2PT/BertVits2.2PT_dec.onnx",
    },
    Providers=["CPUExecutionProvider"],
)

# 创建输入数据x，数据类型为numpy数组
x = np.array(
    [
        0,
        97,
        0,
        8,
        0,
        78,
        0,
        8,
        0,
        76,
        0,
        37,
        0,
        40,
        0,
        97,
        0,
        8,
        0,
        23,
        0,
        8,
        0,
        74,
        0,
        26,
        0,
        104,
        0,
    ]
)

# 创建与x相同形状的全零数组
tone = np.zeros_like(x)
language = np.zeros_like(x)
# 创建包含单个元素0的数组
sid = np.array([0])
# 创建与x相同形状的随机数组
bert = np.random.randn(x.shape[0], 1024)
ja_bert = np.random.randn(x.shape[0], 1024)
en_bert = np.random.randn(x.shape[0], 1024)
emo = np.random.randn(512, 1)

# 调用Session对象，传入输入数据，获取输出结果
audio = Session(x, tone, language, bert, ja_bert, en_bert, emo, sid)

# 打印输出结果
print(audio)

```