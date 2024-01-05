# `d:/src/tocomm/Bert-VITS2\onnx_infer.py`

```
# 导入OnnxInferenceSession类和numpy库
from onnx_modules.V220_OnnxInference import OnnxInferenceSession
import numpy as np

# 创建OnnxInferenceSession对象Session，传入模型文件路径和执行提供者
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

# 创建输入数据x，使用numpy.array方法创建一个包含多个数字的数组
x = np.array(
    [
        0,
        97,
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制内容，并封装成字节流
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
        0,  # 创建一个值为0的整数
        74,  # 创建一个值为74的整数
        0,   # 创建一个值为0的整数
        26,  # 创建一个值为26的整数
        0,   # 创建一个值为0的整数
        104, # 创建一个值为104的整数
        0,   # 创建一个值为0的整数
    ]
)
tone = np.zeros_like(x)  # 创建一个与x相同形状的全零数组，并赋值给tone
language = np.zeros_like(x)  # 创建一个与x相同形状的全零数组，并赋值给language
sid = np.array([0])  # 创建一个包含单个元素0的数组，并赋值给sid
bert = np.random.randn(x.shape[0], 1024)  # 创建一个形状为(x.shape[0], 1024)的随机数组，并赋值给bert
ja_bert = np.random.randn(x.shape[0], 1024)  # 创建一个形状为(x.shape[0], 1024)的随机数组，并赋值给ja_bert
en_bert = np.random.randn(x.shape[0], 1024)  # 创建一个形状为(x.shape[0], 1024)的随机数组，并赋值给en_bert
emo = np.random.randn(512, 1)  # 创建一个形状为(512, 1)的随机数组，并赋值给emo

audio = Session(x, tone, language, bert, ja_bert, en_bert, emo, sid)  # 使用上述创建的数组和sid创建一个Session对象，并赋值给audio

print(audio)  # 打印audio对象
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制数据，并封装成字节流对象
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流创建 ZIP 对象，'r'表示以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名，读取文件数据，组成字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
```