# `so-vits-svc\vencoder\ContentVec768L12_Onnx.py`

```py
# 导入 onnxruntime 和 torch 模块
import onnxruntime
import torch

# 从 vencoder.encoder 模块中导入 SpeechEncoder 类
from vencoder.encoder import SpeechEncoder

# 定义一个名为 ContentVec768L12_Onnx 的类，继承自 SpeechEncoder 类
class ContentVec768L12_Onnx(SpeechEncoder):
    # 初始化方法，接受 vec_path 和 device 两个参数
    def __init__(self, vec_path="pretrain/vec-768-layer-12.onnx", device=None):
        # 调用父类的初始化方法
        super().__init__()
        # 打印加载模型的信息
        print("load model(s) from {}".format(vec_path))
        # 设置隐藏维度为 768
        self.hidden_dim = 768
        # 如果 device 为 None，则将 self.dev 设置为 CPU 设备
        if device is None:
            self.dev = torch.device("cpu")
        # 否则将 self.dev 设置为传入的 device
        else:
            self.dev = torch.device(device)

        # 如果 device 为 'cuda' 或者是 torch.device("cuda")，则设置 providers 为包含 'CUDAExecutionProvider' 和 'CPUExecutionProvider' 的列表
        if device == 'cuda' or device == torch.device("cuda"):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        # 否则设置 providers 为只包含 'CPUExecutionProvider' 的列表
        else:
            providers = ['CPUExecutionProvider']
            
        # 使用 onnxruntime.InferenceSession 创建模型，传入 vec_path 和 providers 参数
        self.model = onnxruntime.InferenceSession(vec_path, providers=providers)

    # 定义一个名为 encoder 的方法，接受 wav 作为输入
    def encoder(self, wav):
        # 将输入的 wav 赋值给 feats
        feats = wav
        # 如果 feats 的维度为 2，即双声道
        if feats.dim() == 2:  
            # 取平均值，将双声道转换为单声道
            feats = feats.mean(-1)
        # 断言 feats 的维度为 1
        assert feats.dim() == 1, feats.dim()
        # 将 feats 转换为形状为 (1, -1) 的张量
        feats = feats.view(1, -1)
        # 在第 0 维度上增加一个维度
        feats = feats.unsqueeze(0).cpu().detach().numpy()
        # 创建一个字典，键为模型的输入名称，值为 feats
        onnx_input = {self.model.get_inputs()[0].name: feats}
        # 运行模型，传入输入字典，得到 logits
        logits = self.model.run(None, onnx_input)
        # 将 logits 转置，然后转换为指定设备上的张量，并返回
        return torch.tensor(logits[0]).transpose(1, 2).to(self.dev)
```