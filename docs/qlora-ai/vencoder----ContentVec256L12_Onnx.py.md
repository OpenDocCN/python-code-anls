# `so-vits-svc\vencoder\ContentVec256L12_Onnx.py`

```
# 导入 onnxruntime 和 torch 模块
import onnxruntime
import torch

# 从 vencoder.encoder 模块中导入 SpeechEncoder 类
from vencoder.encoder import SpeechEncoder

# 定义 ContentVec256L12_Onnx 类，继承自 SpeechEncoder 类
class ContentVec256L12_Onnx(SpeechEncoder):
    # 初始化方法，接受 vec_path 和 device 两个参数
    def __init__(self, vec_path="pretrain/vec-256-layer-12.onnx", device=None):
        # 调用父类的初始化方法
        super().__init__()
        # 打印加载模型的信息
        print("load model(s) from {}".format(vec_path))
        # 设置隐藏维度为 256
        self.hidden_dim = 256
        # 如果 device 为 None，则使用 CPU 设备
        if device is None:
            self.dev = torch.device("cpu")
        # 否则使用指定的设备
        else:
            self.dev = torch.device(device)

        # 根据设备类型选择执行提供程序
        if device == 'cuda' or device == torch.device("cuda"):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        # 使用 onnxruntime.InferenceSession 加载模型，并指定执行提供程序
        self.model = onnxruntime.InferenceSession(vec_path, providers=providers)

    # 定义 encoder 方法，接受 wav 作为输入
    def encoder(self, wav):
        # 将输入特征赋值给 feats
        feats = wav
        # 如果特征的维度为 2，则取平均值，处理双声道音频
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        # 断言特征的维度为 1
        assert feats.dim() == 1, feats.dim()
        # 将特征重塑为 1 行多列的形式
        feats = feats.view(1, -1)
        # 在第 0 维度上增加一个维度
        feats = feats.unsqueeze(0).cpu().detach().numpy()
        # 构建输入字典，键为模型的输入名称，值为特征
        onnx_input = {self.model.get_inputs()[0].name: feats}
        # 运行模型，获取输出 logits
        logits = self.model.run(None, onnx_input)
        # 将 logits 转置，并转换为指定设备上的张量
        return torch.tensor(logits[0]).transpose(1, 2).to(self.dev)
```