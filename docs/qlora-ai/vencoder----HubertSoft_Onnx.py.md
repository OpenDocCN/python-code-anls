# `so-vits-svc\vencoder\HubertSoft_Onnx.py`

```
# 导入 onnxruntime 库
import onnxruntime
# 导入 torch 库
import torch

# 导入 SpeechEncoder 类
from vencoder.encoder import SpeechEncoder

# 创建 HubertSoft_Onnx 类，继承自 SpeechEncoder 类
class HubertSoft_Onnx(SpeechEncoder):
    # 初始化方法
    def __init__(self, vec_path="pretrain/hubert-soft.onnx", device=None):
        # 调用父类的初始化方法
        super().__init__()
        # 打印加载模型的信息
        print("load model(s) from {}".format(vec_path))
        # 设置隐藏维度为 256
        self.hidden_dim = 256
        # 如果设备为空，则使用 CPU
        if device is None:
            self.dev = torch.device("cpu")
        else:
            self.dev = torch.device(device)

        # 如果设备为 'cuda' 或者是 torch.device("cuda")，则使用 CUDAExecutionProvider 和 CPUExecutionProvider
        if device == 'cuda' or device == torch.device("cuda"):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        # 否则只使用 CPUExecutionProvider
        else:
            providers = ['CPUExecutionProvider']
            
        # 使用 onnxruntime.InferenceSession 加载模型，并指定 providers
        self.model = onnxruntime.InferenceSession(vec_path, providers=providers)

    # 编码器方法
    def encoder(self, wav):
        # 将输入的音频特征赋值给 feats
        feats = wav
        # 如果特征的维度为 2，表示有双声道，则取平均值
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        # 断言特征的维度为 1
        assert feats.dim() == 1, feats.dim()
        # 将特征重塑为 1 行，-1 列的形状
        feats = feats.view(1, -1)
        # 在第 0 维度上增加一个维度，并转移到 CPU 上，并转换为 numpy 数组
        feats = feats.unsqueeze(0).cpu().detach().numpy()
        # 构建输入字典，键为模型的输入名称，值为特征
        onnx_input = {self.model.get_inputs()[0].name: feats}
        # 运行模型，获取输出 logits
        logits = self.model.run(None, onnx_input)
        # 将 logits 转置，并转移到指定设备上
        return torch.tensor(logits[0]).transpose(1, 2).to(self.dev)
```