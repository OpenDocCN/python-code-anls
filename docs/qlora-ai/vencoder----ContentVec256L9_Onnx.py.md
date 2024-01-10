# `so-vits-svc\vencoder\ContentVec256L9_Onnx.py`

```
# 导入 onnxruntime 和 torch 库
import onnxruntime
import torch

# 从 vencoder.encoder 中导入 SpeechEncoder 类
from vencoder.encoder import SpeechEncoder

# 定义 ContentVec256L9_Onnx 类，继承自 SpeechEncoder 类
class ContentVec256L9_Onnx(SpeechEncoder):
    # 初始化方法，接受参数 vec_path 和 device，默认为 None
    def __init__(self, vec_path="pretrain/vec-256-layer-9.onnx", device=None):
        # 调用父类的初始化方法
        super().__init__()
        # 打印加载模型的信息
        print("load model(s) from {}".format(vec_path))
        # 设置隐藏维度为 256
        self.hidden_dim = 256
        # 如果 device 为 None，则设备为 CPU
        if device is None:
            self.dev = torch.device("cpu")
        # 否则设备为传入的 device
        else:
            self.dev = torch.device(device)
        # 根据设备类型设置 providers
        if device == 'cpu' or device == torch.device("cpu") or device is None:
            providers = ['CPUExecutionProvider']
        elif device == 'cuda' or device == torch.device("cuda"):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        # 使用 onnxruntime 创建推理会话，传入 vec_path 和 providers
        self.model = onnxruntime.InferenceSession(vec_path, providers=providers)

    # 定义 encoder 方法，接受参数 wav
    def encoder(self, wav):
        # 将 wav 赋值给 feats
        feats = wav
        # 如果 feats 的维度为 2，则取平均值，将其转为一维
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        # 断言 feats 的维度为 1
        assert feats.dim() == 1, feats.dim()
        # 将 feats 转为二维，再转为 numpy 数组
        feats = feats.view(1, -1)
        feats = feats.unsqueeze(0).cpu().detach().numpy()
        # 构建 onnx_input 字典，键为模型输入的名称，值为 feats
        onnx_input = {self.model.get_inputs()[0].name: feats}
        # 运行模型，得到 logits
        logits = self.model.run(None, onnx_input)
        # 将 logits 转为 torch 张量，并进行维度转置，最后转到指定设备
        return torch.tensor(logits[0]).transpose(1, 2).to(self.dev)
```