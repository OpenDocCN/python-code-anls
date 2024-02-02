# `so-vits-svc\vencoder\ContentVec768L12.py`

```py
# 导入 torch 库
import torch
# 从 fairseq 库中导入 checkpoint_utils 模块
from fairseq import checkpoint_utils
# 从 vencoder.encoder 模块中导入 SpeechEncoder 类
from vencoder.encoder import SpeechEncoder

# 定义 ContentVec768L12 类，继承自 SpeechEncoder 类
class ContentVec768L12(SpeechEncoder):
    # 初始化方法，接受 vec_path 和 device 两个参数
    def __init__(self, vec_path="pretrain/checkpoint_best_legacy_500.pt", device=None):
        # 调用父类的初始化方法
        super().__init__()
        # 打印加载模型的信息
        print("load model(s) from {}".format(vec_path))
        # 设置隐藏维度为 768
        self.hidden_dim = 768
        # 使用 checkpoint_utils.load_model_ensemble_and_task 方法加载模型
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
          [vec_path],
          suffix="",
        )
        # 如果 device 为 None，则根据是否有 CUDA 设备自动选择设备
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 否则，根据传入的 device 参数选择设备
        else:
            self.dev = torch.device(device)
        # 将模型移动到选择的设备上
        self.model = models[0].to(self.dev)
        # 设置模型为评估模式
        self.model.eval()

    # 定义 encoder 方法，接受 wav 作为输入
    def encoder(self, wav):
        # 将输入的 wav 赋值给 feats
        feats = wav
        # 如果 feats 的维度为 2，表示有双声道，将其取平均值
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        # 断言 feats 的维度为 1
        assert feats.dim() == 1, feats.dim()
        # 将 feats 转换为形状为 (1, -1) 的张量
        feats = feats.view(1, -1)
        # 创建与 feats 相同形状的填充掩码张量
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        # 构建输入字典，包括源数据、填充掩码和输出层编号
        inputs = {
          "source": feats.to(wav.device),
          "padding_mask": padding_mask.to(wav.device),
          "output_layer": 12,  # layer 12
        }
        # 使用 torch.no_grad() 上下文，禁用梯度计算
        with torch.no_grad():
            # 提取特征
            logits = self.model.extract_features(**inputs)
        # 返回转置后的 logits[0]
        return logits[0].transpose(1, 2)
```