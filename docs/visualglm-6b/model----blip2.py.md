# `.\VisualGLM-6B\model\blip2.py`

```
# 导入 torch 库
import torch
# 导入 torch 中的神经网络模块
import torch.nn as nn

# 从 sat.model 模块中导入 ViTModel、BaseModel 类
from sat.model import ViTModel, BaseModel
# 从 sat.model 模块中导入 BaseMixin 类
from sat.model import BaseMixin
# 从 sat 模块中导入 AutoModel 类
from sat import AutoModel
# 从 copy 模块中导入 deepcopy 函数
from copy import deepcopy
# 从 torchvision 模块中导入 transforms 类
from torchvision import transforms
# 从 torchvision.transforms.functional 模块中导入 InterpolationMode 类
from torchvision.transforms.functional import InterpolationMode

# 定义 LNFinalyMixin 类，继承自 BaseMixin 类
class LNFinalyMixin(BaseMixin):
    # 初始化方法
    def __init__(self, hidden_size):
        super().__init__()
        # 创建一个 LayerNorm 层，对输入进行归一化处理
        self.ln_vision = nn.LayerNorm(hidden_size)

    # 定义 final_forward 方法
    def final_forward(self, logits, **kw_args):
        # 返回经过 LayerNorm 处理后的 logits
        return self.ln_vision(logits)

# 定义 EVAViT 类，继承自 ViTModel 类
class EVAViT(ViTModel):
    # 初始化方法
    def __init__(self, args, transformer=None, parallel_output=True, **kwargs):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output, **kwargs)
        # 删除 cls mixin
        self.del_mixin("cls")
        # 添加 LNFinalyMixin 类作为 cls mixin
        self.add_mixin("cls", LNFinalyMixin(args.hidden_size))
    
    # 定义 forward 方法
    def forward(self, image):
        # 获取 batch_size
        batch_size = image.size(0)
        # 创建一个全零的 input_ids 张量
        input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=image.device)
        # 创建一个全一的 attention_mask 张量
        attention_mask = torch.tensor([[1.]], dtype=image.dtype, device=image.device)
        # 调用父类的 forward 方法，传入参数并返回结果
        return super().forward(input_ids=input_ids, position_ids=None, attention_mask=attention_mask, image=image)

# 定义 QFormer 类，继承自 BaseModel 类
class QFormer(BaseModel):
    # 初始化方法
    def __init__(self, args, transformer=None, parallel_output=True, **kwargs):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output, activation_func=nn.functional.gelu, **kwargs)
        # 将 transformer 的 position_embeddings 属性设为 None
        self.transformer.position_embeddings = None
    
    # 定义 final_forward 方法
    def final_forward(self, logits, **kw_args):
        # 直接返回 logits
        return logits

    # 定义 position_embedding_forward 方法
    def position_embedding_forward(self, position_ids, **kw_args):
        # 返回 None
        return None
    # 定义一个前向传播函数，接收编码器输出作为输入
    def forward(self, encoder_outputs):
        # 获取编码器输出的批量大小
        batch_size = encoder_outputs.size(0)
        # 生成一个长为32的长整型张量，设备为编码器输出设备，然后在第0维度上增加一个维度，扩展为(batch_size, 32)的张量
        input_ids = torch.arange(32, dtype=torch.long, device=encoder_outputs.device).unsqueeze(0).expand(batch_size, -1)
        # 创建一个形状为(1, 1)的张量，数据类型与编码器输出相同，设备为编码器输出设备
        attention_mask = torch.tensor([[1.]], dtype=encoder_outputs.dtype, device=encoder_outputs.device)
        # 创建一个形状为(1, 1)的张量，数据类型与编码器输出相同，设备为编码器输出设备
        cross_attention_mask = torch.tensor([[1.]], dtype=encoder_outputs.dtype, device=encoder_outputs.device)
        # 调用父类的forward方法，传入输入张量、位置张量为None、注意力掩码、编码器输出和交叉注意力掩码
        return super().forward(input_ids=input_ids, position_ids=None, attention_mask=attention_mask, encoder_outputs=encoder_outputs, cross_attention_mask=cross_attention_mask)
class BLIP2(torch.nn.Module):
    def __init__(self, eva_args, qformer_args, vit=None, qformer=None, **kwargs):
        super().__init__()
        # 如果传入了vit参数，则使用传入的vit，否则创建一个EVAViT对象
        if vit is not None:
            self.vit = vit
        else:
            self.vit = EVAViT(EVAViT.get_args(**eva_args))
        # 如果传入了qformer参数，则使用传入的qformer，否则创建一个QFormer对象
        if qformer is not None:
            self.qformer = qformer
        else:
            self.qformer = QFormer(QFormer.get_args(**qformer_args))
        
        # 创建一个线性层，输入维度为768，输出维度为4096
        self.glm_proj = nn.Linear(768, 4096).to(self.qformer.parameters().__next__().device).to(self.qformer.parameters().__next__().dtype)

    def forward(self, image, **kwargs):
        # 使用vit对图像进行编码
        enc = self.vit(image)[0]
        # 使用qformer对编码结果进行处理
        out = self.qformer(enc)[0]
        # 将处理后的结果通过线性层进行投影
        return self.glm_proj(out)
    
class BlipImageBaseProcessor():
    def __init__(self, mean=None, std=None):
        # 如果未传入均值和标准差，则使用默认值
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        # 创建一个归一化操作
        self.normalize = transforms.Normalize(mean, std)

class BlipImageEvalProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=384, mean=None, std=None):
        super().__init__(mean=mean, std=std)

        # 创建图像处理的转换操作序列
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        # 对输入的图像进行处理
        return self.transform(item)
```