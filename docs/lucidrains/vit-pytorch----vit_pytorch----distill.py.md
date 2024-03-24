# `.\lucidrains\vit-pytorch\vit_pytorch\distill.py`

```
import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 中的函数模块
from torch import nn  # 从 PyTorch 中导入 nn 模块
from vit_pytorch.vit import ViT  # 从 vit_pytorch 库中导入 ViT 类
from vit_pytorch.t2t import T2TViT  # 从 vit_pytorch 库中导入 T2TViT 类
from vit_pytorch.efficient import ViT as EfficientViT  # 从 vit_pytorch 库中导入 EfficientViT 类

from einops import rearrange, repeat  # 从 einops 库中导入 rearrange 和 repeat 函数

# helpers

def exists(val):  # 定义 exists 函数，用于判断变量是否存在
    return val is not None  # 返回变量是否不为 None

# classes

class DistillMixin:  # 定义 DistillMixin 类
    def forward(self, img, distill_token = None):  # 定义 forward 方法，接收图像和 distill_token 参数
        distilling = exists(distill_token)  # 判断 distill_token 是否存在
        x = self.to_patch_embedding(img)  # 将图像转换为 patch embedding
        b, n, _ = x.shape  # 获取 x 的形状信息

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)  # 重复添加 cls_token
        x = torch.cat((cls_tokens, x), dim = 1)  # 在维度 1 上拼接 cls_tokens 和 x
        x += self.pos_embedding[:, :(n + 1)]  # 添加位置编码

        if distilling:  # 如果进行蒸馏
            distill_tokens = repeat(distill_token, '() n d -> b n d', b = b)  # 重复添加 distill_token
            x = torch.cat((x, distill_tokens), dim = 1)  # 在维度 1 上拼接 x 和 distill_tokens

        x = self._attend(x)  # 调用 _attend 方法进行注意力计算

        if distilling:  # 如果进行蒸馏
            x, distill_tokens = x[:, :-1], x[:, -1]  # 分割出 distill_tokens

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]  # 计算平均值或取第一个值

        x = self.to_latent(x)  # 转换为 latent 表示
        out = self.mlp_head(x)  # 经过 MLP 头部处理得到输出

        if distilling:  # 如果进行蒸馏
            return out, distill_tokens  # 返回输出和 distill_tokens

        return out  # 返回输出

class DistillableViT(DistillMixin, ViT):  # 定义 DistillableViT 类，继承自 DistillMixin 和 ViT
    def __init__(self, *args, **kwargs):  # 初始化方法
        super(DistillableViT, self).__init__(*args, **kwargs)  # 调用父类的初始化方法
        self.args = args  # 保存参数
        self.kwargs = kwargs  # 保存关键字参数
        self.dim = kwargs['dim']  # 保存维度信息
        self.num_classes = kwargs['num_classes']  # 保存类别数

    def to_vit(self):  # 定义 to_vit 方法
        v = ViT(*self.args, **self.kwargs)  # 创建 ViT 对象
        v.load_state_dict(self.state_dict())  # 加载当前状态字典
        return v  # 返回 ViT 对象

    def _attend(self, x):  # 定义 _attend 方法
        x = self.dropout(x)  # 使用 dropout
        x = self.transformer(x)  # 经过 transformer 处理
        return x  # 返回处理后的结果

class DistillableT2TViT(DistillMixin, T2TViT):  # 定义 DistillableT2TViT 类，继承自 DistillMixin 和 T2TViT
    def __init__(self, *args, **kwargs):  # 初始化方法
        super(DistillableT2TViT, self).__init__(*args, **kwargs)  # 调用父类的初始化方法
        self.args = args  # 保存参数
        self.kwargs = kwargs  # 保存关键字参数
        self.dim = kwargs['dim']  # 保存维度信息
        self.num_classes = kwargs['num_classes']  # 保存类别数

    def to_vit(self):  # 定义 to_vit 方法
        v = T2TViT(*self.args, **self.kwargs)  # 创建 T2TViT 对象
        v.load_state_dict(self.state_dict())  # 加载当前状态字典
        return v  # 返回 T2TViT 对象

    def _attend(self, x):  # 定义 _attend 方法
        x = self.dropout(x)  # 使用 dropout
        x = self.transformer(x)  # 经过 transformer 处理
        return x  # 返回处理后的结果

class DistillableEfficientViT(DistillMixin, EfficientViT):  # 定义 DistillableEfficientViT 类，继承自 DistillMixin 和 EfficientViT
    def __init__(self, *args, **kwargs):  # 初始化方法
        super(DistillableEfficientViT, self).__init__(*args, **kwargs)  # 调用父类的初始化方法
        self.args = args  # 保存参数
        self.kwargs = kwargs  # 保存关键字参数
        self.dim = kwargs['dim']  # 保存维度信息
        self.num_classes = kwargs['num_classes']  # 保存类别数

    def to_vit(self):  # 定义 to_vit 方法
        v = EfficientViT(*self.args, **self.kwargs)  # 创建 EfficientViT 对象
        v.load_state_dict(self.state_dict())  # 加载当前状态字典
        return v  # 返回 EfficientViT 对象

    def _attend(self, x):  # 定义 _attend 方法
        return self.transformer(x)  # 经过 transformer 处理

# knowledge distillation wrapper

class DistillWrapper(nn.Module):  # 定义 DistillWrapper 类，继承自 nn.Module
    def __init__(  # 初始化方法
        self,
        *,
        teacher,  # 教师模型
        student,  # 学生模型
        temperature = 1.,  # 温度参数
        alpha = 0.5,  # alpha 参数
        hard = False  # 是否硬蒸馏
    ):
        super().__init__()  # 调用父类的初始化方法
        assert (isinstance(student, (DistillableViT, DistillableT2TViT, DistillableEfficientViT))) , 'student must be a vision transformer'  # 断言学生模型必须是视觉 transformer

        self.teacher = teacher  # 保存教师模型
        self.student = student  # 保存学生模型

        dim = student.dim  # 获取学生模型的维度信息
        num_classes = student.num_classes  # 获取学生模型的类别数
        self.temperature = temperature  # 保存温度参数
        self.alpha = alpha  # 保存 alpha 参数
        self.hard = hard  # 保存是否硬蒸馏

        self.distillation_token = nn.Parameter(torch.randn(1, 1, dim))  # 创建蒸馏 token

        self.distill_mlp = nn.Sequential(  # 创建 MLP 处理蒸馏信息
            nn.LayerNorm(dim),  # LayerNorm 处理
            nn.Linear(dim, num_classes)  # 线性层处理
        )
    # 定义一个前向传播函数，接受输入图像、标签、温度和权重参数
    def forward(self, img, labels, temperature = None, alpha = None, **kwargs):
        # 获取输入图像的批量大小
        b, *_ = img.shape
        # 如果 alpha 参数存在，则使用传入的值，否则使用类属性中的值
        alpha = alpha if exists(alpha) else self.alpha
        # 如果 temperature 参数存在，则使用传入的值，否则使用类属性中的值
        T = temperature if exists(temperature) else self.temperature

        # 在不计算梯度的情况下，通过教师模型获取教师网络的输出
        with torch.no_grad():
            teacher_logits = self.teacher(img)

        # 通过学生模型获取学生网络的输出和蒸馏 token
        student_logits, distill_tokens = self.student(img, distill_token = self.distillation_token, **kwargs)
        # 通过蒸馏 token 获取蒸馏网络的输出
        distill_logits = self.distill_mlp(distill_tokens)

        # 计算学生网络的交叉熵损失
        loss = F.cross_entropy(student_logits, labels)

        # 如果不是硬蒸馏，则计算软蒸馏损失
        if not self.hard:
            distill_loss = F.kl_div(
                F.log_softmax(distill_logits / T, dim = -1),
                F.softmax(teacher_logits / T, dim = -1).detach(),
            reduction = 'batchmean')
            distill_loss *= T ** 2

        # 如果是硬蒸馏，则计算交叉熵损失
        else:
            teacher_labels = teacher_logits.argmax(dim = -1)
            distill_loss = F.cross_entropy(distill_logits, teacher_labels)

        # 返回加权损失值，结合了学生网络的损失和蒸馏损失
        return loss * (1 - alpha) + distill_loss * alpha
```