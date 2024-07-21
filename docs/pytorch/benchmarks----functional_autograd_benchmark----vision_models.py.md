# `.\pytorch\benchmarks\functional_autograd_benchmark\vision_models.py`

```py
# 导入必要的类型提示工具
from typing import cast

# 导入 TorchVision 中的预训练模型
import torchvision_models as models

# 从自定义的 utils 模块中导入函数和类型
from utils import check_for_functorch, extract_weights, GetterReturnType, load_weights

# 导入 PyTorch 库
import torch
from torch import Tensor

# 检查是否安装了 functorch 库
has_functorch = check_for_functorch()

# 定义获取 ResNet-18 模型及其前向方法的函数
def get_resnet18(device: torch.device) -> GetterReturnType:
    N = 32
    # 加载不含预训练权重的 ResNet-18 模型
    model = models.resnet18(pretrained=False)

    # 如果安装了 functorch，则替换模型中的所有 BatchNorm 层
    if has_functorch:
        from functorch.experimental import replace_all_batch_norm_modules_
        replace_all_batch_norm_modules_(model)

    # 定义交叉熵损失函数
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    # 提取模型的参数及其名称
    params, names = extract_weights(model)

    # 生成随机输入和标签
    inputs = torch.rand([N, 3, 224, 224], device=device)
    labels = torch.rand(N, device=device).mul(10).long()

    # 定义前向传播函数
    def forward(*new_params: Tensor) -> Tensor:
        # 载入新的权重到模型中
        load_weights(model, names, new_params)
        # 运行模型进行推理
        out = model(inputs)
        # 计算模型输出与标签之间的交叉熵损失
        loss = criterion(out, labels)
        return loss

    return forward, params


# 定义获取 FCN-ResNet50 模型及其前向方法的函数
def get_fcn_resnet(device: torch.device) -> GetterReturnType:
    N = 8
    # 加载不含预训练权重的 FCN-ResNet50 模型
    model = models.fcn_resnet50(pretrained=False, pretrained_backbone=False)

    # 如果安装了 functorch，则替换模型中的所有 BatchNorm 层并禁用 Dropout 层
    if has_functorch:
        from functorch.experimental import replace_all_batch_norm_modules_
        replace_all_batch_norm_modules_(model)
        model.eval()  # 为了一致性检查，禁用模型中的 Dropout 层

    model.to(device)
    # 提取模型的参数及其名称
    params, names = extract_weights(model)

    # 生成随机输入和标签
    inputs = torch.rand([N, 3, 480, 480], device=device)
    labels = torch.rand([N, 21, 480, 480], device=device)

    # 定义前向传播函数
    def forward(*new_params: Tensor) -> Tensor:
        # 载入新的权重到模型中
        load_weights(model, names, new_params)
        # 运行模型进行推理，获取输出中的 "out" 部分
        out = model(inputs)["out"]
        # 计算模型输出与标签之间的均方误差损失
        loss = criterion(out, labels)
        return loss

    return forward, params


# 定义获取 DETR 模型及其前向方法的函数
def get_detr(device: torch.device) -> GetterReturnType:
    # 从 CLI 默认参数创建 DETR 模型
    N = 2
    num_classes = 91
    hidden_dim = 256
    nheads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6

    model = models.DETR(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        nheads=nheads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
    )

    # 如果安装了 functorch，则替换模型中的所有 BatchNorm 层
    if has_functorch:
        from functorch.experimental import replace_all_batch_norm_modules_
        replace_all_batch_norm_modules_(model)

    # 定义损失函数的组件和权重
    losses = ["labels", "boxes", "cardinality"]
    eos_coef = 0.1
    bbox_loss_coef = 5
    giou_loss_coef = 2
    weight_dict = {
        "loss_ce": 1,
        "loss_bbox": bbox_loss_coef,
        "loss_giou": giou_loss_coef,
    }

    # 使用 HungarianMatcher 创建匹配器
    matcher = models.HungarianMatcher(1, 5, 2)
    # 使用 SetCriterion 定义整体损失函数
    criterion = models.SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=eos_coef,
        losses=losses,
    )

    # 将模型及损失函数移动到指定的设备上
    model = model.to(device)
    criterion = criterion.to(device)
    # 提取模型的参数及其名称
    params, names = extract_weights(model)
    # 生成一个 N x 3 x 800 x 1200 的随机张量，表示输入数据，使用指定的设备（如 GPU）
    inputs = torch.rand(N, 3, 800, 1200, device=device)
    # 初始化一个空列表，用于存储每个样本的标签和框信息
    labels = []
    # 遍历每个样本
    for idx in range(N):
        # 初始化一个空字典，用于存储当前样本的标签和框信息
        targets = {}
        # 从区间 [5, 10) 中随机选择一个整数作为目标数
        n_targets: int = int(torch.randint(5, 10, size=tuple()).item())
        # 随机生成 n_targets 个标签，作为当前样本的标签信息，使用指定的设备
        label = torch.randint(5, 10, size=(n_targets,), device=device)
        targets["labels"] = label
        # 随机生成 n_targets 个框的坐标信息，每个框包含四个整数值，使用指定的设备
        boxes = torch.randint(100, 800, size=(n_targets, 4), device=device)
        # 对每个框的坐标进行调整，确保第一点在左上角，第二点在右下角
        for t in range(n_targets):
            if boxes[t, 0] > boxes[t, 2]:
                boxes[t, 0], boxes[t, 2] = boxes[t, 2], boxes[t, 0]
            if boxes[t, 1] > boxes[t, 3]:
                boxes[t, 1], boxes[t, 3] = boxes[t, 3], boxes[t, 1]
        targets["boxes"] = boxes.float()
        # 将当前样本的标签和框信息添加到 labels 列表中
        labels.append(targets)

    # 定义一个 forward 函数，接受一系列新的参数张量，并返回计算得到的损失张量
    def forward(*new_params: Tensor) -> Tensor:
        # 调用 load_weights 函数，加载模型权重到指定的参数张量中
        load_weights(model, names, new_params)
        # 使用输入数据计算模型的输出结果
        out = model(inputs)

        # 计算模型输出和真实标签之间的损失
        loss = criterion(out, labels)
        # 获取损失函数的权重字典
        weight_dict = criterion.weight_dict
        # 计算加权损失，仅考虑权重字典中包含的损失项
        final_loss = cast(
            Tensor,
            sum(loss[k] * weight_dict[k] for k in loss.keys() if k in weight_dict),
        )
        # 返回最终的加权损失作为输出
        return final_loss

    # 返回定义的 forward 函数和参数列表作为结果
    return forward, params
```