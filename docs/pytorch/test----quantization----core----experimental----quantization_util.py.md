# `.\pytorch\test\quantization\core\experimental\quantization_util.py`

```py
# 导入PyTorch和相关模块
import torch
import torchvision
import torchvision.transforms.transforms as transforms
import os
import torch.ao.quantization  # 导入PyTorch量化相关模块
from torchvision.models.quantization.resnet import resnet18  # 导入ResNet18模型
from torch.autograd import Variable

# 设置警告过滤器
import warnings
warnings.filterwarnings(
    action='ignore',  # 忽略特定类型的警告
    category=DeprecationWarning,  # 警告类型为DeprecationWarning
    module=r'.*'  # 所有模块中的该类型警告
)
warnings.filterwarnings(
    action='default',  # 恢复默认警告处理方式
    module=r'torch.ao.quantization'  # 对于特定模块的警告
)

"""
Define helper functions for APoT PTQ and QAT
"""

# 指定随机种子以便重复结果
_ = torch.manual_seed(191009)

train_batch_size = 30  # 训练批量大小
eval_batch_size = 50  # 评估批量大小

class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self, name, fmt=':f'):
        self.name = name  # 指定名称
        self.fmt = fmt  # 指定格式
        self.reset()  # 重置计数器

    def reset(self):
        self.val = 0  # 当前值
        self.avg = 0.0  # 平均值
        self.sum = 0  # 总和
        self.count = 0  # 计数器

    def update(self, val, n=1):
        self.val = val  # 更新当前值
        self.sum += val * n  # 累加总和
        self.count += n  # 增加计数
        self.avg = self.sum / self.count  # 计算平均值

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """计算指定topk预测下的准确率"""
    with torch.no_grad():
        maxk = max(topk)  # 获取最大的k值
        batch_size = target.size(0)  # 获取批量大小

        _, pred = output.topk(maxk, 1, True, True)  # 获取前maxk个预测结果
        pred = pred.t()  # 转置预测结果
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # 检查预测是否正确

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # 计算前k个的正确预测数
            res.append(correct_k.mul_(100.0 / batch_size))  # 计算并添加到结果列表中
        return res  # 返回结果列表


def evaluate(model, criterion, data_loader):
    model.eval()  # 将模型设置为评估模式
    top1 = AverageMeter('Acc@1', ':6.2f')  # 计算Top-1准确率
    top5 = AverageMeter('Acc@5', ':6.2f')  # 计算Top-5准确率
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)  # 模型推断
            loss = criterion(output, target)  # 计算损失
            acc1, acc5 = accuracy(output, target, topk=(1, 5))  # 计算Top-1和Top-5准确率
            top1.update(acc1[0], image.size(0))  # 更新Top-1准确率
            top5.update(acc5[0], image.size(0))  # 更新Top-5准确率
    print('')  # 打印空行

    return top1, top5  # 返回Top-1和Top-5准确率


def load_model(model_file):
    model = resnet18(pretrained=False)  # 加载ResNet18模型
    state_dict = torch.load(model_file)  # 加载模型权重
    model.load_state_dict(state_dict)  # 加载权重到模型
    model.to("cpu")  # 将模型移动到CPU
    return model  # 返回加载的模型


def print_size_of_model(model):
    if isinstance(model, torch.jit.RecursiveScriptModule):
        torch.jit.save(model, "temp.p")  # 保存脚本模型
    else:
        torch.jit.save(torch.jit.script(model), "temp.p")  # 保存脚本模型
    print("Size (MB):", os.path.getsize("temp.p") / 1e6)  # 打印模型大小（MB）
    os.remove("temp.p")  # 删除临时文件


def prepare_data_loaders(data_path):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 归一化图像
                                     std=[0.229, 0.224, 0.225])  # 标准化图像
    # 使用 torchvision 加载 ImageNet 数据集中的训练数据集，应用随机裁剪、水平翻转、转换为张量和归一化处理
    dataset = torchvision.datasets.ImageNet(data_path,
                                            split="train",
                                            transform=transforms.Compose([transforms.RandomResizedCrop(224),
                                                                          transforms.RandomHorizontalFlip(),
                                                                          transforms.ToTensor(),
                                                                          normalize]))
    
    # 使用 torchvision 加载 ImageNet 数据集中的验证数据集，应用尺寸调整、中心裁剪、转换为张量和归一化处理
    dataset_test = torchvision.datasets.ImageNet(data_path,
                                                 split="val",
                                                 transform=transforms.Compose([transforms.Resize(256),
                                                                               transforms.CenterCrop(224),
                                                                               transforms.ToTensor(),
                                                                               normalize]))
    
    # 创建训练数据集的随机采样器
    train_sampler = torch.utils.data.RandomSampler(dataset)
    
    # 创建验证数据集的顺序采样器
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    
    # 创建训练数据加载器，指定批量大小和采样器
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler)
    
    # 创建验证数据加载器，指定批量大小和采样器
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)
    
    # 返回训练数据加载器和验证数据加载器
    return data_loader, data_loader_test
# 定义训练循环函数，接受模型、损失函数和数据加载器作为参数
def training_loop(model, criterion, data_loader):
    # 使用Adam优化器来优化模型参数，学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 初始化训练损失、正确预测数量和总样本数量为0
    train_loss, correct, total = 0, 0, 0
    # 将模型设置为训练模式
    model.train()
    # 进行10个周期的训练循环
    for i in range(10):
        # 遍历数据加载器中的数据批次
        for data, target in data_loader:
            # 梯度置零
            optimizer.zero_grad()
            # 前向传播，计算模型输出
            output = model(data)
            # 计算损失值
            loss = criterion(output, target)
            # 将损失值封装成变量，标记为需要计算梯度
            loss = Variable(loss, requires_grad=True)
            # 反向传播，计算梯度
            loss.backward()
            # 使用优化器更新模型参数
            optimizer.step()
            # 累加训练损失
            train_loss += loss.item()
            # 计算预测的类别
            _, predicted = torch.max(output, 1)
            # 累加样本总数
            total += target.size(0)
            # 计算正确预测的数量
            correct += (predicted == target).sum().item()
    # 返回训练过程中的总损失、正确预测的数量和总样本数量
    return train_loss, correct, total
```