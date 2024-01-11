# `yolov5-DNF\utils\torch_utils.py`

```
# 导入日志模块
import logging
# 导入数学模块
import math
# 导入操作系统模块
import os
# 导入时间模块
import time
# 从复制模块中导入深拷贝函数
from copy import deepcopy
# 导入 PyTorch 模块
import torch
# 从 PyTorch 的 cudnn 模块中导入相关函数
import torch.backends.cudnn as cudnn
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 导入 PyTorch 的函数模块
import torch.nn.functional as F
# 导入 PyTorch 的视觉模块
import torchvision

# 获取日志记录器
logger = logging.getLogger(__name__)

# 初始化 PyTorch 的随机种子
def init_torch_seeds(seed=0):
    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)

    # 根据不同的种子值设置 cudnn 的参数
    if seed == 0:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

# 选择设备
def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    # 判断是否请求使用 CPU
    cpu_request = device.lower() == 'cpu'
    # 如果请求的设备不是 CPU
    if device and not cpu_request:  # if device requested other than 'cpu'
        # 设置 CUDA 可见设备
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        # 检查 CUDA 是否可用
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    # 判断是否使用 CUDA
    cuda = False if cpu_request else torch.cuda.is_available()
    # 如果使用 CUDA
    if cuda:
        c = 1024 ** 2  # bytes to MB
        # 获取 CUDA 设备数量
        ng = torch.cuda.device_count()
        # 检查 batch_size 是否与设备数量兼容
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        # 获取每个 CUDA 设备的属性
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA '
        # 打印每个 CUDA 设备的信息
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            logger.info("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                        (s, i, x[i].name, x[i].total_memory / c))
    else:
        # 如果不使用 CUDA，则打印使用 CPU
        logger.info('Using CPU')

    # 打印空行
    logger.info('')  # skip a line
    # 返回所选设备
    return torch.device('cuda:0' if cuda else 'cpu')

# 同步时间
def time_synchronized():
    # 如果 CUDA 可用，则同步 CUDA
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    # 返回当前时间
    return time.time()

# 判断模型是否并行
def is_parallel(model):
    # 检查 model 的类型是否为 nn.parallel.DataParallel 或 nn.parallel.DistributedDataParallel，返回布尔值
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
# 对两个字典进行交集操作，匹配键和形状，省略 'exclude' 键，使用 da 的值
def intersect_dicts(da, db, exclude=()):
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


# 初始化模型权重
def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


# 查找模型中匹配模块类 'mclass' 的层索引
def find_modules(model, mclass=nn.Conv2d):
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


# 返回全局模型稀疏度
def sparsity(model):
    a, b = 0., 0.
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


# 将模型修剪到请求的全局稀疏度
def prune(model, amount=0.3):
    import torch.nn.utils.prune as prune
    print('Pruning model... ', end='')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name='weight', amount=amount)  # prune
            prune.remove(m, 'weight')  # make permanent
    print(' %.3g global sparsity' % sparsity(model))


# 融合卷积和批量归一化层 https://tehnokv.com/posts/fusing-batchnorm-and-conv/
def fuse_conv_and_bn(conv, bn):
    # 初始化
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # 准备滤波器
    # 将卷积层的权重进行克隆，并且重新调整形状为 (输出通道数, -1)
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    # 根据批归一化层的权重和 running_var 计算出对角矩阵
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    # 将批归一化后的权重和卷积层的权重相乘，并且重新调整形状后赋值给融合后的卷积层的权重
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size())

    # 准备空间偏置
    # 如果卷积层没有偏置，则创建一个全零的偏置
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    # 根据批归一化层的偏置、权重和 running_mean 计算出空间偏置
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    # 将批归一化后的偏置和卷积层的偏置相乘，并且加上空间偏置后赋值给融合后的卷积层的偏置
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    # 返回融合后的卷积层
    return fusedconv
def model_info(model, verbose=False):
    # 绘制 PyTorch 模型的逐行描述
    n_p = sum(x.numel() for x in model.parameters())  # 参数数量
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # 梯度数量
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
        from thop import profile
        flops = profile(deepcopy(model), inputs=(torch.zeros(1, 3, 64, 64),), verbose=False)[0] / 1E9 * 2
        fs = ', %.1f GFLOPS' % (flops * 100)  # 640x640 FLOPS
    except:
        fs = ''

    logger.info(
        'Model Summary: %g layers, %g parameters, %g gradients%s' % (len(list(model.parameters())), n_p, n_g, fs))


def load_classifier(name='resnet101', n=2):
    # 加载预训练模型并重塑为 n 类输出
    model = torchvision.models.__dict__[name](pretrained=True)

    # ResNet 模型属性
    # input_size = [3, 224, 224]
    # input_space = 'RGB'
    # input_range = [0, 1]
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # 重塑输出为 n 类
    filters = model.fc.weight.shape[1]
    model.fc.bias = nn.Parameter(torch.zeros(n), requires_grad=True)
    model.fc.weight = nn.Parameter(torch.zeros(n, filters), requires_grad=True)
    model.fc.out_features = n
    return model


def scale_img(img, ratio=1.0, same_shape=False):  # img(16,3,256,416), r=ratio
    # 按比例缩放图像 img(bs,3,y,x)
    if ratio == 1.0:
        return img
    else:
        # 获取图像的高度和宽度
        h, w = img.shape[2:]
        # 计算新的大小
        s = (int(h * ratio), int(w * ratio))  # new size
        # 调整图像大小
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        # 如果不是相同的形状，则进行填充/裁剪图像
        if not same_shape:  # pad/crop img
            # 网格大小
            gs = 32  # (pixels) grid size
            # 计算填充/裁剪后的高度和宽度
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
        # 对图像进行填充
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean
# 复制属性从 b 到 a，可以选择只包括 [...] 和排除 [...]
def copy_attr(a, b, include=(), exclude=()):
    # 遍历 b 对象的属性字典
    for k, v in b.__dict__.items():
        # 如果 include 不为空且 k 不在 include 中，或者 k 以 '_' 开头，或者 k 在 exclude 中，则跳过
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            # 将 b 对象的属性设置到 a 对象
            setattr(a, k, v)


class ModelEMA:
    """ 模型指数移动平均值，来源 https://github.com/rwightman/pytorch-image-models
    保持模型 state_dict 中所有内容的移动平均值（参数和缓冲区）。
    这旨在允许类似 https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage 的功能。
    对于某些训练方案，平滑的权重版本对于表现良好是必要的。
    这个类在模型初始化、GPU 分配和分布式训练包装器的顺序中很敏感。
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # 创建指数移动平均值
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 指数移动平均值
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 指数移动平均值
        self.updates = updates  # 指数移动平均值更新次数
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # 衰减指数斜坡（帮助早期时期）
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # 更新指数移动平均值参数
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # 模型 state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()
    # 定义一个方法，用于更新指定模型的指定属性
    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # 调用外部函数copy_attr，将模型的指定属性值复制给EMA对象
        copy_attr(self.ema, model, include, exclude)
```