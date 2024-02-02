# `yolov5-DNF\models\yolo.py`

```py
# 导入必要的库
import argparse  # 用于解析命令行参数
import logging  # 用于记录日志
import math  # 用于数学计算
import sys  # 用于系统相关操作
from copy import deepcopy  # 用于深拷贝
from pathlib import Path  # 用于处理文件路径

sys.path.append('./')  # 将当前目录添加到系统路径中，以便在子目录中运行 '*.py' 文件
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 中的神经网络模块

# 导入自定义模块
from models.common import Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, Concat, NMS, RGA_Module
from models.experimental import MixConv2d, CrossConv, C3
from utils.general import check_anchor_order, make_divisible, check_file, set_logging  # 导入通用工具函数
from utils.torch_utils import (
    time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, select_device)  # 导入 PyTorch 工具函数


class Detect(nn.Module):  # 定义 Detect 类，继承自 nn.Module
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # 初始化方法，接受参数 nc、anchors、ch
        super(Detect, self).__init__()  # 调用父类的初始化方法
        self.nc = nc  # 类别数
        self.no = nc + 5  # 每个锚点的输出数量
        self.nl = len(anchors)  # 检测层的数量
        self.na = len(anchors[0]) // 2  # 锚点的数量
        self.grid = [torch.zeros(1)] * self.nl  # 初始化网格
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)  # 将锚点转换为张量，并重新形状
        self.register_buffer('anchors', a)  # 注册锚点张量为模型的缓冲区
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # 注册锚点网格张量为模型的缓冲区
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # 输出卷积层
    # 定义一个前向传播函数，接受输入 x
    def forward(self, x):
        # x = x.copy()  # for profiling
        # 初始化推断输出列表
        z = []  # inference output
        # 更新训练状态
        self.training |= self.export
        # 遍历网络层
        for i in range(self.nl):
            # 对输入 x[i] 进行卷积操作
            x[i] = self.m[i](x[i])  # conv
            # 获取 x[i] 的形状信息
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            # 重新调整 x[i] 的形状
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            # 如果不是训练状态，进行推断
            if not self.training:  # inference
                # 如果网格形状与 x[i] 的形状不同，重新生成网格
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                # 对 x[i] 进行 sigmoid 操作
                y = x[i].sigmoid()
                # 更新坐标信息
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                # 更新宽高信息
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                # 将更新后的结果添加到输出列表
                z.append(y.view(bs, -1, self.no))

        # 如果是训练状态，返回 x；否则返回拼接后的 z 和 x
        return x if self.training else (torch.cat(z, 1), x)

    # 定义一个静态方法，用于生成网格
    @staticmethod
    def _make_grid(nx=20, ny=20):
        # 生成网格坐标
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        # 将坐标堆叠并转换为浮点型
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
class Model(nn.Module):
    def __init__(self, img_size, cfg='yolov5s.yaml', ch=3, nc=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        if nc and nc != self.yaml['nc']:
            print('Overriding model.yaml nc=%g with nc=%g' % (self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), img_size, ch=[ch])  # model, savelist, ch_out
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = img_size[0]  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        print('')
    # 前向传播函数，接受输入 x，是否进行数据增强 augment，是否进行性能分析 profile
    def forward(self, x, augment=False, profile=False):
        # 如果进行数据增强
        if augment:
            # 获取输入图片的尺寸
            img_size = x.shape[-2:]  # height, width
            # 定义尺度变换参数
            s = [1, 0.83, 0.67]  # scales
            # 定义翻转参数
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            # 遍历尺度变换参数和翻转参数
            for si, fi in zip(s, f):
                # 根据翻转参数和尺度参数对输入图片进行处理
                xi = scale_img(x.flip(fi) if fi else x, si)
                # 对处理后的图片进行前向传播
                yi = self.forward_once(xi)[0]  # forward
                # 对输出结果进行反缩放
                yi[..., :4] /= si  # de-scale
                # 如果进行上下翻转
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                # 如果进行左右翻转
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                # 将处理后的输出结果添加到列表中
                y.append(yi)
            # 将处理后的输出结果拼接在一起返回
            return torch.cat(y, 1), None  # augmented inference, train
        # 如果不进行数据增强
        else:
            # 调用单尺度前向传播函数
            return self.forward_once(x, profile)  # single-scale inference, train

    # 单尺度前向传播函数，接受输入 x，是否进行性能分析 profile
    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        i = 1
        # 遍历模型的每一层
        for m in self.model:
            # 如果不是来自上一层
            if m.f != -1:  # if not from previous layer
                # 如果来自于之前的层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            # 如果进行性能分析
            if profile:
                try:
                    import thop
                    # 计算模型的 FLOPS
                    o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # FLOPS
                except:
                    o = 0
                # 记录模型运行时间
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                # 打印模型的 FLOPS、参数量、运行时间和类型
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))
            # 运行模型
            x = m(x)  # run
            #print('层数：',i,'特征图大小：',x.shape)
            i+=1
            # 将输出结果保存在列表中
            y.append(x if m.i in self.save else None)  # save output

        # 如果进行性能分析
        if profile:
            # 打印总运行时间
            print('%.1fms total' % sum(dt))
        # 返回最终输出结果
        return x
    # 初始化偏置项到Detect()中，cf是类频率
    def _initialize_biases(self, cf=None):
        # 计算类频率，使用torch.bincount函数统计标签中每个类别的出现次数
        cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # 获取Detect()模块
        # 遍历Detect()模块中的m和stride
        for mi, s in zip(m.m, m.stride):
            # 将偏置项转换为(3,85)的形状
            b = mi.bias.view(m.na, -1)
            # 更新偏置项中的obj部分
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # 每张640像素的图片中有8个对象
            # 更新偏置项中的cls部分
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            # 将更新后的偏置项转换为可训练的参数
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    # 打印偏置项
    def _print_biases(self):
        m = self.model[-1]  # 获取Detect()模块
        # 遍历Detect()模块中的m
        for mi in m.m:
            # 将偏置项转换为(3,85)的形状并打印
            b = mi.bias.detach().view(m.na, -1).T
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # 融合模型中的Conv2d()和BatchNorm2d()层
    def fuse(self):
        print('Fusing layers... ')
        # 遍历模型中的每个模块
        for m in self.model.modules():
            # 如果是Conv并且有bn属性
            if type(m) is Conv and hasattr(m, 'bn'):
                m._non_persistent_buffers_set = set()  # 适配pytorch 1.6.0
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # 更新conv
                delattr(m, 'bn')  # 移除batchnorm
                m.forward = m.fuseforward  # 更新forward
        # 打印模型信息
        self.info()
        return self
    # 添加非极大值抑制（NMS）模块到模型中，用于融合 Conv2d() 和 BatchNorm2d() 层
    def add_nms(self):
        # 如果模型的最后一层不是 NMS 模块
        if type(self.model[-1]) is not NMS:
            # 打印提示信息
            print('Adding NMS module... ')
            # 创建 NMS 模块
            m = NMS()  # module
            # 设置 NMS 模块的位置为最后一层的位置加一
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            # 将 NMS 模块添加到模型中
            self.model.add_module(name='%s' % m.i, module=m)  # add
        # 返回修改后的模型
        return self
    
    # 打印模型信息
    def info(self, verbose=False):
        # 调用 model_info 函数打印模型信息
        model_info(self, verbose)
# 解析模型参数，生成模型的层和保存列表
def parse_model(d, img_size, ch):  # model_dict, input_channels(3)
    # 打印日志信息
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    # 从模型参数字典中获取锚点、类别数、深度倍数和宽度倍数
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    # 如果锚点是列表，则计算锚点数，否则直接使用锚点数
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    # 计算输出的通道数，输出通道数 = 锚点数 * (类别数 + 5)
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    # 初始化层列表、保存列表和输出通道数
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # 返回模型的序列化层和排序后的保存列表
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数--cfg，默认值为'yolov5s.yaml'，帮助信息为'model.yaml'
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    # 添加命令行参数--device，默认值为空，帮助信息为'cuda device, i.e. 0 or 0,1,2,3 or cpu'
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 解析命令行参数
    opt = parser.parse_args()
    # 检查文件是否存在
    opt.cfg = check_file(opt.cfg)  # check file
    # 设置日志记录
    set_logging()
    # 选择设备
    device = select_device(opt.device)

    # 创建模型
    model = Model(opt.cfg).to(device)
    model.train()

    # 以下为一些注释掉的代码，包括模型的性能分析、ONNX导出和Tensorboard相关操作
```