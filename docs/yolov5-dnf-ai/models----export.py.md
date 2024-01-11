# `yolov5-DNF\models\export.py`

```
"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""
#首先pip install onnx
import argparse  # 导入命令行参数解析模块
import sys  # 导入系统模块
import time  # 导入时间模块

sys.path.append('./')  # 将当前目录添加到系统路径中，以便在子目录中运行 '$ python *.py' 文件
sys.path.append('../')  # 将上级目录添加到系统路径中
import torch  # 导入PyTorch模块
import torch.nn as nn  # 导入PyTorch神经网络模块

import models  # 导入模型模块
from models.experimental import attempt_load  # 从模型实验模块中导入attempt_load函数
from utils.activations import Hardswish  # 从激活函数工具模块中导入Hardswish函数
from utils.general import set_logging, check_img_size  # 从通用工具模块中导入set_logging和check_img_size函数

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument('--weights', type=str, default='./yolov5s.pt', help='weights path')  # 添加权重路径参数
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # 添加图像大小参数
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')  # 添加批处理大小参数
    opt = parser.parse_args()  # 解析参数
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # 如果图像大小参数长度为1，则扩展为原来的两倍
    print(opt)  # 打印参数
    set_logging()  # 设置日志记录
    t = time.time()  # 记录当前时间

    # Load PyTorch model
    model = attempt_load(opt.weights, map_location=torch.device('cpu'))  # 加载FP32模型
    labels = model.names  # 获取模型标签

    # Checks
    gs = int(max(model.stride))  # 获取最大步长作为网格大小
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # 验证图像大小是否为网格大小的倍数

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size)  # 创建一个全零张量作为输入图像

    # Update model
    for k, m in model.named_modules():  # 遍历模型的所有模块
        m._non_persistent_buffers_set = set()  # 设置非持久缓冲区集合，兼容PyTorch 1.6.0
        if isinstance(m, models.common.Conv) and isinstance(m.act, nn.Hardswish):  # 如果是卷积模块且激活函数是Hardswish
            m.act = Hardswish()  # 将激活函数替换为Hardswish
        # if isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # 分配前向传播（可选）
    model.model[-1].export = True  # 设置Detect()层的export=True
    y = model(img)  # 运行模型进行预测

    # TorchScript export
    # 尝试进行 TorchScript 导出
    try:
        # 打印 Torch 版本信息
        print('\nStarting TorchScript export with torch %s...' % torch.__version__)
        # 根据权重文件名生成导出文件名
        f = opt.weights.replace('.pt', '.torchscript.pt')  # filename
        # 使用 torch.jit.trace 方法将模型转换为 TorchScript 格式
        ts = torch.jit.trace(model, img)
        # 保存 TorchScript 模型
        ts.save(f)
        # 打印导出成功信息
        print('TorchScript export success, saved as %s' % f)
    # 捕获异常
    except Exception as e:
        # 打印导出失败信息
        print('TorchScript export failure: %s' % e)

    # ONNX 导出
    try:
        # 导入 onnx 模块
        import onnx
        # 打印 ONNX 版本信息
        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        # 根据权重文件名生成导出文件名
        f = opt.weights.replace('.pt', '.onnx')  # filename
        # 使用 torch.onnx.export 方法将模型转换为 ONNX 格式
        torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
                          output_names=['classes', 'boxes'] if y is None else ['output'])
        # 检查导出的 ONNX 模型
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        # 打印可读的模型图
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        # 打印导出成功信息
        print('ONNX export success, saved as %s' % f)
    # 捕获异常
    except Exception as e:
        # 打印导出失败信息
        print('ONNX export failure: %s' % e)

    # CoreML 导出
    try:
        # 导入 coremltools 模块
        import coremltools as ct
        # 打印 CoreML 版本信息
        print('\nStarting CoreML export with coremltools %s...' % ct.__version__)
        # 将模型从 TorchScript 转换为 CoreML 格式，并根据 detect.py 中的像素缩放应用像素缩放
        model = ct.convert(ts, inputs=[ct.ImageType(name='image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
        # 根据权重文件名生成导出文件名
        f = opt.weights.replace('.pt', '.mlmodel')  # filename
        # 保存 CoreML 模型
        model.save(f)
        # 打印导出成功信息
        print('CoreML export success, saved as %s' % f)
    # 捕获异常
    except Exception as e:
        # 打印导出失败信息
        print('CoreML export failure: %s' % e)

    # 完成导出过程，打印导出所花费的时间
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))
```