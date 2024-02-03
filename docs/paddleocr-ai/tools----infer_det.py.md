# `.\PaddleOCR\tools\infer_det.py`

```py
# 导入必要的库
import numpy as np
import os
import sys
# 获取当前文件所在目录的绝对路径
__dir__ = os.path.dirname(os.path.abspath(__file__))
# 将当前文件所在目录添加到系统路径中
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
# 设置环境变量 FLAGS_allocator_strategy 为 'auto_growth'
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

# 导入 OpenCV、JSON、PaddlePaddle 相关库
import cv2
import json
import paddle

# 导入自定义模块
from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
import tools.program as program

# 定义函数 draw_det_res 用于在图像上绘制检测结果
def draw_det_res(dt_boxes, config, img, img_name, save_path):
    import cv2
    src_im = img
    # 遍历检测到的文本框，绘制多边形
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    # 如果保存路径不存在，则创建
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 保存绘制结果的图像
    save_path = os.path.join(save_path, os.path.basename(img_name))
    cv2.imwrite(save_path, src_im)
    logger.info("The detected Image saved in {}".format(save_path))

# 使用 paddle.no_grad() 装饰器，关闭梯度计算
@paddle.no_grad()
def main():
    global_config = config['Global']

    # 构建模型
    model = build_model(config['Architecture'])

    # 加载模型参数
    load_model(config, model)
    # 构建后处理模块
    # 根据配置文件中的 'PostProcess' 构建后处理类
    post_process_class = build_post_process(config['PostProcess'])

    # 创建数据操作列表
    transforms = []
    # 遍历配置文件中 'Eval' 下 'dataset' 中的 transforms
    for op in config['Eval']['dataset']['transforms']:
        # 获取操作名
        op_name = list(op)[0]
        # 如果操作名中包含 'Label'，则跳过
        if 'Label' in op_name:
            continue
        # 如果操作名为 'KeepKeys'，则设置 keep_keys 为 ['image', 'shape']
        elif op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['image', 'shape']
        # 将操作添加到 transforms 列表中
        transforms.append(op)

    # 根据 transforms 和全局配置创建操作符
    ops = create_operators(transforms, global_config)

    # 获取保存结果路径
    save_res_path = config['Global']['save_res_path']
    # 如果保存结果路径的父目录不存在，则创建父目录
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))

    # 将模型设置为评估模式
    model.eval()
    # 打印日志信息
    logger.info("success!")
# 如果当前脚本作为主程序执行
if __name__ == '__main__':
    # 调用 program 模块的 preprocess 函数，获取 config, device, logger, vdl_writer 四个变量
    config, device, logger, vdl_writer = program.preprocess()
    # 调用主函数 main()
    main()
```