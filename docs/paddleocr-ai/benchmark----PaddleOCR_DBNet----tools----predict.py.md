# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\tools\predict.py`

```
# -*- coding: utf-8 -*-
# 定义文件编码格式为 utf-8
# @Time    : 2019/8/24 12:06
# 定义代码最后修改时间为 2019 年 8 月 24 日 12 点 6 分
# @Author  : zhoujun
# 定义作者为 zhoujun

import os
# 导入操作系统模块
import sys
# 导入系统模块
import pathlib
# 导入路径操作模块
__dir__ = pathlib.Path(os.path.abspath(__file__))
# 获取当前文件的绝对路径，并转换为路径对象
sys.path.append(str(__dir__))
# 将当前文件所在路径添加到系统路径中
sys.path.append(str(__dir__.parent.parent))
# 将当前文件的父目录的父目录添加到系统路径中

import time
# 导入时间模块
import cv2
# 导入 OpenCV 模块
import paddle
# 导入 Paddle 模块

from data_loader import get_transforms
# 从 data_loader 模块中导入 get_transforms 函数
from models import build_model
# 从 models 模块中导入 build_model 函数
from post_processing import get_post_processing
# 从 post_processing 模块中导入 get_post_processing 函数

def resize_image(img, short_size):
    # 定义一个函数，用于调整图像大小
    height, width, _ = img.shape
    # 获取图像的高度和宽度
    if height < width:
        # 如果高度小于宽度
        new_height = short_size
        # 新的高度为指定的 short_size
        new_width = new_height / height * width
        # 根据比例计算新的宽度
    else:
        # 如果宽度小于高度
        new_width = short_size
        # 新的宽度为指定的 short_size
        new_height = new_width / width * height
        # 根据比例计算新的高度
    new_height = int(round(new_height / 32) * 32)
    # 将新的高度调整为最接近的 32 的倍数
    new_width = int(round(new_width / 32) * 32)
    # 将新的宽度调整为最接近的 32 的倍数
    resized_img = cv2.resize(img, (new_width, new_height))
    # 调整图像大小
    return resized_img
    # 返回调整后的图像

class PaddleModel:
# 定义一个类 PaddleModel
    # 初始化模型类，设置模型地址、后处理阈值和 GPU ID
    def __init__(self, model_path, post_p_thre=0.7, gpu_id=None):
        '''
        初始化模型
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
        :param gpu_id: 在哪一块gpu上运行
        '''
        # 设置 GPU ID
        self.gpu_id = gpu_id

        # 如果指定了 GPU ID 并且是整数类型，并且 paddle 编译时支持 CUDA
        if self.gpu_id is not None and isinstance(
                self.gpu_id, int) and paddle.device.is_compiled_with_cuda():
            # 设置设备为指定的 GPU
            paddle.device.set_device("gpu:{}".format(self.gpu_id))
        else:
            # 否则设置设备为 CPU
            paddle.device.set_device("cpu")
        
        # 加载模型参数
        checkpoint = paddle.load(model_path)

        # 获取模型配置
        config = checkpoint['config']
        config['arch']['backbone']['pretrained'] = False
        
        # 构建模型
        self.model = build_model(config['arch'])
        
        # 获取后处理方法
        self.post_process = get_post_processing(config['post_processing'])
        self.post_process.box_thresh = post_p_thre
        
        # 获取图像模式
        self.img_mode = config['dataset']['train']['dataset']['args'][
            'img_mode']
        
        # 设置模型参数
        self.model.set_state_dict(checkpoint['state_dict'])
        self.model.eval()

        # 获取数据转换方法
        self.transform = []
        for t in config['dataset']['train']['dataset']['args']['transforms']:
            if t['type'] in ['ToTensor', 'Normalize']:
                self.transform.append(t)
        
        # 获取数据转换方法
        self.transform = get_transforms(self.transform)
    def predict(self,
                img_path: str,
                is_output_polygon=False,
                short_size: int=1024):
        '''
        对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
        :param img_path: 图像地址
        :param is_numpy:
        :return:
        '''
        # 断言传入的图像地址存在
        assert os.path.exists(img_path), 'file is not exists'
        # 使用opencv读取图像，根据图像模式选择读取方式
        img = cv2.imread(img_path, 1 if self.img_mode != 'GRAY' else 0)
        # 如果图像模式为RGB，则转换颜色通道
        if self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 获取图像的高度和宽度
        h, w = img.shape[:2]
        # 调整图像大小为指定的短边长度
        img = resize_image(img, short_size)
        # 将图像转换为模型输入的张量格式
        tensor = self.transform(img)
        tensor = tensor.unsqueeze_(0)

        # 构建批次信息
        batch = {'shape': [(h, w)]}
        # 关闭梯度计算
        with paddle.no_grad():
            start = time.time()
            # 使用模型进行预测
            preds = self.model(tensor)
            # 对预测结果进行后处理，得到边界框和得分
            box_list, score_list = self.post_process(
                batch, preds, is_output_polygon=is_output_polygon)
            box_list, score_list = box_list[0], score_list[0]
            # 如果存在边界框
            if len(box_list) > 0:
                if is_output_polygon:
                    # 对多边形边界框进行处理
                    idx = [x.sum() > 0 for x in box_list]
                    box_list = [box_list[i] for i, v in enumerate(idx) if v]
                    score_list = [score_list[i] for i, v in enumerate(idx) if v]
                else:
                    # 对矩形边界框进行处理，去掉全为0的框
                    idx = box_list.reshape(box_list.shape[0], -1).sum(
                        axis=1) > 0
                    box_list, score_list = box_list[idx], score_list[idx]
            else:
                box_list, score_list = [], []
            # 计算预测时间
            t = time.time() - start
        # 返回预测结果
        return preds[0, 0, :, :].detach().cpu().numpy(), box_list, score_list, t
# 保存部署模型，用于直接推理
def save_depoly(net, input, save_path):
    # 定义输入规格，指定输入形状和数据类型
    input_spec = [
        paddle.static.InputSpec(
            shape=[None, 3, None, None], dtype="float32")
    ]
    # 将动态图模型转换为静态图模型
    net = paddle.jit.to_static(net, input_spec=input_spec)

    # 保存静态图模型，用于推理
    paddle.jit.save(net, save_path)


# 初始化命令行参数
def init_args():
    import argparse
    parser = argparse.ArgumentParser(description='DBNet.paddle')
    # 添加命令行参数
    parser.add_argument('--model_path', default=r'model_best.pth', type=str)
    parser.add_argument(
        '--input_folder',
        default='./test/input',
        type=str,
        help='img path for predict')
    parser.add_argument(
        '--output_folder',
        default='./test/output',
        type=str,
        help='img path for output')
    parser.add_argument('--gpu', default=0, type=int, help='gpu for inference')
    parser.add_argument(
        '--thre', default=0.3, type=float, help='the thresh of post_processing')
    parser.add_argument(
        '--polygon', action='store_true', help='output polygon or box')
    parser.add_argument('--show', action='store_true', help='show result')
    parser.add_argument(
        '--save_result',
        action='store_true',
        help='save box and score to txt file')
    # 解析命令行参数
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import pathlib
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from utils.util import show_img, draw_bbox, save_result, get_image_file_list

    # 初始化命令行参数
    args = init_args()
    # 打印命令行参数
    print(args)
    # 初始化Paddle模型
    model = PaddleModel(args.model_path, post_p_thre=args.thre, gpu_id=args.gpu)
    # 获取输入图像文件夹路径
    img_folder = pathlib.Path(args.input_folder)
    # 遍历输入文件夹中的所有图片路径，并显示进度条
    for img_path in tqdm(get_image_file_list(args.input_folder)):
        # 使用模型对图片进行预测，获取预测结果、边界框、得分和处理时间
        preds, boxes_list, score_list, t = model.predict(
            img_path, is_output_polygon=args.polygon)
        # 读取图片并绘制边界框
        img = draw_bbox(cv2.imread(img_path)[:, :, ::-1], boxes_list)
        # 如果指定显示图片，则显示预测结果和带有边界框的图片
        if args.show:
            show_img(preds)
            show_img(img, title=os.path.basename(img_path))
            plt.show()
        # 创建输出文件夹（如果不存在）
        os.makedirs(args.output_folder, exist_ok=True)
        # 获取图片路径的路径对象
        img_path = pathlib.Path(img_path)
        # 构建输出结果图片的路径
        output_path = os.path.join(args.output_folder,
                                   img_path.stem + '_result.jpg')
        # 构建预测结果图片的路径
        pred_path = os.path.join(args.output_folder,
                                 img_path.stem + '_pred.jpg')
        # 保存带有边界框的图片
        cv2.imwrite(output_path, img[:, :, ::-1])
        # 保存预测结果图片
        cv2.imwrite(pred_path, preds * 255)
        # 保存结果到文本文件
        save_result(
            output_path.replace('_result.jpg', '.txt'), boxes_list, score_list,
            args.polygon)
```