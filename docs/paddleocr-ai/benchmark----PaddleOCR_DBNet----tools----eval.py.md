# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\tools\eval.py`

```
# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 15:54
# @Author  : zhoujun
# 导入必要的模块
import os
import sys
import pathlib
# 获取当前文件的绝对路径
__dir__ = pathlib.Path(os.path.abspath(__file__))
# 将当前文件所在目录添加到系统路径中
sys.path.append(str(__dir__))
# 将当前文件的父目录的父目录添加到系统路径中
sys.path.append(str(__dir__.parent.parent))

# 导入 argparse 模块
import argparse
# 导入 time 模块
import time
# 导入 paddle 模块
import paddle
# 从 tqdm.auto 模块中导入 tqdm 函数
from tqdm.auto import tqdm

# 定义 EVAL 类
class EVAL():
    # 初始化方法
    def __init__(self, model_path, gpu_id=0):
        # 导入 build_model 函数、get_dataloader 函数、get_post_processing 函数和 get_metric 函数
        from models import build_model
        from data_loader import get_dataloader
        from post_processing import get_post_processing
        from utils import get_metric
        # 设置 GPU ID
        self.gpu_id = gpu_id
        # 如果 GPU ID 不为 None 且为整数类型且编译时使用了 CUDA，则设置设备为 GPU
        if self.gpu_id is not None and isinstance(
                self.gpu_id, int) and paddle.device.is_compiled_with_cuda():
            paddle.device.set_device("gpu:{}".format(self.gpu_id))
        else:
            # 否则设置设备为 CPU
            paddle.device.set_device("cpu")
        # 加载模型
        checkpoint = paddle.load(model_path)
        # 获取配置信息
        config = checkpoint['config']
        # 设置预训练为 False
        config['arch']['backbone']['pretrained'] = False

        # 获取验证数据加载器
        self.validate_loader = get_dataloader(config['dataset']['validate'],
                                              config['distributed'])

        # 构建模型
        self.model = build_model(config['arch'])
        # 设置模型参数
        self.model.set_state_dict(checkpoint['state_dict'])

        # 获取后处理方法
        self.post_process = get_post_processing(config['post_processing'])
        # 获取评估指标
        self.metric_cls = get_metric(config['metric'])
    # 执行模型的评估模式
    def eval(self):
        # 将模型设置为评估模式
        self.model.eval()
        # 初始化原始指标列表、总帧数和总时间
        raw_metrics = []
        total_frame = 0.0
        total_time = 0.0
        # 遍历验证数据加载器中的每个批次
        for i, batch in tqdm(
                enumerate(self.validate_loader),
                total=len(self.validate_loader),
                desc='test model'):
            # 禁用梯度计算
            with paddle.no_grad():
                # 记录开始时间
                start = time.time()
                # 使用模型进行推理
                preds = self.model(batch['img'])
                # 对推理结果进行后处理，得到边界框和分数
                boxes, scores = self.post_process(
                    batch,
                    preds,
                    is_output_polygon=self.metric_cls.is_output_polygon)
                # 更新总帧数和总时间
                total_frame += batch['img'].shape[0]
                total_time += time.time() - start
                # 计算原始指标
                raw_metric = self.metric_cls.validate_measure(batch,
                                                              (boxes, scores))
                raw_metrics.append(raw_metric)
        # 汇总原始指标
        metrics = self.metric_cls.gather_measure(raw_metrics)
        # 打印每秒帧数
        print('FPS:{}'.format(total_frame / total_time))
        # 返回包含召回率、精确率和 F 值的字典
        return {
            'recall': metrics['recall'].avg,
            'precision': metrics['precision'].avg,
            'fmeasure': metrics['fmeasure'].avg
        }
# 初始化命令行参数解析器
def init_args():
    # 创建参数解析器对象，设置描述信息
    parser = argparse.ArgumentParser(description='DBNet.paddle')
    # 添加命令行参数，指定模型路径，默认为指定路径
    parser.add_argument(
        '--model_path',
        required=False,
        default='output/DBNet_resnet18_FPN_DBHead/checkpoint/1.pth',
        type=str)
    # 解析命令行参数
    args = parser.parse_args()
    # 返回解析后的参数对象
    return args

# 主程序入口
if __name__ == '__main__':
    # 初始化命令行参数
    args = init_args()
    # 创建 EVAL 对象，传入模型路径
    eval = EVAL(args.model_path)
    # 进行评估操作，获取评估结果
    result = eval.eval()
    # 打印评估结果
    print(result)
```