# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\trainer\trainer.py`

```py
# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:58
# @Author  : zhoujun
# 导入所需的模块和库
import time

import paddle
from tqdm import tqdm

# 导入自定义的模块和类
from base import BaseTrainer
from utils import runningScore, cal_text_score, Polynomial, profiler

# 定义 Trainer 类，继承自 BaseTrainer 类
class Trainer(BaseTrainer):
    # 初始化方法，接收配置、模型、损失函数、训练数据加载器、验证数据加载器、度量类、后处理函数和性能分析选项
    def __init__(self,
                 config,
                 model,
                 criterion,
                 train_loader,
                 validate_loader,
                 metric_cls,
                 post_process=None,
                 profiler_options=None):
        # 调用父类的初始化方法
        super(Trainer, self).__init__(config, model, criterion, train_loader,
                                      validate_loader, metric_cls, post_process)
        # 设置性能分析选项
        self.profiler_options = profiler_options
        # 根据配置文件设置是否启用评估
        self.enable_eval = config['trainer'].get('enable_eval', True)
    # 在指定的 epoch 上评估模型
    def _eval(self, epoch):
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
                # 如果启用混合精度训练
                if self.amp:
                    # 在 GPU 上自动转换数据类型
                    with paddle.amp.auto_cast(
                            enable='gpu' in paddle.device.get_device(),
                            custom_white_list=self.amp.get('custom_white_list',
                                                           []),
                            custom_black_list=self.amp.get('custom_black_list',
                                                           []),
                            level=self.amp.get('level', 'O2')):
                        # 获取模型预测结果
                        preds = self.model(batch['img'])
                    # 将预测结果转换为 paddle.float32 类型
                    preds = preds.astype(paddle.float32)
                else:
                    # 获取模型预测结果
                    preds = self.model(batch['img'])
                # 对预测结果进行后处理，得到边界框和分数
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
        # 计算并记录 FPS
        self.logger_info('FPS:{}'.format(total_frame / total_time))
        # 返回召回率、精确率和 F1 值的平均值
        return metrics['recall'].avg, metrics['precision'].avg, metrics[
            'fmeasure'].avg

    # 训练结束时的操作
    def _on_train_finish(self):
        # 如果启用评估
        if self.enable_eval:
            # 遍历指标字典，记录每个指标的值
            for k, v in self.metrics.items():
                self.logger_info('{}:{}'.format(k, v))
        # 记录训练结束
        self.logger_info('finish train')
    # 初始化学习率调度器
    def _initialize_scheduler(self):
        # 如果配置中的学习率调度器类型为多项式
        if self.config['lr_scheduler']['type'] == 'Polynomial':
            # 设置学习率调度器参数中的 epochs 为训练的总轮数
            self.config['lr_scheduler']['args']['epochs'] = self.config[
                'trainer']['epochs']
            # 设置学习率调度器参数中的 step_each_epoch 为每个 epoch 的步数
            self.config['lr_scheduler']['args']['step_each_epoch'] = len(
                self.train_loader)
            # 使用多项式学习率调度器初始化 lr_scheduler
            self.lr_scheduler = Polynomial(
                **self.config['lr_scheduler']['args'])()
        # 如果配置中的学习率调度器类型不是多项式
        else:
            # 使用默认的初始化函数初始化 lr_scheduler
            self.lr_scheduler = self._initialize('lr_scheduler',
                                                 paddle.optimizer.lr)
```