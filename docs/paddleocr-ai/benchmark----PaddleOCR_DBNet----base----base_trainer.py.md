# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\base\base_trainer.py`

```py
# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:50
# @Author  : zhoujun

# 导入所需的模块
import os
import pathlib
import shutil
from pprint import pformat

import anyconfig
import paddle
import numpy as np
import random
from paddle.jit import to_static
from paddle.static import InputSpec

# 导入自定义的工具函数
from utils import setup_logger

# 定义一个基础的训练器类
class BaseTrainer:
    # 将模型转换为静态图模式
    def apply_to_static(self, model):
        # 获取配置中是否支持静态图模式
        support_to_static = self.config['trainer'].get('to_static', False)
        if support_to_static:
            specs = None
            print('static')
            # 定义输入规格
            specs = [InputSpec([None, 3, -1, -1])]
            # 将模型转换为静态图模式
            model = to_static(model, input_spec=specs)
            self.logger_info(
                "Successfully to apply @to_static with specs: {}".format(specs))
        return model

    # 完整的训练逻辑
    def train(self):
        """
        Full training logic
        """
        # 遍历每个 epoch
        for epoch in range(self.start_epoch + 1, self.epochs + 1):
            # 训练一个 epoch
            self.epoch_result = self._train_epoch(epoch)
            # 每个 epoch 结束时的操作
            self._on_epoch_finish()
        # 如果是分布式训练的主进程且启用了可视化工具，则关闭可视化工具
        if paddle.distributed.get_rank() == 0 and self.visualdl_enable:
            self.writer.close()
        # 训练结束时的操作
        self._on_train_finish()

    # 单个 epoch 的训练逻辑
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        # 抽象方法，需要在子类中实现
        raise NotImplementedError

    # 单个 epoch 的评估逻辑
    def _eval(self, epoch):
        """
        eval logic for an epoch

        :param epoch: Current epoch number
        """
        # 抽象方法，需要在子类中实现
        raise NotImplementedError

    # 每个 epoch 结束时的操作
    def _on_epoch_finish(self):
        # 抽象方法，需要在子类中实现
        raise NotImplementedError

    # 训练结束时的操作
    def _on_train_finish(self):
        # 抽象方法，需要在子类中实现
        raise NotImplementedError
    # 保存模型的检查点信息
    def _save_checkpoint(self, epoch, file_name):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth.tar'
        """
        # 获取模型的状态字典
        state_dict = self.model.state_dict()
        # 构建包含当前 epoch、全局步数、模型状态字典、优化器状态字典、配置信息和指标信息的字典
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'metrics': self.metrics
        }
        # 拼接保存路径
        filename = os.path.join(self.checkpoint_dir, file_name)
        # 保存状态字典到文件
        paddle.save(state, filename)

    # 加载检查点信息以恢复训练
    def _load_checkpoint(self, checkpoint_path, resume):
        """
        Resume from saved checkpoints
        :param checkpoint_path: Checkpoint path to be resumed
        """
        # 打印加载检查点信息的日志
        self.logger_info("Loading checkpoint: {} ...".format(checkpoint_path))
        # 加载检查点文件
        checkpoint = paddle.load(checkpoint_path)
        # 设置模型状态字典
        self.model.set_state_dict(checkpoint['state_dict'])
        # 如果需要恢复训练
        if resume:
            # 更新全局步数和起始 epoch
            self.global_step = checkpoint['global_step']
            self.start_epoch = checkpoint['epoch']
            # 更新学习率调度器的最后一个 epoch
            self.config['lr_scheduler']['args']['last_epoch'] = self.start_epoch
            # 设置优化器状态字典
            self.optimizer.set_state_dict(checkpoint['optimizer'])
            # 如果检查点中包含指标信息，则更新指标信息
            if 'metrics' in checkpoint:
                self.metrics = checkpoint['metrics']
            # 打印从检查点中恢复训练的日志信息
            self.logger_info("resume from checkpoint {} (epoch {})".format(
                checkpoint_path, self.start_epoch))
        else:
            # 打印从检查点中进行微调的日志信息
            self.logger_info("finetune from checkpoint {}".format(
                checkpoint_path))
    # 初始化函数，用于初始化模块
    def _initialize(self, name, module, *args, **kwargs):
        # 获取模块名称
        module_name = self.config[name]['type']
        # 获取模块参数
        module_args = self.config[name].get('args', {})
        # 检查是否有重复的参数
        assert all([k not in module_args for k in kwargs
                    ]), 'Overwriting kwargs given in config file is not allowed'
        # 更新参数
        module_args.update(kwargs)
        # 返回初始化后的模块对象
        return getattr(module, module_name)(*args, **module_args)

    # 初始化学习率调度器
    def _initialize_scheduler(self):
        # 使用_initialize函数初始化学习率调度器
        self.lr_scheduler = self._initialize('lr_scheduler',
                                             paddle.optimizer.lr)

    # 初始化优化器
    def _initialize_optimizer(self):
        # 使用_initialize函数初始化优化器
        self.optimizer = self._initialize(
            'optimizer',
            paddle.optimizer,
            parameters=self.model.parameters(),
            learning_rate=self.lr_scheduler)

    # 反向归一化函数
    def inverse_normalize(self, batch_img):
        # 如果需要反向归一化
        if self.UN_Normalize:
            # 对每个通道进行反向归一化
            batch_img[:, 0, :, :] = batch_img[:, 0, :, :] * self.normalize_std[0] + self.normalize_mean[0]
            batch_img[:, 1, :, :] = batch_img[:, 1, :, :] * self.normalize_std[1] + self.normalize_mean[1]
            batch_img[:, 2, :, :] = batch_img[:, 2, :, :] * self.normalize_std[2] + self.normalize_mean[2]

    # 日志信息函数
    def logger_info(self, s):
        # 如果是主进程
        if paddle.distributed.get_rank() == 0:
            # 记录日志信息
            self.logger.info(s)
```