# `so-vits-svc\diffusion\logger\saver.py`

```
'''
author: wayn391@mastertones
'''

# 导入所需的库
import datetime
import os
import time

import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

# 定义 Saver 类
class Saver(object):
    def __init__(
            self, 
            args,
            initial_global_step=-1):

        # 设置实验目录和采样率
        self.expdir = args.env.expdir
        self.sample_rate = args.data.sampling_rate
        
        # 冷启动
        self.global_step = initial_global_step
        self.init_time = time.time()
        self.last_time = time.time()

        # 创建实验目录
        os.makedirs(self.expdir, exist_ok=True)       

        # 设置日志信息文件路径
        self.path_log_info = os.path.join(self.expdir, 'log_info.txt')

        # 创建检查点目录
        os.makedirs(self.expdir, exist_ok=True)       

        # 创建 SummaryWriter 对象
        self.writer = SummaryWriter(os.path.join(self.expdir, 'logs'))
        
        # 保存配置信息到 config.yaml 文件
        path_config = os.path.join(self.expdir, 'config.yaml')
        with open(path_config, "w") as out_config:
            yaml.dump(dict(args), out_config)


    # 记录日志信息的方法
    def log_info(self, msg):
        '''log method'''
        if isinstance(msg, dict):
            msg_list = []
            for k, v in msg.items():
                tmp_str = ''
                if isinstance(v, int):
                    tmp_str = '{}: {:,}'.format(k, v)
                else:
                    tmp_str = '{}: {}'.format(k, v)

                msg_list.append(tmp_str)
            msg_str = '\n'.join(msg_list)
        else:
            msg_str = msg
        
        # 显示日志信息
        print(msg_str)

        # 保存日志信息到文件
        with open(self.path_log_info, 'a') as fp:
            fp.write(msg_str+'\n')

    # 记录数值的方法
    def log_value(self, dict):
        for k, v in dict.items():
            self.writer.add_scalar(k, v, self.global_step)
    # 记录模型的规范化输出和期望输出之间的差异，并将其拼接成一个张量
    def log_spec(self, name, spec, spec_out, vmin=-14, vmax=3.5):  
        spec_cat = torch.cat([(spec_out - spec).abs() + vmin, spec, spec_out], -1)
        # 从拼接的张量中取出第一个元素
        spec = spec_cat[0]
        # 如果 spec 是 torch.Tensor 类型，则转换成 numpy 数组
        if isinstance(spec, torch.Tensor):
            spec = spec.cpu().numpy()
        # 创建一个图形对象
        fig = plt.figure(figsize=(12, 9))
        # 绘制颜色编码的矩阵
        plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
        plt.tight_layout()
        # 将图形添加到写入器中
        self.writer.add_figure(name, fig, self.global_step)
    
    # 记录音频数据
    def log_audio(self, dict):
        # 遍历字典中的键值对
        for k, v in dict.items():
            # 将音频数据添加到写入器中
            self.writer.add_audio(k, v, global_step=self.global_step, sample_rate=self.sample_rate)
    
    # 获取时间间隔
    def get_interval_time(self, update=True):
        # 获取当前时间
        cur_time = time.time()
        # 计算时间间隔
        time_interval = cur_time - self.last_time
        # 如果需要更新上次记录的时间，则更新
        if update:
            self.last_time = cur_time
        return time_interval

    # 获取总时间
    def get_total_time(self, to_str=True):
        # 计算总时间
        total_time = time.time() - self.init_time
        # 如果需要转换成字符串格式，则进行转换
        if to_str:
            total_time = str(datetime.timedelta(
                seconds=total_time))[:-5]
        return total_time

    # 保存模型
    def save_model(
            self,
            model, 
            optimizer,
            name='model',
            postfix='',
            to_json=False):
        # 路径
        if postfix:
            postfix = '_' + postfix
        # 拼接保存模型的路径
        path_pt = os.path.join(
            self.expdir , name+postfix+'.pt')
       
        # 检查
        print(' [*] model checkpoint saved: {}'.format(path_pt))

        # 保存模型
        if optimizer is not None:
            torch.save({
                'global_step': self.global_step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, path_pt)
        else:
            torch.save({
                'global_step': self.global_step,
                'model': model.state_dict()}, path_pt)
    # 删除模型文件，可以指定模型名称和后缀
    def delete_model(self, name='model', postfix=''):
        # 如果有指定后缀，则在模型名称后添加下划线和后缀
        if postfix:
            postfix = '_' + postfix
        # 拼接模型文件路径
        path_pt = os.path.join(
            self.expdir , name+postfix+'.pt')
       
        # 如果模型文件存在，则删除
        if os.path.exists(path_pt):
            os.remove(path_pt)
            # 打印删除模型文件的信息
            print(' [*] model checkpoint deleted: {}'.format(path_pt))
        
    # 全局步数增加
    def global_step_increment(self):
        # 全局步数加一
        self.global_step += 1
```