# `Bert-VITS2\utils.py`

```

# 导入所需的库
import os  # 用于操作文件和目录
import glob  # 用于查找文件路径模式匹配
import argparse  # 用于解析命令行参数
import logging  # 用于记录日志
import json  # 用于处理 JSON 数据
import shutil  # 用于高级文件操作
import subprocess  # 用于创建子进程
import numpy as np  # 用于数值计算
from huggingface_hub import hf_hub_download  # 用于从 Hugging Face Hub 下载模型
from scipy.io.wavfile import read  # 用于读取 WAV 文件
import torch  # 用于构建神经网络和进行深度学习
import re  # 用于处理正则表达式

MATPLOTLIB_FLAG = False  # 设置全局变量 MATPLOTLIB_FLAG 为 False

logger = logging.getLogger(__name__)  # 创建一个记录器对象

# 下面是一系列函数的定义，包括下载模型、加载检查点、保存检查点、汇总数据、获取最新检查点路径、绘制频谱图等

# 最后一个函数是用于从模型路径中获取步数的函数

```