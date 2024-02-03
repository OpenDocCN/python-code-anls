# `.\PaddleOCR\deploy\hubserving\structure_table\module.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 只有在遵守许可证的情况下才能使用此文件
# 可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证“按原样”分发
# 没有任何形式的担保或条件，无论是明示的还是暗示的
# 有关特定语言的特定权限和限制，请参阅许可证
#
# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.insert(0, ".")
import copy

import time
import paddlehub
from paddlehub.common.logger import logger
from paddlehub.module.module import moduleinfo, runnable, serving
import cv2
import numpy as np
import paddlehub as hub

# 导入自定义工具函数
from tools.infer.utility import base64_to_cv2
# 导入表格结构预测模块
from ppstructure.table.predict_table import TableSystem as _TableSystem
# 导入保存结构化结果的函数
from ppstructure.predict_system import save_structure_res
# 导入参数读取函数
from ppstructure.utility import parse_args
from deploy.hubserving.structure_table.params import read_params

# 定义模块信息
@moduleinfo(
    name="structure_table",
    version="1.0.0",
    summary="PP-Structure table service",
    author="paddle-dev",
    author_email="paddle-dev@baidu.com",
    type="cv/structure_table")
class TableSystem(hub.Module):
    # 初始化函数，用于初始化必要的元素
    def _initialize(self, use_gpu=False, enable_mkldnn=False):
        """
        initialize with the necessary elements
        """
        # 合并配置参数
        cfg = self.merge_configs()
        # 设置是否使用 GPU
        cfg.use_gpu = use_gpu
        # 如果使用 GPU
        if use_gpu:
            try:
                # 获取环境变量 CUDA_VISIBLE_DEVICES
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                # 检查是否能转换为整数
                int(_places[0])
                # 打印使用 GPU 信息
                print("use gpu: ", use_gpu)
                print("CUDA_VISIBLE_DEVICES: ", _places)
                # 设置 GPU 内存
                cfg.gpu_mem = 8000
            except:
                # 抛出运行时错误，提示设置 CUDA_VISIBLE_DEVICES 环境变量
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES via export CUDA_VISIBLE_DEVICES=cuda_device_id."
                )
        # 设置 IR 优化
        cfg.ir_optim = True
        # 设置是否启用 MKLDNN
        cfg.enable_mkldnn = enable_mkldnn

        # 初始化表格系统
        self.table_sys = _TableSystem(cfg)

    # 合并配置参数
    def merge_configs(self):
        # 备份命令行参数
        backup_argv = copy.deepcopy(sys.argv)
        sys.argv = sys.argv[:1]
        # 解析默认配置参数
        cfg = parse_args()

        # 更新配置参数映射
        update_cfg_map = vars(read_params())

        # 更新配置参数
        for key in update_cfg_map:
            cfg.__setattr__(key, update_cfg_map[key])

        # 恢复命令行参数
        sys.argv = copy.deepcopy(backup_argv)
        return cfg

    # 读取图片函数
    def read_images(self, paths=[]):
        # 存储图片列表
        images = []
        # 遍历图片路径
        for img_path in paths:
            # 检查图片文件是否存在
            assert os.path.isfile(
                img_path), "The {} isn't a valid file.".format(img_path)
            # 读取图片
            img = cv2.imread(img_path)
            # 如果图片为空
            if img is None:
                # 记录加载图片错误信息
                logger.info("error in loading image:{}".format(img_path))
                continue
            # 将图片添加到列表中
            images.append(img)
        return images
    # 定义一个方法用于预测图片中的中文文本
    def predict(self, images=[], paths=[]):
        """
        Get the chinese texts in the predicted images.
        Args:
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C]. If images not paths
            paths (list[str]): The paths of images. If paths not images
        Returns:
            res (list): The result of chinese texts and save path of images.
        """

        # 如果传入的是图片数据且不为空，并且是列表形式，而路径为空
        if images != [] and isinstance(images, list) and paths == []:
            # 将传入的图片数据赋值给predicted_data
            predicted_data = images
        # 如果传入的是空列表且路径不为空，并且是列表形式
        elif images == [] and isinstance(paths, list) and paths != []:
            # 调用read_images方法读取路径中的图片数据，赋值给predicted_data
            predicted_data = self.read_images(paths)
        else:
            # 抛出类型错误异常
            raise TypeError("The input data is inconsistent with expectations.")

        # 断言predicted_data不为空，如果为空则抛出异常
        assert predicted_data != [], "There is not any image to be predicted. Please check the input data."

        # 存储所有预测结果的列表
        all_results = []
        # 遍历预测数据
        for img in predicted_data:
            # 如果图片为空
            if img is None:
                # 记录日志信息
                logger.info("error in loading image")
                # 添加空列表到结果中
                all_results.append([])
                continue
            # 记录开始时间
            starttime = time.time()
            # 调用table_sys方法进行预测
            res, _ = self.table_sys(img)
            # 计算预测时间
            elapse = time.time() - starttime
            # 记录预测时间
            logger.info("Predict time: {}".format(elapse))

            # 将预测结果中的html内容添加到结果列表中
            all_results.append({'html': res['html']})
        # 返回所有结果
        return all_results

    # 作为服务运行的方法
    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        # 将base64编码的图片转换为cv2格式
        images_decode = [base64_to_cv2(image) for image in images]
        # 调用predict方法进行预测
        results = self.predict(images_decode, **kwargs)
        # 返回预测结果
        return results
# 如果当前脚本作为主程序运行
if __name__ == '__main__':
    # 创建一个表格系统对象
    table_system = TableSystem()
    # 初始化表格系统对象
    table_system._initialize()
    # 定义一个包含图片路径的列表
    image_path = ['./ppstructure/docs/table/table.jpg']
    # 使用表格系统对象对指定路径的图片进行预测
    res = table_system.predict(paths=image_path)
    # 打印预测结果
    print(res)
```