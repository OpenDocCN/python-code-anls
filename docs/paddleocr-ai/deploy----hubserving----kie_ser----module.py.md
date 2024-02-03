# `.\PaddleOCR\deploy\hubserving\kie_ser\module.py`

```
# 版权声明和许可证信息
# 该代码版权归 PaddlePaddle 作者所有，遵循 Apache License, Version 2.0 许可证
# 只有在遵守许可证的情况下才能使用该文件
# 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证的副本
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息

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
from tools.infer.utility import base64_to_cv2
from ppstructure.kie.predict_kie_token_ser import SerPredictor
from ppstructure.utility import parse_args
from deploy.hubserving.kie_ser.params import read_params

# 模块信息注解
@moduleinfo(
    name="kie_ser",
    version="1.0.0",
    summary="kie ser service",
    author="paddle-dev",
    author_email="paddle-dev@baidu.com",
    type="cv/KIE_SER")
class KIESer(hub.Module):
    # 初始化函数，用于初始化必要的元素
    def _initialize(self, use_gpu=False, enable_mkldnn=False):
        """
        initialize with the necessary elements
        """
        # 合并配置参数
        cfg = self.merge_configs()

        # 设置是否使用 GPU
        cfg.use_gpu = use_gpu
        if use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
                print("use gpu: ", use_gpu)
                print("CUDA_VISIBLE_DEVICES: ", _places)
                cfg.gpu_mem = 8000
            except:
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES via export CUDA_VISIBLE_DEVICES=cuda_device_id."
                )
        cfg.ir_optim = True
        cfg.enable_mkldnn = enable_mkldnn

        # 创建 SerPredictor 对象
        self.ser_predictor = SerPredictor(cfg)

    # 合并配置参数函数
    def merge_configs(self, ):
        # 备份命令行参数
        backup_argv = copy.deepcopy(sys.argv)
        sys.argv = sys.argv[:1]
        # 解析默认配置参数
        cfg = parse_args()

        # 更新配置参数
        update_cfg_map = vars(read_params())

        for key in update_cfg_map:
            cfg.__setattr__(key, update_cfg_map[key])

        sys.argv = copy.deepcopy(backup_argv)
        return cfg

    # 读取图片函数
    def read_images(self, paths=[]):
        images = []
        for img_path in paths:
            assert os.path.isfile(
                img_path), "The {} isn't a valid file.".format(img_path)
            # 读取图片
            img = cv2.imread(img_path)
            if img is None:
                logger.info("error in loading image:{}".format(img_path))
                continue
            images.append(img)
        return images
    # 定义一个预测方法，用于获取预测图像中的中文文本
    def predict(self, images=[], paths=[]):
        """
        Get the chinese texts in the predicted images.
        Args:
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C]. If images not paths
            paths (list[str]): The paths of images. If paths not images
        Returns:
            res (list): The result of chinese texts and save path of images.
        """
        # 如果传入的是图像数据而不是路径，并且路径为空，则将图像数据赋值给predicted_data
        if images != [] and isinstance(images, list) and paths == []:
            predicted_data = images
        # 如果传入的是路径而不是图像数据，并且路径不为空，则调用read_images方法读取图像数据并赋值给predicted_data
        elif images == [] and isinstance(paths, list) and paths != []:
            predicted_data = self.read_images(paths)
        else:
            # 如果传入数据不符合预期，则抛出类型错误异常
            raise TypeError("The input data is inconsistent with expectations.")

        # 断言predicted_data不为空，如果为空则抛出异常
        assert predicted_data != [], "There is not any image to be predicted. Please check the input data."

        # 存储所有预测结果的列表
        all_results = []
        # 遍历预测数据
        for img in predicted_data:
            # 如果图像为空，则记录日志并将空列表添加到结果中
            if img is None:
                logger.info("error in loading image")
                all_results.append([])
                continue
            # 记录开始时间
            starttime = time.time()
            # 调用ser_predictor方法进行预测
            ser_res, _, elapse = self.ser_predictor(img)
            # 计算预测时间
            elapse = time.time() - starttime
            logger.info("Predict time: {}".format(elapse))
            # 将预测结果添加到结果列表中
            all_results.append(ser_res)
        # 返回所有预测结果
        return all_results

    # 将方法标记为服务方法
    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        # 将base64编码的图像解码为OpenCV格式的图像
        images_decode = [base64_to_cv2(image) for image in images]
        # 调用predict方法进行预测
        results = self.predict(images_decode, **kwargs)
        # 返回预测结果
        return results
# 如果当前脚本作为主程序运行
if __name__ == '__main__':
    # 创建 OCR 系统对象
    ocr = OCRSystem()
    # 初始化 OCR 系统
    ocr._initialize()
    # 定义图片路径列表
    image_path = [
        './doc/imgs/11.jpg',
        './doc/imgs/12.jpg',
    ]
    # 使用 OCR 系统预测给定图片路径的内容
    res = ocr.predict(paths=image_path)
    # 打印预测结果
    print(res)
```