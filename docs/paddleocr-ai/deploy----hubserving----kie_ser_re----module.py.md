# `.\PaddleOCR\deploy\hubserving\kie_ser_re\module.py`

```py
# 版权声明和许可证信息
# 该代码版权归 PaddlePaddle 作者所有，保留所有权利
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关特定语言的权限和限制

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
from ppstructure.kie.predict_kie_token_ser_re import SerRePredictor
from ppstructure.utility import parse_args

# 导入参数读取函数
from deploy.hubserving.kie_ser_re.params import read_params

# 定义模块信息
@moduleinfo(
    name="kie_ser_re",
    version="1.0.0",
    summary="kie ser re service",
    author="paddle-dev",
    author_email="paddle-dev@baidu.com",
    type="cv/KIE_SER_RE")
class KIESerRE(hub.Module):
```  
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

        # 初始化 SerRePredictor 对象
        self.ser_re_predictor = SerRePredictor(cfg)

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
        # 如果传入的是图像数据且不是空列表，并且传入的路径是空列表
        if images != [] and isinstance(images, list) and paths == []:
            # 将预测数据设置为传入的图像数据
            predicted_data = images
        # 如果传入的是空列表且路径是列表，并且路径不是空列表
        elif images == [] and isinstance(paths, list) and paths != []:
            # 通过路径读取图像数据
            predicted_data = self.read_images(paths)
        else:
            # 抛出类型错误异常
            raise TypeError("The input data is inconsistent with expectations.")

        # 断言预测数据不为空，如果为空则抛出异常
        assert predicted_data != [], "There is not any image to be predicted. Please check the input data."

        # 存储所有预测结果
        all_results = []
        # 遍历预测数据中的每个图像
        for img in predicted_data:
            # 如果图像为空
            if img is None:
                # 记录日志信息
                logger.info("error in loading image")
                # 将空列表添加到结果中
                all_results.append([])
                continue
            # 打印图像的形状
            print(img.shape)
            # 记录开始时间
            starttime = time.time()
            # 进行单个图像的预测
            re_res, _ = self.ser_re_predictor(img)
            # 打印预测结果
            print(re_res)
            # 计算预测时间
            elapse = time.time() - starttime
            # 记录预测时间
            logger.info("Predict time: {}".format(elapse))
            # 将预测结果添加到结果列表中
            all_results.append(re_res)
        # 返回所有预测结果
        return all_results

    # 将方法标记为服务方法
    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        # 将base64编码的图像数据解码为cv2图像
        images_decode = [base64_to_cv2(image) for image in images]
        # 对解码后的图像进行预测
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