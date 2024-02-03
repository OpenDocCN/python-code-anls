# `.\PaddleOCR\deploy\hubserving\ocr_cls\module.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本使用此文件；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是基于“按原样”提供的，没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制。
#
# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.insert(0, ".")
import copy
import paddlehub
from paddlehub.common.logger import logger
from paddlehub.module.module import moduleinfo, runnable, serving
import cv2
import paddlehub as hub

# 导入自定义工具函数
from tools.infer.utility import base64_to_cv2
from tools.infer.predict_cls import TextClassifier
from tools.infer.utility import parse_args
from deploy.hubserving.ocr_cls.params import read_params

# 定义 OCR 文本角度分类模块
@moduleinfo(
    name="ocr_cls",
    version="1.0.0",
    summary="ocr angle cls service",
    author="paddle-dev",
    author_email="paddle-dev@baidu.com",
    type="cv/text_angle_cls")
class OCRCls(hub.Module):
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

        # 初始化文本分类器
        self.text_classifier = TextClassifier(cfg)

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
    # 定义一个预测方法，用于获取预测图像中的文本角度
    def predict(self, images=[], paths=[]):
        """
        Get the text angle in the predicted images.
        Args:
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C]. If images not paths
            paths (list[str]): The paths of images. If paths not images
        Returns:
            res (list): The result of text detection box and save path of images.
        """

        # 如果传入的是图像数据而不是路径，并且是列表形式，而路径为空
        if images != [] and isinstance(images, list) and paths == []:
            predicted_data = images
        # 如果传入的是路径而不是图像数据，并且是列表形式，而图像数据为空
        elif images == [] and isinstance(paths, list) and paths != []:
            predicted_data = self.read_images(paths)
        else:
            raise TypeError("The input data is inconsistent with expectations.")

        # 断言预测数据不为空
        assert predicted_data != [], "There is not any image to be predicted. Please check the input data."

        # 初始化图像列表
        img_list = []
        # 遍历预测数据，将非空图像添加到图像列表中
        for img in predicted_data:
            if img is None:
                continue
            img_list.append(img)

        # 初始化最终识别结果列表
        rec_res_final = []
        try:
            # 调用文本分类器，获取分类结果和预测时间
            img_list, cls_res, predict_time = self.text_classifier(img_list)
            # 遍历分类结果，将角度和置信度添加到最终识别结果列表中
            for dno in range(len(cls_res)):
                angle, score = cls_res[dno]
                rec_res_final.append({
                    'angle': angle,
                    'confidence': float(score),
                })
        except Exception as e:
            # 捕获异常并打印错误信息
            print(e)
            return [[]]

        # 返回最终识别结果列表
        return [rec_res_final]

    # 定义一个服务方法，用于作为服务运行
    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        # 将base64编码的图像解码为OpenCV格式的图像
        images_decode = [base64_to_cv2(image) for image in images]
        # 调用预测方法，获取结果
        results = self.predict(images_decode, **kwargs)
        # 返回结果
        return results
# 如果当前脚本作为主程序运行
if __name__ == '__main__':
    # 创建 OCR 类的实例
    ocr = OCRCls()
    # 初始化 OCR 实例
    ocr._initialize()
    # 定义包含多个图片路径的列表
    image_path = [
        './doc/imgs_words/ch/word_1.jpg',
        './doc/imgs_words/ch/word_2.jpg',
        './doc/imgs_words/ch/word_3.jpg',
    ]
    # 使用 OCR 实例对图片路径列表进行预测
    res = ocr.predict(paths=image_path)
    # 打印预测结果
    print(res)
```