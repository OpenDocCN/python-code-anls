# `.\PaddleOCR\deploy\hubserving\structure_layout\module.py`

```
# 版权声明和许可证信息
# 该代码版权归 PaddlePaddle 作者所有，仅限于遵守 Apache 许可证 2.0 使用
# 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证副本
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”基础分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

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
import paddlehub as hub

# 导入自定义工具函数
from tools.infer.utility import base64_to_cv2
from ppstructure.layout.predict_layout import LayoutPredictor as _LayoutPredictor
from ppstructure.utility import parse_args
from deploy.hubserving.structure_layout.params import read_params

# 模块信息注解
@moduleinfo(
    name="structure_layout",
    version="1.0.0",
    summary="PP-Structure layout service",
    author="paddle-dev",
    author_email="paddle-dev@baidu.com",
    type="cv/structure_layout")
class LayoutPredictor(hub.Module):
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
                # 获取环境变量 CUDA_VISIBLE_DEVICES
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                # 检查是否能转换为整数
                int(_places[0])
                print("use gpu: ", use_gpu)
                print("CUDA_VISIBLE_DEVICES: ", _places)
                cfg.gpu_mem = 8000
            except:
                # 抛出异常，提示 CUDA_VISIBLE_DEVICES 环境变量未正确设置
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES via export CUDA_VISIBLE_DEVICES=cuda_device_id."
                )
        # 设置 IR 优化和是否启用 MKLDNN
        cfg.ir_optim = True
        cfg.enable_mkldnn = enable_mkldnn

        # 初始化布局预测器
        self.layout_predictor = _LayoutPredictor(cfg)

    # 合并配置参数
    def merge_configs(self):
        # 备份命令行参数
        backup_argv = copy.deepcopy(sys.argv)
        sys.argv = sys.argv[:1]
        # 解析默认配置参数
        cfg = parse_args()

        # 读取更新的配置参数
        update_cfg_map = vars(read_params())

        # 更新配置参数
        for key in update_cfg_map:
            cfg.__setattr__(key, update_cfg_map[key])

        # 恢复命令行参数
        sys.argv = copy.deepcopy(backup_argv)
        return cfg

    # 读取图片函数
    def read_images(self, paths=[]):
        images = []
        for img_path in paths:
            # 检查图片路径是否为有效文件
            assert os.path.isfile(
                img_path), "The {} isn't a valid file.".format(img_path)
            # 读取图片
            img = cv2.imread(img_path)
            if img is None:
                logger.info("error in loading image:{}".format(img_path))
                continue
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
            res (list): The layout results of images.
        """

        # 如果传入的是图片数据且不为空，并且是一个列表，而路径为空
        if images != [] and isinstance(images, list) and paths == []:
            # 将预测数据设置为传入的图片数据
            predicted_data = images
        # 如果传入的是空列表且路径不为空，并且是一个列表
        elif images == [] and isinstance(paths, list) and paths != []:
            # 通过路径读取图片数据
            predicted_data = self.read_images(paths)
        else:
            # 抛出类型错误异常
            raise TypeError("The input data is inconsistent with expectations.")

        # 断言预测数据不为空
        assert predicted_data != [], "There is not any image to be predicted. Please check the input data."

        # 存储所有结果的列表
        all_results = []
        # 遍历预测数据中的每张图片
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
            # 进行布局预测
            res, _ = self.layout_predictor(img)
            # 计算预测时间
            elapse = time.time() - starttime
            # 记录预测时间
            logger.info("Predict time: {}".format(elapse))

            # 将结果中的边界框转换为列表形式
            for item in res:
                item['bbox'] = item['bbox'].tolist()
            # 将结果添加到所有结果中
            all_results.append({'layout': res})
        # 返回所有结果
        return all_results

    # 作为服务运行的方法
    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        # 将base64编码的图片转换为OpenCV格式的图片
        images_decode = [base64_to_cv2(image) for image in images]
        # 对解码后的图片进行预测
        results = self.predict(images_decode, **kwargs)
        # 返回预测结果
        return results
# 如果当前脚本作为主程序运行
if __name__ == '__main__':
    # 创建一个 LayoutPredictor 实例
    layout = LayoutPredictor()
    # 初始化 LayoutPredictor 实例
    layout._initialize()
    # 定义一个包含图片路径的列表
    image_path = ['./ppstructure/docs/table/1.png']
    # 使用 LayoutPredictor 实例对指定路径的图片进行预测
    res = layout.predict(paths=image_path)
    # 打印预测结果
    print(res)
```