# `.\PaddleOCR\StyleText\engine\synthesisers.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证，除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件，包括但不限于特定用途的适用性保证
# 请查看许可证以获取有关权限和限制的详细信息
import os
import numpy as np
import cv2

# 导入自定义模块
from utils.config import ArgsParser, load_config, override_config
from utils.logging import get_logger
from engine import style_samplers, corpus_generators, text_drawers, predictors, writers

# 定义图像合成器类
class ImageSynthesiser(object):
    def __init__(self):
        # 解析命令行参数
        self.FLAGS = ArgsParser().parse_args()
        # 加载配置文件
        self.config = load_config(self.FLAGS.config)
        # 覆盖配置文件中的选项
        self.config = override_config(self.config, options=self.FLAGS.override)
        # 设置输出目录
        self.output_dir = self.config["Global"]["output_dir"]
        # 如果输出目录不存在，则创建
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        # 获取日志记录器
        self.logger = get_logger(log_file='{}/predict.log'.format(self.output_dir))

        # 初始化文本绘制器
        self.text_drawer = text_drawers.StdTextDrawer(self.config)

        # 获取预测器方法并实例化预测器
        predictor_method = self.config["Predictor"]["method"]
        assert predictor_method is not None
        self.predictor = getattr(predictors, predictor_method)(self.config)

    # 合成图像的方法
    def synth_image(self, corpus, style_input, language="en"):
        # 绘制文本并获取文本输入列表
        corpus_list, text_input_list = self.text_drawer.draw_text(
            corpus, language, style_input_width=style_input.shape[1])
        # 使用预测器预测结果
        synth_result = self.predictor.predict(style_input, text_input_list)
        return synth_result

# DatasetSynthesiser 类继承自 ImageSynthesiser 类
class DatasetSynthesiser(ImageSynthesiser):
    # 初始化 DatasetSynthesiser 类
    def __init__(self):
        # 调用父类的初始化方法
        super(DatasetSynthesiser, self).__init__()
        # 获取标签信息
        self.tag = self.FLAGS.tag
        # 获取输出数量
        self.output_num = self.config["Global"]["output_num"]
        # 获取语料生成器的方法
        corpus_generator_method = self.config["CorpusGenerator"]["method"]
        # 根据配置文件中的方法名获取对应的语料生成器对象
        self.corpus_generator = getattr(corpus_generators, corpus_generator_method)(self.config)

        # 获取样式采样器的方法
        style_sampler_method = self.config["StyleSampler"]["method"]
        # 确保样式采样器方法不为空
        assert style_sampler_method is not None
        # 创建样式采样器对象
        self.style_sampler = style_samplers.DatasetSampler(self.config)
        # 创建简单写入器对象
        self.writer = writers.SimpleWriter(self.config, self.tag)

    # 合成数据集
    def synth_dataset(self):
        # 循环生成指定数量的输出
        for i in range(self.output_num):
            # 采样样式数据
            style_data = self.style_sampler.sample()
            style_input = style_data["image"]
            # 生成语料语言和文本输入标签
            corpus_language, text_input_label = self.corpus_generator.generate()
            # 生成文本输入标签列表和文本输入列表
            text_input_label_list, text_input_list = self.text_drawer.draw_text(
                text_input_label,
                corpus_language,
                style_input_width=style_input.shape[1])

            # 将文本输入标签列表合并为字符串
            text_input_label = "".join(text_input_label_list)

            # 使用预测器预测结果
            synth_result = self.predictor.predict(style_input, text_input_list)
            fake_fusion = synth_result["fake_fusion"]
            # 保存合成图像和文本输入标签
            self.writer.save_image(fake_fusion, text_input_label)
        # 保存标签信息
        self.writer.save_label()
        # 合并标签信息
        self.writer.merge_label()
```