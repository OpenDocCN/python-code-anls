# `.\PaddleOCR\tools\infer_kie_token_ser_re.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权使用本文件
# 只有在遵守许可证的情况下才能使用本文件
# 可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 基于“按原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
import sys

# 获取当前文件所在目录
__dir__ = os.path.dirname(os.path.abspath(__file__))
# 将当前目录添加到系统路径中
sys.path.append(__dir__)
# 将当前目录的上一级目录添加到系统路径中
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

# 设置环境变量 FLAGS_allocator_strategy 为 'auto_growth'
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
# 导入 OpenCV 库
import cv2
# 导入 JSON 库
import json
# 导入 PaddlePaddle 库
import paddle
# 导入分布式训练模块
import paddle.distributed as dist

# 导入数据处理相关模块
from ppocr.data import create_operators, transform
# 导入模型构建相关模块
from ppocr.modeling.architectures import build_model
# 导入后处理相关模块
from ppocr.postprocess import build_post_process
# 导入模型加载保存相关模块
from ppocr.utils.save_load import load_model
# 导入可视化相关模块
from ppocr.utils.visual import draw_re_results
# 导入日志记录相关模块
from ppocr.utils.logging import get_logger
# 导入实用工具相关模块
from ppocr.utils.utility import get_image_file_list, load_vqa_bio_label_maps, print_dict
# 导入程序相关模块
from tools.program import ArgsParser, load_config, merge_config
# 导入推理相关模块
from tools.infer_kie_token_ser import SerPredictor

# 定义 ReArgsParser 类，继承自 ArgsParser 类
class ReArgsParser(ArgsParser):
    def __init__(self):
        super(ReArgsParser, self).__init__()
        # 添加参数 -c_ser，用于指定 ser 配置文件
        self.add_argument(
            "-c_ser", "--config_ser", help="ser configuration file to use")
        # 添加参数 -o_ser，用于设置 ser 配置选项
        self.add_argument(
            "-o_ser",
            "--opt_ser",
            nargs='+',
            help="set ser configuration options ")
    # 解析命令行参数，如果没有传入参数则使用默认值
    def parse_args(self, argv=None):
        # 调用父类的 parse_args 方法解析参数
        args = super(ReArgsParser, self).parse_args(argv)
        # 断言确保配置文件路径不为空
        assert args.config_ser is not None, \
            "Please specify --config_ser=ser_configure_file_path."
        # 解析参数中的 opt_ser 字段
        args.opt_ser = self._parse_opt(args.opt_ser)
        # 返回解析后的参数
        return args
# 创建输入数据，包括实体和关系信息
def make_input(ser_inputs, ser_results):
    # 定义实体类型标签
    entities_labels = {'HEADER': 0, 'QUESTION': 1, 'ANSWER': 2}
    # 获取批处理大小和最大序列长度
    batch_size, max_seq_len = ser_inputs[0].shape[:2]
    # 获取实体信息
    entities = ser_inputs[8][0]
    # 获取结果信息
    ser_results = ser_results[0]
    # 确保实体和结果数量一致
    assert len(entities) == len(ser_results)

    # 处理实体信息
    start = []
    end = []
    label = []
    entity_idx_dict = {}
    for i, (res, entity) in enumerate(zip(ser_results, entities)):
        if res['pred'] == 'O':
            continue
        entity_idx_dict[len(start)] = i
        start.append(entity['start'])
        end.append(entity['end'])
        label.append(entities_labels[res['pred']])

    # 创建实体数组
    entities = np.full([max_seq_len + 1, 3], fill_value=-1, dtype=np.int64)
    entities[0, 0] = len(start)
    entities[1:len(start) + 1, 0] = start
    entities[0, 1] = len(end)
    entities[1:len(end) + 1, 1] = end
    entities[0, 2] = len(label)
    entities[1:len(label) + 1, 2] = label

    # 处理关系信息
    head = []
    tail = []
    for i in range(len(label)):
        for j in range(len(label)):
            if label[i] == 1 and label[j] == 2:
                head.append(i)
                tail.append(j)

    # 创建关系数组
    relations = np.full([len(head) + 1, 2], fill_value=-1, dtype=np.int64)
    relations[0, 0] = len(head)
    relations[1:len(head) + 1, 0] = head
    relations[0, 1] = len(tail)
    relations[1:len(tail) + 1, 1] = tail

    # 扩展实体和关系数组维度，并重复以匹配批处理大小
    entities = np.expand_dims(entities, axis=0)
    entities = np.repeat(entities, batch_size, axis=0)
    relations = np.expand_dims(relations, axis=0)
    relations = np.repeat(relations, batch_size, axis=0)

    # 如果输入是 paddle.Tensor 类型，则转换为张量
    if isinstance(ser_inputs[0], paddle.Tensor):
        entities = paddle.to_tensor(entities)
        relations = paddle.to_tensor(relations)
    # 更新输入数据，移除 OCR 信息中的 segment_offset_id 和标签
    ser_inputs = ser_inputs[:5] + [entities, relations]

    # 为每个批次保存实体索引字典
    entity_idx_dict_batch = []
    for b in range(batch_size):
        entity_idx_dict_batch.append(entity_idx_dict)
    # 返回 ser_inputs 和 entity_idx_dict_batch 两个变量
    return ser_inputs, entity_idx_dict_batch
class SerRePredictor(object):
    # 初始化序列标注和关系抽取模型预测器
    def __init__(self, config, ser_config):
        # 获取全局配置
        global_config = config['Global']
        # 如果全局配置中包含'infer_mode'，则将其赋值给序列标注配置中的'infer_mode'
        if "infer_mode" in global_config:
            ser_config["Global"]["infer_mode"] = global_config["infer_mode"]

        # 初始化序列标注模型预测器
        self.ser_engine = SerPredictor(ser_config)

        # 初始化关系抽取模型

        # 构建后处理过程
        self.post_process_class = build_post_process(config['PostProcess'],
                                                     global_config)

        # 构建模型
        self.model = build_model(config['Architecture'])

        # 加载模型
        load_model(
            config, self.model, model_type=config['Architecture']["model_type"])

        # 设置模型为评估模式
        self.model.eval()

    # 调用函数，进行预测
    def __call__(self, data):
        # 获取序列标注结果和输入数据
        ser_results, ser_inputs = self.ser_engine(data)
        # 构建关系抽取模型的输入和实体索引字典
        re_input, entity_idx_dict_batch = make_input(ser_inputs, ser_results)
        # 如果模型的backbone不使用视觉backbone，则移除第4个元素
        if self.model.backbone.use_visual_backbone is False:
            re_input.pop(4)
        # 进行模型预测
        preds = self.model(re_input)
        # 进行后处理
        post_result = self.post_process_class(
            preds,
            ser_results=ser_results,
            entity_idx_dict_batch=entity_idx_dict_batch)
        return post_result


# 预处理函数
def preprocess():
    # 解析命令行参数
    FLAGS = ReArgsParser().parse_args()
    # 加载配置文件
    config = load_config(FLAGS.config)
    # 合并配置文件
    config = merge_config(config, FLAGS.opt)

    # 加载序列标注配置文件
    ser_config = load_config(FLAGS.config_ser)
    # 合并序列标注配置文件
    ser_config = merge_config(ser_config, FLAGS.opt_ser)

    # 获取日志记录器
    logger = get_logger()

    # 检查是否在paddlepaddle的cpu版本中设置了use_gpu=True
    use_gpu = config['Global']['use_gpu']

    # 设置设备为GPU或CPU
    device = 'gpu:{}'.format(dist.ParallelEnv().dev_id) if use_gpu else 'cpu'
    device = paddle.set_device(device)

    # 打印关系抽取配置信息
    logger.info('{} re config {}'.format('*' * 10, '*' * 10))
    print_dict(config, logger)
    logger.info('\n')
    # 打印序列标注配置信息
    logger.info('{} ser config {}'.format('*' * 10, '*' * 10))
    print_dict(ser_config, logger)
    # 使用 logger 记录训练信息，包括 PaddlePaddle 版本和设备信息
    logger.info('train with paddle {} and device {}'.format(paddle.__version__,
                                                            device))
    # 返回配置信息、序列化配置信息、设备信息和 logger
    return config, ser_config, device, logger
# 如果当前脚本作为主程序运行
if __name__ == '__main__':
    # 预处理，获取配置信息、序列化配置信息、设备信息和日志记录器
    config, ser_config, device, logger = preprocess()
    # 创建保存结果路径
    os.makedirs(config['Global']['save_res_path'], exist_ok=True)

    # 创建序列识别预测器对象
    ser_re_engine = SerRePredictor(config, ser_config)

    # 根据配置判断推理模式
    if config["Global"].get("infer_mode", None) is False:
        # 获取数据集目录和推理图片列表
        data_dir = config['Eval']['dataset']['data_dir']
        with open(config['Global']['infer_img'], "rb") as f:
            infer_imgs = f.readlines()
    else:
        # 获取推理图片列表
        infer_imgs = get_image_file_list(config['Global']['infer_img'])

    # 打开保存推理结果的文件
    with open(
            os.path.join(config['Global']['save_res_path'],
                         "infer_results.txt"),
            "w",
            encoding='utf-8') as fout:
        # 遍历推理图片列表
        for idx, info in enumerate(infer_imgs):
            # 根据推理模式处理数据
            if config["Global"].get("infer_mode", None) is False:
                data_line = info.decode('utf-8')
                substr = data_line.strip("\n").split("\t")
                img_path = os.path.join(data_dir, substr[0])
                data = {'img_path': img_path, 'label': substr[1]}
            else:
                img_path = info
                data = {'img_path': img_path}

            # 构建保存结果图片路径
            save_img_path = os.path.join(
                config['Global']['save_res_path'],
                os.path.splitext(os.path.basename(img_path))[0] + "_ser_re.jpg")

            # 进行序列识别预测
            result = ser_re_engine(data)
            result = result[0]
            # 将结果写入文件
            fout.write(img_path + "\t" + json.dumps(
                result, ensure_ascii=False) + "\n")
            # 绘制结果图片并保存
            img_res = draw_re_results(img_path, result)
            cv2.imwrite(save_img_path, img_res)

            # 记录处理进度
            logger.info("process: [{}/{}], save result to {}".format(
                idx, len(infer_imgs), save_img_path))
```