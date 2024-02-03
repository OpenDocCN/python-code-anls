# `.\PaddleOCR\ppstructure\table\predict_structure.py`

```
# 导入必要的库
import os
import sys

# 获取当前文件所在目录路径
__dir__ = os.path.dirname(os.path.abspath(__file__))
# 将当前文件所在目录路径添加到系统路径中
sys.path.append(__dir__)
# 将当前文件所在目录路径的上一级目录路径添加到系统路径中
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

# 设置环境变量 FLAGS_allocator_strategy 为 'auto_growth'
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

# 导入必要的库
import cv2
import numpy as np
import time
import json

# 导入自定义模块
import tools.infer.utility as utility
from ppocr.data import create_operators, transform
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.utils.visual import draw_rectangle
from ppstructure.utility import parse_args

# 获取日志记录器
logger = get_logger()

# 构建预处理操作列表
def build_pre_process_list(args):
    # 定义图像缩放操作
    resize_op = {'ResizeTableImage': {'max_len': args.table_max_len, }}
    # 定义图像填充操作
    pad_op = {
        'PaddingTableImage': {
            'size': [args.table_max_len, args.table_max_len]
        }
    }
    # 定义图像归一化操作
    normalize_op = {
        'NormalizeImage': {
            'std': [0.229, 0.224, 0.225] if
            args.table_algorithm not in ['TableMaster'] else [0.5, 0.5, 0.5],
            'mean': [0.485, 0.456, 0.406] if
            args.table_algorithm not in ['TableMaster'] else [0.5, 0.5, 0.5],
            'scale': '1./255.',
            'order': 'hwc'
        }
    }
    # 定义图像通道转换操作
    to_chw_op = {'ToCHWImage': None}
    # 定义保留关键字操作
    keep_keys_op = {'KeepKeys': {'keep_keys': ['image', 'shape']}}
    # 如果参数中的表格算法不在指定的列表中
    if args.table_algorithm not in ['TableMaster']:
        # 如果不是TableMaster算法，按照指定顺序进行预处理操作
        pre_process_list = [
            resize_op, normalize_op, pad_op, to_chw_op, keep_keys_op
        ]
    else:
        # 如果是TableMaster算法，按照指定顺序进行预处理操作
        pre_process_list = [
            resize_op, pad_op, normalize_op, to_chw_op, keep_keys_op
        ]
    # 返回预处理操作列表
    return pre_process_list
# 定义一个名为 TableStructurer 的类，继承自 object 类
class TableStructurer(object):
    # 初始化方法，接收参数 args
    def __init__(self, args):
        # 将参数 args 赋值给实例变量 self.args
        self.args = args
        # 根据参数 args 中的 use_onnx 属性判断是否使用 ONNX
        self.use_onnx = args.use_onnx
        # 根据参数 args 构建预处理操作列表
        pre_process_list = build_pre_process_list(args)
        # 根据参数 args 中的 table_algorithm 属性判断表格算法类型
        if args.table_algorithm not in ['TableMaster']:
            # 如果不是 TableMaster 算法，则设置后处理参数
            postprocess_params = {
                'name': 'TableLabelDecode',
                "character_dict_path": args.table_char_dict_path,
                'merge_no_span_structure': args.merge_no_span_structure
            }
        else:
            # 如果是 TableMaster 算法，则设置后处理参数
            postprocess_params = {
                'name': 'TableMasterLabelDecode',
                "character_dict_path": args.table_char_dict_path,
                'box_shape': 'pad',
                'merge_no_span_structure': args.merge_no_span_structure
            }

        # 创建预处理操作
        self.preprocess_op = create_operators(pre_process_list)
        # 构建后处理操作
        self.postprocess_op = build_post_process(postprocess_params)
        # 创建预测器、输入张量、输出张量和配置信息
        self.predictor, self.input_tensor, self.output_tensors, self.config = \
            utility.create_predictor(args, 'table', logger)

        # 如果需要进行基准测试
        if args.benchmark:
            # 导入 auto_log 模块
            import auto_log
            # 获取当前进程 ID
            pid = os.getpid()
            # 获取推理 GPU ID
            gpu_id = utility.get_infer_gpuid()
            # 创建自动记录器对象
            self.autolog = auto_log.AutoLogger(
                model_name="table",
                model_precision=args.precision,
                batch_size=1,
                data_shape="dynamic",
                save_path=None,  #args.save_log_path,
                inference_config=self.config,
                pids=pid,
                process_name=None,
                gpu_ids=gpu_id if args.use_gpu else None,
                time_keys=[
                    'preprocess_time', 'inference_time', 'postprocess_time'
                ],
                warmup=0,
                logger=logger)
    # 定义一个类方法，用于对输入的图像进行处理并返回结果
    def __call__(self, img):
        # 记录方法开始时间
        starttime = time.time()
        # 如果设置了 benchmark 标志，则开始记录时间
        if self.args.benchmark:
            self.autolog.times.start()

        # 复制原始图像
        ori_im = img.copy()
        # 创建包含图像的字典
        data = {'image': img}
        # 使用预处理操作对数据进行转换
        data = transform(data, self.preprocess_op)
        # 从转换后的数据中获取图像
        img = data[0]
        # 如果图像为空，则返回空结果和时间为0
        if img is None:
            return None, 0
        # 在第0维度上扩展图像
        img = np.expand_dims(img, axis=0)
        # 复制图像
        img = img.copy()
        # 如果设置了 benchmark 标志，则记录时间
        if self.args.benchmark:
            self.autolog.times.stamp()
        # 如果使用 ONNX 模型
        if self.use_onnx:
            # 创建输入字典
            input_dict = {}
            input_dict[self.input_tensor.name] = img
            # 运行预测器并获取输出
            outputs = self.predictor.run(self.output_tensors, input_dict)
        else:
            # 将图像数据复制到输入张量
            self.input_tensor.copy_from_cpu(img)
            # 运行预测器
            self.predictor.run()
            outputs = []
            # 将输出张量复制到 CPU 并存储在列表中
            for output_tensor in self.output_tensors:
                output = output_tensor.copy_to_cpu()
                outputs.append(output)
            # 如果设置了 benchmark 标志，则记录时间
            if self.args.benchmark:
                self.autolog.times.stamp()

        # 创建包含预测结果的字典
        preds = {}
        preds['structure_probs'] = outputs[1]
        preds['loc_preds'] = outputs[0]

        # 在第0维度上扩展数据的形状列表
        shape_list = np.expand_dims(data[-1], axis=0)
        # 对预测结果进行后处理
        post_result = self.postprocess_op(preds, [shape_list])

        # 获取结构字符串列表和边界框列表
        structure_str_list = post_result['structure_batch_list'][0]
        bbox_list = post_result['bbox_batch_list'][0]
        # 对结构字符串列表进行处理
        structure_str_list = structure_str_list[0]
        structure_str_list = [
            '<html>', '<body>', '<table>'
        ] + structure_str_list + ['</table>', '</body>', '</html>']
        # 计算方法执行时间
        elapse = time.time() - starttime
        # 如果设置了 benchmark 标志，则结束记录时间
        if self.args.benchmark:
            self.autolog.times.end(stamp=True)
        # 返回结构字符串列表、边界框列表和执行时间
        return (structure_str_list, bbox_list), elapse
# 主函数，接收参数并执行主要逻辑
def main(args):
    # 获取图片文件列表
    image_file_list = get_image_file_list(args.image_dir)
    # 创建表格结构化器对象
    table_structurer = TableStructurer(args)
    # 初始化计数器和总时间
    count = 0
    total_time = 0
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    # 打开输出文件，准备写入推理结果
    with open(
            os.path.join(args.output, 'infer.txt'), mode='w',
            encoding='utf-8') as f_w:
        # 遍历图片文件列表
        for image_file in image_file_list:
            # 检查并读取图片文件
            img, flag, _ = check_and_read(image_file)
            # 如果读取失败，使用OpenCV重新读取图片
            if not flag:
                img = cv2.imread(image_file)
            # 如果图片为空，记录错误信息并继续下一张图片
            if img is None:
                logger.info("error in loading image:{}".format(image_file))
                continue
            # 对图片进行表格结构化处理，记录结果和耗时
            structure_res, elapse = table_structurer(img)
            structure_str_list, bbox_list = structure_res
            bbox_list_str = json.dumps(bbox_list.tolist())
            logger.info("result: {}, {}".format(structure_str_list,
                                                bbox_list_str))
            f_w.write("result: {}, {}\n".format(structure_str_list,
                                                bbox_list_str))

            # 如果检测到边界框信息，绘制矩形框
            if len(bbox_list) > 0 and len(bbox_list[0]) == 4:
                img = draw_rectangle(image_file, bbox_list)
            else:
                img = utility.draw_boxes(img, bbox_list)
            # 保存处理后的图片
            img_save_path = os.path.join(args.output,
                                         os.path.basename(image_file))
            cv2.imwrite(img_save_path, img)
            logger.info("save vis result to {}".format(img_save_path))
            # 累加总时间
            if count > 0:
                total_time += elapse
            count += 1
            logger.info("Predict time of {}: {}".format(image_file, elapse))
    # 如果需要进行基准测试，生成报告
    if args.benchmark:
        table_structurer.autolog.report()


if __name__ == "__main__":
    # 解析参数并执行主函数
    main(parse_args())
```