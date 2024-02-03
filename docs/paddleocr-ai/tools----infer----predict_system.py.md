# `.\PaddleOCR\tools\infer\predict_system.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本使用此文件
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“按原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
import os
import sys
import subprocess

# 获取当前文件所在目录的绝对路径
__dir__ = os.path.dirname(os.path.abspath(__file__))
# 将当前文件所在目录添加到系统路径中
sys.path.append(__dir__)
# 将当前文件所在目录的上一级目录添加到系统路径中
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

# 设置环境变量 FLAGS_allocator_strategy 为 'auto_growth'
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

# 导入必要的库
import cv2
import copy
import numpy as np
import json
import time
import logging
from PIL import Image
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.utils.logging import get_logger
from tools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image, get_minarea_rect_crop
# 获取日志记录器
logger = get_logger()

# 定义 TextSystem 类
class TextSystem(object):
    def __init__(self, args):
        # 如果不显示日志，则将日志级别设置为 INFO
        if not args.show_log:
            logger.setLevel(logging.INFO)

        # 初始化文本检测器
        self.text_detector = predict_det.TextDetector(args)
        # 初始化文本识别器
        self.text_recognizer = predict_rec.TextRecognizer(args)
        # 是否使用角度分类器
        self.use_angle_cls = args.use_angle_cls
        # 是否丢弃得分
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            # 初始化文本角度分类器
            self.text_classifier = predict_cls.TextClassifier(args)

        self.args = args
        self.crop_image_res_index = 0
    # 定义一个方法用于在指定目录下绘制裁剪后的图像和识别结果
    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        # 如果输出目录不存在，则创建
        os.makedirs(output_dir, exist_ok=True)
        # 获取裁剪图像列表的长度
        bbox_num = len(img_crop_list)
        # 遍历裁剪图像列表
        for bno in range(bbox_num):
            # 将裁剪后的图像保存到输出目录中，文件名包含裁剪图像索引
            cv2.imwrite(
                os.path.join(output_dir,
                             f"mg_crop_{bno+self.crop_image_res_index}.jpg"),
                img_crop_list[bno])
            # 记录调试信息，包括裁剪图像索引和识别结果
            logger.debug(f"{bno}, {rec_res[bno]}")
        # 更新裁剪图像结果索引
        self.crop_image_res_index += bbox_num
# 对检测到的文本框按照从上到下，从左到右的顺序进行排序
def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    # 获取文本框的数量
    num_boxes = dt_boxes.shape[0]
    # 根据文本框的左上角坐标进行排序
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    # 对排序后的文本框进行进一步处理，确保从上到下，从左到右的顺序
    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


def main(args):
    # 获取图像文件列表
    image_file_list = get_image_file_list(args.image_dir)
    # 根据进程ID和总进程数筛选图像文件列表
    image_file_list = image_file_list[args.process_id::args.total_process_num]
    # 初始化文本系统
    text_sys = TextSystem(args)
    is_visualize = True
    font_path = args.vis_font_path
    drop_score = args.drop_score
    draw_img_save_dir = args.draw_img_save_dir
    # 创建保存结果的文件夹
    os.makedirs(draw_img_save_dir, exist_ok=True)
    save_results = []

    logger.info(
        "In PP-OCRv3, rec_image_shape parameter defaults to '3, 48, 320', "
        "if you are using recognition model with PP-OCRv2 or an older version, please set --rec_image_shape='3,32,320"
    )

    # 预热模型，随机生成图像进行预测
    if args.warmup:
        img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for i in range(10):
            res = text_sys(img)

    total_time = 0
    cpu_mem, gpu_mem, gpu_util = 0, 0, 0
    _st = time.time()
    count = 0
    logger.info("The predict total time is {}".format(time.time() - _st))
    # 如果需要进行基准测试，则记录性能数据
    if args.benchmark:
        text_sys.text_detector.autolog.report()
        text_sys.text_recognizer.autolog.report()

    # 将结果保存到文件中
    with open(
            os.path.join(draw_img_save_dir, "system_results.txt"),
            'w',
            encoding='utf-8') as f:
        f.writelines(save_results)
# 如果当前脚本被当作主程序执行
if __name__ == "__main__":
    # 解析命令行参数
    args = utility.parse_args()
    # 如果使用多进程
    if args.use_mp:
        # 创建进程列表
        p_list = []
        # 获取总进程数
        total_process_num = args.total_process_num
        # 遍历每个进程
        for process_id in range(total_process_num):
            # 构建子进程的命令
            cmd = [sys.executable, "-u"] + sys.argv + [
                "--process_id={}".format(process_id),
                "--use_mp={}".format(False)
            ]
            # 启动子进程
            p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
            # 将子进程添加到进程列表
            p_list.append(p)
        # 等待所有子进程执行完毕
        for p in p_list:
            p.wait()
    # 如果不使用多进程
    else:
        # 调用主函数
        main(args)
```