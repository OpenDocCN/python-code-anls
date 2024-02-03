# `.\PaddleOCR\ppstructure\predict_system.py`

```py
# 导入所需的库
import os
import sys
import subprocess

# 获取当前文件所在目录的绝对路径
__dir__ = os.path.dirname(os.path.abspath(__file__))
# 将当前文件所在目录添加到系统路径中
sys.path.append(__dir__)
# 将当前文件所在目录的上一级目录添加到系统路径中
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../')))

# 设置环境变量 FLAGS_allocator_strategy 为 'auto_growth'
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
# 导入 OpenCV 库
import cv2
# 导入 json 库
import json
# 导入 numpy 库
import numpy as np
# 导入 time 库
import time
# 导入 logging 库
import logging
# 导入 deepcopy 函数
from copy import deepcopy

# 导入自定义的函数和类
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.utils.logging import get_logger
from ppocr.utils.visual import draw_ser_results, draw_re_results
from tools.infer.predict_system import TextSystem
from ppstructure.layout.predict_layout import LayoutPredictor
from ppstructure.table.predict_table import TableSystem, to_excel
from ppstructure.utility import parse_args, draw_structure_result

# 获取日志记录器
logger = get_logger()

# 定义 StructureSystem 类
class StructureSystem(object):
    # 初始化函数，接受参数 args
    def __init__(self, args):
        # 将参数中的 mode 赋值给 self.mode
        self.mode = args.mode
        # 将参数中的 recovery 赋值给 self.recovery

        # 初始化图像方向预测器为 None
        self.image_orientation_predictor = None
        # 如果参数中包含 image_orientation
        if args.image_orientation:
            # 导入 paddleclas 模块
            import paddleclas
            # 创建 PaddleClas 对象，使用模型名称 "text_image_orientation"
            self.image_orientation_predictor = paddleclas.PaddleClas(
                model_name="text_image_orientation")

        # 如果 mode 为 'structure'
        if self.mode == 'structure':
            # 如果不展示日志
            if not args.show_log:
                # 设置日志级别为 INFO
                logger.setLevel(logging.INFO)
            # 如果 layout 为 False 且 ocr 为 True
            if args.layout == False and args.ocr == True:
                # 将 ocr 设置为 False
                args.ocr = False
                # 输出警告信息
                logger.warning(
                    "When args.layout is false, args.ocr is automatically set to false"
                )
            # 将 drop_score 设置为 0
            args.drop_score = 0
            # 初始化模型
            # 初始化 layout_predictor、text_system、table_system 为 None
            self.layout_predictor = None
            self.text_system = None
            self.table_system = None
            # 如果 layout 为 True
            if args.layout:
                # 创建 LayoutPredictor 对象
                self.layout_predictor = LayoutPredictor(args)
                # 如果 ocr 为 True
                if args.ocr:
                    # 创建 TextSystem 对象
                    self.text_system = TextSystem(args)
            # 如果 table 为 True
            if args.table:
                # 如果 text_system 不为 None
                if self.text_system is not None:
                    # 创建 TableSystem 对象，传入参数和 text_detector、text_recognizer
                    self.table_system = TableSystem(
                        args, self.text_system.text_detector,
                        self.text_system.text_recognizer)
                else:
                    # 创建 TableSystem 对象，只传入参数
                    self.table_system = TableSystem(args)

        # 如果 mode 为 'kie'
        elif self.mode == 'kie':
            # 导入 SerRePredictor 模块中的 SerRePredictor 类
            from ppstructure.kie.predict_kie_token_ser_re import SerRePredictor
            # 创建 SerRePredictor 对象，传入参数 args
            self.kie_predictor = SerRePredictor(args)
# 保存结构化结果到指定文件夹中
def save_structure_res(res, save_folder, img_name, img_idx=0):
    # 创建保存 Excel 文件的文件夹
    excel_save_folder = os.path.join(save_folder, img_name)
    os.makedirs(excel_save_folder, exist_ok=True)
    # 深拷贝结构化结果
    res_cp = deepcopy(res)
    # 保存结构化结果到文件
    with open(
            os.path.join(excel_save_folder, 'res_{}.txt'.format(img_idx)),
            'w',
            encoding='utf8') as f:
        for region in res_cp:
            # 弹出图像数据并写入文件
            roi_img = region.pop('img')
            f.write('{}\n'.format(json.dumps(region)))

            # 如果是表格类型且包含 HTML 数据，则保存为 Excel 文件
            if region['type'].lower() == 'table' and len(region[
                    'res']) > 0 and 'html' in region['res']:
                excel_path = os.path.join(
                    excel_save_folder,
                    '{}_{}.xlsx'.format(region['bbox'], img_idx))
                to_excel(region['res']['html'], excel_path)
            # 如果是图像类型，则保存为 JPG 文件
            elif region['type'].lower() == 'figure':
                img_path = os.path.join(
                    excel_save_folder,
                    '{}_{}.jpg'.format(region['bbox'], img_idx))
                cv2.imwrite(img_path, roi_img)


# 主函数
def main(args):
    # 获取图像文件列表
    image_file_list = get_image_file_list(args.image_dir)
    # 根据进程 ID 和总进程数筛选图像文件列表
    image_file_list = image_file_list[args.process_id::args.total_process_num]

    # 如果不使用 PDF2Docx API，则创建结构化系统对象并指定保存文件夹
    if not args.use_pdf2docx_api:
        structure_sys = StructureSystem(args)
        save_folder = os.path.join(args.output, structure_sys.mode)
        os.makedirs(save_folder, exist_ok=True)
    # 计算图像文件数量
    img_num = len(image_file_list)

# 解析命令行参数并执行主函数
if __name__ == "__main__":
    args = parse_args()
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
        # 等待所有子进程结束
        for p in p_list:
            p.wait()
    # 如果不使用多进程
    else:
        # 调用主函数
        main(args)
```