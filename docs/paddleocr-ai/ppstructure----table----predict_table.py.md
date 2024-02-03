# `.\PaddleOCR\ppstructure\table\predict_table.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本使用此文件；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是基于“按原样”分发的，没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制。

# 导入必要的库
import os
import sys

# 获取当前文件所在目录
__dir__ = os.path.dirname(os.path.abspath(__file__))
# 将当前目录添加到系统路径中
sys.path.append(__dir__)
# 将上级目录添加到系统路径中
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

# 设置环境变量 FLAGS_allocator_strategy 为 'auto_growth'
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
# 导入 OpenCV 库
import cv2
import copy
import logging
import numpy as np
import time
# 导入预测文本识别的模块
import tools.infer.predict_rec as predict_rec
# 导入预测文本检测的模块
import tools.infer.predict_det as predict_det
# 导入工具模块
import tools.infer.utility as utility
# 导入排序边框的函数
from tools.infer.predict_system import sorted_boxes
# 导入获取图像文件列表和检查读取图像的函数
from ppocr.utils.utility import get_image_file_list, check_and_read
# 导入日志记录器
from ppocr.utils.logging import get_logger
# 导入表格匹配器
from ppstructure.table.matcher import TableMatch
# 导入表格主匹配器
from ppstructure.table.table_master_match import TableMasterMatcher
# 导入参数解析函数
from ppstructure.utility import parse_args
# 导入预测表格结构的模块
import ppstructure.table.predict_structure as predict_strture

# 获取日志记录器
logger = get_logger()

# 定义一个函数用于扩展边框
def expand(pix, det_box, shape):
    x0, y0, x1, y1 = det_box
    h, w, c = shape
    tmp_x0 = x0 - pix
    tmp_x1 = x1 + pix
    tmp_y0 = y0 - pix
    tmp_y1 = y1 + pix
    x0_ = tmp_x0 if tmp_x0 >= 0 else 0
    x1_ = tmp_x1 if tmp_x1 <= w else w
    y0_ = tmp_y0 if tmp_y0 >= 0 else 0
    y1_ = tmp_y1 if tmp_y1 <= h else h
    return x0_, y0_, x1_, y1_

# 定义一个表格系统类
class TableSystem(object):
    # 初始化方法，接受参数、文本检测器和文本识别器作为输入
    def __init__(self, args, text_detector=None, text_recognizer=None):
        # 将参数保存到实例变量中
        self.args = args
        # 如果不显示日志，则将日志级别设置为 INFO
        if not args.show_log:
            logger.setLevel(logging.INFO)
        # 检查是否需要进行基准测试
        benchmark_tmp = False
        if args.benchmark:
            benchmark_tmp = args.benchmark
            args.benchmark = False
        # 初始化文本检测器和文本识别器
        self.text_detector = predict_det.TextDetector(copy.deepcopy(
            args)) if text_detector is None else text_detector
        self.text_recognizer = predict_rec.TextRecognizer(copy.deepcopy(
            args)) if text_recognizer is None else text_recognizer
        # 恢复基准测试设置
        if benchmark_tmp:
            args.benchmark = True
        # 初始化表格结构化器
        self.table_structurer = predict_strture.TableStructurer(args)
        # 根据参数选择表格匹配器
        if args.table_algorithm in ['TableMaster']:
            self.match = TableMasterMatcher()
        else:
            self.match = TableMatch(filter_ocr_result=True)

        # 创建预测器、输入张量、输出张量和配置
        self.predictor, self.input_tensor, self.output_tensors, self.config = utility.create_predictor(
            args, 'table', logger)

    # 调用方法，处理图像并返回结果和时间消耗
    def __call__(self, img, return_ocr_result_in_table=False):
        # 初始化结果字典和时间消耗字典
        result = dict()
        time_dict = {'det': 0, 'rec': 0, 'table': 0, 'all': 0, 'match': 0}
        start = time.time()
        # 结构化处理图像，获取结果和耗时
        structure_res, elapse = self._structure(copy.deepcopy(img))
        result['cell_bbox'] = structure_res[1].tolist()
        time_dict['table'] = elapse

        # 进行文本检测和文本识别，获取结果和耗时
        dt_boxes, rec_res, det_elapse, rec_elapse = self._ocr(
            copy.deepcopy(img))
        time_dict['det'] = det_elapse
        time_dict['rec'] = rec_elapse

        # 如果需要返回OCR结果在表格中
        if return_ocr_result_in_table:
            result['boxes'] = [x.tolist() for x in dt_boxes]
            result['rec_res'] = rec_res

        # 进行表格匹配，获取匹配结果和耗时
        tic = time.time()
        pred_html = self.match(structure_res, dt_boxes, rec_res)
        toc = time.time()
        time_dict['match'] = toc - tic
        result['html'] = pred_html
        end = time.time()
        time_dict['all'] = end - start
        return result, time_dict
    # 对输入的图像进行结构化处理，返回结构化结果和处理时间
    def _structure(self, img):
        # 调用表格结构化函数，传入图像的深拷贝
        structure_res, elapse = self.table_structurer(copy.deepcopy(img))
        # 返回结构化结果和处理时间
        return structure_res, elapse

    # 对输入的图像进行OCR识别
    def _ocr(self, img):
        # 获取图像的高度和宽度
        h, w = img.shape[:2]
        # 使用文本检测器检测文本框，传入图像的深拷贝
        dt_boxes, det_elapse = self.text_detector(copy.deepcopy(img))
        # 对检测到的文本框进行排序
        dt_boxes = sorted_boxes(dt_boxes)

        # 对每个文本框进行处理
        r_boxes = []
        for box in dt_boxes:
            # 计算文本框的边界
            x_min = max(0, box[:, 0].min() - 1)
            x_max = min(w, box[:, 0].max() + 1)
            y_min = max(0, box[:, 1].min() - 1)
            y_max = min(h, box[:, 1].max() + 1)
            box = [x_min, y_min, x_max, y_max]
            r_boxes.append(box)
        dt_boxes = np.array(r_boxes)
        
        # 记录文本框数量和检测时间
        logger.debug("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), det_elapse))
        # 如果没有检测到文本框，则返回空
        if dt_boxes is None:
            return None, None

        # 对每个文本框进行裁剪
        img_crop_list = []
        for i in range(len(dt_boxes)):
            det_box = dt_boxes[i]
            x0, y0, x1, y1 = expand(2, det_box, img.shape)
            text_rect = img[int(y0):int(y1), int(x0):int(x1), :]
            img_crop_list.append(text_rect)
        
        # 使用文本识别器对裁剪后的图像进行识别
        rec_res, rec_elapse = self.text_recognizer(img_crop_list)
        # 记录识别结果数量和识别时间
        logger.debug("rec_res num  : {}, elapse : {}".format(
            len(rec_res), rec_elapse))
        # 返回文本框、识别结果、检测时间和识别时间
        return dt_boxes, rec_res, det_elapse, rec_elapse
# 将 HTML 表格转换为 Excel 文件
def to_excel(html_table, excel_path):
    # 导入 tablepyxl 模块
    from tablepyxl import tablepyxl
    # 调用 tablepyxl 模块中的函数将 HTML 表格转换为 Excel 文件
    tablepyxl.document_to_xl(html_table, excel_path)

# 主函数
def main(args):
    # 获取图像文件列表
    image_file_list = get_image_file_list(args.image_dir)
    # 根据进程 ID 和总进程数筛选图像文件列表
    image_file_list = image_file_list[args.process_id::args.total_process_num]
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 创建 TableSystem 对象
    table_sys = TableSystem(args)
    # 获取图像文件数量
    img_num = len(image_file_list)

    # 打开一个用于写入的 HTML 文件
    f_html = open(
        os.path.join(args.output, 'show.html'), mode='w', encoding='utf-8')
    # 写入 HTML 文件头部
    f_html.write('<html>\n<body>\n')
    # 写入 HTML 表格开始标签
    f_html.write('<table border="1">\n')
    # 写入 HTML 文件编码信息
    f_html.write(
        "<meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\" />"
    )
    # 写入 HTML 表格行开始标签
    f_html.write("<tr>\n")
    # 写入 HTML 表格列标题
    f_html.write('<td>img name\n')
    f_html.write('<td>ori image</td>')
    f_html.write('<td>table html</td>')
    f_html.write('<td>cell box</td>')
    f_html.write("</tr>\n")
    # 遍历图像文件列表，使用enumerate获取索引和图像文件
    for i, image_file in enumerate(image_file_list):
        # 记录日志信息，显示当前处理的图像文件信息
        logger.info("[{}/{}] {}".format(i, img_num, image_file))
        # 检查并读取图像文件，获取图像数据、标志和其他信息
        img, flag, _ = check_and_read(image_file)
        # 构建输出的Excel文件路径
        excel_path = os.path.join(
            args.output, os.path.basename(image_file).split('.')[0] + '.xlsx')
        # 如果标志为False，重新读取图像文件
        if not flag:
            img = cv2.imread(image_file)
        # 如果图像为空，记录错误信息并继续下一轮循环
        if img is None:
            logger.error("error in loading image:{}".format(image_file))
            continue
        # 记录开始时间
        starttime = time.time()
        # 对图像进行表格识别，获取预测结果
        pred_res, _ = table_sys(img)
        # 获取预测结果的HTML格式
        pred_html = pred_res['html']
        # 记录预测结果的HTML信息
        logger.info(pred_html)
        # 将预测结果保存为Excel文件
        to_excel(pred_html, excel_path)
        # 记录Excel文件保存路径
        logger.info('excel saved to {}'.format(excel_path))
        # 计算预测时间
        elapse = time.time() - starttime
        # 记录预测时间
        logger.info("Predict time : {:.3f}s".format(elapse))

        # 如果预测结果中存在单元格边界信息且每个单元格边界有4个点
        if len(pred_res['cell_bbox']) > 0 and len(pred_res['cell_bbox'][
                0]) == 4:
            # 在图像上绘制矩形框
            img = predict_strture.draw_rectangle(image_file,
                                                 pred_res['cell_bbox'])
        else:
            # 在图像上绘制边界框
            img = utility.draw_boxes(img, pred_res['cell_bbox'])
        # 构建保存图像的路径
        img_save_path = os.path.join(args.output, os.path.basename(image_file))
        # 保存处理后的图像
        cv2.imwrite(img_save_path, img)

        # 写入HTML文件中的表格行标签
        f_html.write("<tr>\n")
        # 写入HTML文件中的图像文件名和图像
        f_html.write(f'<td> {os.path.basename(image_file)} <br/>\n')
        f_html.write(f'<td><img src="{image_file}" width=640></td>\n')
        # 写入HTML文件中的预测结果表格
        f_html.write('<td><table  border="1">' + pred_html.replace(
            '<html><body><table>', '').replace('</table></body></html>', '') +
                     '</table></td>\n')
        f_html.write(
            f'<td><img src="{os.path.basename(image_file)}" width=640></td>\n')
        f_html.write("</tr>\n")
    # 写入HTML文件结束标签
    f_html.write("</table>\n")
    # 关闭HTML文件
    f_html.close()

    # 如果设置了benchmark参数，自动记录表格结构化的性能信息
    if args.benchmark:
        table_sys.table_structurer.autolog.report()
# 如果当前脚本被当作主程序执行
if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    # 如果使用多进程
    if args.use_mp:
        # 导入 subprocess 模块
        import subprocess
        # 创建进程列表
        p_list = []
        # 获取总进程数
        total_process_num = args.total_process_num
        # 遍历进程编号范围
        for process_id in range(total_process_num):
            # 构建子进程命令
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