# `.\PaddleOCR\ppstructure\utility.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“按原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的具体语言
import random
import ast
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tools.infer.utility import draw_ocr_box_txt, str2bool, str2int_tuple, init_args as infer_args

# 初始化参数
def init_args():
    parser = infer_args()

    # 输出参数
    parser.add_argument("--output", type=str, default='./output')
    # 表结构参数
    parser.add_argument("--table_max_len", type=int, default=488)
    parser.add_argument("--table_algorithm", type=str, default='TableAttn')
    parser.add_argument("--table_model_dir", type=str)
    parser.add_argument(
        "--merge_no_span_structure", type=str2bool, default=True)
    parser.add_argument(
        "--table_char_dict_path",
        type=str,
        default="../ppocr/utils/dict/table_structure_dict_ch.txt")
    # 布局参数
    parser.add_argument("--layout_model_dir", type=str)
    parser.add_argument(
        "--layout_dict_path",
        type=str,
        default="../ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt")
    parser.add_argument(
        "--layout_score_threshold",
        type=float,
        default=0.5,
        help="Threshold of score.")
    parser.add_argument(
        "--layout_nms_threshold",
        type=float,
        default=0.5,
        help="Threshold of nms.")
    # 关键信息提取参数
    parser.add_argument("--kie_algorithm", type=str, default='LayoutXLM')
    # 添加参数 --ser_model_dir，指定序列识别模型的目录
    parser.add_argument("--ser_model_dir", type=str)
    # 添加参数 --re_model_dir，指定关系抽取模型的目录
    parser.add_argument("--re_model_dir", type=str)
    # 添加参数 --use_visual_backbone，指定是否使用视觉骨干网络，默认为True
    parser.add_argument("--use_visual_backbone", type=str2bool, default=True)
    # 添加参数 --ser_dict_path，指定序列识别的字典路径，默认为 "../train_data/XFUND/class_list_xfun.txt"
    parser.add_argument(
        "--ser_dict_path",
        type=str,
        default="../train_data/XFUND/class_list_xfun.txt")
    # 添加参数 --ocr_order_method，指定OCR的排序方法，默认为None
    # 需要为 None 或者 tb-yx
    parser.add_argument("--ocr_order_method", type=str, default=None)
    # 添加参数 --mode，指定模式为结构化或者知识抽取，默认为'structure'
    parser.add_argument(
        "--mode",
        type=str,
        choices=['structure', 'kie'],
        default='structure',
        help='structure and kie is supported')
    # 添加参数 --image_orientation，指定是否启用图像方向识别，默认为False
    parser.add_argument(
        "--image_orientation",
        type=bool,
        default=False,
        help='Whether to enable image orientation recognition')
    # 添加参数 --layout，指定是否启用布局分析，默认为True
    parser.add_argument(
        "--layout",
        type=str2bool,
        default=True,
        help='Whether to enable layout analysis')
    # 添加参数 --table，指定是否在前向过程中使用表格识别，默认为True
    parser.add_argument(
        "--table",
        type=str2bool,
        default=True,
        help='In the forward, whether the table area uses table recognition')
    # 添加参数 --ocr，指定是否在前向过程中使用OCR识别非表格区域，默认为True
    parser.add_argument(
        "--ocr",
        type=str2bool,
        default=True,
        help='In the forward, whether the non-table area is recognition by ocr')
    # 添加参数 --recovery，指定是否启用恢复布局，默认为False
    parser.add_argument(
        "--recovery",
        type=str2bool,
        default=False,
        help='Whether to enable layout of recovery')
    # 添加参数 --use_pdf2docx_api，指定是否使用pdf2docx api，默认为False
    parser.add_argument(
        "--use_pdf2docx_api",
        type=str2bool,
        default=False,
        help='Whether to use pdf2docx api')
    # 添加参数 --invert，指定是否在处理前反转图像，默认为False
    parser.add_argument(
        "--invert",
        type=str2bool,
        default=False,
        help='Whether to invert image before processing')
    # 添加参数 --binarize，指定是否在处理前对图像进行二值化，默认为False
    parser.add_argument(
        "--binarize",
        type=str2bool,
        default=False,
        help='Whether to threshold binarize image before processing')
    # 添加一个命令行参数，用于指定 alpha 通道的替换颜色，如果 alpha 通道存在的话；默认为 (255, 255, 255)
    parser.add_argument(
        "--alphacolor",
        type=str2int_tuple,
        default=(255, 255, 255),
        help='Replacement color for the alpha channel, if the latter is present; R,G,B integers')
    
    # 返回更新后的参数解析器
    return parser
# 解析命令行参数并返回解析结果
def parse_args():
    # 初始化参数解析器
    parser = init_args()
    # 解析并返回参数
    return parser.parse_args()

# 绘制结构化结果的函数
def draw_structure_result(image, result, font_path):
    # 如果输入的图像是 numpy 数组，则转换为 PIL 图像
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    # 初始化空列表用于存储框、文本和置信度
    boxes, txts, scores = []

    # 复制输入图像用于绘制布局
    img_layout = image.copy()
    draw_layout = ImageDraw.Draw(img_layout)
    text_color = (255, 255, 255)
    text_background_color = (80, 127, 255)
    catid2color = {}  # 存储类别到颜色的映射
    font_size = 15
    # 使用指定字体路径和大小创建字体对象
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    # 遍历识别结果中的每个区域
    for region in result:
        # 如果当前区域类型不在映射中，则随机生成颜色并添加到映射中
        if region['type'] not in catid2color:
            box_color = (random.randint(0, 255), random.randint(0, 255),
                         random.randint(0, 255))
            catid2color[region['type']] = box_color
        else:
            box_color = catid2color[region['type']]
        
        # 获取当前区域的边界框坐标
        box_layout = region['bbox']
        # 在布局图像上绘制边界框
        draw_layout.rectangle(
            [(box_layout[0], box_layout[1]), (box_layout[2], box_layout[3])],
            outline=box_color,
            width=3)
        
        # 获取文本的边界框大小
        left, top, right, bottom = font.getbbox(region['type'])
        text_w, text_h = right - left, bottom - top
        # 在布局图像上绘制文本背景
        draw_layout.rectangle(
            [(box_layout[0], box_layout[1]),
             (box_layout[0] + text_w, box_layout[1] + text_h)],
            fill=text_background_color)
        # 在布局图像上绘制文本
        draw_layout.text(
            (box_layout[0], box_layout[1]),
            region['type'],
            fill=text_color,
            font=font)

        # 如果区域类型不是表格，则遍历区域内的文本结果
        if region['type'] == 'table':
            pass
        else:
            for text_result in region['res']:
                # 将文本区域、文本内容和置信度添加到相应列表中
                boxes.append(np.array(text_result['text_region']))
                txts.append(text_result['text'])
                scores.append(text_result['confidence'])

    # 绘制带有文本框和文本的图像
    im_show = draw_ocr_box_txt(
        img_layout, boxes, txts, scores, font_path=font_path, drop_score=0)
    return im_show
```