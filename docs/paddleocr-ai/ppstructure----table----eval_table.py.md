# `.\PaddleOCR\ppstructure\table\eval_table.py`

```py
# 版权声明和许可信息
# 导入必要的库
import os
import sys

# 获取当前文件所在目录的绝对路径
__dir__ = os.path.dirname(os.path.abspath(__file__))
# 将当前文件所在目录添加到系统路径中
sys.path.append(__dir__)
# 将当前文件所在目录的上级目录添加到系统路径中
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

# 导入所需的库和模块
import cv2
import pickle
import paddle
from tqdm import tqdm
from ppstructure.table.table_metric import TEDS
from ppstructure.table.predict_table import TableSystem
from ppstructure.utility import init_args
from ppocr.utils.logging import get_logger

# 获取日志记录器
logger = get_logger()

# 解析命令行参数
def parse_args():
    parser = init_args()
    parser.add_argument("--gt_path", type=str)
    return parser.parse_args()

# 加载文本文件内容到字典中
def load_txt(txt_path):
    pred_html_dict = {}
    # 如果文件不存在，则返回空字典
    if not os.path.exists(txt_path):
        return pred_html_dict
    with open(txt_path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            img_name, pred_html = line
            pred_html_dict[img_name] = pred_html
    return pred_html_dict

# 加载结果数据
def load_result(path):
    data = {}
    # 如果文件存在，则加载数据
    if os.path.exists(path):
        data = pickle.load(open(path, 'rb'))
    return data

# 保存结果数据
def save_result(path, data):
    old_data = load_result(path)
    old_data.update(data)
    with open(path, 'wb') as f:
        pickle.dump(old_data, f)

# 主函数
def main(gt_path, img_root, args):
    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)
    # 初始化 TableSystem 对象
    text_sys = TableSystem(args)
    # 加载 gt 和 preds 的 HTML 结果
    gt_html_dict = load_txt(gt_path)

    # 加载 OCR 结果
    ocr_result = load_result(os.path.join(args.output, 'ocr.pickle'))
    # 加载结构化结果
    structure_result = load_result(
        os.path.join(args.output, 'structure.pickle'))

    # 初始化预测的 HTML 结果列表和 GT 的 HTML 结果列表
    pred_htmls = []
    gt_htmls = []
    # 遍历 GT HTML 字典中的每个项目
    for img_name, gt_html in tqdm(gt_html_dict.items()):
        # 读取图像
        img = cv2.imread(os.path.join(img_root, img_name))
        # 运行 OCR 并保存结果
        if img_name not in ocr_result:
            dt_boxes, rec_res, _, _ = text_sys._ocr(img)
            ocr_result[img_name] = [dt_boxes, rec_res]
            save_result(os.path.join(args.output, 'ocr.pickle'), ocr_result)
        # 运行结构化并保存结果
        if img_name not in structure_result:
            structure_res, _ = text_sys._structure(img)
            structure_result[img_name] = structure_res
            save_result(
                os.path.join(args.output, 'structure.pickle'), structure_result)
        dt_boxes, rec_res = ocr_result[img_name]
        structure_res = structure_result[img_name]
        # 匹配 OCR 和结构化结果
        pred_html = text_sys.match(structure_res, dt_boxes, rec_res)

        pred_htmls.append(pred_html)
        gt_htmls.append(gt_html)

    # 计算 TEDS
    teds = TEDS(n_jobs=16)
    scores = teds.batch_evaluate_html(gt_htmls, pred_htmls)
    logger.info('teds: {}'.format(sum(scores) / len(scores)))
# 如果当前脚本被直接执行，则执行以下代码
if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()
    # 调用主函数，传入参数：gt_path, image_dir, args
    main(args.gt_path, args.image_dir, args)
```