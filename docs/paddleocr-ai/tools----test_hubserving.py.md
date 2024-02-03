# `.\PaddleOCR\tools\test_hubserving.py`

```py
# 导入必要的库
import os
import sys
# 获取当前文件所在目录的绝对路径
__dir__ = os.path.dirname(os.path.abspath(__file__))
# 将当前文件所在目录添加到系统路径中
sys.path.append(__dir__)
# 将当前文件所在目录的上级目录添加到系统路径中
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

# 导入日志记录器
from ppocr.utils.logging import get_logger
# 获取日志记录器
logger = get_logger()

# 导入 OpenCV 库
import cv2
# 导入 NumPy 库
import numpy as np
# 导入时间库
import time
# 导入 PIL 库中的 Image 模块
from PIL import Image
# 导入获取图片文件列表的函数
from ppocr.utils.utility import get_image_file_list
# 导入绘制 OCR 结果的函数
from tools.infer.utility import draw_ocr, draw_boxes, str2bool
# 导入绘制结构化结果的函数
from ppstructure.utility import draw_structure_result
# 导入将结果输出到 Excel 的函数
from ppstructure.predict_system import to_excel

# 导入 requests 库
import requests
# 导入 json 库
import json
# 导入 base64 编码库
import base64

# 将 OpenCV 图像转换为 base64 编码
def cv2_to_base64(image):
    return base64.b64encode(image).decode('utf8')

# 绘制服务器返回的结果
def draw_server_result(image_file, res):
    # 读取图像文件
    img = cv2.imread(image_file)
    # 将图像转换为 PIL Image 对象
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 如果结果为空，则返回原图像
    if len(res) == 0:
        return np.array(image)
    # 获取结果中的键
    keys = res[0].keys()
    # 如果结果中不包含'text_region'键，则绘制函数无效
    if 'text_region' not in keys:
        logger.info("draw function is invalid for ocr_rec!")
        return None
    # 如果结果中不包含'text'键，则为 ocr_det
    elif 'text' not in keys:
        logger.info("draw text boxes only!")
        # 提取文本框坐标
        boxes = []
        for dno in range(len(res)):
            boxes.append(res[dno]['text_region'])
        boxes = np.array(boxes)
        # 绘制文本框
        draw_img = draw_boxes(image, boxes)
        return draw_img
    else:  # 如果条件不满足，执行以下代码（用于 OCR 系统）
        logger.info("draw boxes and texts!")  # 记录信息到日志，表示正在绘制框和文本
        boxes = []  # 初始化一个空列表，用于存储框的坐标信息
        texts = []  # 初始化一个空列表，用于存储文本信息
        scores = []  # 初始化一个空列表，用于存储置信度信息
        for dno in range(len(res)):  # 遍历结果列表的长度
            boxes.append(res[dno]['text_region'])  # 将每个结果中的文本区域信息添加到框列表中
            texts.append(res[dno]['text'])  # 将每个结果中的文本信息添加到文本列表中
            scores.append(res[dno]['confidence'])  # 将每个结果中的置信度信息添加到置信度列表中
        boxes = np.array(boxes)  # 将框列表转换为 NumPy 数组
        scores = np.array(scores)  # 将置信度列表转换为 NumPy 数组
        draw_img = draw_ocr(  # 调用绘制 OCR 结果的函数
            image, boxes, texts, scores, draw_txt=True, drop_score=0.5)  # 传入图像、框、文本、置信度等参数进行绘制，设置绘制文本和置信度阈值
        return draw_img  # 返回绘制后的图像
# 保存结构化结果到指定文件夹中
def save_structure_res(res, save_folder, image_file):
    # 读取指定图片文件
    img = cv2.imread(image_file)
    # 创建保存 Excel 文件的文件夹路径
    excel_save_folder = os.path.join(save_folder, os.path.basename(image_file))
    # 如果文件夹不存在则创建
    os.makedirs(excel_save_folder, exist_ok=True)
    
    # 保存 res
    with open(
            os.path.join(excel_save_folder, 'res.txt'), 'w',
            encoding='utf8') as f:
        # 遍历结构化结果中的每个区域
        for region in res:
            # 如果区域类型为表格
            if region['type'] == 'Table':
                # 创建 Excel 文件路径
                excel_path = os.path.join(excel_save_folder,
                                          '{}.xlsx'.format(region['bbox']))
                # 将区域结果保存到 Excel 文件中
                to_excel(region['res'], excel_path)
            # 如果区域类型为图像
            elif region['type'] == 'Figure':
                # 获取图像区域的坐标
                x1, y1, x2, y2 = region['bbox']
                print(region['bbox'])
                # 截取原始图像中的 ROI 区域
                roi_img = img[y1:y2, x1:x2, :]
                # 创建图像文件路径
                img_path = os.path.join(excel_save_folder,
                                        '{}.jpg'.format(region['bbox']))
                # 将 ROI 图像保存为 JPG 文件
                cv2.imwrite(img_path, roi_img)
            # 如果区域类型为其他
            else:
                # 遍历区域中的每个文本结果，将其写入文件
                for text_result in region['res']:
                    f.write('{}\n'.format(json.dumps(text_result)))


# 主函数
def main(args):
    # 获取图片文件列表
    image_file_list = get_image_file_list(args.image_dir)
    # 是否可视化结果
    is_visualize = False
    # 请求头信息
    headers = {"Content-type": "application/json"}
    # 计数器初始化
    cnt = 0
    # 总时间初始化
    total_time = 0
    # 遍历图片文件列表
    for image_file in image_file_list:
        # 读取图片文件的二进制数据
        img = open(image_file, 'rb').read()
        # 如果图片数据为空，则记录错误信息并继续下一个图片
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        # 获取图片文件名
        img_name = os.path.basename(image_file)
        # 发送 HTTP 请求的时间起点
        starttime = time.time()
        # 构造包含图片数据的字典
        data = {'images': [cv2_to_base64(img)]}
        # 发送 POST 请求到指定的服务器地址
        r = requests.post(
            url=args.server_url, headers=headers, data=json.dumps(data))
        # 计算请求响应时间
        elapse = time.time() - starttime
        # 累加总响应时间
        total_time += elapse
        logger.info("Predict time of %s: %.3fs" % (image_file, elapse))
        # 获取响应结果
        res = r.json()["results"][0]
        logger.info(res)

        # 如果需要可视化结果
        if args.visualize:
            draw_img = None
            # 根据不同的服务器地址进行不同的处理
            if 'structure_table' in args.server_url:
                # 将结果保存为 Excel 文件
                to_excel(res['html'], './{}.xlsx'.format(img_name))
            elif 'structure_system' in args.server_url:
                # 保存结构化结果
                save_structure_res(res['regions'], args.output, image_file)
            else:
                # 绘制服务器返回的结果
                draw_img = draw_server_result(image_file, res)
            # 如果有绘制的图片，则保存
            if draw_img is not None:
                # 如果输出目录不存在，则创建
                if not os.path.exists(args.output):
                    os.makedirs(args.output)
                # 保存绘制的图片
                cv2.imwrite(
                    os.path.join(args.output, os.path.basename(image_file)),
                    draw_img[:, :, ::-1])
                logger.info("The visualized image saved in {}".format(
                    os.path.join(args.output, os.path.basename(image_file))))
        # 计数器加一
        cnt += 1
        # 每处理100张图片输出一次信息
        if cnt % 100 == 0:
            logger.info("{} processed".format(cnt))
    # 输出平均处理时间
    logger.info("avg time cost: {}".format(float(total_time) / cnt))
# 解析命令行参数
def parse_args():
    # 导入 argparse 模块，用于解析命令行参数
    import argparse
    # 创建 ArgumentParser 对象，设置描述信息
    parser = argparse.ArgumentParser(description="args for hub serving")
    # 添加命令行参数，指定服务器 URL，类型为字符串，必须提供
    parser.add_argument("--server_url", type=str, required=True)
    # 添加命令行参数，指定图片目录，类型为字符串，必须提供
    parser.add_argument("--image_dir", type=str, required=True)
    # 添加命令行参数，指定是否可视化，类型为自定义函数 str2bool，默认为 False
    parser.add_argument("--visualize", type=str2bool, default=False)
    # 添加命令行参数，指定输出路径，类型为字符串，默认为 './hubserving_result'
    parser.add_argument("--output", type=str, default='./hubserving_result')
    # 解析命令行参数并返回结果
    args = parser.parse_args()
    return args

# 如果作为脚本直接运行
if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()
    # 调用主函数，传入参数
    main(args)
```