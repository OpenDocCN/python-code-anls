# `.\PaddleOCR\tools\infer_e2e.py`

```py
# 导入必要的库
import numpy as np
import os
import sys
import cv2
import json
import paddle
from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
import tools.program as program
from PIL import Image, ImageDraw, ImageFont
import math

# 定义函数，用于在中文图像上绘制端到端识别结果
def draw_e2e_res_for_chinese(image,
                             boxes,
                             txts,
                             config,
                             img_name,
                             font_path="./doc/simfang.ttf"):
    # 获取图像的高度和宽度
    h, w = image.height, image.width
    # 复制原始图像，分别用于绘制左侧和右侧的结果
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    # 设置随机种子
    import random
    random.seed(0)
    # 创建用于在左侧图像上绘制的画笔和在右侧图像上绘制的画笔
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    # 遍历框和文本的列表，同时获取索引和元素
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        # 将框转换为 NumPy 数组
        box = np.array(box)
        # 将框中的坐标转换为元组
        box = [tuple(x) for x in box]
        # 生成随机颜色
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # 在左侧图像上绘制填充的多边形
        draw_left.polygon(box, fill=color)
        # 在右侧图像上绘制轮廓的多边形
        draw_right.polygon(box, outline=color)
        # 加载指定字体文件
        font = ImageFont.truetype(font_path, 15, encoding="utf-8")
        # 在右侧图像上绘制文本
        draw_right.text([box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    # 将两个图像进行混合
    img_left = Image.blend(image, img_left, 0.5)
    # 创建新的 RGB 图像
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    # 将左侧图像粘贴到新图像的左侧
    img_show.paste(img_left, (0, 0, w, h))
    # 将右侧图像粘贴到新图像的右侧
    img_show.paste(img_right, (w, 0, w * 2, h))

    # 设置保存结果的路径
    save_e2e_path = os.path.dirname(config['Global']['save_res_path']) + "/e2e_results/"
    # 如果保存路径不存在，则创建
    if not os.path.exists(save_e2e_path):
        os.makedirs(save_e2e_path)
    # 构建保存文件的完整路径
    save_path = os.path.join(save_e2e_path, os.path.basename(img_name))
    # 将图像保存到指定路径
    cv2.imwrite(save_path, np.array(img_show)[:, :, ::-1])
    # 打印保存路径信息
    logger.info("The e2e Image saved in {}".format(save_path))
# 绘制端到端识别结果
def draw_e2e_res(dt_boxes, strs, config, img, img_name):
    # 如果检测到文本框数量大于0
    if len(dt_boxes) > 0:
        # 保存原始图像
        src_im = img
        # 遍历文本框和对应的文本字符串
        for box, str in zip(dt_boxes, strs):
            # 将文本框转换为整数类型的坐标，并重塑为(-1, 1, 2)的形状
            box = box.astype(np.int32).reshape((-1, 1, 2))
            # 绘制多边形文本框
            cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
            # 在图像上绘制文本字符串
            cv2.putText(
                src_im,
                str,
                org=(int(box[0, 0, 0]), int(box[0, 0, 1])),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.7,
                color=(0, 255, 0),
                thickness=1)
        # 设置保存检测结果的路径
        save_det_path = os.path.dirname(config['Global']['save_res_path']) + "/e2e_results/"
        # 如果路径不存在，则创建
        if not os.path.exists(save_det_path):
            os.makedirs(save_det_path)
        # 设置保存结果的完整路径
        save_path = os.path.join(save_det_path, os.path.basename(img_name))
        # 保存图像
        cv2.imwrite(save_path, src_im)
        # 打印保存路径信息
        logger.info("The e2e Image saved in {}".format(save_path))

# 主函数
def main():
    # 获取全局配置
    global_config = config['Global']

    # 构建模型
    model = build_model(config['Architecture'])

    # 加载模型
    load_model(config, model)

    # 构建后处理
    post_process_class = build_post_process(config['PostProcess'], global_config)

    # 创建数据操作
    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        # 如果操作名中包含'Label'，则跳过
        if 'Label' in op_name:
            continue
        # 如果操作名为'KeepKeys'，则设置保留的键为'image'和'shape'
        elif op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['image', 'shape']
        transforms.append(op)

    # 创建操作符
    ops = create_operators(transforms, global_config)

    # 设置保存结果的路径
    save_res_path = config['Global']['save_res_path']
    # 如果路径不存在，则创建
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))

    # 将模型设置为评估模式
    model.eval()
    # 以二进制写模式打开保存结果的文件
    with open(save_res_path, "wb") as fout:
        # 遍历获取推理图片文件列表
        for file in get_image_file_list(config['Global']['infer_img']):
            # 记录日志，输出当前处理的推理图片文件名
            logger.info("infer_img: {}".format(file))
            # 以二进制读模式打开当前图片文件
            with open(file, 'rb') as f:
                # 读取图片文件内容
                img = f.read()
                # 将图片数据封装成字典
                data = {'image': img}
            # 对数据进行转换操作
            batch = transform(data, ops)
            # 将图片数据扩展一个维度
            images = np.expand_dims(batch[0], axis=0)
            # 将形状数据扩展一个维度
            shape_list = np.expand_dims(batch[1], axis=0)
            # 将图片数据转换为张量
            images = paddle.to_tensor(images)
            # 使用模型进行推理
            preds = model(images)
            # 对推理结果进行后处理
            post_result = post_process_class(preds, shape_list)
            # 获取检测到的文本框坐标和文本内容
            points, strs = post_result['points'], post_result['texts']
            # 写入结果
            dt_boxes_json = []
            for poly, str in zip(points, strs):
                tmp_json = {"transcription": str}
                tmp_json['points'] = poly.tolist()
                dt_boxes_json.append(tmp_json)
            otstr = file + "\t" + json.dumps(dt_boxes_json) + "\n"
            fout.write(otstr.encode())
            # 读取原始图片
            src_img = cv2.imread(file)
            # 根据配置选择可视化类型为英文时的处理方式
            if global_config['infer_visual_type'] == 'EN':
                draw_e2e_res(points, strs, config, src_img, file)
            # 根据配置选择可视化类型为中文时的处理方式
            elif global_config['infer_visual_type'] == 'CN':
                src_img = Image.fromarray(
                    cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB))
                draw_e2e_res_for_chinese(
                    src_img,
                    points,
                    strs,
                    config,
                    file,
                    font_path="./doc/fonts/simfang.ttf")

    # 记录日志，输出处理成功信息
    logger.info("success!")
# 如果当前脚本作为主程序执行
if __name__ == '__main__':
    # 调用 program 模块的 preprocess 函数，获取 config, device, logger, vdl_writer 四个变量
    config, device, logger, vdl_writer = program.preprocess()
    # 调用主函数 main()
    main()
```