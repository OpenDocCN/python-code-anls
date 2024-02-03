# `.\PaddleOCR\tools\infer_kie.py`

```py
# 版权声明
# 本代码版权归 PaddlePaddle 作者所有
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的
# 没有任何明示或暗示的保证或条件，无论是明示的还是暗示的
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import paddle.nn.functional as F
import os
import sys

# 获取当前文件所在目录路径
__dir__ = os.path.dirname(os.path.abspath(__file__))
# 将当前目录添加到系统路径中
sys.path.append(__dir__)
# 将上级目录添加到系统路径中
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

# 设置环境变量 FLAGS_allocator_strategy 为 'auto_growth'
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

# 导入 OpenCV 和 PaddlePaddle 库
import cv2
import paddle

# 导入数据处理相关的模块
from ppocr.data import create_operators, transform
# 导入模型构建相关的模块
from ppocr.modeling.architectures import build_model
# 导入模型保存和加载相关的函数
from ppocr.utils.save_load import load_model
# 导入程序相关的工具函数
import tools.program as program
import time

# 从文件中读取类别列表
def read_class_list(filepath):
    ret = {}
    with open(filepath, "r") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            ret[idx] = line.strip("\n")
    return ret

# 绘制关键信息提取结果
def draw_kie_result(batch, node, idx_to_cls, count):
    # 复制输入图像
    img = batch[6].copy()
    # 获取检测框
    boxes = batch[7]
    h, w = img.shape[:2]
    # 创建预测图像
    pred_img = np.ones((h, w * 2, 3), dtype=np.uint8) * 255
    # 获取最大值和最大值索引
    max_value, max_idx = paddle.max(node, -1), paddle.argmax(node, -1)
    # 将预测结果转换为 numpy 数组
    node_pred_label = max_idx.numpy().tolist()
    node_pred_score = max_value.numpy().tolist()
    # 遍历每个框和索引
    for i, box in enumerate(boxes):
        # 如果索引超过了节点预测标签的长度，则跳出循环
        if i >= len(node_pred_label):
            break
        # 创建新的框，顺时针四个点的坐标
        new_box = [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
        # 将新框的坐标转换为 NumPy 数组
        Pts = np.array([new_box], np.int32)
        # 在图像上绘制多边形
        cv2.polylines(
            img, [Pts.reshape((-1, 1, 2))],
            True,
            color=(255, 255, 0),
            thickness=1)
        # 计算新框的最小 x 和 y 坐标
        x_min = int(min([point[0] for point in new_box]))
        y_min = int(min([point[1] for point in new_box]))

        # 获取节点预测标签和分数
        pred_label = node_pred_label[i]
        # 如果预测标签在索引到类别的映射中，则替换为对应的类别
        if pred_label in idx_to_cls:
            pred_label = idx_to_cls[pred_label]
        pred_score = '{:.2f}'.format(node_pred_score[i])
        # 构建显示的文本
        text = pred_label + '(' + pred_score + ')'
        # 在预测图像上绘制文本
        cv2.putText(pred_img, text, (x_min * 2, y_min),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # 创建一个三倍宽度的白色图像
    vis_img = np.ones((h, w * 3, 3), dtype=np.uint8) * 255
    # 将原始图像和预测图像放在一起
    vis_img[:, :w] = img
    vis_img[:, w:] = pred_img
    # 设置保存路径
    save_kie_path = os.path.dirname(config['Global']['save_res_path']) + "/kie_results/"
    # 如果保存路径不存在，则创建
    if not os.path.exists(save_kie_path):
        os.makedirs(save_kie_path)
    # 设置保存文件路径
    save_path = os.path.join(save_kie_path, str(count) + ".png")
    # 将合成图像保存到文件
    cv2.imwrite(save_path, vis_img)
    # 打印保存路径信息
    logger.info("The Kie Image saved in {}".format(save_path))
# 将推理结果写入输出文件，按每行的预测标签排序。
# 输出格式与输入格式相同，但增加了分数属性。
def write_kie_result(fout, node, data):
    # 导入json模块
    import json
    # 获取数据中的标签
    label = data['label']
    # 将标签转换为字典
    annotations = json.loads(label)
    # 获取每行中预测标签的最大值和索引
    max_value, max_idx = paddle.max(node, -1), paddle.argmax(node, -1)
    # 将预测标签转换为列表
    node_pred_label = max_idx.numpy().tolist()
    # 将预测分数转换为列表
    node_pred_score = max_value.numpy().tolist()
    # 初始化结果列表
    res = []
    # 遍历预测标签列表
    for i, label in enumerate(node_pred_label):
        # 格式化预测分数
        pred_score = '{:.2f}'.format(node_pred_score[i])
        # 构建预测结果字典
        pred_res = {
                'label': label,
                'transcription': annotations[i]['transcription'],
                'score': pred_score,
                'points': annotations[i]['points'],
            }
        # 将预测结果添加到结果列表
        res.append(pred_res)
    # 根据标签排序结果列表
    res.sort(key=lambda x: x['label'])
    # 将结果列表写入输出文件
    fout.writelines([json.dumps(res, ensure_ascii=False) + '\n'])

# 主函数
def main():
    # 获取全局配置
    global_config = config['Global']

    # 构建模型
    model = build_model(config['Architecture'])
    # 加载模型
    load_model(config, model)

    # 创建数据操作
    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        transforms.append(op)

    # 获取数据目录
    data_dir = config['Eval']['dataset']['data_dir']

    # 创建操作符
    ops = create_operators(transforms, global_config)

    # 保存结果路径
    save_res_path = config['Global']['save_res_path']
    # 类别路径
    class_path = config['Global']['class_path']
    # 读取类别列表
    idx_to_cls = read_class_list(class_path)
    # 创建保存结果路径
    os.makedirs(os.path.dirname(save_res_path), exist_ok=True)

    # 模型评估模式
    model.eval()

    # 初始化变量
    warmup_times = 0
    count_t = []
    # 以写入模式打开保存结果的文件
    with open(save_res_path, "w") as fout:
        # 以二进制读取模式打开推断图片文件
        with open(config['Global']['infer_img'], "rb") as f:
            # 读取文件的所有行
            lines = f.readlines()
            # 遍历文件的每一行数据
            for index, data_line in enumerate(lines):
                # 当索引为10时，记录当前时间作为预热时间
                if index == 10:
                    warmup_t = time.time()
                # 将数据行解码为UTF-8格式
                data_line = data_line.decode('utf-8')
                # 去除换行符并按制表符分割数据行
                substr = data_line.strip("\n").split("\t")
                # 构建图片路径和标签信息
                img_path, label = data_dir + "/" + substr[0], substr[1]
                data = {'img_path': img_path, 'label': label}
                # 以二进制读取模式打开图片文件
                with open(data['img_path'], 'rb') as f:
                    # 读取图片数据
                    img = f.read()
                    data['image'] = img
                # 记录当前时间
                st = time.time()
                # 对数据进行转换和操作
                batch = transform(data, ops)
                # 初始化批量预测结果
                batch_pred = [0] * len(batch)
                # 遍历批量数据
                for i in range(len(batch)):
                    # 将数据转换为Paddle张量
                    batch_pred[i] = paddle.to_tensor(
                        np.expand_dims(
                            batch[i], axis=0))
                # 记录当前时间
                st = time.time()
                # 获取模型的节点和边信息
                node, edge = model(batch_pred)
                # 对节点进行softmax操作
                node = F.softmax(node, -1)
                # 计算预测时间并添加到列表中
                count_t.append(time.time() - st)
                # 绘制KIE结果
                draw_kie_result(batch, node, idx_to_cls, index)
                # 写入KIE结果
                write_kie_result(fout, node, data)
        # 关闭推断图片文件
        fout.close()
    # 输出成功信息
    logger.info("success!")
    # 输出预测图片数量和总耗时
    logger.info("It took {} s for predict {} images.".format(
        np.sum(count_t), len(count_t)))
    # 计算每秒推理图片数
    ips = len(count_t[warmup_times:]) / np.sum(count_t[warmup_times:])
    # 输出IPS信息
    logger.info("The ips is {} images/s".format(ips))
# 如果当前脚本作为主程序执行
if __name__ == '__main__':
    # 调用 program 模块的 preprocess 函数，获取 config, device, logger, vdl_writer 四个变量
    config, device, logger, vdl_writer = program.preprocess()
    # 调用主函数 main()
    main()
```