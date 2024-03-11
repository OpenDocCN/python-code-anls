# `.\YOLO-World\demo.py`

```py
# 导入必要的库
import argparse
import os.path as osp
from functools import partial
import cv2
import torch
import gradio as gr
import numpy as np
import supervision as sv
from PIL import Image
from torchvision.ops import nms
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmengine.dataset import Compose
from mmdet.datasets import CocoDataset
from mmyolo.registry import RUNNERS

# 创建边界框标注器和标签标注器对象
BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-World Demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

# 运行图像处理
def run_image(runner,
              image,
              text,
              max_num_boxes,
              score_thr,
              nms_thr,
              image_path='./work_dirs/demo.png'):
    # 保存图像到指定路径
    image.save(image_path)
    # 将文本分割成列表
    texts = [[t.strip()] for t in text.split(',')] + [[' ']]
    # 构建数据信息字典
    data_info = dict(img_id=0, img_path=image_path, texts=texts)
    # 运行处理管道
    data_info = runner.pipeline(data_info)
    # 构建数据批次
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])
    # 关闭自动混合精度和禁用梯度计算
    with autocast(enabled=False), torch.no_grad():
        # 运行模型的测试步骤，获取输出
        output = runner.model.test_step(data_batch)[0]
        # 获取预测实例
        pred_instances = output.pred_instances

    # 使用非极大值抑制（NMS）筛选预测实例
    keep = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
    pred_instances = pred_instances[keep]
    # 根据置信度阈值筛选预测实例
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

    # 如果预测实例数量超过最大边界框数目限制
    if len(pred_instances.scores) > max_num_boxes:
        # 保留置信度最高的边界框
        indices = pred_instances.scores.float().topk(max_num_boxes)[1]
        pred_instances = pred_instances[indices]

    # 将预测实例转换为 NumPy 数组
    pred_instances = pred_instances.cpu().numpy()
    # 创建检测结果对象
    detections = sv.Detections(
        xyxy=pred_instances['bboxes'],
        class_id=pred_instances['labels'],
        confidence=pred_instances['scores']
    )
    # 生成标签列表
    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    # 将图像转换为 NumPy 数组
    image = np.array(image)
    # 将图像从 RGB 转换为 BGR 格式
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # 在图像上绘制边界框
    image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
    # 在图像上添加标签
    image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
    # 将图像从 BGR 转换为 RGB 格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 创建图像对象
    image = Image.fromarray(image)
    # 返回处理后的图像
    return image
def demo(runner, args):
    # 如果当前脚本作为主程序运行
    if __name__ == '__main__':
        # 解析命令行参数
        args = parse_args()

        # 从配置文件加载配置信息
        cfg = Config.fromfile(args.config)
        # 如果有额外的配置选项，则合并到配置中
        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)

        # 如果命令行参数中指定了工作目录，则使用该目录，否则使用配置中的工作目录，如果配置中也没有，则使用默认目录
        if args.work_dir is not None:
            cfg.work_dir = args.work_dir
        elif cfg.get('work_dir', None) is None:
            cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

        # 加载模型参数
        cfg.load_from = args.checkpoint

        # 如果配置中没有指定运行器类型，则根据配置创建运行器对象
        if 'runner_type' not in cfg:
            runner = Runner.from_cfg(cfg)
        else:
            # 否则根据配置中的运行器类型创建运行器对象
            runner = RUNNERS.build(cfg)

        # 运行前的钩子函数
        runner.call_hook('before_run')
        # 加载或恢复模型参数
        runner.load_or_resume()
        # 获取测试数据集的数据处理流程
        pipeline = cfg.test_dataloader.dataset.pipeline
        # 创建数据处理流程对象
        runner.pipeline = Compose(pipeline)
        # 设置模型为评估模式
        runner.model.eval()
        # 运行演示
        demo(runner, args)
```