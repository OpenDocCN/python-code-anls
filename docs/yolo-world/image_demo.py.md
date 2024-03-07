# `.\YOLO-World\image_demo.py`

```
# 版权声明
# 导入必要的库
import os
import cv2
import argparse
import os.path as osp

import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmengine.dataset import Compose
from mmengine.utils import ProgressBar
from mmyolo.registry import RUNNERS

# 定义BOUNDING_BOX_ANNOTATOR对象
BOUNDING_BOX_ANNOTATOR = None
# 定义LABEL_ANNOTATOR对象
LABEL_ANNOTATOR = None

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-World Demo')
    # 添加命令行参数
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('image', help='image path, include image file or dir.')
    parser.add_argument(
        'text',
        help='text prompts, including categories separated by a comma or a txt file with each line as a prompt.'
    )
    parser.add_argument('--topk',
                        default=100,
                        type=int,
                        help='keep topk predictions.')
    parser.add_argument('--threshold',
                        default=0.0,
                        type=float,
                        help='confidence score threshold for predictions.')
    parser.add_argument('--device',
                        default='cuda:0',
                        help='device used for inference.')
    parser.add_argument('--show',
                        action='store_true',
                        help='show the detection results.')
    parser.add_argument('--annotation',
                        action='store_true',
                        help='save the annotated detection results as yolo text format.')
    parser.add_argument('--amp',
                        action='store_true',
                        help='use mixed precision for inference.')
    # 添加一个名为'--output-dir'的命令行参数，用于指定保存输出的目录，默认为'demo_outputs'
    parser.add_argument('--output-dir',
                        default='demo_outputs',
                        help='the directory to save outputs')
    # 添加一个名为'--cfg-options'的命令行参数，用于覆盖配置文件中的一些设置，支持键值对形式的参数
    # 如果要覆盖的值是列表，则应该以 key="[a,b]" 或 key=a,b 的格式提供
    # 还支持嵌套列表/元组值，例如 key="[(a,b),(c,d)]"
    # 注意引号是必要的，不允许有空格
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
    # 解析命令行参数
    args = parser.parse_args()
    # 返回解析后的参数
    return args
# 推断检测器，运行模型进行推断
def inference_detector(runner,
                       image_path,
                       texts,
                       max_dets,
                       score_thr,
                       output_dir,
                       use_amp=False,
                       show=False,
                       annotation=False):
    # 创建包含图像信息的字典
    data_info = dict(img_id=0, img_path=image_path, texts=texts)
    # 运行数据处理管道
    data_info = runner.pipeline(data_info)
    # 创建包含数据批次信息的字典
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])

    # 使用自动混合精度和禁用梯度计算
    with autocast(enabled=use_amp), torch.no_grad():
        # 运行模型的测试步骤
        output = runner.model.test_step(data_batch)[0]
        pred_instances = output.pred_instances
        # 通过设置阈值过滤预测实例
        pred_instances = pred_instances[
            pred_instances.scores.float() > score_thr]
    # 如果预测实例数量超过最大检测数
    if len(pred_instances.scores) > max_dets:
        # 选择得分最高的前 max_dets 个预测实例
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]

    # 将预测实例转换为 numpy 数组
    pred_instances = pred_instances.cpu().numpy()
    # 定义检测对象
    detections = None

    # 为每个检测结果添加标签
    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]

    # 读取图像
    image = cv2.imread(image_path)
    anno_image = image.copy()
    # 在图像上绘制边界框
    image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
    # 在图像上添加标签
    image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
    # 将标记后的图像保存到输出目录
    cv2.imwrite(osp.join(output_dir, osp.basename(image_path)), image)
    # 如果有注释
    if annotation:
        # 创建空字典用于存储图像和注释
        images_dict = {}
        annotations_dict = {}

        # 将图像路径的基本名称作为键，注释图像作为值存储在图像字典中
        images_dict[osp.basename(image_path)] = anno_image
        # 将图像路径的基本名称作为键，检测结果作为值存储在注释字典中
        annotations_dict[osp.basename(image_path)] = detections
        
        # 创建一个名为ANNOTATIONS_DIRECTORY的目录，如果目录已存在则不创建
        ANNOTATIONS_DIRECTORY =  os.makedirs(r"./annotations", exist_ok=True)

        # 设置最小图像面积百分比
        MIN_IMAGE_AREA_PERCENTAGE = 0.002
        # 设置最大图像面积百分比
        MAX_IMAGE_AREA_PERCENTAGE = 0.80
        # 设置近似百分比
        APPROXIMATION_PERCENTAGE = 0.75
        
        # 创建一个DetectionDataset对象，传入类别、图像字典和注释字典，然后转换为YOLO格式
        sv.DetectionDataset(
            classes=texts,
            images=images_dict,
            annotations=annotations_dict
        ).as_yolo(
            annotations_directory_path=ANNOTATIONS_DIRECTORY,
            min_image_area_percentage=MIN_IMAGE_AREA_PERCENTAGE,
            max_image_area_percentage=MAX_IMAGE_AREA_PERCENTAGE,
            approximation_percentage=APPROXIMATION_PERCENTAGE
        )

    # 如果需要展示图像
    if show:
        # 在窗口中展示图像，提供窗口名称
        cv2.imshow('Image', image)
        # 等待按键输入，0表示一直等待
        k = cv2.waitKey(0)
        # 如果按下ESC键（ASCII码为27），关闭所有窗口
        if k == 27:
            cv2.destroyAllWindows()
if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()

    # 加载配置文件
    cfg = Config.fromfile(args.config)
    # 如果有额外的配置选项，则合并到配置文件中
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 设置工作目录为当前目录下的 work_dirs 文件夹中，使用配置文件名作为子目录名
    cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    # 加载模型检查点
    cfg.load_from = args.checkpoint

    # 根据配置文件中是否包含 runner_type 字段来选择不同的 Runner 类型
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    # 加载文本数据
    if args.text.endswith('.txt'):
        with open(args.text) as f:
            lines = f.readlines()
        # 将文本数据转换为列表形式
        texts = [[t.rstrip('\r\n')] for t in lines] + [[' ']]
    else:
        # 将命令行参数中的文本数据转换为列表形式
        texts = [[t.strip()] for t in args.text.split(',')] + [[' ']]

    # 设置输出目录
    output_dir = args.output_dir
    # 如果输出目录不存在，则创建
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    # 在运行之前调用钩子函数
    runner.call_hook('before_run')
    # 加载或恢复模型
    runner.load_or_resume()
    # 获取数据处理流程
    pipeline = cfg.test_dataloader.dataset.pipeline
    runner.pipeline = Compose(pipeline)
    # 设置模型为评估模式
    runner.model.eval()

    # 检查输入的图像路径是否为文件夹
    if not osp.isfile(args.image):
        # 获取文件夹中所有以 .png 或 .jpg 结尾的图像文件路径
        images = [
            osp.join(args.image, img) for img in os.listdir(args.image)
            if img.endswith('.png') or img.endswith('.jpg')
        ]
    else:
        # 将输入的图像路径转换为列表形式
        images = [args.image]

    # 创建进度条对象，用于显示处理进度
    progress_bar = ProgressBar(len(images))
    # 遍历每张图像进行目标检测
    for image_path in images:
        # 调用目标检测函数进行推理
        inference_detector(runner,
                           image_path,
                           texts,
                           args.topk,
                           args.threshold,
                           output_dir=output_dir,
                           use_amp=args.amp,
                           show=args.show,
                           annotation=args.annotation)
        # 更新进度条
        progress_bar.update()
```