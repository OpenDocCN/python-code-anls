# `.\PaddleOCR\tools\infer\utility.py`

```py
# 导入必要的库
import argparse  # 用于解析命令行参数
import os  # 用于操作系统相关功能
import sys  # 提供对 Python 解释器的访问
import platform  # 用于获取操作系统的信息
import cv2  # OpenCV 库，用于图像处理
import numpy as np  # NumPy 库，用于数值计算
import paddle  # PaddlePaddle 深度学习框架
from PIL import Image, ImageDraw, ImageFont  # Python 图像处理库
import math  # 数学库
from paddle import inference  # PaddlePaddle 推理模块
import time  # 用于时间相关操作
import random  # 生成随机数
from ppocr.utils.logging import get_logger  # 获取日志记录器

# 定义将字符串转换为布尔值的函数
def str2bool(v):
    return v.lower() in ("true", "yes", "t", "y", "1")

# 定义将字符串转换为整数元组的函数
def str2int_tuple(v):
    return tuple([int(i.strip()) for i in v.split(",")])

# 初始化命令行参数解析器
def init_args():
    parser = argparse.ArgumentParser()
    # 预测引擎参数
    parser.add_argument("--use_gpu", type=str2bool, default=True)  # 是否使用 GPU
    parser.add_argument("--use_xpu", type=str2bool, default=False)  # 是否使用 XPU
    parser.add_argument("--use_npu", type=str2bool, default=False)  # 是否使用 NPU
    parser.add_argument("--ir_optim", type=str2bool, default=True)  # 是否进行 IR 优化
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)  # 是否使用 TensorRT
    parser.add_argument("--min_subgraph_size", type=int, default=15)  # 最小子图大小
    parser.add_argument("--precision", type=str, default="fp32")  # 精度
    parser.add_argument("--gpu_mem", type=int, default=500)  # GPU 内存
    parser.add_argument("--gpu_id", type=int, default=0)  # GPU ID

    # 文本检测器参数
    parser.add_argument("--image_dir", type=str)  # 图像目录
    parser.add_argument("--page_num", type=int, default=0)  # 页数
    parser.add_argument("--det_algorithm", type=str, default='DB')  # 检测算法
    parser.add_argument("--det_model_dir", type=str)  # 检测模型目录
    # 添加命令行参数 det_limit_side_len，表示检测限制的边长，默认值为 960
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    # 添加命令行参数 det_limit_type，表示检测限制的类型，默认值为 'max'
    parser.add_argument("--det_limit_type", type=str, default='max')
    # 添加命令行参数 det_box_type，表示检测框的类型，默认值为 'quad'
    parser.add_argument("--det_box_type", type=str, default='quad')

    # 添加命令行参数 det_db_thresh，表示 DB 检测的阈值，默认值为 0.3
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    # 添加命令行参数 det_db_box_thresh，表示 DB 检测框的阈值，默认值为 0.6
    parser.add_argument("--det_db_box_thresh", type=float, default=0.6)
    # 添加命令行参数 det_db_unclip_ratio，表示 DB 检测的 unclip 比例，默认值为 1.5
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.5)
    # 添加命令行参数 max_batch_size，表示最大批处理大小，默认值为 10
    parser.add_argument("--max_batch_size", type=int, default=10)
    # 添加命令行参数 use_dilation，表示是否使用膨胀，默认值为 False
    parser.add_argument("--use_dilation", type=str2bool, default=False)
    # 添加命令行参数 det_db_score_mode，表示 DB 检测的得分模式，默认值为 "fast"
    parser.add_argument("--det_db_score_mode", type=str, default="fast")

    # 添加命令行参数 det_east_score_thresh，表示 EAST 检测的得分阈值，默认值为 0.8
    parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
    # 添加命令行参数 det_east_cover_thresh，表示 EAST 检测的覆盖率阈值，默认值为 0.1
    parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
    # 添加命令行参数 det_east_nms_thresh，表示 EAST 检测的非极大值抑制阈值，默认值为 0.2
    parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)

    # 添加命令行参数 det_sast_score_thresh，表示 SAST 检测的得分阈值，默认值为 0.5
    parser.add_argument("--det_sast_score_thresh", type=float, default=0.5)
    # 添加命令行参数 det_sast_nms_thresh，表示 SAST 检测的非极大值抑制阈值，默认值为 0.2
    parser.add_argument("--det_sast_nms_thresh", type=float, default=0.2)

    # 添加命令行参数 det_pse_thresh，表示 PSE 检测的阈值，默认值为 0
    parser.add_argument("--det_pse_thresh", type=float, default=0)
    # 添加命令行参数 det_pse_box_thresh，表示 PSE 检测框的阈值，默认值为 0.85
    parser.add_argument("--det_pse_box_thresh", type=float, default=0.85)
    # 添加命令行参数 det_pse_min_area，表示 PSE 检测的最小区域面积，默认值为 16
    parser.add_argument("--det_pse_min_area", type=float, default=16)
    # 添加命令行参数 det_pse_scale，表示 PSE 检测的缩放比例，默认值为 1
    parser.add_argument("--det_pse_scale", type=int, default=1)

    # 添加命令行参数 scales，表示 FCE 检测的尺度列表，默认值为 [8, 16, 32]
    parser.add_argument("--scales", type=list, default=[8, 16, 32])
    # 添加命令行参数 alpha，表示 FCE 检测的 alpha 参数，默认值为 1.0
    parser.add_argument("--alpha", type=float, default=1.0)
    # 添加命令行参数 beta，表示 FCE 检测的 beta 参数，默认值为 1.0
    parser.add_argument("--beta", type=float, default=1.0)
    # 添加命令行参数 fourier_degree，表示 FCE 检测的傅立叶级数，默认值为 5
    parser.add_argument("--fourier_degree", type=int, default=5)

    # 添加命令行参数 rec_algorithm，表示文本识别的算法，默认值为 'SVTR_LCNet'
    parser.add_argument("--rec_algorithm", type=str, default='SVTR_LCNet')
    # 添加命令行参数 rec_model_dir，表示文本识别模型的目录
    parser.add_argument("--rec_model_dir", type=str)
    # 添加命令行参数 rec_image_inverse，表示是否对图像进行反转，默认值为 True
    parser.add_argument("--rec_image_inverse", type=str2bool, default=True)
    # 添加命令行参数 rec_image_shape，表示文本识别的图像形状，默认为 "3, 48, 320"
    parser.add_argument("--rec_image_shape", type=str, default="3, 48, 320")
    # 添加参数--rec_batch_num，指定识别模型的批处理大小，默认为6
    parser.add_argument("--rec_batch_num", type=int, default=6)
    # 添加参数--max_text_length，指定最大文本长度，默认为25
    parser.add_argument("--max_text_length", type=int, default=25)
    # 添加参数--rec_char_dict_path，指定识别模型的字符字典路径，默认为"./ppocr/utils/ppocr_keys_v1.txt"
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default="./ppocr/utils/ppocr_keys_v1.txt")
    # 添加参数--use_space_char，指定是否使用空格字符，默认为True
    parser.add_argument("--use_space_char", type=str2bool, default=True)
    # 添加参数--vis_font_path，指定可视化字体路径，默认为"./doc/fonts/simfang.ttf"
    parser.add_argument(
        "--vis_font_path", type=str, default="./doc/fonts/simfang.ttf")
    # 添加参数--drop_score，指定丢弃分数的阈值，默认为0.5
    parser.add_argument("--drop_score", type=float, default=0.5)

    # params for e2e
    # 添加参数--e2e_algorithm，指定端到端模型的算法，默认为'PGNet'
    parser.add_argument("--e2e_algorithm", type=str, default='PGNet')
    # 添加参数--e2e_model_dir，指定端到端模型的目录路径
    parser.add_argument("--e2e_model_dir", type=str)
    # 添加参数--e2e_limit_side_len，指定端到端模型的限制边长，默认为768
    parser.add_argument("--e2e_limit_side_len", type=float, default=768)
    # 添加参数--e2e_limit_type，指定端到端模型的限制类型，默认为'max'
    parser.add_argument("--e2e_limit_type", type=str, default='max')

    # PGNet parmas
    # 添加参数--e2e_pgnet_score_thresh，指定PGNet模型的分数阈值，默认为0.5
    parser.add_argument("--e2e_pgnet_score_thresh", type=float, default=0.5)
    # 添加参数--e2e_char_dict_path，指定PGNet模型的字符字典路径，默认为"./ppocr/utils/ic15_dict.txt"
    parser.add_argument(
        "--e2e_char_dict_path", type=str, default="./ppocr/utils/ic15_dict.txt")
    # 添加参数--e2e_pgnet_valid_set，指定PGNet模型的验证集，默认为'totaltext'
    parser.add_argument("--e2e_pgnet_valid_set", type=str, default='totaltext')
    # 添加参数--e2e_pgnet_mode，指定PGNet模型的模式，默认为'fast'
    parser.add_argument("--e2e_pgnet_mode", type=str, default='fast')

    # params for text classifier
    # 添加参数--use_angle_cls，指定是否使用角度分类器，默认为False
    parser.add_argument("--use_angle_cls", type=str2bool, default=False)
    # 添加参数--cls_model_dir，指定分类器模型的目录路径
    parser.add_argument("--cls_model_dir", type=str)
    # 添加参数--cls_image_shape，指定分类器模型的图像形状，默认为"3, 48, 192"
    parser.add_argument("--cls_image_shape", type=str, default="3, 48, 192")
    # 添加参数--label_list，指定标签列表，默认为['0', '180']
    parser.add_argument("--label_list", type=list, default=['0', '180'])
    # 添加参数--cls_batch_num，指定分类器模型的批处理大小，默认为6
    parser.add_argument("--cls_batch_num", type=int, default=6)
    # 添加参数--cls_thresh，指定分类器模型的阈值，默认为0.9
    parser.add_argument("--cls_thresh", type=float, default=0.9)

    # 添加参数--enable_mkldnn，指定是否启用MKL-DNN，默认为False
    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    # 添加参数--cpu_threads，指定CPU线程数，默认为10
    parser.add_argument("--cpu_threads", type=int, default=10)
    # 添加参数--use_pdserving，指定是否使用PaddleServing，默认为False
    parser.add_argument("--use_pdserving", type=str2bool, default=False)
    # 添加参数--warmup，指定是否进行预热，默认为False
    parser.add_argument("--warmup", type=str2bool, default=False)

    # SR parmas
    # 添加参数--sr_model_dir，指定SR模型的目录路径
    parser.add_argument("--sr_model_dir", type=str)
    # 添加参数--sr_image_shape，指定SR模型的图像形状，默认为"3, 32, 128"
    parser.add_argument("--sr_image_shape", type=str, default="3, 32, 128")
    # 添加一个参数，用于指定超分辨率模型的批处理大小，默认为1
    parser.add_argument("--sr_batch_num", type=int, default=1)

    #
    # 添加一个参数，用于指定保存绘制图像的目录，默认为"./inference_results"
    parser.add_argument(
        "--draw_img_save_dir", type=str, default="./inference_results")
    # 添加一个参数，用于指定是否保存裁剪结果，默认为False
    parser.add_argument("--save_crop_res", type=str2bool, default=False)
    # 添加一个参数，用于指定裁剪结果保存的目录，默认为"./output"

    parser.add_argument("--crop_res_save_dir", type=str, default="./output")

    # 多进程
    # 添加一个参数，用于指定是否使用多进程，默认为False
    parser.add_argument("--use_mp", type=str2bool, default=False)
    # 添加一个参数，用于指定总进程数，默认为1
    parser.add_argument("--total_process_num", type=int, default=1)
    # 添加一个参数，用于指定进程ID，默认为0
    parser.add_argument("--process_id", type=int, default=0)

    # 添加一个参数，用于指定是否进行基准测试，默认为False
    parser.add_argument("--benchmark", type=str2bool, default=False)
    # 添加一个参数，用于指定保存日志的路径，默认为"./log_output/"
    parser.add_argument("--save_log_path", type=str, default="./log_output/")

    # 添加一个参数，用于指定是否显示日志，默认为True
    parser.add_argument("--show_log", type=str2bool, default=True)
    # 添加一个参数，用于指定是否使用ONNX，默认为False
    parser.add_argument("--use_onnx", type=str2bool, default=False)
    # 返回参数解析器
    return parser
# 解析命令行参数并返回解析结果
def parse_args():
    # 初始化参数解析器
    parser = init_args()
    return parser.parse_args()

# 根据模式和参数创建预测器
def create_predictor(args, mode, logger):
    # 根据不同的模式选择对应的模型目录
    if mode == "det":
        model_dir = args.det_model_dir
    elif mode == 'cls':
        model_dir = args.cls_model_dir
    elif mode == 'rec':
        model_dir = args.rec_model_dir
    elif mode == 'table':
        model_dir = args.table_model_dir
    elif mode == 'ser':
        model_dir = args.ser_model_dir
    elif mode == 're':
        model_dir = args.re_model_dir
    elif mode == "sr":
        model_dir = args.sr_model_dir
    elif mode == 'layout':
        model_dir = args.layout_model_dir
    else:
        model_dir = args.e2e_model_dir

    # 如果模型目录为空，则打印日志并退出程序
    if model_dir is None:
        logger.info("not find {} model file path {}".format(mode, model_dir))
        sys.exit(0)
    
    # 如果使用 ONNX 模型，则导入 onnxruntime 库
    if args.use_onnx:
        import onnxruntime as ort
        model_file_path = model_dir
        if not os.path.exists(model_file_path):
            raise ValueError("not find model file path {}".format(
                model_file_path))
        # 创建 ONNX 推理会话
        sess = ort.InferenceSession(model_file_path)
        return sess, sess.get_inputs()[0], None, None

# 获取预测器的输出张量
def get_output_tensors(args, mode, predictor):
    output_names = predictor.get_output_names()
    output_tensors = []
    # 如果是文本识别模式并且使用 CRNN、SVTR_LCNet 或 SVTR_HGNet 算法
    if mode == "rec" and args.rec_algorithm in [
            "CRNN", "SVTR_LCNet", "SVTR_HGNet"
    ]:
        output_name = 'softmax_0.tmp_0'
        if output_name in output_names:
            return [predictor.get_output_handle(output_name)]
        else:
            for output_name in output_names:
                output_tensor = predictor.get_output_handle(output_name)
                output_tensors.append(output_tensor)
    else:
        for output_name in output_names:
            output_tensor = predictor.get_output_handle(output_name)
            output_tensors.append(output_tensor)
    return output_tensors

# 获取推理 GPU ID
def get_infer_gpuid():
    sysstr = platform.system()
    if sysstr == "Windows":
        return 0
    # 检查是否使用 ROCm 编译，选择不同的环境变量命令
    if not paddle.device.is_compiled_with_rocm:
        cmd = "env | grep CUDA_VISIBLE_DEVICES"
    else:
        cmd = "env | grep HIP_VISIBLE_DEVICES"
    # 执行环境变量命令，获取 CUDA 或 HIP 可见设备信息
    env_cuda = os.popen(cmd).readlines()
    # 如果未找到可见设备信息，则返回 0
    if len(env_cuda) == 0:
        return 0
    else:
        # 从可见设备信息中提取 GPU ID，并转换为整数返回
        gpu_id = env_cuda[0].strip().split("=")[1]
        return int(gpu_id[0])
# 绘制端到端识别结果，将检测到的文本框和文本字符串绘制在图像上
def draw_e2e_res(dt_boxes, strs, img_path):
    # 读取图像
    src_im = cv2.imread(img_path)
    # 遍历文本框和文本字符串
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
    # 返回绘制后的图像
    return src_im


# 绘制文本检测结果
def draw_text_det_res(dt_boxes, img):
    # 遍历文本框
    for box in dt_boxes:
        # 将文本框转换为整数类型的坐标，并重塑为(-1, 2)的形状
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        # 绘制多边形文本框
        cv2.polylines(img, [box], True, color=(255, 255, 0), thickness=2)
    # 返回绘制后的图像
    return img


# 调整图像大小
def resize_img(img, input_size=600):
    """
    resize img and limit the longest side of the image to input_size
    """
    # 将图像转换为数组
    img = np.array(img)
    # 获取图像形状
    im_shape = img.shape
    # 获取图像最大边长
    im_size_max = np.max(im_shape[0:2])
    # 计算缩放比例
    im_scale = float(input_size) / float(im_size_max)
    # 调整图像大小
    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
    # 返回调整大小后的图像
    return img


# 绘制OCR检测和识别结果
def draw_ocr(image,
             boxes,
             txts=None,
             scores=None,
             drop_score=0.5,
             font_path="./doc/fonts/simfang.ttf"):
    """
    Visualize the results of OCR detection and recognition
    args:
        image(Image|array): RGB image
        boxes(list): boxes with shape(N, 4, 2)
        txts(list): the texts
        scores(list): txxs corresponding scores
        drop_score(float): only scores greater than drop_threshold will be visualized
        font_path: the path of font which is used to draw text
    return(array):
        the visualized img
    """
    # 如果未提供分数，则默认为1
    if scores is None:
        scores = [1] * len(boxes)
    # 获取文本框数量
    box_num = len(boxes)
    # 遍历框的数量
    for i in range(box_num):
        # 如果存在分数并且分数小于阈值或者是 NaN，则跳过当前框
        if scores is not None and (scores[i] < drop_score or
                                   math.isnan(scores[i])):
            continue
        # 将框的坐标转换为 numpy 数组，并重塑为 [-1, 1, 2] 的形状，转换为 int64 类型
        box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
        # 在图像上绘制多边形，连接框的四个点，颜色为 (255, 0, 0)，线宽为 2
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    # 如果存在文本信息
    if txts is not None:
        # 将图像调整大小为 600，返回调整后的图像
        img = np.array(resize_img(image, input_size=600))
        # 生成文本可视化图像
        txt_img = text_visual(
            txts,
            scores,
            img_h=img.shape[0],
            img_w=600,
            threshold=drop_score,
            font_path=font_path)
        # 将原图像和文本可视化图像拼接在一起
        img = np.concatenate([np.array(img), np.array(txt_img)], axis=1)
        # 返回拼接后的图像
        return img
    # 如果不存在文本信息，则返回原图像
    return image
# 绘制OCR框和文本到图像上
def draw_ocr_box_txt(image,
                     boxes,
                     txts=None,
                     scores=None,
                     drop_score=0.5,
                     font_path="./doc/fonts/simfang.ttf"):
    # 获取图像的高度和宽度
    h, w = image.height, image.width
    # 复制原始图像
    img_left = image.copy()
    # 创建一个与原始图像相同大小的白色图像
    img_right = np.ones((h, w, 3), dtype=np.uint8) * 255
    random.seed(0)

    # 在左侧图像上绘制OCR框
    draw_left = ImageDraw.Draw(img_left)
    # 如果文本列表为空或长度与框列表不匹配，则初始化为None
    if txts is None or len(txts) != len(boxes):
        txts = [None] * len(boxes)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        # 如果存在置信度分数并且小于设定的阈值，则跳过
        if scores is not None and scores[idx] < drop_score:
            continue
        # 生成随机颜色
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        # 在左侧图像上绘制多边形框
        draw_left.polygon(box, fill=color)
        # 在右侧图像上绘制文本框
        img_right_text = draw_box_txt_fine((w, h), box, txt, font_path)
        # 将多边形框转换为点坐标
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        # 在文本框上绘制多边形边框
        cv2.polylines(img_right_text, [pts], True, color, 1)
        # 对右侧图像进行按位与操作
        img_right = cv2.bitwise_and(img_right, img_right_text)
    # 将左侧图像与原始图像混合
    img_left = Image.blend(image, img_left, 0.5)
    # 创建一个新的RGB图像，用于显示左右两个图像
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    # 将左侧图像粘贴到新图像的左侧
    img_show.paste(img_left, (0, 0, w, h))
    # 将右侧图像转换为Image对象后粘贴到新图像的右侧
    img_show.paste(Image.fromarray(img_right), (w, 0, w * 2, h))
    return np.array(img_show)


# 绘制文本框和文本到图像上
def draw_box_txt_fine(img_size, box, txt, font_path="./doc/fonts/simfang.ttf"):
    # 计算文本框的高度和宽度
    box_height = int(
        math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][1])**2))
    box_width = int(
        math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2))

    # 如果文本框高度大于两倍宽度且高度大于30，则创建一个新的文本图像
    if box_height > 2 * box_width and box_height > 30:
        img_text = Image.new('RGB', (box_height, box_width), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        # 如果存在文本内容，则根据文本内容和字体路径创建字体对象并绘制文本
        if txt:
            font = create_font(txt, (box_height, box_width), font_path)
            draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)
        # 将文本图像旋转270度
        img_text = img_text.transpose(Image.ROTATE_270)
    # 如果条件不成立，创建一个白色背景的图像
    else:
        img_text = Image.new('RGB', (box_width, box_height), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        # 如果有文本内容，根据指定字体路径创建字体对象，并在图像上绘制文本
        if txt:
            font = create_font(txt, (box_width, box_height), font_path)
            draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)

    # 定义原始图像和目标图像的四个顶点坐标
    pts1 = np.float32(
        [[0, 0], [box_width, 0], [box_width, box_height], [0, box_height]])
    pts2 = np.array(box, dtype=np.float32)
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(pts1, pts2)

    # 将 PIL 图像转换为 NumPy 数组
    img_text = np.array(img_text, dtype=np.uint8)
    # 对图像进行透视变换
    img_right_text = cv2.warpPerspective(
        img_text,
        M,
        img_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255))
    # 返回经过透视变换后的图像
    return img_right_text
# 创建指定文本的字体对象
def create_font(txt, sz, font_path="./doc/fonts/simfang.ttf"):
    # 计算字体大小
    font_size = int(sz[1] * 0.99)
    # 使用指定字体路径创建字体对象
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    # 获取文本长度
    length = font.getlength(txt)
    # 如果文本长度超过指定大小，则重新计算字体大小
    if length > sz[0]:
        font_size = int(font_size * sz[0] / length)
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    return font

# 统计字符串中的中文字符、英文字符和数字个数
def str_count(s):
    """
    Count the number of Chinese characters,
    a single English character and a single number
    equal to half the length of Chinese characters.
    args:
        s(string): the input of string
    return(int):
        the number of Chinese characters
    """
    import string
    count_zh = count_pu = 0
    s_len = len(s)
    en_dg_count = 0
    for c in s:
        if c in string.ascii_letters or c.isdigit() or c.isspace():
            en_dg_count += 1
        elif c.isalpha():
            count_zh += 1
        else:
            count_pu += 1
    return s_len - math.ceil(en_dg_count / 2)

# 在空白图像上绘制文本
def text_visual(texts,
                scores,
                img_h=400,
                img_w=600,
                threshold=0.,
                font_path="./doc/simfang.ttf"):
    """
    create new blank img and draw txt on it
    args:
        texts(list): the text will be draw
        scores(list|None): corresponding score of each txt
        img_h(int): the height of blank img
        img_w(int): the width of blank img
        font_path: the path of font which is used to draw text
    return(array):
    """
    if scores is not None:
        assert len(texts) == len(
            scores), "The number of txts and corresponding scores must match"

    # 创建空白图像和绘制文本对象
    def create_blank_img():
        blank_img = np.ones(shape=[img_h, img_w], dtype=np.int8) * 255
        blank_img[:, img_w - 1:] = 0
        blank_img = Image.fromarray(blank_img).convert("RGB")
        draw_txt = ImageDraw.Draw(blank_img)
        return blank_img, draw_txt

    # 调用函数创建空白图像和绘制文本对象
    blank_img, draw_txt = create_blank_img()

    # 设置字体大小
    font_size = 20
    # 设置文本颜色为黑色
    txt_color = (0, 0, 0)
    # 使用指定字体路径、字体大小和编码创建字体对象
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    
    # 计算行间距
    gap = font_size + 5
    # 初始化文本图片列表、计数器和索引
    txt_img_list = []
    count, index = 1, 0
    
    # 遍历文本和对应的分数
    for idx, txt in enumerate(texts):
        index += 1
        # 如果分数低于阈值或为 NaN，则跳过当前文本
        if scores[idx] < threshold or math.isnan(scores[idx]):
            index -= 1
            continue
        
        # 初始化是否为第一行标志
        first_line = True
        # 当文本长度超过图片宽度时，进行分割处理
        while str_count(txt) >= img_w // font_size - 4:
            tmp = txt
            txt = tmp[:img_w // font_size - 4]
            # 根据是否为第一行添加行号
            if first_line:
                new_txt = str(index) + ': ' + txt
                first_line = False
            else:
                new_txt = '    ' + txt
            # 在图片上绘制文本
            draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
            txt = tmp[img_w // font_size - 4:]
            # 判断是否需要添加新的空白图片
            if count >= img_h // gap - 1:
                txt_img_list.append(np.array(blank_img))
                blank_img, draw_txt = create_blank_img()
                count = 0
            count += 1
        
        # 根据是否为第一行添加分数信息
        if first_line:
            new_txt = str(index) + ': ' + txt + '   ' + '%.3f' % (scores[idx])
        else:
            new_txt = "  " + txt + "  " + '%.3f' % (scores[idx])
        draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
        
        # 判断是否需要添加新的空白图片
        if count >= img_h // gap - 1 and idx + 1 < len(texts):
            txt_img_list.append(np.array(blank_img))
            blank_img, draw_txt = create_blank_img()
            count = 0
        count += 1
    
    # 将最后一个空白图片添加到文本图片列表中
    txt_img_list.append(np.array(blank_img))
    # 根据文本图片列表合并生成最终图片
    if len(txt_img_list) == 1:
        blank_img = np.array(txt_img_list[0])
    else:
        blank_img = np.concatenate(txt_img_list, axis=1)
    # 返回最终图片数据
    return np.array(blank_img)
# 将 base64 编码的字符串转换为 OpenCV 图像格式
def base64_to_cv2(b64str):
    # 导入 base64 模块
    import base64
    # 将 base64 编码的字符串解码为二进制数据
    data = base64.b64decode(b64str.encode('utf8'))
    # 将二进制数据转换为 numpy 数组
    data = np.frombuffer(data, np.uint8)
    # 使用 OpenCV 解码图像数据
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


# 在图像上绘制边界框
def draw_boxes(image, boxes, scores=None, drop_score=0.5):
    # 如果未提供分数列表，则默认为所有边界框的分数为1
    if scores is None:
        scores = [1] * len(boxes)
    # 遍历边界框和对应的分数
    for (box, score) in zip(boxes, scores):
        # 如果分数低于阈值，则跳过该边界框
        if score < drop_score:
            continue
        # 将边界框转换为 OpenCV 可接受的格式
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
        # 在图像上绘制多边形边界框
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    return image


# 获取旋转裁剪后的图像
def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    # 确保提供的点的数量为4
    assert len(points) == 4, "shape of points must be 4*2"
    # 计算裁剪后图像的宽度和高度
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    # 定义标准的四个点
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    # 获取透视变换矩阵
    M = cv2.getPerspectiveTransform(points, pts_std)
    # 进行透视变换
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    # 获取裁剪后图像的高度和宽度
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    # 如果裁剪后图像的高宽比大于1.5，则旋转图像
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


# 获取最小外接矩形裁剪后的图像
def get_minarea_rect_crop(img, points):
    # 计算包围盒的最小外接矩形
    bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
    # 根据边界框的顶点坐标，按照 x 坐标进行排序
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    # 初始化四个顶点的索引
    index_a, index_b, index_c, index_d = 0, 1, 2, 3
    # 根据顶点的 y 坐标大小确定顶点索引
    if points[1][1] > points[0][1]:
        index_a = 0
        index_d = 1
    else:
        index_a = 1
        index_d = 0
    # 根据顶点的 y 坐标大小确定顶点索引
    if points[3][1] > points[2][1]:
        index_b = 2
        index_c = 3
    else:
        index_b = 3
        index_c = 2

    # 根据确定的顶点索引构建新的边界框
    box = [points[index_a], points[index_b], points[index_c], points[index_d]]
    # 根据新的边界框裁剪图像
    crop_img = get_rotate_crop_image(img, np.array(box))
    # 返回裁剪后的图像
    return crop_img
# 检查是否使用 GPU，如果使用 GPU 但当前环境未编译支持 CUDA，则将 use_gpu 设为 False
def check_gpu(use_gpu):
    # 如果 use_gpu 为 True 且当前环境未编译支持 CUDA，则将 use_gpu 设为 False
    if use_gpu and not paddle.is_compiled_with_cuda():
        use_gpu = False
    # 返回最终的 use_gpu 值
    return use_gpu

# 如果当前脚本作为主程序运行，则执行以下代码
if __name__ == '__main__':
    pass
```