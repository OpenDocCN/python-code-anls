# `.\PaddleOCR\deploy\fastdeploy\cpu-gpu\python\infer_det.py`

```
# 导入所需的库
import fastdeploy as fd
import cv2
import os

# 解析命令行参数
def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--det_model", required=True, help="Path of Detection model of PPOCR.")
    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file.")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Type of inference device, support 'cpu', 'kunlunxin' or 'gpu'.")
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="Define which GPU card used to run model.")
    return parser.parse_args()

# 构建运行时选项
def build_option(args):

    det_option = fd.RuntimeOption()

    if args.device.lower() == "gpu":
        det_option.use_gpu(args.device_id)

    return det_option

# 解析命令行参数
args = parse_arguments()

# 拼接检测模型文件路径
det_model_file = os.path.join(args.det_model, "inference.pdmodel")
det_params_file = os.path.join(args.det_model, "inference.pdiparams")

# 设置运行时选项
det_option = build_option(args)

# 创建检测模型
det_model = fd.vision.ocr.DBDetector(
    det_model_file, det_params_file, runtime_option=det_option)

# 设置预处理参数
det_model.preprocessor.max_side_len = 960
det_model.postprocessor.det_db_thresh = 0.3
det_model.postprocessor.det_db_box_thresh = 0.6
# 设置检测模型后处理器的解除裁剪比例为1.5
det_model.postprocessor.det_db_unclip_ratio = 1.5
# 设置检测模型后处理器的得分模式为"slow"
det_model.postprocessor.det_db_score_mode = "slow"
# 设置检测模型后处理器的使用膨胀为False
det_model.postprocessor.use_dilation = False

# 读取图像文件
im = cv2.imread(args.image)

# 使用检测模型对图像进行预测并返回结果
result = det_model.predict(im)

# 打印预测结果
print(result)

# 可视化结果
vis_im = fd.vision.vis_ppocr(im, result)
# 将可视化结果保存为"visualized_result.jpg"
cv2.imwrite("visualized_result.jpg", vis_im)
# 打印提示信息，说明可视化结果已保存在"./visualized_result.jpg"中
print("Visualized result save in ./visualized_result.jpg")
```