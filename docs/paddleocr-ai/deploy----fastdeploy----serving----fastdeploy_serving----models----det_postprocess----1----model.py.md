# `.\PaddleOCR\deploy\fastdeploy\serving\fastdeploy_serving\models\det_postprocess\1\model.py`

```
# 导入所需的库
import json
import numpy as np
import time
import math
import cv2
import fastdeploy as fd

# 导入 Triton Python 模型中的 triton_python_backend_utils 模块
# 该模块用于创建推理请求和响应，提供一些实用函数用于从 model_config 中提取信息，以及将 Triton 输入/输出类型转换为 numpy 类型
import triton_python_backend_utils as pb_utils

# 定义一个函数，用于获取旋转裁剪后的图像
def get_rotate_crop_image(img, box):
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
    # 将边界框的坐标转换为四个点的坐标
    points = []
    for i in range(4):
        points.append([box[2 * i], box[2 * i + 1]])
    points = np.array(points, dtype=np.float32)
    img = img.astype(np.float32)
    # 断言确保点的数量为4
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
    # 定义标准四个点的坐标，用于透视变换
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    # 获取透视变换矩阵，将原始图像的四个点映射到标准四个点
    M = cv2.getPerspectiveTransform(points, pts_std)
    # 进行透视变换，得到目标图像
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    # 获取目标图像的高度和宽度
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    # 如果目标图像的高宽比大于等于1.5，进行旋转
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    # 返回处理后的目标图像
    return dst_img
class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # You must parse model_config. JSON string is not parsed here
        self.model_config = json.loads(args['model_config'])
        print("model_config:", self.model_config)

        self.input_names = []
        # Iterate through input configurations in model_config and append input names to input_names list
        for input_config in self.model_config["input"]:
            self.input_names.append(input_config["name"])
        print("postprocess input names:", self.input_names)

        self.output_names = []
        self.output_dtype = []
        # Iterate through output configurations in model_config
        for output_config in self.model_config["output"]:
            # Append output names to output_names list
            self.output_names.append(output_config["name"])
            # Convert Triton data type to NumPy data type and append to output_dtype list
            dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])
            self.output_dtype.append(dtype)
        print("postprocess output names:", self.output_names)
        # Initialize postprocessor, cls_preprocessor, rec_preprocessor, and set cls_threshold
        self.postprocessor = fd.vision.ocr.DBDetectorPostprocessor()
        self.cls_preprocessor = fd.vision.ocr.ClassifierPreprocessor()
        self.rec_preprocessor = fd.vision.ocr.RecognizerPreprocessor()
        self.cls_threshold = 0.9
    # `finalize`函数在模型被卸载时仅调用一次
    # 实现`finalize`函数是可选的。该函数允许模型在退出之前执行任何必要的清理工作
    def finalize(self):
        # 打印清理信息
        print('Cleaning up...')
```