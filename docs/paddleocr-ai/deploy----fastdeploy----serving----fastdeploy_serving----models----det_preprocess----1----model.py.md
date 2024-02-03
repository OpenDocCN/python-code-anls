# `.\PaddleOCR\deploy\fastdeploy\serving\fastdeploy_serving\models\det_preprocess\1\model.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“按原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的具体语言

# 导入所需的库
import json
import numpy as np
import time

# 导入 fastdeploy 库
import fastdeploy as fd

# triton_python_backend_utils 在每个 Triton Python 模型中都可用
# 您需要使用此模块来创建推理请求和响应
# 它还包含一些实用函数，用于从 model_config 中提取信息
# 并将 Triton 输入/输出类型转换为 numpy 类型
import triton_python_backend_utils as pb_utils

# 定义 TritonPythonModel 类
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
        # 解析 model_config，这里不会解析 JSON 字符串
        self.model_config = json.loads(args['model_config'])
        print("model_config:", self.model_config)

        self.input_names = []
        # 遍历模型配置中的输入，将输入名称添加到 input_names 列表中
        for input_config in self.model_config["input"]:
            self.input_names.append(input_config["name"])
        print("preprocess input names:", self.input_names)

        self.output_names = []
        self.output_dtype = []
        # 遍历模型配置中的输出，将输出名称添加到 output_names 列表中，并将数据类型转换为 numpy 类型后添加到 output_dtype 列表中
        for output_config in self.model_config["output"]:
            self.output_names.append(output_config["name"])
            dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])
            self.output_dtype.append(dtype)
        print("preprocess output names:", self.output_names)
        # 实例化一个 DBDetectorPreprocessor 对象作为预处理器
        self.preprocessor = fd.vision.ocr.DBDetectorPreprocessor()
    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        # 初始化一个空列表用于存储每个请求的响应
        responses = []
        # 遍历每个请求
        for request in requests:
            # 通过请求和输入名称获取输入张量数据
            data = pb_utils.get_input_tensor_by_name(request,
                                                     self.input_names[0])
            # 将输入张量数据转换为 numpy 数组
            data = data.as_numpy()
            # 调用预处理器处理数据，获取输出和图像信息
            outputs, im_infos = self.preprocessor.run(data)
            # 将输出张量转换为 DLpack 张量
            dlpack_tensor = outputs[0].to_dlpack()
            # 创建输出张量 0
            output_tensor_0 = pb_utils.Tensor.from_dlpack(self.output_names[0],
                                                          dlpack_tensor)
            # 创建输出张量 1，包含图像信息
            output_tensor_1 = pb_utils.Tensor(
                self.output_names[1], np.array(
                    im_infos, dtype=np.int32))
            # 创建推理响应对象，包含输出张量
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor_0, output_tensor_1])
            # 将推理响应对象添加到响应列表中
            responses.append(inference_response)
        # 返回所有响应对象的列表
        return responses
    # `finalize`函数在模型被卸载时仅调用一次
    # 实现`finalize`函数是可选的。该函数允许模型在退出之前执行任何必要的清理工作
    def finalize(self):
        # 打印清理信息
        print('Cleaning up...')
```