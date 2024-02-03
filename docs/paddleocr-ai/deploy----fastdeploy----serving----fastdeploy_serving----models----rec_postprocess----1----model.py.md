# `.\PaddleOCR\deploy\fastdeploy\serving\fastdeploy_serving\models\rec_postprocess\1\model.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“按原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关特定语言的权限和限制

# 导入所需的库
import json
import numpy as np
import time
import os
import sys
import codecs
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
        # 解析模型配置，这里不会解析 JSON 字符串
        self.model_config = json.loads(args['model_config'])
        print("model_config:", self.model_config)

        self.input_names = []
        # 遍历模型配置中的输入，将输入名称添加到列表中
        for input_config in self.model_config["input"]:
            self.input_names.append(input_config["name"])
        print("postprocess input names:", self.input_names)

        self.output_names = []
        self.output_dtype = []
        # 遍历模型配置中的输出，将输出名称添加到列表中，并将数据类型转换为 numpy 类型
        for output_config in self.model_config["output"]:
            self.output_names.append(output_config["name"])
            dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])
            self.output_dtype.append(dtype)
        print("postprocess output names:", self.output_names)

        # 获取当前文件所在目录，并拼接文件名
        dir_name = os.path.dirname(os.path.realpath(__file__)) + "/"
        file_name = dir_name + "ppocr_keys_v1.txt"
        #self.label_list = load_dict()
        # 初始化后处理器，加载识别器后处理器
        self.postprocessor = fd.vision.ocr.RecognizerPostprocessor(file_name)
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
            # 通过输入名称获取推理输出张量
            infer_outputs = pb_utils.get_input_tensor_by_name(
                request, self.input_names[0])
            # 将推理输出张量转换为 numpy 数组
            infer_outputs = infer_outputs.as_numpy()
            # 运行后处理器，得到结果
            results = self.postprocessor.run([infer_outputs])
            # 创建输出张量 0
            out_tensor_0 = pb_utils.Tensor(
                self.output_names[0], np.array(
                    results[0], dtype=np.object_))
            # 创建输出张量 1
            out_tensor_1 = pb_utils.Tensor(self.output_names[1],
                                           np.array(results[1]))
            # 创建推理响应对象，包含输出张量
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_tensor_1])
            # 将推理响应对象添加到响应列表中
            responses.append(inference_response)
        # 返回所有响应对象的列表
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        # 在模型卸载时调用 finalize 函数，用于清理工作
        print('Cleaning up...')
```