# `.\yolov8\ultralytics\utils\triton.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

# 引入必要的类型
from typing import List
# 解析 URL 的组件
from urllib.parse import urlsplit

# 引入 NumPy 库
import numpy as np


class TritonRemoteModel:
    """
    用于与远程 Triton 推理服务器模型进行交互的客户端类。

    Attributes:
        endpoint (str): Triton 服务器上模型的名称。
        url (str): Triton 服务器的 URL。
        triton_client: Triton 客户端（可以是 HTTP 或 gRPC）。
        InferInput: Triton 客户端的输入类。
        InferRequestedOutput: Triton 客户端的输出请求类。
        input_formats (List[str]): 模型输入的数据类型。
        np_input_formats (List[type]): 模型输入的 NumPy 数据类型。
        input_names (List[str]): 模型输入的名称列表。
        output_names (List[str]): 模型输出的名称列表。
    """

    def __init__(self, url: str, endpoint: str = "", scheme: str = ""):
        """
        初始化 TritonRemoteModel。

        参数可以单独提供，也可以从形如 <scheme>://<netloc>/<endpoint>/<task_name> 的 'url' 参数中解析。

        Args:
            url (str): Triton 服务器的 URL。
            endpoint (str): Triton 服务器上模型的名称。
            scheme (str): 通信协议（'http' 或 'grpc'）。
        """
        if not endpoint and not scheme:  # 从 URL 字符串中解析所有参数
            splits = urlsplit(url)
            endpoint = splits.path.strip("/").split("/")[0]
            scheme = splits.scheme
            url = splits.netloc

        self.endpoint = endpoint
        self.url = url

        # 根据通信协议选择 Triton 客户端
        if scheme == "http":
            import tritonclient.http as client  # noqa

            self.triton_client = client.InferenceServerClient(url=self.url, verbose=False, ssl=False)
            config = self.triton_client.get_model_config(endpoint)
        else:
            import tritonclient.grpc as client  # noqa

            self.triton_client = client.InferenceServerClient(url=self.url, verbose=False, ssl=False)
            config = self.triton_client.get_model_config(endpoint, as_json=True)["config"]

        # 按字母顺序对输出名称进行排序，例如 'output0', 'output1' 等。
        config["output"] = sorted(config["output"], key=lambda x: x.get("name"))

        # 定义模型属性
        type_map = {"TYPE_FP32": np.float32, "TYPE_FP16": np.float16, "TYPE_UINT8": np.uint8}
        self.InferRequestedOutput = client.InferRequestedOutput
        self.InferInput = client.InferInput
        self.input_formats = [x["data_type"] for x in config["input"]]
        self.np_input_formats = [type_map[x] for x in self.input_formats]
        self.input_names = [x["name"] for x in config["input"]]
        self.output_names = [x["name"] for x in config["output"]]
    # 定义一个特殊方法 __call__，允许将实例对象像函数一样调用，接受多个 numpy 数组作为输入，并返回多个 numpy 数组作为输出
    def __call__(self, *inputs: np.ndarray) -> List[np.ndarray]:
        """
        Call the model with the given inputs.

        Args:
            *inputs (List[np.ndarray]): Input data to the model.

        Returns:
            (List[np.ndarray]): Model outputs.
        """
        # 初始化一个空列表，用于存储推理输入
        infer_inputs = []
        # 获取输入数组的数据类型，假设所有输入数组的数据类型相同
        input_format = inputs[0].dtype
        # 遍历所有输入数组
        for i, x in enumerate(inputs):
            # 如果当前输入数组的数据类型与预期的输入数据类型不匹配，将其转换为预期的数据类型
            if x.dtype != self.np_input_formats[i]:
                x = x.astype(self.np_input_formats[i])
            # 创建一个推理输入对象，指定输入名称、形状和数据类型
            infer_input = self.InferInput(self.input_names[i], [*x.shape], self.input_formats[i].replace("TYPE_", ""))
            # 将 numpy 数组的数据复制到推理输入对象中
            infer_input.set_data_from_numpy(x)
            # 将推理输入对象添加到推理输入列表中
            infer_inputs.append(infer_input)

        # 根据输出名称列表创建推理输出对象列表
        infer_outputs = [self.InferRequestedOutput(output_name) for output_name in self.output_names]
        # 调用 Triton 客户端进行推理，传入模型名称、输入列表和输出列表，并获取推理结果
        outputs = self.triton_client.infer(model_name=self.endpoint, inputs=infer_inputs, outputs=infer_outputs)

        # 将每个输出结果转换回原始输入数据类型，并存储在列表中返回
        return [outputs.as_numpy(output_name).astype(input_format) for output_name in self.output_names]
```