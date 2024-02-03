# `.\PaddleOCR\deploy\fastdeploy\serving\fastdeploy_serving\client.py`

```py
# 导入 logging 模块，用于记录日志
import logging
# 导入 numpy 模块，用于科学计算
import numpy as np
# 导入 time 模块，用于时间相关操作
import time
# 导入 Optional 类型提示，用于指定可选参数类型
from typing import Optional
# 导入 OpenCV 模块，用于图像处理
import cv2
# 导入 json 模块，用于处理 JSON 数据
import json

# 导入 tritonclient 中的 utils 模块，用于 Triton 客户端工具
from tritonclient import utils as client_utils
# 导入 tritonclient 中的 InferenceServerClient、InferInput、InferRequestedOutput 类
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput, service_pb2_grpc, service_pb2

# 创建名为 LOGGER 的日志记录器对象
LOGGER = logging.getLogger("run_inference_on_triton")

# 定义同步的 GRPC Triton 运行器类
class SyncGRPCTritonRunner:
    # 默认最大等待响应时间为 120 秒
    DEFAULT_MAX_RESP_WAIT_S = 120
    # 初始化函数，接受服务器 URL、模型名称、模型版本等参数
    def __init__(
            self,
            server_url: str,
            model_name: str,
            model_version: str,
            *,
            verbose=False,
            resp_wait_s: Optional[float]=None, ):
        # 初始化对象的属性
        self._server_url = server_url
        self._model_name = model_name
        self._model_version = model_version
        self._verbose = verbose
        # 设置响应等待时间，默认为 None
        self._response_wait_t = self.DEFAULT_MAX_RESP_WAIT_S if resp_wait_s is None else resp_wait_s

        # 创建 Triton 客户端对象
        self._client = InferenceServerClient(
            self._server_url, verbose=self._verbose)
        # 验证 Triton 服务器状态
        error = self._verify_triton_state(self._client)
        if error:
            # 如果验证失败，抛出运行时错误
            raise RuntimeError(
                f"Could not communicate to Triton Server: {error}")

        # 记录 Triton 服务器和模型的状态信息
        LOGGER.debug(
            f"Triton server {self._server_url} and model {self._model_name}:{self._model_version} "
            f"are up and ready!")

        # 获取模型配置和元数据
        model_config = self._client.get_model_config(self._model_name,
                                                     self._model_version)
        model_metadata = self._client.get_model_metadata(self._model_name,
                                                         self._model_version)
        # 记录模型配置和元数据信息
        LOGGER.info(f"Model config {model_config}")
        LOGGER.info(f"Model metadata {model_metadata}")

        # 初始化输入和输出信息
        self._inputs = {tm.name: tm for tm in model_metadata.inputs}
        self._input_names = list(self._inputs)
        self._outputs = {tm.name: tm for tm in model_metadata.outputs}
        self._output_names = list(self._outputs)
        self._outputs_req = [
            InferRequestedOutput(name) for name in self._outputs
        ]
    # 运行推理任务，根据输入数据返回推理结果
    def Run(self, inputs):
        """
        Args:
            inputs: list, Each value corresponds to an input name of self._input_names
        Returns:
            results: dict, {name : numpy.array}
        """
        # 创建推理输入列表
        infer_inputs = []
        # 遍历输入数据，创建推理输入对象并设置数据
        for idx, data in enumerate(inputs):
            infer_input = InferInput(self._input_names[idx], data.shape,
                                     "UINT8")
            infer_input.set_data_from_numpy(data)
            infer_inputs.append(infer_input)

        # 进行推理，获取结果
        results = self._client.infer(
            model_name=self._model_name,
            model_version=self._model_version,
            inputs=infer_inputs,
            outputs=self._outputs_req,
            client_timeout=self._response_wait_t, )
        # 将结果转换为字典形式
        results = {name: results.as_numpy(name) for name in self._output_names}
        # 返回结果字典
        return results

    # 验证 Triton 状态
    def _verify_triton_state(self, triton_client):
        # 检查 Triton 服务器是否存活
        if not triton_client.is_server_live():
            return f"Triton server {self._server_url} is not live"
        # 检查 Triton 服务器是否准备就绪
        elif not triton_client.is_server_ready():
            return f"Triton server {self._server_url} is not ready"
        # 检查模型是否准备就绪
        elif not triton_client.is_model_ready(self._model_name,
                                              self._model_version):
            return f"Model {self._model_name}:{self._model_version} is not ready"
        # 返回空表示 Triton 状态正常
        return None
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 模型名称
    model_name = "pp_ocr"
    # 模型版本
    model_version = "1"
    # Triton 服务器的 URL
    url = "localhost:8001"
    # 创建一个同步的 Triton 运行器对象
    runner = SyncGRPCTritonRunner(url, model_name, model_version)
    # 读取图像文件 "12.jpg"
    im = cv2.imread("12.jpg")
    # 将图像转换为 numpy 数组
    im = np.array([im, ])
    # 循环运行模型
    for i in range(1):
        # 运行模型，获取结果
        result = runner.Run([im, ])
        # 获取识别文本、识别分数和检测框
        batch_texts = result['rec_texts']
        batch_scores = result['rec_scores']
        batch_bboxes = result['det_bboxes']
        # 遍历每个批次的结果
        for i_batch in range(len(batch_texts)):
            # 获取文本、分数和框
            texts = batch_texts[i_batch]
            scores = batch_scores[i_batch]
            bboxes = batch_bboxes[i_batch]
            # 遍历每个框
            for i_box in range(len(texts)):
                # 打印文本、分数和框信息
                print('text=', texts[i_box].decode('utf-8'), '  score=',
                      scores[i_box], '  bbox=', bboxes[i_box])
```