# `.\yolov8\ultralytics\hub\session.py`

```py
# Ultralytics YOLO , AGPL-3.0 license

import threading  # 导入多线程支持模块
import time  # 导入时间模块
from http import HTTPStatus  # 导入HTTP状态码模块
from pathlib import Path  # 导入路径操作模块

import requests  # 导入HTTP请求模块

from ultralytics.hub.utils import HELP_MSG, HUB_WEB_ROOT, PREFIX, TQDM  # 导入Ultralytics HUB的工具模块
from ultralytics.utils import IS_COLAB, LOGGER, SETTINGS, __version__, checks, emojis  # 导入Ultralytics的工具函数和常量
from ultralytics.utils.errors import HUBModelError  # 导入自定义的错误类

AGENT_NAME = f"python-{__version__}-colab" if IS_COLAB else f"python-{__version__}-local"  # 根据是否在Colab环境中设置代理名称


class HUBTrainingSession:
    """
    HUB training session for Ultralytics HUB YOLO models. Handles model initialization, heartbeats, and checkpointing.

    Attributes:
        model_id (str): Identifier for the YOLO model being trained.
        model_url (str): URL for the model in Ultralytics HUB.
        rate_limits (dict): Rate limits for different API calls (in seconds).
        timers (dict): Timers for rate limiting.
        metrics_queue (dict): Queue for the model's metrics.
        model (dict): Model data fetched from Ultralytics HUB.
    """

    def __init__(self, identifier):
        """
        Initialize the HUBTrainingSession with the provided model identifier.

        Args:
            identifier (str): Model identifier used to initialize the HUB training session.
                It can be a URL string or a model key with specific format.

        Raises:
            ValueError: If the provided model identifier is invalid.
            ConnectionError: If connecting with global API key is not supported.
            ModuleNotFoundError: If hub-sdk package is not installed.
        """
        from hub_sdk import HUBClient  # 导入HUBClient类来进行与Ultralytics HUB的API交互

        self.rate_limits = {"metrics": 3, "ckpt": 900, "heartbeat": 300}  # 设置API调用的速率限制（秒）
        self.metrics_queue = {}  # 存储每个epoch的指标，直到上传
        self.metrics_upload_failed_queue = {}  # 存储上传失败的每个epoch的指标
        self.timers = {}  # 在ultralytics/utils/callbacks/hub.py中保存计时器
        self.model = None  # 初始化模型数据为None
        self.model_url = None  # 初始化模型URL为None
        self.model_file = None  # 初始化模型文件为None

        # 解析输入的标识符
        api_key, model_id, self.filename = self._parse_identifier(identifier)

        # 获取凭证
        active_key = api_key or SETTINGS.get("api_key")
        credentials = {"api_key": active_key} if active_key else None  # 设置凭证信息

        # 初始化客户端
        self.client = HUBClient(credentials)

        # 如果认证成功则加载模型
        if self.client.authenticated:
            if model_id:
                self.load_model(model_id)  # 加载现有模型
            else:
                self.model = self.client.model()  # 加载空模型

    @classmethod
    def create_session(cls, identifier, args=None):
        """Class method to create an authenticated HUBTrainingSession or return None."""
        try:
            # 尝试创建一个指定标识符的会话对象
            session = cls(identifier)
            # 检查客户端是否已认证
            if not session.client.authenticated:
                # 如果未认证且标识符以指定路径开始，则警告并退出程序
                if identifier.startswith(f"{HUB_WEB_ROOT}/models/"):
                    LOGGER.warning(f"{PREFIX}WARNING ⚠️ Login to Ultralytics HUB with 'yolo hub login API_KEY'.")
                    exit()
                return None
            # 如果提供了参数且标识符不是 HUB 模型的 URL，则创建模型
            if args and not identifier.startswith(f"{HUB_WEB_ROOT}/models/"):  # not a HUB model URL
                session.create_model(args)
                # 断言模型已加载正确
                assert session.model.id, "HUB model not loaded correctly"
            # 返回创建的会话对象
            return session
        # 处理权限错误或模块未找到异常，表明 hub-sdk 未安装
        except (PermissionError, ModuleNotFoundError, AssertionError):
            return None

    def load_model(self, model_id):
        """Loads an existing model from Ultralytics HUB using the provided model identifier."""
        # 通过提供的模型标识符加载现有模型
        self.model = self.client.model(model_id)
        # 如果模型数据不存在，则抛出值错误异常
        if not self.model.data:  # then model does not exist
            raise ValueError(emojis("❌ The specified HUB model does not exist"))  # TODO: improve error handling

        # 设置模型的 URL
        self.model_url = f"{HUB_WEB_ROOT}/models/{self.model.id}"
        # 如果模型已经训练完成
        if self.model.is_trained():
            # 输出加载已训练的 HUB 模型的信息
            print(emojis(f"Loading trained HUB model {self.model_url} "))
            # 获取模型权重的 URL
            self.model_file = self.model.get_weights_url("best")
            return

        # 设置训练参数并启动 HUB 监控代理的心跳
        self._set_train_args()
        self.model.start_heartbeat(self.rate_limits["heartbeat"])
        # 输出模型的 URL
        LOGGER.info(f"{PREFIX}View model at {self.model_url} ")
    def create_model(self, model_args):
        """Initializes a HUB training session with the specified model identifier."""
        # 构造包含训练参数的 payload 对象
        payload = {
            "config": {
                "batchSize": model_args.get("batch", -1),  # 设置批量大小，默认为-1
                "epochs": model_args.get("epochs", 300),   # 设置训练周期数，默认为300
                "imageSize": model_args.get("imgsz", 640),  # 设置图像大小，默认为640
                "patience": model_args.get("patience", 100),  # 设置训练耐心值，默认为100
                "device": str(model_args.get("device", "")),  # 设置设备类型，将None转换为字符串
                "cache": str(model_args.get("cache", "ram")),  # 设置缓存类型，将True、False、None转换为字符串
            },
            "dataset": {"name": model_args.get("data")},  # 设置数据集名称
            "lineage": {
                "architecture": {"name": self.filename.replace(".pt", "").replace(".yaml", "")},  # 设置模型架构名称
                "parent": {},  # 初始化父模型信息
            },
            "meta": {"name": self.filename},  # 设置模型元数据名称
        }

        if self.filename.endswith(".pt"):
            payload["lineage"]["parent"]["name"] = self.filename  # 如果文件名以.pt结尾，设置父模型名称为文件名

        self.model.create_model(payload)  # 调用模型对象的创建模型方法，使用payload作为参数

        # Model could not be created
        # TODO: improve error handling
        # 如果模型未成功创建，记录错误并返回None
        if not self.model.id:
            return None

        self.model_url = f"{HUB_WEB_ROOT}/models/{self.model.id}"  # 构造模型的URL链接

        # Start heartbeats for HUB to monitor agent
        # 启动心跳以便HUB监控代理
        self.model.start_heartbeat(self.rate_limits["heartbeat"])

        LOGGER.info(f"{PREFIX}View model at {self.model_url} ")  # 记录模型的访问链接
    def _parse_identifier(identifier):
        """
        Parses the given identifier to determine the type of identifier and extract relevant components.
        
        The method supports different identifier formats:
            - A HUB URL, which starts with HUB_WEB_ROOT followed by '/models/'
            - An identifier containing an API key and a model ID separated by an underscore
            - An identifier that is solely a model ID of a fixed length
            - A local filename that ends with '.pt' or '.yaml'
        
        Args:
            identifier (str): The identifier string to be parsed.
        
        Returns:
            (tuple): A tuple containing the API key, model ID, and filename as applicable.
        
        Raises:
            HUBModelError: If the identifier format is not recognized.
        """

        # Initialize variables to None
        api_key, model_id, filename = None, None, None

        # Check if identifier is a HUB URL
        if identifier.startswith(f"{HUB_WEB_ROOT}/models/"):
            # Extract the model_id after the HUB_WEB_ROOT URL
            model_id = identifier.split(f"{HUB_WEB_ROOT}/models/")[-1]
        else:
            # Split the identifier based on underscores only if it's not a HUB URL
            parts = identifier.split("_")

            # Check if identifier is in the format of API key and model ID
            if len(parts) == 2 and len(parts[0]) == 42 and len(parts[1]) == 20:
                api_key, model_id = parts
            # Check if identifier is a single model ID
            elif len(parts) == 1 and len(parts[0]) == 20:
                model_id = parts[0]
            # Check if identifier is a local filename
            elif identifier.endswith(".pt") or identifier.endswith(".yaml"):
                filename = identifier
            else:
                # Raise an error if identifier format does not match any supported format
                raise HUBModelError(
                    f"model='{identifier}' could not be parsed. Check format is correct. "
                    f"Supported formats are Ultralytics HUB URL, apiKey_modelId, modelId, local pt or yaml file."
                )

        # Return the extracted components as a tuple
        return api_key, model_id, filename
    def _set_train_args(self):
        """
        Initializes training arguments and creates a model entry on the Ultralytics HUB.

        This method sets up training arguments based on the model's state and updates them with any additional
        arguments provided. It handles different states of the model, such as whether it's resumable, pretrained,
        or requires specific file setup.

        Raises:
            ValueError: If the model is already trained, if required dataset information is missing, or if there are
                issues with the provided training arguments.
        """

        if self.model.is_resumable():
            # Model has saved weights
            self.train_args = {"data": self.model.get_dataset_url(), "resume": True}
            self.model_file = self.model.get_weights_url("last")
        else:
            # Model has no saved weights
            self.train_args = self.model.data.get("train_args")  # 从模型数据中获取训练参数
            # 设置模型文件，可以是 *.pt 或 *.yaml 文件
            self.model_file = (
                self.model.get_weights_url("parent") if self.model.is_pretrained() else self.model.get_architecture()
            )

        if "data" not in self.train_args:
            # RF bug - datasets are sometimes not exported
            raise ValueError("Dataset may still be processing. Please wait a minute and try again.")

        self.model_file = checks.check_yolov5u_filename(self.model_file, verbose=False)  # 检查并纠正文件名
        self.model_id = self.model.id

    def request_queue(
        self,
        request_func,
        retry=3,
        timeout=30,
        thread=True,
        verbose=True,
        progress_total=None,
        stream_response=None,
        *args,
        **kwargs,
    ):
        """
        Attempts to execute `request_func` with retries, timeout handling, optional threading, and progress.
        """

        def retry_request():
            """
            Attempts to call `request_func` with retries, timeout, and optional threading.
            """
            t0 = time.time()  # Record the start time for the timeout
            response = None
            for i in range(retry + 1):
                if (time.time() - t0) > timeout:
                    LOGGER.warning(f"{PREFIX}Timeout for request reached. {HELP_MSG}")
                    break  # Timeout reached, exit loop

                response = request_func(*args, **kwargs)
                if response is None:
                    LOGGER.warning(f"{PREFIX}Received no response from the request. {HELP_MSG}")
                    time.sleep(2**i)  # Exponential backoff before retrying
                    continue  # Skip further processing and retry

                if progress_total:
                    self._show_upload_progress(progress_total, response)
                elif stream_response:
                    self._iterate_content(response)

                if HTTPStatus.OK <= response.status_code < HTTPStatus.MULTIPLE_CHOICES:
                    # if request related to metrics upload
                    if kwargs.get("metrics"):
                        self.metrics_upload_failed_queue = {}
                    return response  # Success, no need to retry

                if i == 0:
                    # Initial attempt, check status code and provide messages
                    message = self._get_failure_message(response, retry, timeout)

                    if verbose:
                        LOGGER.warning(f"{PREFIX}{message} {HELP_MSG} ({response.status_code})")

                if not self._should_retry(response.status_code):
                    LOGGER.warning(f"{PREFIX}Request failed. {HELP_MSG} ({response.status_code})")
                    break  # Not an error that should be retried, exit loop

                time.sleep(2**i)  # Exponential backoff for retries

            # if request related to metrics upload and exceed retries
            if response is None and kwargs.get("metrics"):
                self.metrics_upload_failed_queue.update(kwargs.get("metrics", None))

            return response

        if thread:
            # Start a new thread to run the retry_request function
            threading.Thread(target=retry_request, daemon=True).start()
        else:
            # If running in the main thread, call retry_request directly
            return retry_request()

    @staticmethod
    def _should_retry(status_code):
        """
        Determines if a request should be retried based on the HTTP status code.
        """
        retry_codes = {
            HTTPStatus.REQUEST_TIMEOUT,
            HTTPStatus.BAD_GATEWAY,
            HTTPStatus.GATEWAY_TIMEOUT,
        }
        return status_code in retry_codes
    def _get_failure_message(self, response: requests.Response, retry: int, timeout: int):
        """
        Generate a retry message based on the response status code.

        Args:
            response: The HTTP response object.
            retry: The number of retry attempts allowed.
            timeout: The maximum timeout duration.

        Returns:
            (str): The retry message.
        """
        # 如果应该重试，返回重试信息，包括重试次数和超时时间
        if self._should_retry(response.status_code):
            return f"Retrying {retry}x for {timeout}s." if retry else ""
        # 如果响应状态码为429（太多请求），则显示速率限制信息
        elif response.status_code == HTTPStatus.TOO_MANY_REQUESTS:  # rate limit
            headers = response.headers
            return (
                f"Rate limit reached ({headers['X-RateLimit-Remaining']}/{headers['X-RateLimit-Limit']}). "
                f"Please retry after {headers['Retry-After']}s."
            )
        else:
            try:
                # 尝试从响应中读取JSON格式的消息，如果无法读取则返回默认消息
                return response.json().get("message", "No JSON message.")
            except AttributeError:
                # 如果无法读取JSON，则返回无法读取JSON的提示信息
                return "Unable to read JSON."

    def upload_metrics(self):
        """Upload model metrics to Ultralytics HUB."""
        # 将模型指标上传到Ultralytics HUB，并返回请求队列的结果
        return self.request_queue(self.model.upload_metrics, metrics=self.metrics_queue.copy(), thread=True)

    def upload_model(
        self,
        epoch: int,
        weights: str,
        is_best: bool = False,
        map: float = 0.0,
        final: bool = False,
    ) -> None:
        """
        Upload a model checkpoint to Ultralytics HUB.

        Args:
            epoch (int): The current training epoch.
            weights (str): Path to the model weights file.
            is_best (bool): Indicates if the current model is the best one so far.
            map (float): Mean average precision of the model.
            final (bool): Indicates if the model is the final model after training.
        """
        # 如果指定的模型权重文件存在
        if Path(weights).is_file():
            # 获取模型文件的总大小（仅在最终上传时显示进度）
            progress_total = Path(weights).stat().st_size if final else None  # Only show progress if final
            # 请求队列将模型上传到Ultralytics HUB，包括各种参数和选项
            self.request_queue(
                self.model.upload_model,
                epoch=epoch,
                weights=weights,
                is_best=is_best,
                map=map,
                final=final,
                retry=10,
                timeout=3600,
                thread=not final,
                progress_total=progress_total,
                stream_response=True,
            )
        else:
            # 如果指定的模型权重文件不存在，则记录警告信息
            LOGGER.warning(f"{PREFIX}WARNING ⚠️ Model upload issue. Missing model {weights}.")

    @staticmethod
    # 显示文件下载进度条，用于跟踪文件下载过程中的进度
    def _show_upload_progress(content_length: int, response: requests.Response) -> None:
        """
        Display a progress bar to track the upload progress of a file download.

        Args:
            content_length (int): The total size of the content to be downloaded in bytes.
            response (requests.Response): The response object from the file download request.

        Returns:
            None
        """
        # 使用 tqdm 创建进度条，总大小为 content_length，单位为 B，自动缩放单位
        with TQDM(total=content_length, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
            # 遍历响应中的数据块，更新进度条
            for data in response.iter_content(chunk_size=1024):
                pbar.update(len(data))

    @staticmethod
    # 静态方法：处理流式 HTTP 响应数据
    def _iterate_content(response: requests.Response) -> None:
        """
        Process the streamed HTTP response data.

        Args:
            response (requests.Response): The response object from the file download request.

        Returns:
            None
        """
        # 遍历响应中的数据块，但不对数据块做任何操作
        for _ in response.iter_content(chunk_size=1024):
            pass  # Do nothing with data chunks
```