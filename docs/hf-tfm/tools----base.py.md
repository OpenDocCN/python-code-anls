# `.\transformers\tools\base.py`

```
#!/usr/bin/env python
# coding=utf-8

# 版权声明

# 导入模块和库
import base64
import importlib
import inspect
import io
import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Union

# 从 huggingface_hub 模块中导入函数
from huggingface_hub import create_repo, hf_hub_download, metadata_update, upload_folder
from huggingface_hub.utils import RepositoryNotFoundError, build_hf_headers, get_session

# 从自定义的模块中导入函数和类
from ..dynamic_module_utils import custom_object_save, get_class_from_dynamic_module, get_imports
from ..image_utils import is_pil_image
from ..models.auto import AutoProcessor
from ..utils import (
    CONFIG_NAME,
    cached_file,
    is_accelerate_available,
    is_torch_available,
    is_vision_available,
    logging,
)

# 从 gradio 模块中导入函数
from .agent_types import handle_agent_inputs, handle_agent_outputs


# 获取日志记录器
logger = logging.get_logger(__name__)

# 如果 Torch 可用则导入 Torch
if is_torch_available():
    import torch

# 如果 Accelerate 可用则导入 send_to_device 函数
if is_accelerate_available():
    from accelerate.utils import send_to_device

# 工具配置文件名
TOOL_CONFIG_FILE = "tool_config.json"

# 获取 repo 类型的函数
def get_repo_type(repo_id, repo_type=None, **hub_kwargs):
    if repo_type is not None:
        return repo_type
    try:
        # 尝试下载 space 类型的仓库信息，如果不存在则返回 model 类型
        hf_hub_download(repo_id, TOOL_CONFIG_FILE, repo_type="space", **hub_kwargs)
        return "space"
    except RepositoryNotFoundError:
        try:
            # 尝试下载 model 类型的仓库信息，如果不存在则抛出错误
            hf_hub_download(repo_id, TOOL_CONFIG_FILE, repo_type="model", **hub_kwargs)
            return "model"
        except RepositoryNotFoundError:
            raise EnvironmentError(f"`{repo_id}` does not seem to be a valid repo identifier on the Hub.")
        except Exception:
            return "model"
    except Exception:
        return "space"


# 根据模板生成应用文件内容的函数
APP_FILE_TEMPLATE = """from transformers import launch_gradio_demo
from {module_name} import {class_name}

launch_gradio_demo({class_name})
"""


# 工具类
class Tool:
    """
    A base class for the functions used by the agent. Subclass this and implement the `__call__` method as well as the
    following class attributes:

    - **description** (`str`) -- A short description of what your tool does, the inputs it expects and the output(s) it
      will return. For instance 'This is a tool that downloads a file from a `url`. It takes the `url` as input, and
      returns the text contained in the file'.
    """
    # 定义 Tool 类，用于创建工具
    class Tool:
        """
        Tool 类是用于创建工具的基类，可以继承并定制化自己的工具。
    
        Args:
            name (str): 工具的名称，在提示中向代理展示的名称，例如 "text-classifier" 或 "image_generator"。
            inputs (List[str]): 输入数据的模态列表（按照调用中的顺序）。模态应为 "text"、"image" 或 "audio"。仅用于 `launch_gradio_demo` 或使您的工具有良好的排版。
            outputs (List[str]): 工具返回的模态列表（与调用方法的返回顺序相同）。模态应为 "text"、"image" 或 "audio"。仅用于 `launch_gradio_demo` 或使您的工具有良好的排版。
    
        您还可以重写方法 [`~Tool.setup`]，如果您的工具在可用之前有昂贵的操作要执行（例如加载模型）。[`~Tool.setup`] 将在首次使用工具时调用，但不会在实例化时调用。
        """
    
        # 描述工具的属性
        description: str = "This is a tool that ..."
        # 工具的名称属性
        name: str = ""
    
        # 输入数据模态列表属性
        inputs: List[str]
        # 输出数据模态列表属性
        outputs: List[str]
    
        # 初始化方法
        def __init__(self, *args, **kwargs):
            # 标记工具是否已初始化
            self.is_initialized = False
    
        # 调用方法，需在 Tool 的子类中实现
        def __call__(self, *args, **kwargs):
            # 返回未实现错误，提示在 Tool 的子类中实现该方法
            return NotImplemented("Write this method in your subclass of `Tool`.")
    
        # 设置方法，用于执行昂贵操作，需在 Tool 的子类中重写
        def setup(self):
            """
            在这里重写此方法，用于执行在开始使用工具之前需要执行的任何昂贵操作。例如加载大型模型。
            """
            # 标记工具已初始化
            self.is_initialized = True
    def save(self, output_dir):
        """
        Saves the relevant code files for your tool so it can be pushed to the Hub. This will copy the code of your
        tool in `output_dir` as well as autogenerate:

        - a config file named `tool_config.json`
        - an `app.py` file so that your tool can be converted to a space
        - a `requirements.txt` containing the names of the module used by your tool (as detected when inspecting its
          code)

        You should only use this method to save tools that are defined in a separate module (not `__main__`).

        Args:
            output_dir (`str`): The folder in which you want to save your tool.
        """
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save module file
        if self.__module__ == "__main__":
            # Raise an error if the tool is defined in __main__ module
            raise ValueError(
                f"We can't save the code defining {self} in {output_dir} as it's been defined in __main__. You "
                "have to put this code in a separate module so we can include it in the saved folder."
            )
        # Save the custom object to the output directory
        module_files = custom_object_save(self, output_dir)

        # Get the module name and class name
        module_name = self.__class__.__module__
        last_module = module_name.split(".")[-1]
        full_name = f"{last_module}.{self.__class__.__name__}"

        # Save config file
        config_file = os.path.join(output_dir, "tool_config.json")
        if os.path.isfile(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                tool_config = json.load(f)
        else:
            tool_config = {}

        # Update the tool config with class information
        tool_config = {"tool_class": full_name, "description": self.description, "name": self.name}
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(tool_config, indent=2, sort_keys=True) + "\n")

        # Save app file
        app_file = os.path.join(output_dir, "app.py")
        with open(app_file, "w", encoding="utf-8") as f:
            f.write(APP_FILE_TEMPLATE.format(module_name=last_module, class_name=self.__class__.__name__))

        # Save requirements file
        requirements_file = os.path.join(output_dir, "requirements.txt")
        imports = []
        for module in module_files:
            imports.extend(get_imports(module))
        imports = list(set(imports))
        with open(requirements_file, "w", encoding="utf-8") as f:
            f.write("\n".join(imports) + "\n")

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        model_repo_id: Optional[str] = None,
        token: Optional[str] = None,
        remote: bool = False,
        **kwargs,
    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload tool",
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        create_pr: bool = False,
    ) -> str:
        """
        Upload the tool to the Hub.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push your tool to. It should contain your organization name when
                pushing to a given organization.
            commit_message (`str`, *optional*, defaults to `"Upload tool"`):
                Message to commit while pushing.
            private (`bool`, *optional`):
                Whether or not the repository created should be private.
            token (`bool` or `str`, *optional`):
                The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            create_pr (`bool`, *optional`, defaults to `False`):
                Whether or not to create a PR with the uploaded files or directly commit.
        """
        # 创建仓库并返回仓库 URL
        repo_url = create_repo(
            repo_id=repo_id, token=token, private=private, exist_ok=True, repo_type="space", space_sdk="gradio"
        )
        # 更新仓库元数据
        repo_id = repo_url.repo_id
        metadata_update(repo_id, {"tags": ["tool"]}, repo_type="space")

        # 使用临时目录保存所有文件
        with tempfile.TemporaryDirectory() as work_dir:
            # 保存所有文件
            self.save(work_dir)
            logger.info(f"Uploading the following files to {repo_id}: {','.join(os.listdir(work_dir))}")
            # 上传文件夹到仓库
            return upload_folder(
                repo_id=repo_id,
                commit_message=commit_message,
                folder_path=work_dir,
                token=token,
                create_pr=create_pr,
                repo_type="space",
            )

    @staticmethod
    def from_gradio(gradio_tool):
        """
        Creates a [`Tool`] from a gradio tool.
        """

        class GradioToolWrapper(Tool):
            def __init__(self, _gradio_tool):
                super().__init__()
                self.name = _gradio_tool.name
                self.description = _gradio_tool.description

        # 将 Gradio 工具包装为 Tool 类
        GradioToolWrapper.__call__ = gradio_tool.run
        return GradioToolWrapper(gradio_tool)
class RemoteTool(Tool):
    """
    A [`Tool`] that will make requests to an inference endpoint.

    Args:
        endpoint_url (`str`, *optional*):
            The url of the endpoint to use.
        token (`str`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated when
            running `huggingface-cli login` (stored in `~/.huggingface`).
        tool_class (`type`, *optional*):
            The corresponding `tool_class` if this is a remote version of an existing tool. Will help determine when
            the output should be converted to another type (like images).
    """

    def __init__(self, endpoint_url=None, token=None, tool_class=None):
        # 初始化 RemoteTool 类，设置属性值
        self.endpoint_url = endpoint_url
        # 创建 EndpointClient 对象，用于与指定的端点进行通信
        self.client = EndpointClient(endpoint_url, token=token)
        # 设置 tool_class 属性
        self.tool_class = tool_class

    def prepare_inputs(self, *args, **kwargs):
        """
        Prepare the inputs received for the HTTP client sending data to the endpoint. Positional arguments will be
        matched with the signature of the `tool_class` if it was provided at instantation. Images will be encoded into
        bytes.

        You can override this method in your custom class of [`RemoteTool`].
        """
        # 复制关键字参数
        inputs = kwargs.copy()
        # 处理位置参数
        if len(args) > 0:
            if self.tool_class is not None:
                # 匹配参数与签名
                if issubclass(self.tool_class, PipelineTool):
                    call_method = self.tool_class.encode
                else:
                    call_method = self.tool_class.__call__
                signature = inspect.signature(call_method).parameters
                parameters = [
                    k
                    for k, p in signature.items()
                    if p.kind not in [inspect._ParameterKind.VAR_POSITIONAL, inspect._ParameterKind.VAR_KEYWORD]
                ]
                if parameters[0] == "self":
                    parameters = parameters[1:]
                if len(args) > len(parameters):
                    raise ValueError(
                        f"{self.tool_class} only accepts {len(parameters)} arguments but {len(args)} were given."
                    )
                for arg, name in zip(args, parameters):
                    inputs[name] = arg
            elif len(args) > 1:
                raise ValueError("A `RemoteTool` can only accept one positional input.")
            elif len(args) == 1:
                if is_pil_image(args[0]):
                    return {"inputs": self.client.encode_image(args[0])}
                return {"inputs": args[0]}

        # 处理输入中的图像数据
        for key, value in inputs.items():
            if is_pil_image(value):
                inputs[key] = self.client.encode_image(value)

        return {"inputs": inputs}
    # 定义一个方法，用于提取输出结果
    def extract_outputs(self, outputs):
        """
        You can override this method in your custom class of [`RemoteTool`] to apply some custom post-processing of the
        outputs of the endpoint.
        """
        # 返回原始输出结果
        return outputs

    # 定义一个方法，用于调用远程工具
    def __call__(self, *args, **kwargs):
        # 处理输入参数
        args, kwargs = handle_agent_inputs(*args, **kwargs)

        # 检查输出是否为图像
        output_image = self.tool_class is not None and self.tool_class.outputs == ["image"]
        # 准备输入数据
        inputs = self.prepare_inputs(*args, **kwargs)
        # 根据输入数据调用客户端
        if isinstance(inputs, dict):
            outputs = self.client(**inputs, output_image=output_image)
        else:
            outputs = self.client(inputs, output_image=output_image)
        # 处理多层嵌套的输出结果
        if isinstance(outputs, list) and len(outputs) == 1 and isinstance(outputs[0], list):
            outputs = outputs[0]

        # 处理输出结果
        outputs = handle_agent_outputs(outputs, self.tool_class.outputs if self.tool_class is not None else None)

        # 提取输出结果
        return self.extract_outputs(outputs)
class PipelineTool(Tool):
    """
    A [`Tool`] tailored towards Transformer models. On top of the class attributes of the base class [`Tool`], you will
    need to specify:

    - **model_class** (`type`) -- The class to use to load the model in this tool.
    - **default_checkpoint** (`str`) -- The default checkpoint that should be used when the user doesn't specify one.
    - **pre_processor_class** (`type`, *optional*, defaults to [`AutoProcessor`]) -- The class to use to load the
      pre-processor
    - **post_processor_class** (`type`, *optional*, defaults to [`AutoProcessor`]) -- The class to use to load the
      post-processor (when different from the pre-processor).

    Args:
        model (`str` or [`PreTrainedModel`], *optional*):
            The name of the checkpoint to use for the model, or the instantiated model. If unset, will default to the
            value of the class attribute `default_checkpoint`.
        pre_processor (`str` or `Any`, *optional*):
            The name of the checkpoint to use for the pre-processor, or the instantiated pre-processor (can be a
            tokenizer, an image processor, a feature extractor or a processor). Will default to the value of `model` if
            unset.
        post_processor (`str` or `Any`, *optional*):
            The name of the checkpoint to use for the post-processor, or the instantiated pre-processor (can be a
            tokenizer, an image processor, a feature extractor or a processor). Will default to the `pre_processor` if
            unset.
        device (`int`, `str` or `torch.device`, *optional*):
            The device on which to execute the model. Will default to any accelerator available (GPU, MPS etc...), the
            CPU otherwise.
        device_map (`str` or `dict`, *optional*):
            If passed along, will be used to instantiate the model.
        model_kwargs (`dict`, *optional*):
            Any keyword argument to send to the model instantiation.
        token (`str`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated when
            running `huggingface-cli login` (stored in `~/.huggingface`).
        hub_kwargs (additional keyword arguments, *optional*):
            Any additional keyword argument to send to the methods that will load the data from the Hub.
    """

    # 默认使用 AutoProcessor 类作为预处理器
    pre_processor_class = AutoProcessor
    # 模型类需要在子类中指定
    model_class = None
    # 默认检查点为空
    post_processor_class = AutoProcessor
    # 默认��查点为空

    def __init__(
        self,
        model=None,
        pre_processor=None,
        post_processor=None,
        device=None,
        device_map=None,
        model_kwargs=None,
        token=None,
        **hub_kwargs,
        ):
        # 检查是否安装了 torch 库，如果没有则抛出 ImportError 异常
        if not is_torch_available():
            raise ImportError("Please install torch in order to use this tool.")

        # 检查是否安装了 accelerate 库，如果没有则抛出 ImportError 异常
        if not is_accelerate_available():
            raise ImportError("Please install accelerate in order to use this tool.")

        # 如果未传入模型，则使用默认检查点，如果默认检查点也未设置，则抛出 ValueError 异常
        if model is None:
            if self.default_checkpoint is None:
                raise ValueError("This tool does not implement a default checkpoint, you need to pass one.")
            model = self.default_checkpoint
        # 如果未传入预处理器，则使用模型作为预处理器
        if pre_processor is None:
            pre_processor = model

        # 初始化模型、预处理器、后处理器、设备、设备映射、模型参数等属性
        self.model = model
        self.pre_processor = pre_processor
        self.post_processor = post_processor
        self.device = device
        self.device_map = device_map
        self.model_kwargs = {} if model_kwargs is None else model_kwargs
        # 如果设备映射不为空，则将其添加到模型参数中
        if device_map is not None:
            self.model_kwargs["device_map"] = device_map
        self.hub_kwargs = hub_kwargs
        # 将 token 添加到 hub_kwargs 中
        self.hub_kwargs["token"] = token

        # 调用父类的初始化方法
        super().__init__()

    def setup(self):
        """
        Instantiates the `pre_processor`, `model` and `post_processor` if necessary.
        """
        # 如果预处理器是字符串，则根据字符串实例化预处理器类
        if isinstance(self.pre_processor, str):
            self.pre_processor = self.pre_processor_class.from_pretrained(self.pre_processor, **self.hub_kwargs)

        # 如果模型是字符串，则根据字符串实例化模型类
        if isinstance(self.model, str):
            self.model = self.model_class.from_pretrained(self.model, **self.model_kwargs, **self.hub_kwargs)

        # 如果后处理器为空，则使用预处理器作为后处理器；如果后处理器是字符串，则根据字符串实例化后处理器类
        if self.post_processor is None:
            self.post_processor = self.pre_processor
        elif isinstance(self.post_processor, str):
            self.post_processor = self.post_processor_class.from_pretrained(self.post_processor, **self.hub_kwargs)

        # 如果设备为空，则根据设备映射设置设备；如果设备映射为空，则获取默认设备
        if self.device is None:
            if self.device_map is not None:
                self.device = list(self.model.hf_device_map.values())[0]
            else:
                self.device = get_default_device()

        # 如果设备映射为空，则将模型移动到设备
        if self.device_map is None:
            self.model.to(self.device)

        # 调用父类的设置方法
        super().setup()

    def encode(self, raw_inputs):
        """
        Uses the `pre_processor` to prepare the inputs for the `model`.
        """
        # 使用预处理器处理原始输入数据
        return self.pre_processor(raw_inputs)

    def forward(self, inputs):
        """
        Sends the inputs through the `model`.
        """
        # 使用模型处理输入数据
        with torch.no_grad():
            return self.model(**inputs)

    def decode(self, outputs):
        """
        Uses the `post_processor` to decode the model output.
        """
        # 使用后处理器解码模型输出
        return self.post_processor(outputs)
    # 定义一个特殊方法，用于实例对象的调用
    def __call__(self, *args, **kwargs):
        # 处理输入参数，确保参数格式正确
        args, kwargs = handle_agent_inputs(*args, **kwargs)

        # 如果Agent对象还未初始化，则进行初始化
        if not self.is_initialized:
            self.setup()

        # 对输入参数进行编码处理
        encoded_inputs = self.encode(*args, **kwargs)
        # 将编码后的输入发送到指定设备
        encoded_inputs = send_to_device(encoded_inputs, self.device)
        # 对编码后的输入进行前向传播
        outputs = self.forward(encoded_inputs)
        # 将输出发送回CPU设备
        outputs = send_to_device(outputs, "cpu")
        # 对输出进行解码处理
        decoded_outputs = self.decode(outputs)

        # 处理Agent对象的输出结果，并返回
        return handle_agent_outputs(decoded_outputs, self.outputs)
# 启动一个 gradio 演示工具，需要传入一个工具类。工具类需要正确实现类属性 `inputs` 和 `outputs`。
def launch_gradio_demo(tool_class: Tool):
    try:
        # 尝试导入 gradio 库
        import gradio as gr
    except ImportError:
        # 如果导入失败，抛出 ImportError
        raise ImportError("Gradio should be installed in order to launch a gradio demo.")

    # 创建指定工具类的实例
    tool = tool_class()

    # 定义一个函数，用于调用工具类的实例
    def fn(*args, **kwargs):
        return tool(*args, **kwargs)

    # 创建 gr.Interface 对象，传入函数、输入、输出、标题和描述，然后启动演示
    gr.Interface(
        fn=fn,
        inputs=tool_class.inputs,
        outputs=tool_class.outputs,
        title=tool_class.__name__,
        article=tool.description,
    ).launch()


# TODO: Migrate to Accelerate for this once `PartialState.default_device` makes its way into a release.
# 获取默认设备的函数，即返回当前可用的设备
def get_default_device():
    logger.warning(
        "`get_default_device` is deprecated and will be replaced with `accelerate`'s `PartialState().default_device` "
        "in version 4.38 of 🤗 Transformers. "
    )
    # 如果没有安装 torch 库，抛出 ImportError
    if not is_torch_available():
        raise ImportError("Please install torch in order to use this tool.")

    # 检查是否支持 MPS 或 CUDA，返回相应的设备
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# 映射任务名称到工具类的字典
TASK_MAPPING = {
    "document-question-answering": "DocumentQuestionAnsweringTool",
    "image-captioning": "ImageCaptioningTool",
    "image-question-answering": "ImageQuestionAnsweringTool",
    "image-segmentation": "ImageSegmentationTool",
    "speech-to-text": "SpeechToTextTool",
    "summarization": "TextSummarizationTool",
    "text-classification": "TextClassificationTool",
    "text-question-answering": "TextQuestionAnsweringTool",
    "text-to-speech": "TextToSpeechTool",
    "translation": "TranslationTool",
}


# 获取默认的端点配置
def get_default_endpoints():
    # 从缓存文件中读取默认端点配置
    endpoints_file = cached_file("huggingface-tools/default-endpoints", "default_endpoints.json", repo_type="dataset")
    with open(endpoints_file, "r", encoding="utf-8") as f:
        endpoints = json.load(f)
    return endpoints


# 检查任务或仓库 ID 是否支持远程加载
def supports_remote(task_or_repo_id):
    # 获取默认端点配置
    endpoints = get_default_endpoints()
    return task_or_repo_id in endpoints


# 加载工具的主要函数，可以在 Hub 或 Transformers 库中快速加载工具
def load_tool(task_or_repo_id, model_repo_id=None, remote=False, token=None, **kwargs):
    Args:
        task_or_repo_id (`str`):
            要加载工具的任务或 Hub 上工具的存储库 ID。在 Transformers 中实现的任务有：

            - `"document-question-answering"`
            - `"image-captioning"`
            - `"image-question-answering"`
            - `"image-segmentation"`
            - `"speech-to-text"`
            - `"summarization"`
            - `"text-classification"`
            - `"text-question-answering"`
            - `"text-to-speech"`
            - `"translation"`

        model_repo_id (`str`, *可选*):
            使用此参数可以使用与所选工具的默认模型不同的模型。
        remote (`bool`, *可选*, 默认为 `False`):
            是否通过下载模型或（如果可用）使用推理端点来使用您的工具。
        token (`str`, *可选*):
            用于在 hf.co 上识别您的令牌。如果未设置，将使用运行 `huggingface-cli login` 时生成的令牌（存储在 `~/.huggingface` 中）。
        kwargs (其他关键字参数, *可选*):
            将被拆分为两部分的其他关键字参数：所有与 Hub 相关的参数（如 `cache_dir`、`revision`、`subfolder`）将在下载工具文件时使用，其他参数将传递给其初始化。
    """
    if task_or_repo_id in TASK_MAPPING:
        tool_class_name = TASK_MAPPING[task_or_repo_id]
        main_module = importlib.import_module("transformers")
        tools_module = main_module.tools
        tool_class = getattr(tools_module, tool_class_name)

        if remote:
            if model_repo_id is None:
                endpoints = get_default_endpoints()
                if task_or_repo_id not in endpoints:
                    raise ValueError(
                        f"Could not infer a default endpoint for {task_or_repo_id}, you need to pass one using the "
                        "`model_repo_id` argument."
                    )
                model_repo_id = endpoints[task_or_repo_id]
            return RemoteTool(model_repo_id, token=token, tool_class=tool_class)
        else:
            return tool_class(model_repo_id, token=token, **kwargs)
    else:
        return Tool.from_hub(task_or_repo_id, model_repo_id=model_repo_id, token=token, remote=remote, **kwargs)
# 定义一个装饰器，用于为函数添加描述
def add_description(description):
    """
    A decorator that adds a description to a function.
    """

    def inner(func):
        # 将描述信息添加到函数对象的属性中
        func.description = description
        func.name = func.__name__
        return func

    return inner


## Will move to the Hub
# 定义一个 EndpointClient 类
class EndpointClient:
    def __init__(self, endpoint_url: str, token: Optional[str] = None):
        # 构建请求头信息
        self.headers = {**build_hf_headers(token=token), "Content-Type": "application/json"}
        self.endpoint_url = endpoint_url

    @staticmethod
    # 将图像编码为 base64 格式
    def encode_image(image):
        _bytes = io.BytesIO()
        image.save(_bytes, format="PNG")
        b64 = base64.b64encode(_bytes.getvalue())
        return b64.decode("utf-8")

    @staticmethod
    # 将 base64 格式的图像解码为图像对象
    def decode_image(raw_image):
        if not is_vision_available():
            raise ImportError(
                "This tool returned an image but Pillow is not installed. Please install it (`pip install Pillow`)."
            )

        from PIL import Image

        b64 = base64.b64decode(raw_image)
        _bytes = io.BytesIO(b64)
        return Image.open(_bytes)

    def __call__(
        self,
        inputs: Optional[Union[str, Dict, List[str], List[List[str]]] = None,
        params: Optional[Dict] = None,
        data: Optional[bytes] = None,
        output_image: bool = False,
    ) -> Any:
        # 构建请求的 payload
        payload = {}
        if inputs:
            payload["inputs"] = inputs
        if params:
            payload["parameters"] = params

        # 发起 API 调用
        response = get_session().post(self.endpoint_url, headers=self.headers, json=payload, data=data)

        # 默认情况下，解析响应并返回给用户
        if output_image:
            return self.decode_image(response.content)
        else:
            return response.json()
```