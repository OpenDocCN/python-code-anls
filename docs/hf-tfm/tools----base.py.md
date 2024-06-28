# `.\tools\base.py`

```py
# 指定脚本的解释器环境为 Python，并设置编码为 UTF-8

# 导入所需的标准库和第三方库
import base64  # 导入 base64 编解码模块
import importlib  # 导入动态导入模块的模块
import inspect  # 导入检查对象信息的模块
import io  # 导入处理文件流的模块
import json  # 导入处理 JSON 数据的模块
import os  # 导入操作系统相关功能的模块
import tempfile  # 导入临时文件和目录创建功能的模块
from typing import Any, Dict, List, Optional, Union  # 导入类型提示相关模块

# 导入 Hugging Face Hub 相关功能模块
from huggingface_hub import create_repo, hf_hub_download, metadata_update, upload_folder
from huggingface_hub.utils import RepositoryNotFoundError, build_hf_headers, get_session

# 导入自定义模块
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

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 如果 torch 可用，则导入 torch 库
if is_torch_available():
    import torch

# 如果 accelerate 可用，则导入相关功能
if is_accelerate_available():
    from accelerate import PartialState
    from accelerate.utils import send_to_device

# 定义工具配置文件名
TOOL_CONFIG_FILE = "tool_config.json"


# 定义函数：根据 repo_id 获取仓库类型
def get_repo_type(repo_id, repo_type=None, **hub_kwargs):
    # 如果已提供 repo_type，则直接返回
    if repo_type is not None:
        return repo_type
    
    # 尝试下载 repo_id 的配置文件，类型为 "space"
    try:
        hf_hub_download(repo_id, TOOL_CONFIG_FILE, repo_type="space", **hub_kwargs)
        return "space"
    # 如果找不到指定仓库
    except RepositoryNotFoundError:
        # 尝试下载 repo_id 的配置文件，类型为 "model"
        try:
            hf_hub_download(repo_id, TOOL_CONFIG_FILE, repo_type="model", **hub_kwargs)
            return "model"
        # 如果仍然找不到指定仓库，则抛出环境错误
        except RepositoryNotFoundError:
            raise EnvironmentError(f"`{repo_id}` does not seem to be a valid repo identifier on the Hub.")
        # 如果下载过程中出现异常，则默认返回 "model" 类型
        except Exception:
            return "model"
    # 如果下载过程中出现异常，则默认返回 "space" 类型
    except Exception:
        return "space"


# 定义多行字符串模板，用于生成应用文件内容
# docstyle-ignore
APP_FILE_TEMPLATE = """from transformers import launch_gradio_demo
from {module_name} import {class_name}

launch_gradio_demo({class_name})
"""


# 定义工具类：代表代理函数使用的基类
class Tool:
    """
    代理函数使用的基类，实现 `__call__` 方法以及以下类属性：

    - **description** (`str`) -- 工具功能的简要描述，包括预期的输入和输出。例如，'这是一个从 `url` 下载文件的工具。它接受 `url` 作为输入，并返回文件中的文本内容'。
    """
    # 定义一个工具的类，表示一个用户定义的工具
    class Tool:
        # 描述工具的说明
        description: str = "This is a tool that ..."
        # 工具的名称
        name: str = ""
        # 工具接受的输入数据的模态列表
        inputs: List[str]
        # 工具返回的输出数据的模态列表
        outputs: List[str]
    
        # 初始化方法，接受任意数量的位置参数和关键字参数
        def __init__(self, *args, **kwargs):
            # 初始化时标记工具未被初始化
            self.is_initialized = False
    
        # 调用方法，接受任意数量的位置参数和关键字参数
        def __call__(self, *args, **kwargs):
            # 如果未在子类中实现__call__方法，则返回未实现错误
            return NotImplemented("Write this method in your subclass of `Tool`.")
    
        # 执行初始化的方法，用于在使用工具之前执行一些昂贵的操作，比如加载大型模型
        def setup(self):
            # 标记工具被初始化
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

        # Check if the tool's class is defined in the __main__ module
        if self.__module__ == "__main__":
            raise ValueError(
                f"We can't save the code defining {self} in {output_dir} as it's been defined in __main__. You "
                "have to put this code in a separate module so we can include it in the saved folder."
            )

        # Save the module files using a custom function
        module_files = custom_object_save(self, output_dir)

        # Get the name of the module containing the class
        module_name = self.__class__.__module__
        last_module = module_name.split(".")[-1]
        full_name = f"{last_module}.{self.__class__.__name__}"

        # Save or update the tool's configuration file
        config_file = os.path.join(output_dir, "tool_config.json")
        if os.path.isfile(config_file):
            # Load existing configuration if file already exists
            with open(config_file, "r", encoding="utf-8") as f:
                tool_config = json.load(f)
        else:
            tool_config = {}

        # Update tool configuration with class information
        tool_config = {"tool_class": full_name, "description": self.description, "name": self.name}
        with open(config_file, "w", encoding="utf-8") as f:
            # Write updated configuration to file in a human-readable format
            f.write(json.dumps(tool_config, indent=2, sort_keys=True) + "\n")

        # Save the app.py file using a template specific to the tool
        app_file = os.path.join(output_dir, "app.py")
        with open(app_file, "w", encoding="utf-8") as f:
            # Write the app file content based on a predefined template
            f.write(APP_FILE_TEMPLATE.format(module_name=last_module, class_name=self.__class__.__name__))

        # Save the requirements.txt file listing all dependencies
        requirements_file = os.path.join(output_dir, "requirements.txt")
        imports = []
        for module in module_files:
            # Gather all imports used by the modules of the tool
            imports.extend(get_imports(module))
        imports = list(set(imports))  # Ensure uniqueness of imports
        with open(requirements_file, "w", encoding="utf-8") as f:
            # Write each import as a separate line in the requirements file
            f.write("\n".join(imports) + "\n")
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
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether or not to create a PR with the uploaded files or directly commit.
        """
        # 创建仓库并获取仓库 URL
        repo_url = create_repo(
            repo_id=repo_id, token=token, private=private, exist_ok=True, repo_type="space", space_sdk="gradio"
        )
        # 更新仓库的元数据，添加标签 "tool"
        repo_id = repo_url.repo_id
        metadata_update(repo_id, {"tags": ["tool"]}, repo_type="space")

        # 使用临时目录来保存文件
        with tempfile.TemporaryDirectory() as work_dir:
            # 保存所有文件到临时目录
            self.save(work_dir)
            # 记录日志，显示将要上传的文件列表
            logger.info(f"Uploading the following files to {repo_id}: {','.join(os.listdir(work_dir))}")
            # 调用上传函数，将临时目录中的文件夹上传到指定仓库
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
        # 定义一个内部类 GradioToolWrapper，继承自 Tool
        class GradioToolWrapper(Tool):
            def __init__(self, _gradio_tool):
                super().__init__()
                # 初始化名称和描述
                self.name = _gradio_tool.name
                self.description = _gradio_tool.description

        # 将 GradioToolWrapper 的 __call__ 方法设置为 gradio_tool 的 run 方法
        GradioToolWrapper.__call__ = gradio_tool.run
        # 返回创建的 GradioToolWrapper 实例，该实例包装了 gradio_tool
        return GradioToolWrapper(gradio_tool)
# 定义一个名为 RemoteTool 的类，继承自 Tool 类
class RemoteTool(Tool):
    """
    A [`Tool`] that will make requests to an inference endpoint.

    Args:
        endpoint_url (`str`, *optional*):
            The url of the endpoint to use.
        token (`str`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated when
            running `huggingface-cli login` (stored in `~/.huggingface`).
        tool_class (`type`, *optional`):
            The corresponding `tool_class` if this is a remote version of an existing tool. Will help determine when
            the output should be converted to another type (like images).
    """

    # 初始化方法，接收三个可选参数：endpoint_url, token, tool_class
    def __init__(self, endpoint_url=None, token=None, tool_class=None):
        # 设置实例变量 endpoint_url，用于存储端点 URL
        self.endpoint_url = endpoint_url
        # 创建 EndpointClient 对象并存储在实例变量 client 中，用于处理与端点的通信
        self.client = EndpointClient(endpoint_url, token=token)
        # 设置实例变量 tool_class，用于存储工具类别信息
        self.tool_class = tool_class

    # 准备输入数据的方法，接收任意数量的位置参数和关键字参数
    def prepare_inputs(self, *args, **kwargs):
        """
        Prepare the inputs received for the HTTP client sending data to the endpoint. Positional arguments will be
        matched with the signature of the `tool_class` if it was provided at instantiation. Images will be encoded into
        bytes.

        You can override this method in your custom class of [`RemoteTool`].
        """
        # 复制关键字参数到 inputs 字典中
        inputs = kwargs.copy()

        # 如果有位置参数传入
        if len(args) > 0:
            # 如果指定了 tool_class
            if self.tool_class is not None:
                # 匹配位置参数与 tool_class 方法签名
                if issubclass(self.tool_class, PipelineTool):
                    call_method = self.tool_class.encode
                else:
                    call_method = self.tool_class.__call__
                signature = inspect.signature(call_method).parameters
                # 获取方法的参数名，排除可变位置参数和可变关键字参数
                parameters = [
                    k
                    for k, p in signature.items()
                    if p.kind not in [inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD]
                ]
                # 如果方法的第一个参数是 self，则去掉
                if parameters[0] == "self":
                    parameters = parameters[1:]
                # 如果传入的参数多于方法要求的参数数量，抛出 ValueError 异常
                if len(args) > len(parameters):
                    raise ValueError(
                        f"{self.tool_class} only accepts {len(parameters)} arguments but {len(args)} were given."
                    )
                # 将位置参数与参数名对应存入 inputs 字典中
                for arg, name in zip(args, parameters):
                    inputs[name] = arg
            # 如果未指定 tool_class，但传入了多个位置参数，抛出 ValueError 异常
            elif len(args) > 1:
                raise ValueError("A `RemoteTool` can only accept one positional input.")
            # 如果只有一个位置参数，并且是 PIL 图像，则编码为字节流放入 "inputs" 键中
            elif len(args) == 1:
                if is_pil_image(args[0]):
                    return {"inputs": self.client.encode_image(args[0])}
                return {"inputs": args[0]}

        # 对 inputs 中的每个值进行检查，如果是 PIL 图像，则编码为字节流
        for key, value in inputs.items():
            if is_pil_image(value):
                inputs[key] = self.client.encode_image(value)

        # 返回包含编码后数据的字典，键为 "inputs"
        return {"inputs": inputs}
    # 定义一个方法 `extract_outputs`，用于处理端点输出的自定义后处理逻辑
    def extract_outputs(self, outputs):
        """
        You can override this method in your custom class of [`RemoteTool`] to apply some custom post-processing of the
        outputs of the endpoint.
        """
        # 默认情况下，直接返回输出结果
        return outputs

    # 定义 `__call__` 方法，使对象可以像函数一样被调用
    def __call__(self, *args, **kwargs):
        # 处理传入的参数，确保它们符合要求
        args, kwargs = handle_agent_inputs(*args, **kwargs)

        # 检查是否需要输出图片，并准备输入数据
        output_image = self.tool_class is not None and self.tool_class.outputs == ["image"]
        inputs = self.prepare_inputs(*args, **kwargs)

        # 根据输入的类型调用客户端方法，并传递需要输出图片的信息
        if isinstance(inputs, dict):
            outputs = self.client(**inputs, output_image=output_image)
        else:
            outputs = self.client(inputs, output_image=output_image)

        # 如果输出是一个嵌套列表，并且只有一个元素，将其解包
        if isinstance(outputs, list) and len(outputs) == 1 and isinstance(outputs[0], list):
            outputs = outputs[0]

        # 处理从客户端获取的输出，应用工具类定义的输出规范（如果有的话）
        outputs = handle_agent_outputs(outputs, self.tool_class.outputs if self.tool_class is not None else None)

        # 调用 `extract_outputs` 方法处理最终的输出结果，并返回处理后的结果
        return self.extract_outputs(outputs)
# 定义一个 PipelineTool 类，继承自 Tool 类，用于处理 Transformer 模型相关的工具功能
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

    # 默认的预处理器类为 AutoProcessor
    pre_processor_class = AutoProcessor
    # 模型类，需要根据具体情况指定
    model_class = None
    # 默认的后处理器类也为 AutoProcessor
    post_processor_class = AutoProcessor
    # 默认的检查点名称，当用户未指定时应使用该值
    default_checkpoint = None

    # 初始化方法，接收多个可选参数以配置工具的行为
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
        # 检查是否安装了 Torch 库，如果未安装则抛出 ImportError 异常
        if not is_torch_available():
            raise ImportError("Please install torch in order to use this tool.")

        # 检查是否安装了 Accelerate 库，如果未安装则抛出 ImportError 异常
        if not is_accelerate_available():
            raise ImportError("Please install accelerate in order to use this tool.")

        # 如果未提供模型，则尝试使用默认的检查点，如果默认检查点未设置，则抛出 ValueError 异常
        if model is None:
            if self.default_checkpoint is None:
                raise ValueError("This tool does not implement a default checkpoint, you need to pass one.")
            model = self.default_checkpoint

        # 如果未提供预处理器，则使用模型作为预处理器
        if pre_processor is None:
            pre_processor = model

        # 设置对象的模型、预处理器、后处理器、设备、设备映射和模型参数
        self.model = model
        self.pre_processor = pre_processor
        self.post_processor = post_processor
        self.device = device
        self.device_map = device_map
        self.model_kwargs = {} if model_kwargs is None else model_kwargs

        # 如果设备映射不为空，则将其添加到模型参数中
        if device_map is not None:
            self.model_kwargs["device_map"] = device_map

        # 将 hub_kwargs 参数添加到对象的属性中
        self.hub_kwargs = hub_kwargs
        self.hub_kwargs["token"] = token

        # 调用父类的构造函数
        super().__init__()

    def setup(self):
        """
        Instantiates the `pre_processor`, `model` and `post_processor` if necessary.
        """
        # 如果预处理器是字符串，则根据预训练模型名称实例化预处理器对象
        if isinstance(self.pre_processor, str):
            self.pre_processor = self.pre_processor_class.from_pretrained(self.pre_processor, **self.hub_kwargs)

        # 如果模型是字符串，则根据预训练模型名称实例化模型对象
        if isinstance(self.model, str):
            self.model = self.model_class.from_pretrained(self.model, **self.model_kwargs, **self.hub_kwargs)

        # 如果未指定后处理器，则使用预处理器作为后处理器
        if self.post_processor is None:
            self.post_processor = self.pre_processor
        # 如果后处理器是字符串，则根据预训练模型名称实例化后处理器对象
        elif isinstance(self.post_processor, str):
            self.post_processor = self.post_processor_class.from_pretrained(self.post_processor, **self.hub_kwargs)

        # 如果设备未指定，则根据模型的设备映射设置设备
        if self.device is None:
            if self.device_map is not None:
                self.device = list(self.model.hf_device_map.values())[0]
            else:
                self.device = PartialState().default_device

        # 如果设备映射为空，则将模型移动到设备
        if self.device_map is None:
            self.model.to(self.device)

        # 调用父类的 setup 方法
        super().setup()

    def encode(self, raw_inputs):
        """
        Uses the `pre_processor` to prepare the inputs for the `model`.
        """
        # 使用预处理器对原始输入进行编码
        return self.pre_processor(raw_inputs)

    def forward(self, inputs):
        """
        Sends the inputs through the `model`.
        """
        # 使用模型处理输入数据，并返回输出结果
        with torch.no_grad():
            return self.model(**inputs)

    def decode(self, outputs):
        """
        Uses the `post_processor` to decode the model output.
        """
        # 使用后处理器对模型输出进行解码
        return self.post_processor(outputs)
    # 定义对象的调用方法，接受任意位置参数和关键字参数
    def __call__(self, *args, **kwargs):
        # 使用辅助函数处理输入参数，返回处理后的 args 和 kwargs
        args, kwargs = handle_agent_inputs(*args, **kwargs)

        # 如果对象尚未初始化，则调用 setup 方法进行初始化
        if not self.is_initialized:
            self.setup()

        # 对输入参数进行编码处理，返回编码后的结果
        encoded_inputs = self.encode(*args, **kwargs)
        # 将编码后的输入数据发送到指定设备上
        encoded_inputs = send_to_device(encoded_inputs, self.device)
        # 调用对象的 forward 方法进行前向传播，得到输出结果
        outputs = self.forward(encoded_inputs)
        # 将输出结果发送回 CPU
        outputs = send_to_device(outputs, "cpu")
        # 对输出结果进行解码处理，得到最终的解码输出
        decoded_outputs = self.decode(outputs)

        # 使用辅助函数处理解码后的输出，并返回处理后的结果
        return handle_agent_outputs(decoded_outputs, self.outputs)
# 启动一个 gradio 演示界面，展示特定工具的功能。该工具类需要正确实现类属性 `inputs` 和 `outputs`。
def launch_gradio_demo(tool_class: Tool):
    try:
        import gradio as gr  # 尝试导入 gradio 库
    except ImportError:
        raise ImportError("Gradio 应该安装才能启动 gradio 演示。")

    tool = tool_class()  # 实例化给定的工具类对象

    # 定义一个函数 fn，用来调用工具类实例的 __call__ 方法
    def fn(*args, **kwargs):
        return tool(*args, **kwargs)

    # 创建一个 gr.Interface 对象，配置输入输出和界面的标题和文章描述
    gr.Interface(
        fn=fn,
        inputs=tool_class.inputs,
        outputs=tool_class.outputs,
        title=tool_class.__name__,
        article=tool.description,
    ).launch()  # 启动 gradio 演示界面


# 支持的任务映射关系，将任务 ID 映射到工具类的字符串名称
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


def get_default_endpoints():
    # 获取默认的端点配置文件，并读取其中的端点信息
    endpoints_file = cached_file("huggingface-tools/default-endpoints", "default_endpoints.json", repo_type="dataset")
    with open(endpoints_file, "r", encoding="utf-8") as f:
        endpoints = json.load(f)  # 解析 JSON 文件中的端点配置信息
    return endpoints


def supports_remote(task_or_repo_id):
    endpoints = get_default_endpoints()  # 获取默认的端点信息
    return task_or_repo_id in endpoints  # 判断给定的任务或库 ID 是否存在于端点信息中


def load_tool(task_or_repo_id, model_repo_id=None, remote=False, token=None, **kwargs):
    """
    主函数，快速加载一个工具，无论是在 Hub 上还是在 Transformers 库中。

    <Tip warning={true}>

    加载工具意味着你会下载并在本地执行该工具。
    在加载到运行时之前，始终检查你要下载的工具，就像使用 pip/npm/apt 安装软件包时一样。

    </Tip>
    """
    # 如果给定的任务或模型ID在任务映射中已定义
    if task_or_repo_id in TASK_MAPPING:
        # 获取任务对应的工具类名
        tool_class_name = TASK_MAPPING[task_or_repo_id]
        # 动态导入transformers主模块
        main_module = importlib.import_module("transformers")
        # 获取tools子模块
        tools_module = main_module.tools
        # 根据工具类名获取具体的工具类对象
        tool_class = getattr(tools_module, tool_class_name)

        # 如果选择远程加载模型
        if remote:
            # 如果未提供model_repo_id，则获取默认的端点
            if model_repo_id is None:
                endpoints = get_default_endpoints()
                # 如果任务或模型ID不在默认端点列表中，则抛出值错误
                if task_or_repo_id not in endpoints:
                    raise ValueError(
                        f"Could not infer a default endpoint for {task_or_repo_id}, you need to pass one using the "
                        "`model_repo_id` argument."
                    )
                # 使用获取到的默认端点作为模型仓库ID
                model_repo_id = endpoints[task_or_repo_id]
            # 返回一个远程工具对象，包括模型仓库ID和token
            return RemoteTool(model_repo_id, token=token, tool_class=tool_class)
        else:
            # 直接实例化本地工具对象，传入模型仓库ID和额外的关键字参数kwargs
            return tool_class(model_repo_id, token=token, **kwargs)
    else:
        # 如果任务或模型ID未定义在任务映射中，则发出警告
        logger.warning_once(
            f"You're loading a tool from the Hub from {model_repo_id}. Please make sure this is a source that you "
            f"trust as the code within that tool will be executed on your machine. Always verify the code of "
            f"the tools that you load. We recommend specifying a `revision` to ensure you're loading the "
            f"code that you have checked."
        )
        # 从Hub加载工具对象，传入任务或模型ID、模型仓库ID、token、远程标志和其他关键字参数kwargs
        return Tool.from_hub(task_or_repo_id, model_repo_id=model_repo_id, token=token, remote=remote, **kwargs)
# 为函数添加描述信息的装饰器
def add_description(description):
    """
    A decorator that adds a description to a function.
    """

    def inner(func):
        # 将描述信息添加到函数对象的属性中
        func.description = description
        # 记录函数的名称
        func.name = func.__name__
        return func

    return inner


## Will move to the Hub
# EndpointClient 类，用于管理与端点通信的客户端
class EndpointClient:
    def __init__(self, endpoint_url: str, token: Optional[str] = None):
        # 构建 HTTP 请求头部信息，包括访问令牌
        self.headers = {**build_hf_headers(token=token), "Content-Type": "application/json"}
        # 记录端点的 URL 地址
        self.endpoint_url = endpoint_url

    @staticmethod
    def encode_image(image):
        # 将图像编码为 PNG 格式的 Base64 字符串
        _bytes = io.BytesIO()
        image.save(_bytes, format="PNG")
        b64 = base64.b64encode(_bytes.getvalue())
        return b64.decode("utf-8")

    @staticmethod
    def decode_image(raw_image):
        # 解码 Base64 字符串为图像对象
        if not is_vision_available():
            # 如果 Pillow 库不可用，抛出 ImportError 异常
            raise ImportError(
                "This tool returned an image but Pillow is not installed. Please install it (`pip install Pillow`)."
            )

        from PIL import Image

        b64 = base64.b64decode(raw_image)
        _bytes = io.BytesIO(b64)
        return Image.open(_bytes)

    def __call__(
        self,
        inputs: Optional[Union[str, Dict, List[str], List[List[str]]]] = None,
        params: Optional[Dict] = None,
        data: Optional[bytes] = None,
        output_image: bool = False,
    ) -> Any:
        # 构建请求的有效负载
        payload = {}
        if inputs:
            payload["inputs"] = inputs
        if params:
            payload["parameters"] = params

        # 发起 API 调用
        response = get_session().post(self.endpoint_url, headers=self.headers, json=payload, data=data)

        # 根据需要输出图像或解析 JSON 响应
        if output_image:
            # 如果需要输出图像，则解码 API 响应的图像数据
            return self.decode_image(response.content)
        else:
            # 否则解析 API 响应的 JSON 数据
            return response.json()
```