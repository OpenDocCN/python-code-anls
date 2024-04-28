# `.\transformers\tools\agent_types.py`

```py
# 导入必要的库
import os  # 导入操作系统模块
import pathlib  # 导入路径操作模块
import tempfile  # 导入临时文件模块
import uuid  # 导入 UUID 模块
import numpy as np  # 导入 NumPy 库

# 导入自定义的工具函数和日志记录模块
from ..utils import is_soundfile_availble, is_torch_available, is_vision_available, logging  

# 获取日志记录器
logger = logging.get_logger(__name__)

# 如果可用，导入 PIL 库以及相关子模块
if is_vision_available():
    import PIL.Image  # 导入 PIL 图像模块
    from PIL import Image  # 导入 PIL 图像模块的 Image 子模块
    from PIL.Image import Image as ImageType  # 导入 PIL 图像类型

# 否则将 ImageType 设置为 object 类型
else:
    ImageType = object

# 如果可用，导入 PyTorch 库
if is_torch_available():
    import torch  # 导入 PyTorch 库

# 如果可用，导入 soundfile 库
if is_soundfile_availble():
    import soundfile as sf  # 导入 soundfile 库

# 定义抽象类 AgentType
class AgentType:
    """
    Abstract class to be reimplemented to define types that can be returned by agents.

    These objects serve three purposes:

    - They behave as they were the type they're meant to be, e.g., a string for text, a PIL.Image for images
    - They can be stringified: str(object) in order to return a string defining the object
    - They should be displayed correctly in ipython notebooks/colab/jupyter
    """

    # 初始化方法，接受一个 value 参数
    def __init__(self, value):
        # 将 value 赋值给私有属性 _value
        self._value = value

    # 转换为字符串的方法
    def __str__(self):
        return self.to_string()

    # 返回原始值的方法
    def to_raw(self):
        logger.error(
            "This is a raw AgentType of unknown type. Display in notebooks and string conversion will be unreliable"
        )
        return self._value

    # 转换为字符串的方法
    def to_string(self) -> str:
        logger.error(
            "This is a raw AgentType of unknown type. Display in notebooks and string conversion will be unreliable"
        )
        return str(self._value)

# 定义文本类型 AgentText，继承自 AgentType 和 str 类
class AgentText(AgentType, str):
    """
    Text type returned by the agent. Behaves as a string.
    """

    # 返回原始值的方法
    def to_raw(self):
        return self._value

    # 转换为字符串的方法
    def to_string(self):
        return self._value

# 定义图像类型 AgentImage，继承自 AgentType 和 ImageType 类
class AgentImage(AgentType, ImageType):
    """
    Image type returned by the agent. Behaves as a PIL.Image.
    """

    # 初始化方法，接受一个 value 参数
    def __init__(self, value):
        # 调用父类的初始化方法，并传入 value 参数
        super().__init__(value)

        # 如果 PIL 库可用
        if not is_vision_available():
            # 抛出 ImportError 异常
            raise ImportError("PIL must be installed in order to handle images.")

        # 初始化路径、原始图像和张量
        self._path = None
        self._raw = None
        self._tensor = None

        # 检查 value 的类型并进行相应的处理
        if isinstance(value, ImageType):
            # 如果 value 是 ImageType 类型，直接赋值给 _raw
            self._raw = value
        elif isinstance(value, (str, pathlib.Path)):
            # 如果 value 是字符串或路径对象，将其赋值给 _path
            self._path = value
        elif isinstance(value, torch.Tensor):
            # 如果 value 是张量，将其赋值给 _tensor
            self._tensor = value
        else:
            # 如果 value 类型不受支持，抛出 ValueError 异常
            raise ValueError(f"Unsupported type for {self.__class__.__name__}: {type(value)}")
    # 在 ipython notebook 中正确显示这种类型的对象
    def _ipython_display_(self, include=None, exclude=None):
        """
        Displays correctly this type in an ipython notebook (ipython, colab, jupyter, ...)
        """
        # 导入 IPython.display 模块的 Image 和 display 函数
        from IPython.display import Image, display

        # 调用 display 函数显示对象的字符串表示形式
        display(Image(self.to_string()))

    # 返回对象的原始版本。对于 AgentImage 对象来说，是一个 PIL.Image 对象
    def to_raw(self):
        """
        Returns the "raw" version of that object. In the case of an AgentImage, it is a PIL.Image.
        """
        # 如果已经有原始版本则直接返回
        if self._raw is not None:
            return self._raw

        # 如果有路径，使用 Image.open 方法打开图片并返回
        if self._path is not None:
            self._raw = Image.open(self._path)
            return self._raw

    # 返回对象的字符串表示形式。对于 AgentImage 对象来说，是图像的序列化版本的路径
    def to_string(self):
        """
        Returns the stringified version of that object. In the case of an AgentImage, it is a path to the serialized
        version of the image.
        """
        # 如果有路径，直接返回路径
        if self._path is not None:
            return self._path

        # 如果有原始版本，将原始版本保存到临时文件夹中，并返回路径
        if self._raw is not None:
            directory = tempfile.mkdtemp()
            self._path = os.path.join(directory, str(uuid.uuid4()) + ".png")
            self._raw.save(self._path)

            return self._path

        # 如果有张量，将张量转换为图像并保存到临时文件夹中，返回路径
        if self._tensor is not None:
            array = self._tensor.cpu().detach().numpy()

            # 可以简化操作，先把张量转换成图像再保存
            img = Image.fromarray((array * 255).astype(np.uint8))

            directory = tempfile.mkdtemp()
            self._path = os.path.join(directory, str(uuid.uuid4()) + ".png")

            img.save(self._path)

            return self._path
# 定义 AgentAudio 类，继承 AgentType 类
class AgentAudio(AgentType):
    """
    Audio type returned by the agent.
    """

    # 初始化方法
    def __init__(self, value, samplerate=16_000):
        # 调用父类的初始化方法
        super().__init__(value)

        # 检查是否安装了 soundfile 库
        if not is_soundfile_availble():
            raise ImportError("soundfile must be installed in order to handle audio.")

        # 初始化属性
        self._path = None
        self._tensor = None
        self.samplerate = samplerate

        # 根据值的类型进行处理
        if isinstance(value, (str, pathlib.Path)):
            self._path = value
        elif isinstance(value, torch.Tensor):
            self._tensor = value
        else:
            raise ValueError(f"Unsupported audio type: {type(value)}")

    # 在 ipython notebook 中正确显示该类型的方法
    def _ipython_display_(self, include=None, exclude=None):
        """
        Displays correctly this type in an ipython notebook (ipython, colab, jupyter, ...)
        """
        from IPython.display import Audio, display

        display(Audio(self.to_string(), rate=self.samplerate))

    # 返回该对象的“原始”版本，即 `torch.Tensor` 对象
    def to_raw(self):
        """
        Returns the "raw" version of that object. It is a `torch.Tensor` object.
        """
        if self._tensor is not None:
            return self._tensor

        if self._path is not None:
            tensor, self.samplerate = sf.read(self._path)
            self._tensor = torch.tensor(tensor)
            return self._tensor

    # 返回该对象的字符串版本，在 AgentAudio 的情况下，是序列化音频的路径
    def to_string(self):
        """
        Returns the stringified version of that object. In the case of an AgentAudio, it is a path to the serialized
        version of the audio.
        """
        if self._path is not None:
            return self._path

        if self._tensor is not None:
            directory = tempfile.mkdtemp()
            self._path = os.path.join(directory, str(uuid.uuid4()) + ".wav")
            sf.write(self._path, self._tensor, samplerate=self.samplerate)
            return self._path

# 定义 AGENT_TYPE_MAPPING 和 INSTANCE_TYPE_MAPPING 字典
AGENT_TYPE_MAPPING = {"text": AgentText, "image": AgentImage, "audio": AgentAudio}
INSTANCE_TYPE_MAPPING = {str: AgentText}

# 如果安装了 vision 库，则将 PIL.Image 类添加到 INSTANCE_TYPE_MAPPING 字典中
if is_vision_available():
    INSTANCE_TYPE_MAPPING[PIL.Image] = AgentImage

# 处理 agent 输入的函数
def handle_agent_inputs(*args, **kwargs):
    args = [(arg.to_raw() if isinstance(arg, AgentType) else arg) for arg in args]
    kwargs = {k: (v.to_raw() if isinstance(v, AgentType) else v) for k, v in kwargs.items()}
    return args, kwargs

# 处理 agent 输出的函数
def handle_agent_outputs(outputs, output_types=None):
    # 检查输出是否为字典类型
    if isinstance(outputs, dict):
        # 初始化解码后的输出字典
        decoded_outputs = {}
        # 遍历字典中的键值对
        for i, (k, v) in enumerate(outputs.items()):
            # 如果存在输出类型
            if output_types is not None:
                # 如果类定义了输出类型，根据类定义直接映射
                if output_types[i] in AGENT_TYPE_MAPPING:
                    decoded_outputs[k] = AGENT_TYPE_MAPPING[output_types[i]](v)
                else:
                    decoded_outputs[k] = AgentType(v)

            else:
                # 如果类没有定义输出类型，根据值的类型进行映射
                for _k, _v in INSTANCE_TYPE_MAPPING.items():
                    if isinstance(v, _k):
                        decoded_outputs[k] = _v(v)
                # 如果没有找到匹配的类型，使用默认的 AgentType
                if k not in decoded_outputs:
                    decoded_outputs[k] = AgentType[v]

    # 如果输出是列表或元组类型
    elif isinstance(outputs, (list, tuple)):
        # 根据原始输出类型初始化解码后的输出
        decoded_outputs = type(outputs)()
        # 遍历列表或元组中的值
        for i, v in enumerate(outputs):
            # 如果存在输出类型
            if output_types is not None:
                # 如果类定义了输出类型，根据类定义直接映射
                if output_types[i] in AGENT_TYPE_MAPPING:
                    decoded_outputs.append(AGENT_TYPE_MAPPING[output_types[i]](v))
                else:
                    decoded_outputs.append(AgentType(v))
            else:
                # 如果类没有定义输出类型，根据值的类型进行映射
                found = False
                for _k, _v in INSTANCE_TYPE_MAPPING.items():
                    if isinstance(v, _k):
                        decoded_outputs.append(_v(v))
                        found = True
                # 如果没有找到匹配的类型，使用默认的 AgentType
                if not found:
                    decoded_outputs.append(AgentType(v))

    else:
        # 如果类定义了输出类型，根据类定义直接映射
        if output_types[0] in AGENT_TYPE_MAPPING:
            decoded_outputs = AGENT_TYPE_MAPPING[output_types[0]](outputs)
        else:
            # 如果类没有定义输出类型，根据值的类型进行映射
            for _k, _v in INSTANCE_TYPE_MAPPING.items():
                if isinstance(outputs, _k):
                    return _v(outputs)
            return AgentType(outputs)

    # 返回解码后的输出
    return decoded_outputs
```