# `D:\src\scipysrc\matplotlib\lib\matplotlib\animation.pyi`

```
# 导入 abc 模块，用于定义抽象基类
import abc
# 从 collections.abc 中导入 Callable、Collection、Iterable、Sequence、Generator 类
from collections.abc import Callable, Collection, Iterable, Sequence, Generator
# 导入 contextlib 模块，用于上下文管理
import contextlib
# 从 pathlib 模块导入 Path 类
from pathlib import Path
# 从 matplotlib.artist 模块导入 Artist 类
from matplotlib.artist import Artist
# 从 matplotlib.backend_bases 模块导入 TimerBase 类
from matplotlib.backend_bases import TimerBase
# 从 matplotlib.figure 模块导入 Figure 类
from matplotlib.figure import Figure

# 导入 Any 类型提示
from typing import Any

# 声明 subprocess_creation_flags 变量为整数类型
subprocess_creation_flags: int

# 声明 adjusted_figsize 函数，接受 w、h、dpi、n 四个参数，返回类型为元组[float, float]
def adjusted_figsize(w: float, h: float, dpi: float, n: int) -> tuple[float, float]: ...

# 定义 MovieWriterRegistry 类
class MovieWriterRegistry:
    # 初始化方法
    def __init__(self) -> None: ...
    # 注册方法，参数 name 为字符串，返回 Callable 对象
    def register(
        self, name: str
    ) -> Callable[[type[AbstractMovieWriter]], type[AbstractMovieWriter]]: ...
    # 判断指定名称的写入器是否可用，参数 name 为字符串，返回布尔值
    def is_available(self, name: str) -> bool: ...
    # 迭代器方法，返回字符串类型的生成器
    def __iter__(self) -> Generator[str, None, None]: ...
    # 返回所有可用写入器名称的列表
    def list(self) -> list[str]: ...
    # 根据名称获取对应的 AbstractMovieWriter 类型
    def __getitem__(self, name: str) -> type[AbstractMovieWriter]: ...

# 声明 writers 变量为 MovieWriterRegistry 类型
writers: MovieWriterRegistry

# 定义 AbstractMovieWriter 抽象基类
class AbstractMovieWriter(abc.ABC, metaclass=abc.ABCMeta):
    # 帧率属性 fps，整数类型
    fps: int
    # 元数据属性 metadata，字典，键和值为字符串类型
    metadata: dict[str, str]
    # 编解码器属性 codec，字符串类型
    codec: str
    # 比特率属性 bitrate，整数类型
    bitrate: int
    # 初始化方法，参数包括 fps、metadata、codec、bitrate 四个，类型见注释
    def __init__(
        self,
        fps: int = ...,
        metadata: dict[str, str] | None = ...,
        codec: str | None = ...,
        bitrate: int | None = ...,
    ) -> None: ...
    # 输出文件路径 outfile，可以是字符串或 Path 类型
    outfile: str | Path
    # 图形对象 fig，类型为 Figure
    fig: Figure
    # DPI 属性 dpi，浮点数类型
    dpi: float

    # 抽象方法 setup，设置写入器的环境，参数包括 fig（Figure 对象）、outfile（输出文件路径）、dpi（DPI 值）
    @abc.abstractmethod
    def setup(self, fig: Figure, outfile: str | Path, dpi: float | None = ...) -> None: ...
    # 抽象方法 frame_size，返回元组[int, int]，表示帧的尺寸
    @property
    def frame_size(self) -> tuple[int, int]: ...
    # 抽象方法 grab_frame，抓取帧的方法，接受保存帧的关键字参数
    @abc.abstractmethod
    def grab_frame(self, **savefig_kwargs) -> None: ...
    # 抽象方法 finish，完成写入的方法
    @abc.abstractmethod
    def finish(self) -> None: ...
    # 上下文管理器 saving 方法，用于保存帧，参数包括 fig（Figure 对象）、outfile（输出文件路径）、dpi（DPI 值）
    @contextlib.contextmanager
    def saving(
        self, fig: Figure, outfile: str | Path, dpi: float | None, *args, **kwargs
    ) -> Generator[AbstractMovieWriter, None, None]: ...

# 定义 MovieWriter 类，继承自 AbstractMovieWriter
class MovieWriter(AbstractMovieWriter):
    # 支持的格式列表 supported_formats，元素为字符串类型
    supported_formats: list[str]
    # 帧格式 frame_format，字符串类型
    frame_format: str
    # 额外参数列表 extra_args，可以为 None，元素为字符串类型
    extra_args: list[str] | None
    # 初始化方法，参数包括 fps、codec、bitrate、extra_args、metadata
    def __init__(
        self,
        fps: int = ...,
        codec: str | None = ...,
        bitrate: int | None = ...,
        extra_args: list[str] | None = ...,
        metadata: dict[str, str] | None = ...,
    ) -> None: ...
    # 实现 setup 方法，设置写入器的环境，参数包括 fig（Figure 对象）、outfile（输出文件路径）、dpi（DPI 值）
    def setup(self, fig: Figure, outfile: str | Path, dpi: float | None = ...) -> None: ...
    # 实现 grab_frame 方法，抓取帧的方法，接受保存帧的关键字参数
    def grab_frame(self, **savefig_kwargs) -> None: ...
    # 实现 finish 方法，完成写入的方法
    def finish(self) -> None: ...
    # 类方法 bin_path，返回字符串类型，表示二进制路径
    @classmethod
    def bin_path(cls) -> str: ...
    # 类方法 isAvailable，返回布尔值，表示写入器是否可用
    @classmethod
    def isAvailable(cls) -> bool: ...

# 定义 FileMovieWriter 类，继承自 MovieWriter
class FileMovieWriter(MovieWriter):
    # 图形对象 fig，类型为 Figure
    fig: Figure
    # 输出文件路径 outfile，可以是字符串或 Path 类型
    outfile: str | Path
    # DPI 属性 dpi，浮点数类型
    dpi: float
    # 临时前缀 temp_prefix，字符串类型
    temp_prefix: str
    # 文件名格式字符串 fname_format_str，字符串类型
    fname_format_str: str
    # 实现 setup 方法，设置写入器的环境，参数包括 fig（Figure 对象）、outfile（输出文件路径）、dpi（DPI 值）、frame_prefix（帧前缀）
    def setup(
        self,
        fig: Figure,
        outfile: str | Path,
        dpi: float | None = ...,
        frame_prefix: str | Path | None = ...,
    ) -> None: ...
    # 析构方法 __del__，释放资源
    def __del__(self) -> None: ...
    # 属性方法 frame_format，返回帧格式字符串
    @property
    def frame_format(self) -> str: ...
    # frame_format 属性的 setter 方法，设置帧格式
    @frame_format.setter
    def frame_format(self, frame_format: str) -> None: ...

# 定义 PillowWriter 类，继承自 AbstractMovieWriter
class PillowWriter(AbstractMovieWriter):
    # 类方法 isAvailable，返回布尔值，表示写入器是否可用
    @classmethod
    def isAvailable(cls) -> bool: ...
    # 定义一个方法 `setup`，用于设置动画的绘图参数
    def setup(
        self, fig: Figure, outfile: str | Path, dpi: float | None = ...
    ) -> None: ...
    
    # 定义一个方法 `grab_frame`，用于捕获当前帧并保存为图像文件
    def grab_frame(self, **savefig_kwargs) -> None: ...
    
    # 定义一个方法 `finish`，用于完成动画的处理，可能包括清理和关闭资源等操作
    def finish(self) -> None: ...
class FFMpegBase:
    # 定义一个属性 codec，用于存储编解码器信息
    codec: str
    # 定义一个抽象属性 output_args，返回一个字符串列表，由子类实现具体逻辑
    @property
    def output_args(self) -> list[str]: ...

# FFMpegWriter 类继承自 FFMpegBase 和 MovieWriter 接口
class FFMpegWriter(FFMpegBase, MovieWriter): ...

# FFMpegFileWriter 类继承自 FFMpegBase 和 FileMovieWriter 接口
class FFMpegFileWriter(FFMpegBase, FileMovieWriter):
    # 定义一个支持的格式列表
    supported_formats: list[str]

# ImageMagickBase 类，提供 ImageMagick 相关操作的基础功能
class ImageMagickBase:
    # 类方法，返回 ImageMagick 的可执行文件路径
    @classmethod
    def bin_path(cls) -> str: ...
    # 类方法，检查 ImageMagick 是否可用
    @classmethod
    def isAvailable(cls) -> bool: ...

# ImageMagickWriter 类继承自 ImageMagickBase 和 MovieWriter 接口
class ImageMagickWriter(ImageMagickBase, MovieWriter):
    # 定义一个输入名称的属性
    input_names: str

# ImageMagickFileWriter 类继承自 ImageMagickBase 和 FileMovieWriter 接口
class ImageMagickFileWriter(ImageMagickBase, FileMovieWriter):
    # 定义一个支持的格式列表
    supported_formats: list[str]
    # 属性方法，返回输入名称的字符串
    @property
    def input_names(self) -> str: ...

# HTMLWriter 类继承自 FileMovieWriter 接口
class HTMLWriter(FileMovieWriter):
    # 定义一个支持的格式列表
    supported_formats: list[str]
    # 类方法，检查 HTMLWriter 是否可用
    @classmethod
    def isAvailable(cls) -> bool: ...
    # 嵌入帧的布尔属性
    embed_frames: bool
    # 默认模式的字符串属性
    default_mode: str
    # 初始化方法，设置 FPS、编解码器、比特率等参数
    def __init__(
        self,
        fps: int = ...,
        codec: str | None = ...,
        bitrate: int | None = ...,
        extra_args: list[str] | None = ...,
        metadata: dict[str, str] | None = ...,
        embed_frames: bool = ...,
        default_mode: str = ...,
        embed_limit: float | None = ...,
    ) -> None: ...
    # 设置方法，设置图形、输出文件名、分辨率等参数
    def setup(
        self,
        fig: Figure,
        outfile: str | Path,
        dpi: float | None = ...,
        frame_dir: str | Path | None = ...,
    ) -> None: ...
    # 抓取帧的方法，保存图形的关键字参数
    def grab_frame(self, **savefig_kwargs): ...
    # 结束方法，完成写入操作
    def finish(self) -> None: ...

# Animation 类，提供动画生成和管理的基础功能
class Animation:
    # 帧序列的迭代器属性
    frame_seq: Iterable[Artist]
    # 事件源的属性
    event_source: Any
    # 初始化方法，设置图形、事件源等参数
    def __init__(
        self, fig: Figure, event_source: Any | None = ..., blit: bool = ...
    ) -> None: ...
    # 析构方法，清理资源
    def __del__(self) -> None: ...
    # 保存方法，将动画保存为视频文件或其他格式
    def save(
        self,
        filename: str | Path,
        writer: AbstractMovieWriter | str | None = ...,
        fps: int | None = ...,
        dpi: float | None = ...,
        codec: str | None = ...,
        bitrate: int | None = ...,
        extra_args: list[str] | None = ...,
        metadata: dict[str, str] | None = ...,
        extra_anim: list[Animation] | None = ...,
        savefig_kwargs: dict[str, Any] | None = ...,
        *,
        progress_callback: Callable[[int, int], Any] | None = ...
    ) -> None: ...
    # 创建新的帧序列的方法
    def new_frame_seq(self) -> Iterable[Artist]: ...
    # 创建新的保存帧序列的方法
    def new_saved_frame_seq(self) -> Iterable[Artist]: ...
    # 转换为 HTML5 视频的方法，设置嵌入限制参数
    def to_html5_video(self, embed_limit: float | None = ...) -> str: ...
    # 转换为 JavaScript HTML 的方法，设置 FPS、帧嵌入等参数
    def to_jshtml(
        self,
        fps: int | None = ...,
        embed_frames: bool = ...,
        default_mode: str | None = ...,
    ) -> str: ...
    # 表示为 HTML 的方法，返回 HTML 字符串
    def _repr_html_(self) -> str: ...
    # 暂停方法，暂停动画播放
    def pause(self) -> None: ...
    # 恢复方法，恢复动画播放
    def resume(self) -> None: ...

# TimedAnimation 类继承自 Animation，提供基于时间的动画功能
class TimedAnimation(Animation):
    # 初始化方法，设置图形、帧间隔、重复延迟等参数
    def __init__(
        self,
        fig: Figure,
        interval: int = ...,
        repeat_delay: int = ...,
        repeat: bool = ...,
        event_source: TimerBase | None = ...,
        *args,
        **kwargs
    ) -> None: ...

# ArtistAnimation 类继承自 TimedAnimation，用于处理包含艺术家集合的动画
class ArtistAnimation(TimedAnimation):
    # 初始化方法，设置图形、艺术家集合等参数
    def __init__(self, fig: Figure, artists: Sequence[Collection[Artist]], *args, **kwargs) -> None: ...
# 定义 FuncAnimation 类，继承自 TimedAnimation 类
class FuncAnimation(TimedAnimation):
    
    # 初始化函数，接受多个参数
    def __init__(
        self,
        fig: Figure,  # 参数 fig：图形对象，用于绘制动画
        func: Callable[..., Iterable[Artist]],  # 参数 func：可调用对象，生成艺术家对象的迭代器
        frames: Iterable[Artist] | int | Callable[[], Generator] | None = ...,  # 参数 frames：帧的集合或总帧数或生成帧的可调用对象
        init_func: Callable[[], Iterable[Artist]] | None = ...,  # 参数 init_func：初始化函数，返回初始艺术家对象的迭代器
        fargs: tuple[Any, ...] | None = ...,  # 参数 fargs：传递给 func 和 init_func 的额外参数
        save_count: int | None = ...,  # 参数 save_count：指定缓存的帧数
        *,
        cache_frame_data: bool = ...,  # 参数 cache_frame_data：是否缓存帧数据
        **kwargs  # 其他关键字参数，传递给 TimedAnimation 的构造函数
    ) -> None:
        ...
```