# `D:\src\scipysrc\matplotlib\lib\matplotlib\animation.py`

```
# TODO:
# * Documentation -- this will need a new section of the User's Guide.
#      Both for Animations and just timers.
#   - Also need to update
#     https://scipy-cookbook.readthedocs.io/items/Matplotlib_Animations.html
# * Blit
#   * Currently broken with Qt4 for widgets that don't start on screen
#   * Still a few edge cases that aren't working correctly
#   * Can this integrate better with existing matplotlib animation artist flag?
#     - If animated removes from default draw(), perhaps we could use this to
#       simplify initial draw.
# * Example
#   * Frameless animation - pure procedural with no loop
#   * Need example that uses something like inotify or subprocess
#   * Complex syncing examples
# * Movies
#   * Can blit be enabled for movies?
# * Need to consider event sources to allow clicking through multiple figures

import abc  # 引入抽象基类模块
import base64  # 引入base64编解码模块
import contextlib  # 引入上下文管理模块
from io import BytesIO, TextIOWrapper  # 从io模块中引入BytesIO和TextIOWrapper类
import itertools  # 引入迭代工具模块
import logging  # 引入日志记录模块
from pathlib import Path  # 从pathlib模块引入Path类
import shutil  # 引入文件操作模块
import subprocess  # 引入子进程管理模块
import sys  # 引入系统相关模块
from tempfile import TemporaryDirectory  # 从tempfile模块引入TemporaryDirectory类
import uuid  # 引入UUID生成模块
import warnings  # 引入警告模块

import numpy as np  # 引入数值计算模块numpy
from PIL import Image  # 从PIL库引入Image类

import matplotlib as mpl  # 引入matplotlib绘图库
from matplotlib._animation_data import (
    DISPLAY_TEMPLATE, INCLUDED_FRAMES, JS_INCLUDE, STYLE_INCLUDE)  # 从matplotlib._animation_data模块引入常量
from matplotlib import _api, cbook  # 从matplotlib库引入内部API和兼容性模块cbook
import matplotlib.colors as mcolors  # 引入颜色管理模块matplotlib.colors

_log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象

# Process creation flag for subprocess to prevent it raising a terminal
# window. See for example https://stackoverflow.com/q/24130623/
subprocess_creation_flags = (
    subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)  # 根据系统平台设置子进程创建标志，用于防止弹出终端窗口

# Other potential writing methods:
# * http://pymedia.org/
# * libming (produces swf) python wrappers: https://github.com/libming/libming
# * Wrap x264 API:
# (https://stackoverflow.com/q/2940671/)

def adjusted_figsize(w, h, dpi, n):
    """
    Compute figure size so that pixels are a multiple of n.

    Parameters
    ----------
    w, h : float
        Size in inches.

    dpi : float
        The dpi.

    n : int
        The target multiple.

    Returns
    -------
    wnew, hnew : float
        The new figure size in inches.
    """

    # this maybe simplified if / when we adopt consistent rounding for
    # pixel size across the whole library
    def correct_roundoff(x, dpi, n):
        if int(x*dpi) % n != 0:
            if int(np.nextafter(x, np.inf)*dpi) % n == 0:
                x = np.nextafter(x, np.inf)
            elif int(np.nextafter(x, -np.inf)*dpi) % n == 0:
                x = np.nextafter(x, -np.inf)
        return x

    wnew = int(w * dpi / n) * n / dpi  # 计算调整后的宽度
    hnew = int(h * dpi / n) * n / dpi  # 计算调整后的高度
    return correct_roundoff(wnew, dpi, n), correct_roundoff(hnew, dpi, n)  # 返回调整后的尺寸


class MovieWriterRegistry:
    """Registry of available writer classes by human readable name."""

    def __init__(self):
        self._registered = dict()  # 初始化注册字典
    def register(self, name):
        """
        Decorator for registering a class under a name.

        Example use::

            @registry.register(name)
            class Foo:
                pass
        """
        # 定义一个装饰器函数，用于将类注册到给定名称下
        def wrapper(writer_cls):
            self._registered[name] = writer_cls  # 将类名映射到注册字典中的给定名称
            return writer_cls
        return wrapper

    def is_available(self, name):
        """
        Check if given writer is available by name.

        Parameters
        ----------
        name : str
            The name of the writer class to check availability.

        Returns
        -------
        bool
            True if the writer class is available under the given name, False otherwise.
        """
        try:
            cls = self._registered[name]  # 尝试从注册字典中获取给定名称对应的类
        except KeyError:
            return False  # 如果名称不存在于注册字典中，则返回 False
        return cls.isAvailable()  # 调用类的 isAvailable 方法，返回是否可用的布尔值

    def __iter__(self):
        """
        Iterate over names of available writer class.

        Yields
        ------
        str
            The name of each available writer class.
        """
        for name in self._registered:  # 遍历注册字典中的每个名称
            if self.is_available(name):  # 检查每个名称对应的类是否可用
                yield name  # 如果可用，则产生该名称

    def list(self):
        """
        Get a list of available MovieWriters.

        Returns
        -------
        list
            A list of names of available writer classes.
        """
        return [*self]  # 调用 __iter__ 方法获取所有可用 writer 类的名称，并返回列表形式

    def __getitem__(self, name):
        """
        Get an available writer class from its name.

        Parameters
        ----------
        name : str
            The name of the writer class to retrieve.

        Returns
        -------
        class
            The writer class associated with the given name.

        Raises
        ------
        RuntimeError
            If the requested writer class is not available.
        """
        if self.is_available(name):  # 检查给定名称的 writer 类是否可用
            return self._registered[name]  # 返回该名称对应的 writer 类
        raise RuntimeError(f"Requested MovieWriter ({name}) not available")  # 若不可用，则抛出运行时错误
writers = MovieWriterRegistry()

class AbstractMovieWriter(abc.ABC):
    """
    Abstract base class for writing movies, providing a way to grab frames by
    calling `~AbstractMovieWriter.grab_frame`.

    `setup` is called to start the process and `finish` is called afterwards.
    `saving` is provided as a context manager to facilitate this process as ::

        with moviewriter.saving(fig, outfile='myfile.mp4', dpi=100):
            # Iterate over frames
            moviewriter.grab_frame(**savefig_kwargs)

    The use of the context manager ensures that `setup` and `finish` are
    performed as necessary.

    An instance of a concrete subclass of this class can be given as the
    ``writer`` argument of `Animation.save()`.
    """

    def __init__(self, fps=5, metadata=None, codec=None, bitrate=None):
        # 初始化函数，设置帧率、元数据、编解码器和比特率
        self.fps = fps
        self.metadata = metadata if metadata is not None else {}
        self.codec = mpl._val_or_rc(codec, 'animation.codec')
        self.bitrate = mpl._val_or_rc(bitrate, 'animation.bitrate')

    @abc.abstractmethod
    def setup(self, fig, outfile, dpi=None):
        """
        Setup for writing the movie file.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
            The figure object that contains the information for frames.
        outfile : str
            The filename of the resulting movie file.
        dpi : float, default: ``fig.dpi``
            The DPI (or resolution) for the file.  This controls the size
            in pixels of the resulting movie file.
        """
        # 检查路径是否有效并解析
        Path(outfile).parent.resolve(strict=True)
        self.outfile = outfile
        self.fig = fig
        if dpi is None:
            dpi = self.fig.dpi
        self.dpi = dpi

    @property
    def frame_size(self):
        """A tuple ``(width, height)`` in pixels of a movie frame."""
        # 获取电影帧的像素大小，基于图表的英寸尺寸和 DPI
        w, h = self.fig.get_size_inches()
        return int(w * self.dpi), int(h * self.dpi)

    @abc.abstractmethod
    def grab_frame(self, **savefig_kwargs):
        """
        Grab the image information from the figure and save as a movie frame.

        All keyword arguments in *savefig_kwargs* are passed on to the
        `~.Figure.savefig` call that saves the figure.  However, several
        keyword arguments that are supported by `~.Figure.savefig` may not be
        passed as they are controlled by the MovieWriter:

        - *dpi*, *bbox_inches*:  These may not be passed because each frame of the
           animation much be exactly the same size in pixels.
        - *format*: This is controlled by the MovieWriter.
        """

    @abc.abstractmethod
    def finish(self):
        """Finish any processing for writing the movie."""

    @contextlib.contextmanager
    def saving(self, fig, outfile, dpi=None, *args, **kwargs):
        """
        Context manager for saving a sequence of frames as a movie.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
            The figure object that contains the information for frames.
        outfile : str
            The filename of the resulting movie file.
        dpi : float, optional
            The DPI (or resolution) for the file.  This controls the size
            in pixels of the resulting movie file.
        """
        try:
            # 调用 setup 方法设置电影文件的准备工作
            self.setup(fig, outfile, dpi=dpi)
            yield self
        finally:
            # 调用 finish 方法完成电影文件的写入处理
            self.finish()
    # 定义一个方法 `saving`，用于保存动画帧到文件
    def saving(self, fig, outfile, dpi, *args, **kwargs):
        """
        Context manager to facilitate writing the movie file.

        ``*args, **kw`` are any parameters that should be passed to `setup`.
        """
        # 检查是否设置了 `savefig.bbox` 为 'tight'，如果是则记录日志警告
        if mpl.rcParams['savefig.bbox'] == 'tight':
            _log.info("Disabling savefig.bbox = 'tight', as it may cause "
                      "frame size to vary, which is inappropriate for "
                      "animation.")

        # 调用 `setup` 方法来设置保存动画文件的参数，传入动态参数 `*args, **kwargs`
        self.setup(fig, outfile, dpi, *args, **kwargs)
        
        # 使用 `mpl.rc_context` 创建一个上下文管理器，临时修改 `savefig.bbox` 的设置为 `None`
        with mpl.rc_context({'savefig.bbox': None}):
            try:
                # 使用 `yield self` 将当前对象作为上下文管理器的返回值，用于生成器调用
                yield self
            finally:
                # 调用 `finish` 方法完成保存操作
                self.finish()
# MovieWriter 类，继承自 AbstractMovieWriter 抽象类，用于处理影片写入操作。

class MovieWriter(AbstractMovieWriter):
    """
    Base class for writing movies.

    This is a base class for MovieWriter subclasses that write a movie frame
    data to a pipe. You cannot instantiate this class directly.
    See examples for how to use its subclasses.

    Attributes
    ----------
    frame_format : str
        The format used in writing frame data, defaults to 'rgba'.
    fig : `~matplotlib.figure.Figure`
        The figure to capture data from.
        This must be provided by the subclasses.
    """

    # Builtin writer subclasses additionally define the _exec_key and _args_key
    # attributes, which indicate the rcParams entries where the path to the
    # executable and additional command-line arguments to the executable are
    # stored.  Third-party writers cannot meaningfully set these as they cannot
    # extend rcParams with new keys.

    # Pipe-based writers only support RGBA, but file-based ones support more
    # formats.
    
    # 管道写入类型仅支持 'rgba' 格式，而文件写入类型支持更多格式。
    supported_formats = ["rgba"]

    def __init__(self, fps=5, codec=None, bitrate=None, extra_args=None,
                 metadata=None):
        """
        Parameters
        ----------
        fps : int, default: 5
            Movie frame rate (per second).
        codec : str or None, default: :rc:`animation.codec`
            The codec to use.
        bitrate : int, default: :rc:`animation.bitrate`
            The bitrate of the movie, in kilobits per second.  Higher values
            means higher quality movies, but increase the file size.  A value
            of -1 lets the underlying movie encoder select the bitrate.
        extra_args : list of str or None, optional
            Extra command-line arguments passed to the underlying movie encoder. These
            arguments are passed last to the encoder, just before the filename. The
            default, None, means to use :rc:`animation.[name-of-encoder]_args` for the
            builtin writers.
        metadata : dict[str, str], default: {}
            A dictionary of keys and values for metadata to include in the
            output file. Some keys that may be of use include:
            title, artist, genre, subject, copyright, srcform, comment.
        """
        # 如果尝试实例化 MovieWriter 类本身，会引发 TypeError 异常，因为 MovieWriter 是一个抽象类，需要通过混入类扩展。
        if type(self) is MovieWriter:
            raise TypeError(
                'MovieWriter cannot be instantiated directly. Please use one '
                'of its subclasses.')

        # 调用父类构造函数，初始化影片帧率、元数据、编解码器、比特率等参数。
        super().__init__(fps=fps, metadata=metadata, codec=codec,
                         bitrate=bitrate)
        
        # 设置帧数据写入格式，默认为支持的第一个格式 'rgba'。
        self.frame_format = self.supported_formats[0]
        
        # 存储额外的命令行参数，将在影片编码器中最后传递给编码器，直接位于文件名之前。
        self.extra_args = extra_args
    def _adjust_frame_size(self):
        # 如果使用 H.264 编解码器
        if self.codec == 'h264':
            # 获取当前图像尺寸（单位为英寸）
            wo, ho = self.fig.get_size_inches()
            # 调用函数调整图像尺寸，返回调整后的宽度和高度
            w, h = adjusted_figsize(wo, ho, self.dpi, 2)
            # 如果调整后的尺寸与原尺寸不同，则更新图像尺寸
            if (wo, ho) != (w, h):
                self.fig.set_size_inches(w, h, forward=True)
                # 记录调整后的图像尺寸日志
                _log.info('figure size in inches has been adjusted '
                          'from %s x %s to %s x %s', wo, ho, w, h)
        else:
            # 获取当前图像尺寸（单位为英寸）
            w, h = self.fig.get_size_inches()
        # 记录调试日志，显示帧大小（单位为像素）
        _log.debug('frame size in pixels is %s x %s', *self.frame_size)
        # 返回最终的图像宽度和高度
        return w, h

    def setup(self, fig, outfile, dpi=None):
        # 继承的文档字符串说明
        super().setup(fig, outfile, dpi=dpi)
        # 调整帧大小并记录结果
        self._w, self._h = self._adjust_frame_size()
        # 执行 _run() 方法，以便 grab_frame() 方法可以将数据写入管道，避免使用临时文件
        self._run()

    def _run(self):
        # 使用子进程调用程序来将帧组装成电影文件。*args* 返回一些配置选项的命令行参数序列。
        command = self._args()
        # 记录信息日志，显示执行的命令
        _log.info('MovieWriter._run: running command: %s',
                  cbook._pformat_subprocess(command))
        # 定义 PIPE 常量为 subprocess.PIPE，并启动子进程，设定 stdin、stdout、stderr
        self._proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=subprocess_creation_flags)

    def finish(self):
        """完成写入电影的任何处理工作。"""
        # 从进程中获取输出和错误信息
        out, err = self._proc.communicate()
        # 使用与 universal_newlines 相同的编码和错误处理方式
        out = TextIOWrapper(BytesIO(out)).read()
        err = TextIOWrapper(BytesIO(err)).read()
        # 如果有输出信息，记录到日志中
        if out:
            _log.log(
                logging.WARNING if self._proc.returncode else logging.DEBUG,
                "MovieWriter stdout:\n%s", out)
        # 如果有错误信息，记录到日志中
        if err:
            _log.log(
                logging.WARNING if self._proc.returncode else logging.DEBUG,
                "MovieWriter stderr:\n%s", err)
        # 如果进程返回值非零，抛出 CalledProcessError 异常
        if self._proc.returncode:
            raise subprocess.CalledProcessError(
                self._proc.returncode, self._proc.args, out, err)

    def grab_frame(self, **savefig_kwargs):
        # 继承的文档字符串说明
        _validate_grabframe_kwargs(savefig_kwargs)
        # 记录调试日志，显示正在抓取帧
        _log.debug('MovieWriter.grab_frame: Grabbing frame.')
        # 将图像尺寸调整为之前计算的宽度和高度，以确保所有帧保存时尺寸一致
        self.fig.set_size_inches(self._w, self._h)
        # 将图像数据保存到管道中，使用指定的帧格式和 dpi
        self.fig.savefig(self._proc.stdin, format=self.frame_format,
                         dpi=self.dpi, **savefig_kwargs)

    def _args(self):
        """组装特定编码器的命令行参数列表。"""
        # 抛出 NotImplementedError 异常，要求子类实现这个方法
        return NotImplementedError("args needs to be implemented by subclass.")
    def bin_path(cls):
        """
        Return the binary path to the commandline tool used by a specific
        subclass. This is a class method so that the tool can be looked for
        before making a particular MovieWriter subclass available.
        """
        # 返回特定子类使用的命令行工具的二进制路径
        return str(mpl.rcParams[cls._exec_key])

    @classmethod
    def isAvailable(cls):
        """Return whether a MovieWriter subclass is actually available."""
        # 检查特定 MovieWriter 子类的命令行工具是否可用，返回布尔值
        return shutil.which(cls.bin_path()) is not None
    """
    `MovieWriter` for writing to individual files and stitching at the end.

    This must be sub-classed to be useful.
    """
    # 定义一个 MovieWriter 的子类，用于将动画帧写入单独的文件，并在结束时进行拼接
    class FileMovieWriter(MovieWriter):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # 设置帧的格式为当前 matplotlib 默认的动画帧格式
            self.frame_format = mpl.rcParams['animation.frame_format']

        def setup(self, fig, outfile, dpi=None, frame_prefix=None):
            """
            Setup for writing the movie file.

            Parameters
            ----------
            fig : `~matplotlib.figure.Figure`
                The figure to grab the rendered frames from.
            outfile : str
                The filename of the resulting movie file.
            dpi : float, default: ``fig.dpi``
                The dpi of the output file. This, with the figure size,
                controls the size in pixels of the resulting movie file.
            frame_prefix : str, optional
                The filename prefix to use for temporary files.  If *None* (the
                default), files are written to a temporary directory which is
                deleted by `finish`; if not *None*, no temporary files are
                deleted.
            """
            # 检查输出文件路径是否有效
            Path(outfile).parent.resolve(strict=True)
            self.fig = fig
            self.outfile = outfile
            if dpi is None:
                dpi = self.fig.dpi
            self.dpi = dpi
            self._adjust_frame_size()

            if frame_prefix is None:
                # 如果未指定帧文件名前缀，创建临时目录并设置临时文件名前缀
                self._tmpdir = TemporaryDirectory()
                self.temp_prefix = str(Path(self._tmpdir.name, 'tmp'))
            else:
                self._tmpdir = None
                self.temp_prefix = frame_prefix
            self._frame_counter = 0  # 用于生成顺序文件名的计数器
            self._temp_paths = list()
            # 定义文件名格式化字符串，用于生成文件名
            self.fname_format_str = '%s%%07d.%s'

        def __del__(self):
            # 清理临时目录，释放资源
            if hasattr(self, '_tmpdir') and self._tmpdir:
                self._tmpdir.cleanup()

        @property
        def frame_format(self):
            """
            Format (png, jpeg, etc.) to use for saving the frames, which can be
            decided by the individual subclasses.
            """
            # 返回用于保存帧的格式，可以由子类决定
            return self._frame_format

        @frame_format.setter
        def frame_format(self, frame_format):
            # 设置帧的保存格式，并验证其是否在支持的格式列表中
            if frame_format in self.supported_formats:
                self._frame_format = frame_format
            else:
                _api.warn_external(
                    f"Ignoring file format {frame_format!r} which is not "
                    f"supported by {type(self).__name__}; using "
                    f"{self.supported_formats[0]} instead.")
                self._frame_format = self.supported_formats[0]

        def _base_temp_name(self):
            # 根据帧的格式和前缀生成一个不带数字的模板文件名
            return self.fname_format_str % (self.temp_prefix, self.frame_format)
    # 继承自父类的文档字符串
    def grab_frame(self, **savefig_kwargs):
        # 验证并确保保存参数有效性
        _validate_grabframe_kwargs(savefig_kwargs)
        # 使用基本临时文件名和计数器创建文件路径
        path = Path(self._base_temp_name() % self._frame_counter)
        # 将文件路径添加到临时路径列表中，以备后续使用
        self._temp_paths.append(path)
        # 增加帧计数器，确保每个创建的文件名都是唯一的
        self._frame_counter += 1
        # 记录调试信息，指示正在保存帧的路径和计数
        _log.debug('FileMovieWriter.grab_frame: Grabbing frame %d to path=%s',
                   self._frame_counter, path)
        # 使用二进制写入模式打开路径对应的文件，用于保存图形
        with open(path, 'wb') as sink:
            # 将当前图形保存到文件中，使用指定的格式、分辨率和其他保存参数
            self.fig.savefig(sink, format=self.frame_format, dpi=self.dpi,
                             **savefig_kwargs)

    # 完成帧抓取后的清理和后续操作
    def finish(self):
        # 在所有帧抓取完成后调用运行函数，以准备组装所有临时文件
        try:
            self._run()
            # 调用父类的完成方法
            super().finish()
        finally:
            # 如果存在临时目录，则清理并记录调试信息
            if self._tmpdir:
                _log.debug(
                    'MovieWriter: clearing temporary path=%s', self._tmpdir
                )
                self._tmpdir.cleanup()
# 使用装饰器将此类注册为“pillow”写入器的一部分，该类继承自抽象电影写入器
@writers.register('pillow')
class PillowWriter(AbstractMovieWriter):
    
    # 类方法：检查PillowWriter是否可用
    @classmethod
    def isAvailable(cls):
        return True

    # 设置方法：初始化图形、输出文件和dpi
    def setup(self, fig, outfile, dpi=None):
        super().setup(fig, outfile, dpi=dpi)
        # 初始化帧列表
        self._frames = []

    # 抓取帧方法：保存图形帧到字节流中
    def grab_frame(self, **savefig_kwargs):
        # 验证抓取帧参数的有效性
        _validate_grabframe_kwargs(savefig_kwargs)
        buf = BytesIO()
        # 将图形保存为RGBA格式到字节流中，包括指定的保存参数和dpi
        self.fig.savefig(
            buf, **{**savefig_kwargs, "format": "rgba", "dpi": self.dpi})
        # 将从字节流中读取的RGBA数据转换为Image对象，并添加到帧列表中
        self._frames.append(Image.frombuffer(
            "RGBA", self.frame_size, buf.getbuffer(), "raw", "RGBA", 0, 1))

    # 结束方法：将帧列表中的第一帧作为基础保存为输出文件，保存所有帧作为动画，设置持续时间和循环
    def finish(self):
        self._frames[0].save(
            self.outfile, save_all=True, append_images=self._frames[1:],
            duration=int(1000 / self.fps), loop=0)


# ffmpeg信息的基类。包含配置键和控制输出方面的常见参数的基类
class FFMpegBase:
    """
    Mixin class for FFMpeg output.

    This is a base class for the concrete `FFMpegWriter` and `FFMpegFileWriter`
    classes.
    """

    # ffmpeg执行路径配置键
    _exec_key = 'animation.ffmpeg_path'
    # ffmpeg参数配置键
    _args_key = 'animation.ffmpeg_args'

    # 输出参数属性：根据输出文件类型决定编解码器类型和额外参数
    @property
    def output_args(self):
        args = []
        # 如果输出文件是.gif后缀，则编解码器为'gif'
        if Path(self.outfile).suffix == '.gif':
            self.codec = 'gif'
        else:
            # 否则，添加-vcodec参数和相应编解码器类型
            args.extend(['-vcodec', self.codec])
        # 获取额外参数，如果未提供则使用默认配置
        extra_args = (self.extra_args if self.extra_args is not None
                      else mpl.rcParams[self._args_key])
        # 对于h264编解码器，如果额外参数中未指定-pix_fmt，设置为'yuv420p'以提高兼容性
        if self.codec == 'h264' and '-pix_fmt' not in extra_args:
            args.extend(['-pix_fmt', 'yuv420p'])
        # 对于GIF文件，告知FFMPEG分离视频流、生成调色板并使用它进行编码
        elif self.codec == 'gif' and '-filter_complex' not in extra_args:
            args.extend(['-filter_complex',
                         'split [a][b];[a] palettegen [p];[b][p] paletteuse'])
        # 如果设置了比特率，则添加到参数列表中（以kbps为单位）
        if self.bitrate > 0:
            args.extend(['-b', '%dk' % self.bitrate])  # %dk: bitrate in kbps.
        # 添加元数据到参数列表中
        for k, v in self.metadata.items():
            args.extend(['-metadata', f'{k}={v}'])
        # 添加额外参数到参数列表中
        args.extend(extra_args)

        # 返回最终的参数列表，包括强制覆盖输出文件选项('-y')和输出文件路径
        return args + ['-y', self.outfile]


# 结合FFMpeg选项与基于管道的写入器注册为“ffmpeg”写入器的类
@writers.register('ffmpeg')
class FFMpegWriter(FFMpegBase, MovieWriter):
    """
    Pipe-based ffmpeg writer.

    Frames are streamed directly to ffmpeg via a pipe and written in a single pass.

    This effectively works as a slideshow input to ffmpeg with the fps passed as
    ``-framerate``, so see also `their notes on frame rates`_ for further details.

    .. _their notes on frame rates: https://trac.ffmpeg.org/wiki/Slideshow#Framerates
    """
    """
    def _args(self):
        # 返回用于 subprocess 调用 ffmpeg 创建视频的命令行参数列表
        # 使用自身的二进制路径作为 ffmpeg 可执行文件的路径
        args = [self.bin_path(), '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', '%dx%d' % self.frame_size, '-pix_fmt', self.frame_format,
                '-framerate', str(self.fps)]
        # 日志级别被设置为 DEBUG 以上时，将日志级别设置为 error，以避免由于 subprocess.PIPE 的缓冲区限制导致的问题
        # 如果动画帧数较多且日志级别为 DEBUG，则可能发生缓冲区溢出
        if _log.getEffectiveLevel() > logging.DEBUG:
            args += ['-loglevel', 'error']
        # 添加输入参数 '-i pipe:' 和输出参数列表到 args 中
        args += ['-i', 'pipe:'] + self.output_args
        # 返回组装好的参数列表
        return args
# Combine FFMpeg options with temp file-based writing
@writers.register('ffmpeg_file')
class FFMpegFileWriter(FFMpegBase, FileMovieWriter):
    """
    File-based ffmpeg writer.

    Frames are written to temporary files on disk and then stitched together at the end.

    This effectively works as a slideshow input to ffmpeg with the fps passed as
    ``-framerate``, so see also `their notes on frame rates`_ for further details.

    .. _their notes on frame rates: https://trac.ffmpeg.org/wiki/Slideshow#Framerates
    """
    supported_formats = ['png', 'jpeg', 'tiff', 'raw', 'rgba']

    def _args(self):
        # Returns the command line parameters for subprocess to use
        # ffmpeg to create a movie using a collection of temp images
        args = []
        # For raw frames, we need to explicitly tell ffmpeg the metadata.
        if self.frame_format in {'raw', 'rgba'}:
            args += [
                '-f', 'image2', '-vcodec', 'rawvideo',
                '-video_size', '%dx%d' % self.frame_size,
                '-pixel_format', 'rgba',
            ]
        # Include the frame rate as part of the ffmpeg command
        args += ['-framerate', str(self.fps), '-i', self._base_temp_name()]
        # If a temporary directory is not specified, limit the number of frames
        if not self._tmpdir:
            args += ['-frames:v', str(self._frame_counter)]
        # Adjust logging level to prevent buffer overrun
        if _log.getEffectiveLevel() > logging.DEBUG:
            args += ['-loglevel', 'error']
        # Return the full set of ffmpeg command line arguments
        return [self.bin_path(), *args, *self.output_args]


# Base class for animated GIFs with ImageMagick
class ImageMagickBase:
    """
    Mixin class for ImageMagick output.

    This is a base class for the concrete `ImageMagickWriter` and
    `ImageMagickFileWriter` classes, which define an ``input_names`` attribute
    (or property) specifying the input names passed to ImageMagick.
    """

    _exec_key = 'animation.convert_path'
    _args_key = 'animation.convert_args'

    def _args(self):
        # ImageMagick does not recognize "raw". Convert to "rgba" if necessary.
        fmt = "rgba" if self.frame_format == "raw" else self.frame_format
        # Determine additional arguments; use default if none specified
        extra_args = (self.extra_args if self.extra_args is not None
                      else mpl.rcParams[self._args_key])
        # Construct the ImageMagick command line arguments
        return [
            self.bin_path(),
            "-size", "%ix%i" % self.frame_size,
            "-depth", "8",
            "-delay", str(100 / self.fps),  # Calculate delay based on FPS
            "-loop", "0",  # Specify looping behavior
            f"{fmt}:{self.input_names}",  # Format and input names
            *extra_args,  # Additional user-defined arguments
            self.outfile,  # Output file
        ]

    @classmethod
    def bin_path(cls):
        # Determine the path to the ImageMagick executable
        binpath = super().bin_path()
        if binpath == 'convert':
            binpath = mpl._get_executable_info('magick').executable
        return binpath
    # 定义一个类方法 isAvailable(cls)，用于检查某个功能是否可用
    def isAvailable(cls):
        try:
            # 调用父类的 isAvailable 方法，检查功能是否可用
            return super().isAvailable()
        except mpl.ExecutableNotFoundError as _enf:
            # 如果捕获到 ExecutableNotFoundError 异常，通常由 get_executable_info 引起
            # 记录调试信息，说明 ImageMagick 不可用的具体原因
            _log.debug('ImageMagick unavailable due to: %s', _enf)
            # 返回 False，表示功能不可用
            return False
# 注册一个名为 'imagemagick' 的自定义影片写入器，继承自 ImageMagickBase 和 MovieWriter
@writers.register('imagemagick')
class ImageMagickWriter(ImageMagickBase, MovieWriter):
    """
    Pipe-based animated gif writer.

    Frames are streamed directly to ImageMagick via a pipe and written
    in a single pass.
    """
    
    input_names = "-"  # 标识输入是标准输入（stdin）


# 注册一个名为 'imagemagick_file' 的自定义影片写入器，继承自 ImageMagickBase 和 FileMovieWriter
@writers.register('imagemagick_file')
class ImageMagickFileWriter(ImageMagickBase, FileMovieWriter):
    """
    File-based animated gif writer.

    Frames are written to temporary files on disk and then stitched
    together at the end.
    """
    
    supported_formats = ['png', 'jpeg', 'tiff', 'raw', 'rgba']
    # 定义输入名称属性，使用临时文件前缀和帧格式来动态生成
    input_names = property(lambda self: f'{self.temp_prefix}*.{self.frame_format}')


# 从 jakevdp 的 JSAnimation 包中直接获取的函数
def _included_frames(frame_count, frame_format, frame_dir):
    """
    返回包含帧的字符串，用于指定帧数、帧格式和帧目录
    """
    return INCLUDED_FRAMES.format(Nframes=frame_count,
                                  frame_dir=frame_dir,
                                  frame_format=frame_format)


def _embedded_frames(frame_list, frame_format):
    """
    frame_list 应为 base64 编码的 png 文件列表。
    如果帧格式为 'svg'，修正 MIME 类型为 'svg+xml'。
    返回包含每个帧数据的字符串
    """
    template = '  frames[{0}] = "data:image/{1};base64,{2}"\n'
    return "\n" + "".join(
        template.format(i, frame_format, frame_data.replace('\n', '\\\n'))
        for i, frame_data in enumerate(frame_list))


# 注册一个名为 'html' 的自定义影片写入器，继承自 FileMovieWriter
@writers.register('html')
class HTMLWriter(FileMovieWriter):
    """
    Writer for JavaScript-based HTML movies.
    """

    supported_formats = ['png', 'jpeg', 'tiff', 'svg']

    @classmethod
    def isAvailable(cls):
        """
        检查 HTMLWriter 是否可用，始终返回 True
        """
        return True

    def __init__(self, fps=30, codec=None, bitrate=None, extra_args=None,
                 metadata=None, embed_frames=False, default_mode='loop',
                 embed_limit=None):
        """
        初始化 HTMLWriter 对象。

        忽略 'extra_args' 并设置为 None。
        设置 embed_frames 和 default_mode 属性。
        检查 default_mode 是否有效。
        保存 embed_limit 属性，单位为 MB，转换为字节。
        调用父类 FileMovieWriter 的构造函数进行初始化。
        """
        if extra_args:
            _log.warning("HTMLWriter ignores 'extra_args'")
        extra_args = ()  # 不查询不存在的 rcParam[args_key]。
        self.embed_frames = embed_frames
        self.default_mode = default_mode.lower()
        _api.check_in_list(['loop', 'once', 'reflect'],
                           default_mode=self.default_mode)

        # 保存 embed_limit 属性，给定单位为 MB
        self._bytes_limit = mpl._val_or_rc(embed_limit, 'animation.embed_limit')
        # 转换为字节单位
        self._bytes_limit *= 1024 * 1024

        super().__init__(fps, codec, bitrate, extra_args, metadata)
    # 设置函数用于配置动画输出的参数和文件路径
    def setup(self, fig, outfile, dpi=None, frame_dir=None):
        # 将输出文件路径转换为Path对象
        outfile = Path(outfile)
        # 检查输出文件扩展名是否在 ['.html', '.htm'] 中，确保文件格式符合要求
        _api.check_in_list(['.html', '.htm'], outfile_extension=outfile.suffix)

        # 初始化保存帧的列表
        self._saved_frames = []
        # 初始化动画帧总字节数
        self._total_bytes = 0
        # 初始化是否达到字节限制的标志
        self._hit_limit = False

        # 如果不嵌入帧到文件中
        if not self.embed_frames:
            # 如果未指定帧文件夹，则生成一个默认的帧文件夹路径
            if frame_dir is None:
                frame_dir = outfile.with_name(outfile.stem + '_frames')
            # 创建帧文件夹，如果不存在则创建
            frame_dir.mkdir(parents=True, exist_ok=True)
            # 设置帧文件名前缀
            frame_prefix = frame_dir / 'frame'
        else:
            # 如果嵌入帧到文件中，则帧文件名前缀为None
            frame_prefix = None

        # 调用父类方法设置动画参数
        super().setup(fig, outfile, dpi, frame_prefix)
        # 清空临时文件的标志设置为False
        self._clear_temp = False

    # 抓取当前帧并保存为图像数据
    def grab_frame(self, **savefig_kwargs):
        # 验证抓取帧参数的有效性
        _validate_grabframe_kwargs(savefig_kwargs)
        
        # 如果嵌入帧到文件中
        if self.embed_frames:
            # 如果已经达到字节限制，则停止处理
            if self._hit_limit:
                return
            # 创建一个BytesIO对象
            f = BytesIO()
            # 将当前图形保存为图像数据到BytesIO对象中
            self.fig.savefig(f, format=self.frame_format,
                             dpi=self.dpi, **savefig_kwargs)
            # 将图像数据转换为base64编码字符串并解码为ascii格式
            imgdata64 = base64.encodebytes(f.getvalue()).decode('ascii')
            # 累加总字节数
            self._total_bytes += len(imgdata64)
            # 如果总字节数超过设定的字节限制
            if self._total_bytes >= self._bytes_limit:
                # 发出警告，说明动画大小已经超过限制
                _log.warning(
                    "Animation size has reached %s bytes, exceeding the limit "
                    "of %s. If you're sure you want a larger animation "
                    "embedded, set the animation.embed_limit rc parameter to "
                    "a larger value (in MB). This and further frames will be "
                    "dropped.", self._total_bytes, self._bytes_limit)
                # 设置达到字节限制的标志为True
                self._hit_limit = True
            else:
                # 将当前帧的base64编码字符串添加到已保存帧列表中
                self._saved_frames.append(imgdata64)
        else:
            # 如果不嵌入帧到文件中，则调用父类方法处理
            return super().grab_frame(**savefig_kwargs)
    def finish(self):
        # 将帧保存到一个 HTML 文件中

        # 如果要嵌入帧
        if self.embed_frames:
            # 使用 _embedded_frames 函数填充保存的帧，使用指定的帧格式
            fill_frames = _embedded_frames(self._saved_frames,
                                           self.frame_format)
            # 获取帧的数量
            frame_count = len(self._saved_frames)
        else:
            # 否则，temp_paths 是由 FileMovieWriter 填充的临时文件名列表
            frame_count = len(self._temp_paths)
            # 使用 _included_frames 函数填充帧，指定帧数量、帧格式和输出文件路径的相对路径
            fill_frames = _included_frames(
                frame_count, self.frame_format,
                self._temp_paths[0].parent.relative_to(self.outfile.parent))

        # 创建一个模式字典，包括不同模式的复选框状态
        mode_dict = dict(once_checked='',
                         loop_checked='',
                         reflect_checked='')
        # 根据默认模式设置对应模式的复选框为选中状态
        mode_dict[self.default_mode + '_checked'] = 'checked'

        # 计算帧之间的时间间隔
        interval = 1000 // self.fps

        # 打开输出文件，写入 JS_INCLUDE 和 STYLE_INCLUDE
        with open(self.outfile, 'w') as of:
            of.write(JS_INCLUDE + STYLE_INCLUDE)
            # 使用 DISPLAY_TEMPLATE 格式化写入文件，包括随机生成的 ID、帧数量、填充的帧、间隔和模式字典
            of.write(DISPLAY_TEMPLATE.format(id=uuid.uuid4().hex,
                                             Nframes=frame_count,
                                             fill_frames=fill_frames,
                                             interval=interval,
                                             **mode_dict))

        # 复制自 FileMovieWriter.finish 的临时文件清理逻辑。
        # 不能调用继承版本的 finish，因为它假定存在需要调用的子进程来合并许多帧，或者需要清理的子进程调用。
        # 如果存在临时目录 _tmpdir，则清理它
        if self._tmpdir:
            _log.debug('MovieWriter: clearing temporary path=%s', self._tmpdir)
            self._tmpdir.cleanup()
    """
    A base class for Animations.

    This class is not usable as is, and should be subclassed to provide needed
    behavior.

    .. note::

        You must store the created Animation in a variable that lives as long
        as the animation should run. Otherwise, the Animation object will be
        garbage-collected and the animation stops.

    Parameters
    ----------
    fig : `~matplotlib.figure.Figure`
        The figure object used to get needed events, such as draw or resize.

    event_source : object, optional
        A class that can run a callback when desired events
        are generated, as well as be stopped and started.

        Examples include timers (see `TimedAnimation`) and file
        system notifications.

    blit : bool, default: False
        Whether blitting is used to optimize drawing.  If the backend does not
        support blitting, then this parameter has no effect.

    See Also
    --------
    FuncAnimation,  ArtistAnimation
    """
    
    def __init__(self, fig, event_source=None, blit=False):
        self._draw_was_started = False  # 标志动画是否已经开始绘制

        self._fig = fig  # 将传入的图形对象保存到实例变量中
        # 如果后端支持 blitting，则启用 blitting 以优化绘制效率
        self._blit = blit and fig.canvas.supports_blit

        # 动画的帧序列，用于保存每一帧的信息，具体绘制由子类处理
        self.frame_seq = self.new_frame_seq()
        self.event_source = event_source

        # 连接到图形的 draw_event 事件，等待第一次绘制后再开始动画
        self._first_draw_id = fig.canvas.mpl_connect('draw_event', self._start)

        # 连接到图形的 close_event 事件，以防止在已关闭的图形上继续触发事件和尝试绘制
        self._close_id = self._fig.canvas.mpl_connect('close_event',
                                                      self._stop)
        if self._blit:
            self._setup_blit()  # 如果启用了 blitting，则设置 blitting 相关参数

    def __del__(self):
        if not getattr(self, '_draw_was_started', True):
            warnings.warn(
                'Animation was deleted without rendering anything. This is '
                'most likely not intended. To prevent deletion, assign the '
                'Animation to a variable, e.g. `anim`, that exists until you '
                'output the Animation using `plt.show()` or '
                '`anim.save()`.'
            )
    def _start(self, *args):
        """
        Starts interactive animation. Adds the draw frame command to the GUI
        handler, calls show to start the event loop.
        """
        # 如果正在保存，不启动事件源
        if self._fig.canvas.is_saving():
            return
        # 断开首次绘图事件处理器的连接
        self._fig.canvas.mpl_disconnect(self._first_draw_id)

        # 执行初始绘图
        self._init_draw()

        # 添加用于步进动画的回调函数，并启动事件源
        self.event_source.add_callback(self._step)
        self.event_source.start()

    def _stop(self, *args):
        # 在停止时断开所有事件的连接
        if self._blit:
            self._fig.canvas.mpl_disconnect(self._resize_id)
        self._fig.canvas.mpl_disconnect(self._close_id)
        self.event_source.remove_callback(self._step)
        self.event_source = None

    def _step(self, *args):
        """
        Handler for getting events. By default, gets the next frame in the
        sequence and hands the data off to be drawn.
        """
        # 处理事件的处理程序。默认情况下，获取序列中的下一帧数据并传递给绘制函数。
        try:
            framedata = next(self.frame_seq)
            self._draw_next_frame(framedata, self._blit)
            return True  # 返回 True 表示事件源应继续调用 _step
        except StopIteration:
            return False  # 当序列结束时返回 False

    def new_frame_seq(self):
        """Return a new sequence of frame information."""
        # 返回一个新的帧信息序列，默认为 self._framedata 的迭代器
        return iter(self._framedata)

    def new_saved_frame_seq(self):
        """Return a new sequence of saved/cached frame information."""
        # 返回一个新的已保存/缓存帧信息序列，默认与普通帧序列相同
        return self.new_frame_seq()

    def _draw_next_frame(self, framedata, blit):
        # 分解下一帧的绘制过程，包括预绘制、绘制帧本身和后绘制
        self._pre_draw(framedata, blit)
        self._draw_frame(framedata)
        self._post_draw(framedata, blit)

    def _init_draw(self):
        # 初始绘制以清除帧。在 blit 代码需要一个干净的基础时也会使用。
        self._draw_was_started = True

    def _pre_draw(self, framedata, blit):
        # 在绘制帧之前执行任何清理或其他操作。
        # 这个默认实现允许 blit 清除帧。
        if blit:
            self._blit_clear(self._drawn_artists)

    def _draw_frame(self, framedata):
        # 执行帧的实际绘制。
        raise NotImplementedError('Needs to be implemented by subclasses to'
                                  ' actually make an animation.')
    def _post_draw(self, framedata, blit):
        # 在帧渲染完成后，处理实际的绘制刷新操作
        # 如果使用 blit 并且有已绘制的艺术家对象
        if blit and self._drawn_artists:
            # 执行 blit 绘制已绘制的艺术家对象
            self._blit_draw(self._drawn_artists)
        else:
            # 否则，直接在画布上绘制空闲状态
            self._fig.canvas.draw_idle()

    # 这个类的其余代码是为了方便实现简单的 blitting
    def _blit_draw(self, artists):
        # 处理 blit 绘制，只绘制给定的艺术家对象而不是整个图形
        # 更新艺术家对象所在的 Axes
        updated_ax = {a.axes for a in artists}
        # 遍历更新过的 Axes，缓存 Axes 视图的背景
        for ax in updated_ax:
            # 如果当前视图的 Axes 背景没有缓存，则现在缓存它
            cur_view = ax._get_view()
            view, bg = self._blit_cache.get(ax, (object(), None))
            if cur_view != view:
                self._blit_cache[ax] = (
                    cur_view, ax.figure.canvas.copy_from_bbox(ax.bbox))
        # 单独绘制前景
        for a in artists:
            a.axes.draw_artist(a)
        # 绘制所有需要的艺术家后，逐个 Axes 进行 blit
        for ax in updated_ax:
            ax.figure.canvas.blit(ax.bbox)

    def _blit_clear(self, artists):
        # 从已绘制的艺术家列表中获取需要清除的 Axes
        axes = {a.axes for a in artists}
        # 遍历这些 Axes，从缓存中获取正确的背景并进行恢复
        for ax in axes:
            try:
                view, bg = self._blit_cache[ax]
            except KeyError:
                continue
            if ax._get_view() == view:
                ax.figure.canvas.restore_region(bg)
            else:
                self._blit_cache.pop(ax)

    def _setup_blit(self):
        # 设置 blit 需要：Axes 背景的缓存
        self._blit_cache = dict()
        self._drawn_artists = []
        # 首先调用 _post_draw 初始化渲染器
        self._post_draw(None, self._blit)
        # 然后需要清除帧以进行初始绘制
        # 通常在 _on_resize 中处理，因为 QT 和 Tk
        # 在启动时会发出调整大小事件，但 macOS 后端不会，
        # 因此这里为了保持一致性，强制处理所有情况
        self._init_draw()
        # 连接到未来的调整大小事件
        self._resize_id = self._fig.canvas.mpl_connect('resize_event',
                                                       self._on_resize)
    def _on_resize(self, event):
        # 在窗口大小调整时，禁用大小调整事件处理，以避免触发过多事件。
        # 同时停止动画事件，暂停动画播放。
        # 清空缓存并重新初始化绘图。
        # 设置事件处理器以在绘图完成后捕获通知。
        
        # 断开当前窗口大小调整事件的连接
        self._fig.canvas.mpl_disconnect(self._resize_id)
        # 停止动画事件的处理
        self.event_source.stop()
        # 清空缓存
        self._blit_cache.clear()
        # 重新初始化绘图
        self._init_draw()
        # 设置新的窗口大小调整事件处理器，以在绘图完成后调用 _end_redraw 方法
        self._resize_id = self._fig.canvas.mpl_connect('draw_event',
                                                       self._end_redraw)

    def _end_redraw(self, event):
        # 绘图完成后，执行后续的绘图刷新和位块传输处理。
        # 然后重新启用所有原始事件处理。
        
        # 执行绘图后的刷新处理，传入 None 和 False 作为参数
        self._post_draw(None, False)
        # 重新启动事件源处理
        self.event_source.start()
        # 断开当前绘图事件的连接
        self._fig.canvas.mpl_disconnect(self._resize_id)
        # 设置新的窗口大小调整事件处理器，以在窗口大小调整时调用 _on_resize 方法
        self._resize_id = self._fig.canvas.mpl_connect('resize_event',
                                                       self._on_resize)

    def to_html5_video(self, embed_limit=None):
        """
        Convert the animation to an HTML5 ``<video>`` tag.

        This saves the animation as an h264 video, encoded in base64
        directly into the HTML5 video tag. This respects :rc:`animation.writer`
        and :rc:`animation.bitrate`. This also makes use of the
        *interval* to control the speed, and uses the *repeat*
        parameter to decide whether to loop.

        Parameters
        ----------
        embed_limit : float, optional
            Limit, in MB, of the returned animation. No animation is created
            if the limit is exceeded.
            Defaults to :rc:`animation.embed_limit` = 20.0.

        Returns
        -------
        str
            An HTML5 video tag with the animation embedded as base64 encoded
            h264 video.
            If the *embed_limit* is exceeded, this returns the string
            "Video too large to embed."
        """
        # 定义 HTML5 video 标签的模板，用于将动画转换为 HTML5 video 标签
        
        VIDEO_TAG = r'''<video {size} {options}>
  <source type="video/mp4" src="data:video/mp4;base64,{video}">
  Your browser does not support the video tag.
        # Cache the rendering of the video as HTML
        # 将视频的渲染结果缓存为 HTML

        if not hasattr(self, '_base64_video'):
            # Save embed limit, which is given in MB
            # 保存嵌入限制，单位为 MB

            embed_limit = mpl._val_or_rc(embed_limit, 'animation.embed_limit')

            # Convert from MB to bytes
            # 将 MB 转换为字节
            embed_limit *= 1024 * 1024

            # Can't open a NamedTemporaryFile twice on Windows, so use a
            # TemporaryDirectory instead.
            # Windows 上无法两次打开 NamedTemporaryFile，因此使用 TemporaryDirectory

            with TemporaryDirectory() as tmpdir:
                path = Path(tmpdir, "temp.m4v")
                # We create a writer manually so that we can get the
                # appropriate size for the tag
                # 手动创建 writer，以便获取适当的标签大小

                Writer = writers[mpl.rcParams['animation.writer']]
                writer = Writer(codec='h264',
                                bitrate=mpl.rcParams['animation.bitrate'],
                                fps=1000. / self._interval)
                self.save(str(path), writer=writer)
                # Now open and base64 encode.
                # 现在打开并进行 base64 编码

                vid64 = base64.encodebytes(path.read_bytes())

            vid_len = len(vid64)
            if vid_len >= embed_limit:
                _log.warning(
                    "Animation movie is %s bytes, exceeding the limit of %s. "
                    "If you're sure you want a large animation embedded, set "
                    "the animation.embed_limit rc parameter to a larger value "
                    "(in MB).", vid_len, embed_limit)
            else:
                self._base64_video = vid64.decode('ascii')
                self._video_size = 'width="{}" height="{}"'.format(
                        *writer.frame_size)

        # If we exceeded the size, this attribute won't exist
        # 如果超过了大小限制，这个属性将不存在

        if hasattr(self, '_base64_video'):
            # Default HTML5 options are to autoplay and display video controls
            # 默认的 HTML5 选项是自动播放和显示视频控件

            options = ['controls', 'autoplay']

            # If we're set to repeat, make it loop
            # 如果设置为重复播放，则设置为循环播放

            if getattr(self, '_repeat', False):
                options.append('loop')

            return VIDEO_TAG.format(video=self._base64_video,
                                    size=self._video_size,
                                    options=' '.join(options))
        else:
            return 'Video too large to embed.'
    def to_jshtml(self, fps=None, embed_frames=True, default_mode=None):
        """
        Generate HTML representation of the animation.

        Parameters
        ----------
        fps : int, optional
            Movie frame rate (per second). If not set, the frame rate from
            the animation's frame interval.
        embed_frames : bool, optional
            Whether to embed animation frames in the HTML (default is True).
        default_mode : str, optional
            What to do when the animation ends. Must be one of {'loop',
            'once', 'reflect'}. Defaults to 'loop' if the 'repeat'
            parameter is True, otherwise 'once'.
        """
        # 如果未提供 fps，并且对象有 _interval 属性，则计算帧率
        if fps is None and hasattr(self, '_interval'):
            # 将毫秒间隔转换为每秒帧数
            fps = 1000 / self._interval

        # 如果未提供 default_mode，则根据 _repeat 属性的值选择默认模式
        if default_mode is None:
            default_mode = 'loop' if getattr(self, '_repeat',
                                             False) else 'once'

        # 如果尚未生成 HTML 表示，则创建临时目录和文件以保存 HTML
        if not hasattr(self, "_html_representation"):
            # 在 Windows 上无法两次打开 NamedTemporaryFile，因此使用 TemporaryDirectory
            with TemporaryDirectory() as tmpdir:
                path = Path(tmpdir, "temp.html")
                # 创建 HTMLWriter 对象并保存动画到临时文件
                writer = HTMLWriter(fps=fps,
                                    embed_frames=embed_frames,
                                    default_mode=default_mode)
                self.save(str(path), writer=writer)
                # 读取生成的 HTML 文件内容
                self._html_representation = path.read_text()

        # 返回 HTML 表示
        return self._html_representation

    def _repr_html_(self):
        """IPython 显示钩子，用于渲染动画。"""
        # 获取当前设置的动画 HTML 格式
        fmt = mpl.rcParams['animation.html']
        # 根据格式选择返回 HTML5 视频或者 JavaScript HTML
        if fmt == 'html5':
            return self.to_html5_video()
        elif fmt == 'jshtml':
            return self.to_jshtml()

    def pause(self):
        """暂停动画播放。"""
        # 停止事件源
        self.event_source.stop()
        # 如果启用了 blit，取消所有已绘制的艺术家的动画效果
        if self._blit:
            for artist in self._drawn_artists:
                artist.set_animated(False)

    def resume(self):
        """恢复动画播放。"""
        # 启动事件源
        self.event_source.start()
        # 如果启用了 blit，重新启用所有已绘制的艺术家的动画效果
        if self._blit:
            for artist in self._drawn_artists:
                artist.set_animated(True)
class TimedAnimation(Animation):
    """
    `Animation` subclass for time-based animation.

    A new frame is drawn every *interval* milliseconds.

    .. note::

        You must store the created Animation in a variable that lives as long
        as the animation should run. Otherwise, the Animation object will be
        garbage-collected and the animation stops.

    Parameters
    ----------
    fig : `~matplotlib.figure.Figure`
        The figure object used to get needed events, such as draw or resize.
    interval : int, default: 200
        Delay between frames in milliseconds.
    repeat_delay : int, default: 0
        The delay in milliseconds between consecutive animation runs, if
        *repeat* is True.
    repeat : bool, default: True
        Whether the animation repeats when the sequence of frames is completed.
    blit : bool, default: False
        Whether blitting is used to optimize drawing.
    """
    
    def __init__(self, fig, interval=200, repeat_delay=0, repeat=True,
                 event_source=None, *args, **kwargs):
        # 设置动画帧之间的时间间隔
        self._interval = interval
        # 兼容处理，支持旧版 repeat_delay = None
        self._repeat_delay = repeat_delay if repeat_delay is not None else 0
        # 设置动画是否重复播放的标志
        self._repeat = repeat
        # 如果没有指定事件源，则创建一个新的定时器
        if event_source is None:
            event_source = fig.canvas.new_timer(interval=self._interval)
        # 调用父类初始化方法
        super().__init__(fig, event_source=event_source, *args, **kwargs)

    def _step(self, *args):
        """Handler for getting events."""
        # 调用 Animation 类的 _step() 方法处理事件
        still_going = super()._step(*args)
        # 如果动画已经结束
        if not still_going:
            if self._repeat:
                # 重新初始化绘制循环
                self._init_draw()
                # 生成新的帧序列
                self.frame_seq = self.new_frame_seq()
                # 设置事件源的间隔为重复延迟
                self.event_source.interval = self._repeat_delay
                return True
            else:
                # 动画播放结束，暂停动画
                self.pause()
                # 如果使用 blitting 技术，移除 resize 回调
                if self._blit:
                    self._fig.canvas.mpl_disconnect(self._resize_id)
                # 移除 close 回调
                self._fig.canvas.mpl_disconnect(self._close_id)
                # 释放事件源
                self.event_source = None
                return False

        # 恢复事件源的正常间隔
        self.event_source.interval = self._interval
        return True


class ArtistAnimation(TimedAnimation):
    """
    `TimedAnimation` subclass that creates an animation by using a fixed
    set of `.Artist` objects.

    Before creating an instance, all plotting should have taken place
    and the relevant artists saved.

    .. note::

        You must store the created Animation in a variable that lives as long
        as the animation should run. Otherwise, the Animation object will be
        garbage-collected and the animation stops.

    Parameters
    ----------
    fig : `~matplotlib.figure.Figure`
        The figure object used to get needed events, such as draw or resize.
    artists : list
        Each list entry is a collection of `.Artist` objects that are made
        visible on the corresponding frame.  Other artists are made invisible.
    interval : int, default: 200
        Delay between frames in milliseconds.
    repeat_delay : int, default: 0
        The delay in milliseconds between consecutive animation runs, if
        *repeat* is True.
    repeat : bool, default: True
        Whether the animation repeats when the sequence of frames is completed.
    blit : bool, default: False
        Whether blitting is used to optimize drawing.
    """

    def __init__(self, fig, artists, *args, **kwargs):
        # Internal list of artists drawn in the most recent frame.
        self._drawn_artists = []

        # Use the list of artists as the framedata, which will be iterated
        # over by the machinery.
        self._framedata = artists
        super().__init__(fig, *args, **kwargs)

    def _init_draw(self):
        # Initialize the drawing process by making all involved artists invisible
        super()._init_draw()
        
        # Make all the artists involved in *any* frame invisible
        figs = set()
        for f in self.new_frame_seq():
            for artist in f:
                artist.set_visible(False)
                artist.set_animated(self._blit)
                # Assemble a list of unique figures that need flushing
                if artist.get_figure() not in figs:
                    figs.add(artist.get_figure())

        # Flush the needed figures
        for fig in figs:
            fig.canvas.draw_idle()

    def _pre_draw(self, framedata, blit):
        """Clears artists from the last frame."""
        if blit:
            # Let blit handle clearing by calling _blit_clear with drawn artists
            self._blit_clear(self._drawn_artists)
        else:
            # Otherwise, make all the artists from the previous frame invisible
            for artist in self._drawn_artists:
                artist.set_visible(False)

    def _draw_frame(self, artists):
        # Save the artists that were passed in as framedata for the other
        # steps (esp. blitting) to use.
        self._drawn_artists = artists

        # Make all the artists from the current frame visible
        for artist in artists:
            artist.set_visible(True)
# 继承自 TimedAnimation 的 FuncAnimation 类，用于通过重复调用函数来创建动画。

class FuncAnimation(TimedAnimation):
    """
    `TimedAnimation` 的子类，通过重复调用函数 *func* 来创建动画。

    .. note::
    
        必须将创建的 Animation 对象存储在一个生存周期长于动画所需运行时间的变量中。否则，Animation 对象会被垃圾回收，动画会停止。

    Parameters
    ----------
    fig : `~matplotlib.figure.Figure`
        用于获取所需事件（如绘制或调整大小）的图形对象。

    func : callable
        每帧调用的函数。第一个参数是 *frames* 中的下一个值。可以使用 `functools.partial` 或通过 *fargs* 参数提供额外的位置参数。

        函数的必需签名是::

            def func(frame, *fargs) -> iterable_of_artists

        使用 `functools.partial` 提供参数通常更方便。这样还可以传递关键字参数。要传递既有位置参数又有关键字参数的函数，请将所有参数设置为关键字参数，只需留下 *frame* 参数未设置即可::

            def func(frame, art, *, y=None):
                ...

            ani = FuncAnimation(fig, partial(func, art=ln, y='foo'))

        如果 ``blit == True``，*func* 必须返回所有被修改或创建的艺术家的可迭代对象。这些信息将被 blitting 算法用于确定需要更新的图形的哪些部分。如果 ``blit == False``，返回值未使用，在这种情况下可以省略。

    frames : iterable, int, generator function, or None, optional
        用于传递给 *func* 和动画每一帧数据的源头

        - 如果是可迭代对象，则简单地使用提供的值。如果可迭代对象有长度，则会覆盖 *save_count* 关键字参数。

        - 如果是整数，则等同于传递 ``range(frames)``。

        - 如果是生成器函数，则必须具有以下签名::

             def gen_function() -> obj

        - 如果为 *None*，则等同于传递 ``itertools.count``。

        在所有这些情况下，*frames* 中的值仅仅通过给用户提供的 *func* 传递，并且可以是任何类型。

    init_func : callable, optional
        用于绘制清除帧的函数。如果未提供，则使用来自帧序列中第一个项目的绘制结果。此函数将在第一帧之前调用。

        函数的必需签名是::

            def init_func() -> iterable_of_artists

        如果 ``blit == True``，*init_func* 必须返回一个艺术家的可迭代对象，以便重新绘制。这些信息将被 blitting 算法用于确定需要更新的图形的哪些部分。如果 ``blit == False``，返回值未使用，在这种情况下可以省略。
    fargs : tuple or None, optional
        Additional arguments to pass to each call to *func*. Note: the use of
        `functools.partial` is preferred over *fargs*. See *func* for details.



    save_count : int, optional
        Fallback for the number of values from *frames* to cache. This is
        only used if the number of frames cannot be inferred from *frames*,
        i.e. when it's an iterator without length or a generator.



    interval : int, default: 200
        Delay between frames in milliseconds.



    repeat_delay : int, default: 0
        The delay in milliseconds between consecutive animation runs, if
        *repeat* is True.



    repeat : bool, default: True
        Whether the animation repeats when the sequence of frames is completed.



    blit : bool, default: False
        Whether blitting is used to optimize drawing.  Note: when using
        blitting, any animated artists will be drawn according to their zorder;
        however, they will be drawn on top of any previous artists, regardless
        of their zorder.



    cache_frame_data : bool, default: True
        Whether frame data is cached.  Disabling cache might be helpful when
        frames contain large objects.
    """



    def new_frame_seq(self):
        # Use the generating function to generate a new frame sequence
        return self._iter_gen()



    def new_saved_frame_seq(self):
        # Generate an iterator for the sequence of saved data. If there are
        # no saved frames, generate a new frame sequence and take the first
        # save_count entries in it.
        if self._save_seq:
            # While iterating we are going to update _save_seq
            # so make a copy to safely iterate over
            self._old_saved_seq = list(self._save_seq)
            return iter(self._old_saved_seq)
        else:
            if self._save_count is None:
                frame_seq = self.new_frame_seq()

                def gen():
                    try:
                        while True:
                            yield next(frame_seq)
                    except StopIteration:
                        pass
                return gen()
            else:
                return itertools.islice(self.new_frame_seq(), self._save_count)
    def _init_draw(self):
        super()._init_draw()
        # 初始化绘图，可以使用给定的 init_func 或者调用帧序列的第一个帧数据来绘制。
        # 对于 blitting（部分重绘），init_func 应该返回一系列修改过的 artists（图形元素）。
        if self._init_func is None:
            try:
                frame_data = next(self.new_frame_seq())
            except StopIteration:
                # 如果无法开始迭代帧数据，可能是之前的保存操作已经耗尽了帧序列，或者帧序列长度为 0。
                # 发出警告并退出。
                warnings.warn(
                    "Can not start iterating the frames for the initial draw. "
                    "This can be caused by passing in a 0 length sequence "
                    "for *frames*.\n\n"
                    "If you passed *frames* as a generator "
                    "it may be exhausted due to a previous display or save."
                )
                return
            # 绘制第一帧数据
            self._draw_frame(frame_data)
        else:
            # 使用给定的 init_func 来初始化绘图元素
            self._drawn_artists = self._init_func()
            # 如果使用 blitting，则检查返回的 artists 是否为有效的序列
            if self._blit:
                if self._drawn_artists is None:
                    raise RuntimeError('The init_func must return a '
                                       'sequence of Artist objects.')
                # 设置每个 artist 是否支持动画（blit）
                for a in self._drawn_artists:
                    a.set_animated(self._blit)
        # 初始化保存序列为空
        self._save_seq = []

    def _draw_frame(self, framedata):
        if self._cache_frame_data:
            # 如果启用了缓存帧数据，保存当前帧数据以便于后续保存为视频
            self._save_seq.append(framedata)
            self._save_seq = self._save_seq[-self._save_count:]

        # 调用动画函数 func，并传入帧数据及其它参数。如果启用 blitting，func 需要返回修改过的 artists 序列。
        self._drawn_artists = self._func(framedata, *self._args)

        if self._blit:
            # 如果启用 blitting

            err = RuntimeError('The animation function must return a sequence '
                               'of Artist objects.')
            try:
                # 检查返回的 artists 是否为序列
                iter(self._drawn_artists)
            except TypeError:
                raise err from None

            # 检查每个返回的元素是否为 Artist 对象
            for i in self._drawn_artists:
                if not isinstance(i, mpl.artist.Artist):
                    raise err

            # 对返回的 artists 根据其 zorder 属性进行排序
            self._drawn_artists = sorted(self._drawn_artists,
                                         key=lambda x: x.get_zorder())

            # 设置每个 artist 是否支持动画（blit）
            for a in self._drawn_artists:
                a.set_animated(self._blit)
# 定义函数 _validate_grabframe_kwargs，用于验证 savefig_kwargs 参数的合法性
def _validate_grabframe_kwargs(savefig_kwargs):
    # 检查 matplotlib 的 savefig.bbox 设置，如果为 'tight'，则抛出 ValueError 异常
    if mpl.rcParams['savefig.bbox'] == 'tight':
        raise ValueError(
            f"{mpl.rcParams['savefig.bbox']=} must not be 'tight' as it "
            "may cause frame size to vary, which is inappropriate for animation."
        )
    
    # 遍历预期不应出现在 savefig_kwargs 中的关键字参数，如果出现则抛出 TypeError 异常
    for k in ('dpi', 'bbox_inches', 'format'):
        if k in savefig_kwargs:
            raise TypeError(
                f"grab_frame got an unexpected keyword argument {k!r}"
            )
```