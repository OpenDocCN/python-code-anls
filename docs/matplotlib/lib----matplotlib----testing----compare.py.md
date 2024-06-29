# `D:\src\scipysrc\matplotlib\lib\matplotlib\testing\compare.py`

```
"""
Utilities for comparing image results.
"""

import atexit  # 导入用于注册退出时的处理函数的模块
import functools  # 导入用于创建偏函数的模块
import hashlib  # 导入用于计算哈希值的模块
import logging  # 导入日志记录模块
import os  # 导入操作系统相关功能的模块
from pathlib import Path  # 导入处理路径的模块
import shutil  # 导入文件和目录操作相关的模块
import subprocess  # 导入执行外部命令和管道的模块
import sys  # 导入系统相关的参数和函数
from tempfile import TemporaryDirectory, TemporaryFile  # 导入临时文件和目录创建模块
import weakref  # 导入弱引用相关的模块

import numpy as np  # 导入数值计算库 NumPy
from PIL import Image  # 导入 Python Imaging Library（PIL）中的图像模块

import matplotlib as mpl  # 导入绘图库 Matplotlib 的主模块
from matplotlib import cbook  # 导入 Matplotlib 内部使用的基本函数和类
from matplotlib.testing.exceptions import ImageComparisonFailure  # 导入 Matplotlib 的图像比较异常类

_log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象

__all__ = ['calculate_rms', 'comparable_formats', 'compare_images']  # 模块中公开的函数和类的列表


def make_test_filename(fname, purpose):
    """
    Make a new filename by inserting *purpose* before the file's extension.
    """
    base, ext = os.path.splitext(fname)  # 分离文件名和扩展名
    return f'{base}-{purpose}{ext}'  # 构造新的文件名


def _get_cache_path():
    """
    Get the path to the cache directory for storing temporary files.
    """
    cache_dir = Path(mpl.get_cachedir(), 'test_cache')  # 获取 Matplotlib 缓存目录下的测试缓存子目录
    cache_dir.mkdir(parents=True, exist_ok=True)  # 创建缓存目录，如果不存在则创建
    return cache_dir  # 返回缓存目录的路径对象


def get_cache_dir():
    """
    Get the string path to the cache directory for external use.
    """
    return str(_get_cache_path())  # 返回缓存目录的路径字符串表示


def get_file_hash(path, block_size=2 ** 20):
    """
    Calculate the MD5 hash of the file located at *path*.
    """
    md5 = hashlib.md5()  # 创建一个 MD5 哈希对象
    with open(path, 'rb') as fd:
        while True:
            data = fd.read(block_size)  # 每次读取指定大小的数据块
            if not data:
                break
            md5.update(data)  # 更新 MD5 哈希对象

    if Path(path).suffix == '.pdf':
        md5.update(str(mpl._get_executable_info("gs").version)
                   .encode('utf-8'))  # 如果是 PDF 文件，加入 GhostScript 版本信息到哈希计算中
    elif Path(path).suffix == '.svg':
        md5.update(str(mpl._get_executable_info("inkscape").version)
                   .encode('utf-8'))  # 如果是 SVG 文件，加入 Inkscape 版本信息到哈希计算中

    return md5.hexdigest()  # 返回计算得到的文件哈希值的十六进制表示


class _ConverterError(Exception):
    """
    Exception raised for errors in the conversion process.
    """
    pass


class _Converter:
    """
    Base class for handling external conversion processes.
    """
    def __init__(self):
        self._proc = None  # 初始化进程对象为 None
        # 在退出时注册删除操作，以确保进程被正确终止
        atexit.register(self.__del__)

    def __del__(self):
        """
        Cleanup method to ensure proper termination of the process.
        """
        if self._proc:
            self._proc.kill()  # 终止进程
            self._proc.wait()  # 等待进程终止
            for stream in filter(None, [self._proc.stdin,
                                        self._proc.stdout,
                                        self._proc.stderr]):
                stream.close()  # 关闭进程的输入、输出和错误流
            self._proc = None  # 将进程对象置为 None，确保不再引用

    def _read_until(self, terminator):
        """
        Read from stdout until the terminator sequence is reached.
        """
        buf = bytearray()  # 初始化一个字节数组作为缓冲区
        while True:
            c = self._proc.stdout.read(1)  # 从进程的标准输出读取一个字节
            if not c:
                raise _ConverterError(os.fsdecode(bytes(buf)))  # 如果读取到空字节，则抛出转换器错误异常
            buf.extend(c)  # 将读取的字节追加到缓冲区
            if buf.endswith(terminator):  # 如果缓冲区以终止序列结尾
                return bytes(buf)  # 返回缓冲区的内容作为字节串


class _GSConverter(_Converter):
    """
    Converter subclass for handling GhostScript conversions.
    """
    pass  # _GSConverter 类目前没有额外的代码，继承了 Converter 的功能
    def __call__(self, orig, dest):
        # 如果尚未启动子进程，则启动Ghostscript，并初始化子进程对象self._proc
        if not self._proc:
            # 使用subprocess.Popen启动Ghostscript进程
            self._proc = subprocess.Popen(
                [mpl._get_executable_info("gs").executable,
                 "-dNOSAFER", "-dNOPAUSE", "-dEPSCrop", "-sDEVICE=png16m"],
                # 指定标准输入为管道，标准输出也为管道
                stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            try:
                # 调用内部方法_read_until，等待Ghostscript输出"GS"前的内容
                self._read_until(b"\nGS")
            except _ConverterError as e:
                # 若出现转换错误，抛出OSError并显示错误消息
                raise OSError(f"Failed to start Ghostscript:\n\n{e.args[0]}") from None

        # 定义函数encode_and_escape，用于转义文件名中的特殊字符
        def encode_and_escape(name):
            return (os.fsencode(name)
                    .replace(b"\\", b"\\\\")
                    .replace(b"(", br"\(")
                    .replace(b")", br"\)"))

        # 向Ghostscript进程的标准输入写入命令，设定输出和输入文件名，并刷新输入流
        self._proc.stdin.write(
            b"<< /OutputFile ("
            + encode_and_escape(dest)
            + b") >> setpagedevice ("
            + encode_and_escape(orig)
            + b") run flush\n")
        self._proc.stdin.flush()
        
        # 读取Ghostscript进程的输出，判断处理结果
        # 如果输出以"GS<"结尾，表示有堆栈剩余；如果以"GS>"结尾，表示无堆栈剩余
        err = self._read_until((b"GS<", b"GS>"))
        stack = self._read_until(b">") if err.endswith(b"GS<") else b""
        
        # 如果堆栈不为空或目标文件不存在，则抛出ImageComparisonFailure异常
        if stack or not os.path.exists(dest):
            stack_size = int(stack[:-1]) if stack else 0
            # 向Ghostscript进程发送pop命令，清除堆栈
            self._proc.stdin.write(b"pop\n" * stack_size)
            # 使用系统编码解码错误消息和堆栈内容，并抛出异常
            raise ImageComparisonFailure(
                (err + stack).decode(sys.getfilesystemencoding(), "replace"))
class _SVGConverter(_Converter):
    # _SVGConverter 类，继承自 _Converter
    def __del__(self):
        # 调用父类的析构函数
        super().__del__()
        # 如果对象具有 "_tmpdir" 属性，则清理临时目录
        if hasattr(self, "_tmpdir"):
            self._tmpdir.cleanup()


class _SVGWithMatplotlibFontsConverter(_SVGConverter):
    """
    A SVG converter which explicitly adds the fonts shipped by Matplotlib to
    Inkspace's font search path, to better support `svg.fonttype = "none"`
    (which is in particular used by certain mathtext tests).
    """
    
    def __call__(self, orig, dest):
        # 如果不存在 "_tmpdir" 属性，则创建临时目录并复制 Matplotlib 提供的字体到该目录下
        if not hasattr(self, "_tmpdir"):
            self._tmpdir = TemporaryDirectory()
            shutil.copytree(cbook._get_data_path("fonts/ttf"),
                            Path(self._tmpdir.name, "fonts"))
        # 调用父类的 __call__ 方法进行转换操作
        return super().__call__(orig, dest)


def _update_converter():
    # 尝试获取 'gs' 可执行文件信息，若未找到则捕获 ExecutableNotFoundError 异常
    try:
        mpl._get_executable_info("gs")
    except mpl.ExecutableNotFoundError:
        pass
    else:
        # 若成功获取，则将 'pdf' 和 'eps' 格式的转换器更新为 _GSConverter 类的实例
        converter['pdf'] = converter['eps'] = _GSConverter()
    
    # 尝试获取 'inkscape' 可执行文件信息，若未找到则捕获 ExecutableNotFoundError 异常
    try:
        mpl._get_executable_info("inkscape")
    except mpl.ExecutableNotFoundError:
        pass
    else:
        # 若成功获取，则更新 'svg' 格式的转换器为 _SVGConverter 类的实例
        converter['svg'] = _SVGConverter()


#: A dictionary that maps filename extensions to functions which themselves
#: convert between arguments `old` and `new` (filenames).
converter = {}
# 更新转换器字典，根据系统环境配置不同的文件格式转换器
_update_converter()

# 创建一个 _SVGWithMatplotlibFontsConverter 类的实例
_svg_with_matplotlib_fonts_converter = _SVGWithMatplotlibFontsConverter()


def comparable_formats():
    """
    Return the list of file formats that `.compare_images` can compare
    on this system.

    Returns
    -------
    list of str
        E.g. ``['png', 'pdf', 'svg', 'eps']``.

    """
    # 返回比较工具可以处理的文件格式列表，包括预先配置的转换器格式
    return ['png', *converter]


def convert(filename, cache):
    """
    Convert the named file to png; return the name of the created file.

    If *cache* is True, the result of the conversion is cached in
    `matplotlib.get_cachedir() + '/test_cache/'`.  The caching is based on a
    hash of the exact contents of the input file.  Old cache entries are
    automatically deleted as needed to keep the size of the cache capped to
    twice the size of all baseline images.
    """
    path = Path(filename)
    # 如果指定路径文件不存在，则抛出 OSError 异常
    if not path.exists():
        raise OSError(f"{path} does not exist")
    # 如果文件后缀不在转换器字典中，则使用 pytest 跳过测试，并显示相应的提示信息
    if path.suffix[1:] not in converter:
        import pytest
        pytest.skip(f"Don't know how to convert {path.suffix} files to png")
    # 构造新文件路径，用于存储转换后的 PNG 文件
    newpath = path.parent / f"{path.stem}_{path.suffix[1:]}.png"

    # 只有当目标文件不存在或者已过期时，才进行文件转换操作
    # 如果新路径不存在，或者新路径的修改时间早于原路径的修改时间，则执行以下操作
    if not newpath.exists() or newpath.stat().st_mtime < path.stat().st_mtime:
        # 如果需要缓存，则获取缓存路径，否则设置为 None
        cache_dir = _get_cache_path() if cache else None
        
        # 如果有缓存目录，则注册一次清理转换缓存的操作
        if cache_dir is not None:
            _register_conversion_cache_cleaner_once()
            # 获取文件的哈希值作为缓存文件名，并附加新路径的后缀
            hash_value = get_file_hash(path)
            cached_path = cache_dir / (hash_value + newpath.suffix)
            # 如果缓存文件存在，则直接复制到新路径并返回新路径的字符串表示
            if cached_path.exists():
                _log.debug("For %s: reusing cached conversion.", filename)
                shutil.copyfile(cached_path, newpath)
                return str(newpath)
        
        # 打印调试信息，指示正在将文件转换为 PNG 格式
        _log.debug("For %s: converting to png.", filename)
        # 根据文件后缀选择对应的转换器函数
        convert = converter[path.suffix[1:]]
        
        # 如果文件后缀为 ".svg"，则读取文件内容
        if path.suffix == ".svg":
            contents = path.read_text()
            # 如果内容中包含 'style="font:' 字符串，则使用特定的 SVG 转换器函数
            if 'style="font:' in contents:
                # 对于 svg.fonttype = none，我们需要显式地修改字体搜索路径，
                # 以便找到由 Matplotlib 提供的字体。
                convert = _svg_with_matplotlib_fonts_converter
        
        # 执行文件转换操作，将原路径文件转换为新路径文件
        convert(path, newpath)
        
        # 如果有缓存目录，则将新转换的文件复制到缓存路径，并打印调试信息
        if cache_dir is not None:
            _log.debug("For %s: caching conversion result.", filename)
            shutil.copyfile(newpath, cached_path)
    
    # 返回新路径的字符串表示作为函数的输出
    return str(newpath)
def _clean_conversion_cache():
    # 计算基线图像文件夹中所有文件的总大小
    baseline_images_size = sum(
        path.stat().st_size
        for path in Path(mpl.__file__).parent.glob("**/baseline_images/**/*"))

    # 估算最大缓存大小为基线图像总大小的两倍
    max_cache_size = 2 * baseline_images_size

    # 通过加锁获取缓存路径，确保安全地访问和修改缓存
    with cbook._lock_path(_get_cache_path()):
        # 获取缓存路径下所有文件的状态信息
        cache_stat = {
            path: path.stat() for path in _get_cache_path().glob("*")}
        
        # 计算当前缓存的总大小
        cache_size = sum(stat.st_size for stat in cache_stat.values())
        
        # 按照访问时间排序缓存文件，最早访问的文件排在最后
        paths_by_atime = sorted(
            cache_stat, key=lambda path: cache_stat[path].st_atime,
            reverse=True)
        
        # 如果当前缓存大小超过最大缓存大小，则逐步删除最近最少使用的文件，直到缓存大小符合要求
        while cache_size > max_cache_size:
            path = paths_by_atime.pop()
            cache_size -= cache_stat[path].st_size
            path.unlink()


@functools.cache  # 确保该函数只注册一次
def _register_conversion_cache_cleaner_once():
    # 在程序退出时注册清理缓存的函数
    atexit.register(_clean_conversion_cache)


def crop_to_same(actual_path, actual_image, expected_path, expected_image):
    # 如果实际文件和预期文件分别是 eps 和 pdf 格式，则裁剪实际图像和预期图像到相同大小
    if actual_path[-7:-4] == 'eps' and expected_path[-7:-4] == 'pdf':
        aw, ah, ad = actual_image.shape
        ew, eh, ed = expected_image.shape
        actual_image = actual_image[int(aw / 2 - ew / 2):int(
            aw / 2 + ew / 2), int(ah / 2 - eh / 2):int(ah / 2 + eh / 2)]
    return actual_image, expected_image


def calculate_rms(expected_image, actual_image):
    """
    计算每个像素的误差，然后计算均方根误差。
    """
    # 如果预期图像和实际图像的尺寸不匹配，则抛出图像比较失败异常
    if expected_image.shape != actual_image.shape:
        raise ImageComparisonFailure(
            f"Image sizes do not match expected size: {expected_image.shape} "
            f"actual size {actual_image.shape}")
    
    # 将图像转换为浮点数，避免整数溢出问题
    return np.sqrt(((expected_image - actual_image).astype(float) ** 2).mean())


# 注意：compare_image 和 save_diff_image 假设图像的深度不是16位，因为 Pillow 会将这些图像不正确地转换为 RGB。


def _load_image(path):
    # 打开图像文件并将其转换为 numpy 数组表示
    img = Image.open(path)
    
    # 如果图像不是 RGBA 模式，或者 alpha 通道中最小值为 255（完全不透明），则将图像转换为 RGB 模式
    if img.mode != "RGBA" or img.getextrema()[3][0] == 255:
        img = img.convert("RGB")
    
    return np.asarray(img)


def compare_images(expected, actual, tol, in_decorator=False):
    """
    比较两个图像文件，检查在容差范围内的差异。
    
    给定的文件名可以指向可转换为
    """
    Convert and compare two images, returning differences if they exceed a specified tolerance.

    Parameters
    ----------
    expected : str
        The filename of the expected image.
    actual : str
        The filename of the actual image.
    tol : float
        The tolerance (maximum allowable RMS difference between images).
        Test fails if RMS difference exceeds this value.
    in_decorator : bool
        Determines the output format. If True, returns a dictionary with:
            - 'rms': RMS difference between images.
            - 'expected': Filename of the expected image.
            - 'actual': Filename of the actual image.
            - 'diff_image': Filename of the difference image.
            - 'tol': Comparison tolerance.
        If False, returns a human-readable multi-line string.

    Returns
    -------
    None or dict or str
        Returns None if images are equal within tolerance.
        Otherwise, depending on `in_decorator`:
            - Returns dict with detailed differences if `in_decorator` is True.
            - Returns human-readable string if `in_decorator` is False.

    Examples
    --------
    ::

        img1 = "./baseline/plot.png"
        img2 = "./output/plot.png"
        compare_images(img1, img2, 0.001)

    """
    # Ensure `actual` filename is a string path
    actual = os.fspath(actual)
    # Raise exception if `actual` image file does not exist
    if not os.path.exists(actual):
        raise Exception(f"Output image {actual} does not exist.")
    # Raise exception if `actual` image file is empty
    if os.stat(actual).st_size == 0:
        raise Exception(f"Output image file {actual} is empty.")

    # Ensure `expected` filename is a string path
    expected = os.fspath(expected)
    # Raise OSError if `expected` image file does not exist
    if not os.path.exists(expected):
        raise OSError(f'Baseline image {expected!r} does not exist.')
    # Check if `expected` image is in PNG format; if not, convert both images to PNG
    extension = expected.split('.')[-1]
    if extension != 'png':
        actual = convert(actual, cache=True)
        expected = convert(expected, cache=True)

    # Load expected and actual images
    expected_image = _load_image(expected)
    actual_image = _load_image(actual)

    # Crop images to the same dimensions
    actual_image, expected_image = crop_to_same(
        actual, actual_image, expected, expected_image)

    # Generate filename for difference image
    diff_image = make_test_filename(actual, 'failed-diff')

    # If tolerance is non-positive and images are exactly equal, return None
    if tol <= 0:
        if np.array_equal(expected_image, actual_image):
            return None

    # Convert images to signed integers to avoid overflow during subtraction
    expected_image = expected_image.astype(np.int16)
    actual_image = actual_image.astype(np.int16)

    # Calculate RMS difference between images
    rms = calculate_rms(expected_image, actual_image)

    # If RMS difference is within tolerance, return None
    if rms <= tol:
        return None

    # Save difference image
    save_diff_image(expected, actual, diff_image)

    # Return detailed results in dictionary format
    results = dict(rms=rms, expected=str(expected),
                   actual=str(actual), diff=str(diff_image), tol=tol)
    # 如果不在装饰器中，则生成适合输出到标准输出的字符串结果。
    # 定义输出模板，包括错误消息和各种变量的值
    template = ['Error: Image files did not match.',
                'RMS Value: {rms}',
                'Expected:  \n    {expected}',
                'Actual:    \n    {actual}',
                'Difference:\n    {diff}',
                'Tolerance: \n    {tol}', ]
    # 根据模板和结果数据生成格式化后的每行字符串，并用换行符连接
    results = '\n  '.join([line.format(**results) for line in template])
    # 返回生成的结果字符串
    return results
def save_diff_image(expected, actual, output):
    """
    Parameters
    ----------
    expected : str
        期望图像的文件路径。
    actual : str
        实际图像的文件路径。
    output : str
        用于保存差异图像的文件路径。
    """
    # 加载期望图像
    expected_image = _load_image(expected)
    # 加载实际图像
    actual_image = _load_image(actual)
    # 裁剪使得两个图像具有相同的大小和位置
    actual_image, expected_image = crop_to_same(
        actual, actual_image, expected, expected_image)
    # 将期望图像转换为浮点数数组
    expected_image = np.array(expected_image, float)
    # 将实际图像转换为浮点数数组
    actual_image = np.array(actual_image, float)
    # 检查图像尺寸是否匹配，如果不匹配则抛出异常
    if expected_image.shape != actual_image.shape:
        raise ImageComparisonFailure(
            f"Image sizes do not match expected size: {expected_image.shape} "
            f"actual size {actual_image.shape}")
    # 计算图像的绝对差异
    abs_diff = np.abs(expected_image - actual_image)

    # 在亮度领域扩展差异
    abs_diff *= 10
    # 将差异值限制在0到255之间，并转换为无符号8位整数
    abs_diff = np.clip(abs_diff, 0, 255).astype(np.uint8)

    # 如果图像具有四个通道，则硬编码将 alpha 通道设为完全不透明
    if abs_diff.shape[2] == 4:
        abs_diff[:, :, 3] = 255

    # 将数组转换为图像，并保存为 PNG 格式
    Image.fromarray(abs_diff).save(output, format="png")
```