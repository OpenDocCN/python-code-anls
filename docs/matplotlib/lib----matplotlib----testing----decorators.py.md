# `D:\src\scipysrc\matplotlib\lib\matplotlib\testing\decorators.py`

```
# 导入必要的模块和包
import contextlib  # 提供上下文管理功能的模块
import functools  # 提供高阶函数的工具，如部分应用（partial application）
import inspect  # 提供用于检查源代码的函数
import os  # 提供与操作系统交互的功能
from platform import uname  # 导入平台相关信息的函数
from pathlib import Path  # 提供处理文件路径的对象
import shutil  # 提供高级文件操作功能
import string  # 提供字符串处理相关的常量和函数
import sys  # 提供与Python解释器交互的函数和变量
import warnings  # 提供警告相关的功能

from packaging.version import parse as parse_version  # 导入版本解析工具

import matplotlib.style  # 导入Matplotlib的样式设置模块
import matplotlib.units  # 导入Matplotlib的单位设置模块
import matplotlib.testing  # 导入Matplotlib的测试功能模块
from matplotlib import _pylab_helpers, cbook, ft2font, pyplot as plt, ticker  # 导入Matplotlib的多个子模块和函数
from .compare import comparable_formats, compare_images, make_test_filename  # 导入本地的图像比较相关函数
from .exceptions import ImageComparisonFailure  # 导入本地定义的异常类


@contextlib.contextmanager
def _cleanup_cm():
    """
    A context manager that manages cleanup routines.

    It saves the original units registry of Matplotlib, then restores it
    after executing the wrapped code block with a clean slate of warnings
    and Matplotlib's rc parameters.
    """
    orig_units_registry = matplotlib.units.registry.copy()  # 备份当前的单位注册表
    try:
        with warnings.catch_warnings(), matplotlib.rc_context():
            yield  # 执行包裹的代码块
    finally:
        matplotlib.units.registry.clear()  # 清空当前的单位注册表
        matplotlib.units.registry.update(orig_units_registry)  # 恢复原始的单位注册表
        plt.close("all")  # 关闭所有的Matplotlib图形窗口


def _check_freetype_version(ver):
    """
    Check if the installed Freetype version matches the required version.

    Verifies if the installed Freetype version falls within the specified
    range of versions.

    Args:
        ver: A string or tuple of strings specifying the required version range.

    Returns:
        bool: True if the installed Freetype version meets the requirement, False otherwise.
    """
    if ver is None:
        return True

    if isinstance(ver, str):
        ver = (ver, ver)
    ver = [parse_version(x) for x in ver]  # 解析版本字符串为版本对象
    found = parse_version(ft2font.__freetype_version__)  # 获取当前安装的Freetype版本

    return ver[0] <= found <= ver[1]


def _checked_on_freetype_version(required_freetype_version):
    """
    Decorator that marks a test function to expect failure if Freetype version is mismatched.

    Args:
        required_freetype_version (str or tuple): Required Freetype version or range.

    Returns:
        function: Decorated test function.
    """
    import pytest  # 导入pytest模块
    return pytest.mark.xfail(
        not _check_freetype_version(required_freetype_version),
        reason=f"Mismatched version of freetype. "
               f"Test requires '{required_freetype_version}', "
               f"you have '{ft2font.__freetype_version__}'",
        raises=ImageComparisonFailure, strict=False)


def remove_ticks_and_titles(figure):
    """
    Remove ticks and titles from a Matplotlib figure.

    Args:
        figure (matplotlib.figure.Figure): The figure from which to remove ticks and titles.
    """
    figure.suptitle("")  # 清空总标题
    null_formatter = ticker.NullFormatter()  # 创建一个空格式化对象
    def remove_ticks(ax):
        """Remove ticks in *ax* and all its child Axes."""
        ax.set_title("")  # 清空坐标轴标题
        ax.xaxis.set_major_formatter(null_formatter)  # 设置X轴主刻度格式为空
        ax.xaxis.set_minor_formatter(null_formatter)  # 设置X轴次刻度格式为空
        ax.yaxis.set_major_formatter(null_formatter)  # 设置Y轴主刻度格式为空
        ax.yaxis.set_minor_formatter(null_formatter)  # 设置Y轴次刻度格式为空
        try:
            ax.zaxis.set_major_formatter(null_formatter)  # 设置Z轴主刻度格式为空（若存在）
            ax.zaxis.set_minor_formatter(null_formatter)  # 设置Z轴次刻度格式为空（若存在）
        except AttributeError:
            pass
        for child in ax.child_axes:
            remove_ticks(child)  # 递归移除子坐标轴的刻度
    for ax in figure.get_axes():
        remove_ticks(ax)  # 移除每个子图的刻度和标题


@contextlib.contextmanager
def _collect_new_figures():
    """
    Context manager to collect new Matplotlib figures created during its scope.

    Usage:

        with _collect_new_figures() as figs:
            some_code()

    After the context, *figs* will contain the new figures sorted by figure number.
    """
    managers = _pylab_helpers.Gcf.figs  # 获取当前所有图形的管理器
    preexisting = [manager for manager in managers.values()]  # 存储当前已存在的图形管理器列表
    new_figs = []  # 存储新创建的图形对象列表
    try:
        yield new_figs  # 执行包裹的代码块，并传入新创建图形列表
    finally:
        new_managers = sorted([manager for manager in managers.values()
                               if manager not in preexisting],  # 获取所有新创建的图形管理器
                              key=lambda manager: manager.num)
        new_figs[:] = [manager.canvas.figure for manager in new_managers]  # 提取新图形对象，并排序
# 设置 traceback 隐藏标志，不在 traceback 中显示该函数调用的位置信息
def _raise_on_image_difference(expected, actual, tol):
    __tracebackhide__ = True

    # 调用 compare_images 函数比较两张图片的差异，使用给定的容差值
    err = compare_images(expected, actual, tol, in_decorator=True)
    # 如果比较结果存在误差
    if err:
        # 将 actual、expected 和 diff 文件路径转换为相对路径
        for key in ["actual", "expected", "diff"]:
            err[key] = os.path.relpath(err[key])
        # 抛出 ImageComparisonFailure 异常，指示图片不相似，并包含详细信息
        raise ImageComparisonFailure(
            ('images not close (RMS %(rms).3f):'
                '\n\t%(actual)s\n\t%(expected)s\n\t%(diff)s') % err)

class _ImageComparisonBase:
    """
    图片比较的基础类

    该类仅提供与比较相关的功能，避免任何特定于任何测试框架的代码。
    """

    def __init__(self, func, tol, remove_text, savefig_kwargs):
        # 初始化函数，保存函数、容差、移除文本标志和保存图像参数
        self.func = func
        self.baseline_dir, self.result_dir = _image_directories(func)
        self.tol = tol
        self.remove_text = remove_text
        self.savefig_kwargs = savefig_kwargs

    def copy_baseline(self, baseline, extension):
        # 拷贝基准图像文件到测试结果目录下，并返回拷贝后的文件名
        baseline_path = self.baseline_dir / baseline
        orig_expected_path = baseline_path.with_suffix(f'.{extension}')
        # 如果是 eps 格式且文件不存在，则尝试使用 pdf 格式
        if extension == 'eps' and not orig_expected_path.exists():
            orig_expected_path = orig_expected_path.with_suffix('.pdf')
        # 构造测试结果目录下预期文件的完整路径
        expected_fname = make_test_filename(
            self.result_dir / orig_expected_path.name, 'expected')
        try:
            # 尝试删除已存在的符号链接文件
            with contextlib.suppress(OSError):
                os.remove(expected_fname)
            try:
                # 在非 Windows 平台上创建符号链接到原始预期图像文件
                if 'microsoft' in uname().release.lower():
                    raise OSError  # 在 WSL 上，符号链接会无声地中断
                os.symlink(orig_expected_path, expected_fname)
            except OSError:  # 在 Windows 上，符号链接可能不可用
                # 如果创建符号链接失败，在 Windows 上则复制文件
                shutil.copyfile(orig_expected_path, expected_fname)
        except OSError as err:
            # 抛出 ImageComparisonFailure 异常，指示基准图像文件丢失的详细信息
            raise ImageComparisonFailure(
                f"Missing baseline image {expected_fname} because the "
                f"following file cannot be accessed: "
                f"{orig_expected_path}") from err
        # 返回拷贝后的文件名
        return expected_fname
    # 定义一个方法用于比较图形文件和基准文件，可以选择加锁以防止并发访问
    def compare(self, fig, baseline, extension, *, _lock=False):
        # 隐藏调用栈追踪信息，用于测试时更清晰
        __tracebackhide__ = True

        # 如果设置了移除文本标记，调用函数删除图形中的文本信息
        if self.remove_text:
            remove_ticks_and_titles(fig)

        # 构建实际保存路径，根据指定的基准文件名和文件扩展名生成路径
        actual_path = (self.result_dir / baseline).with_suffix(f'.{extension}')

        # 复制保存图形的参数字典，用于后续保存图形时传递参数
        kwargs = self.savefig_kwargs.copy()

        # 如果保存为 PDF 格式，设置元数据以便生成文件与基准比较
        if extension == 'pdf':
            kwargs.setdefault('metadata',
                              {'Creator': None, 'Producer': None,
                               'CreationDate': None})

        # 如果设置了锁标志，则创建一个上下文管理器来锁定文件访问
        lock = (cbook._lock_path(actual_path)
                if _lock else contextlib.nullcontext())

        # 使用上下文管理器锁定文件访问
        with lock:
            try:
                # 保存图形到实际指定路径下，使用给定的参数字典
                fig.savefig(actual_path, **kwargs)
            finally:
                # 无论如何最终关闭图形，确保资源释放
                # 对于第三方用户，这使得关闭更加方便，即使 Matplotlib 有自动使用的 fixture
                plt.close(fig)

            # 拷贝基准文件到当前工作目录，并生成相应的扩展名文件
            expected_path = self.copy_baseline(baseline, extension)

            # 检查实际生成的图像文件与基准文件之间的差异，并根据指定的容差抛出异常
            _raise_on_image_difference(expected_path, actual_path, self.tol)
def _pytest_image_comparison(baseline_images, extensions, tol,
                             freetype_version, remove_text, savefig_kwargs,
                             style):
    """
    Decorate function with image comparison for pytest.

    This function creates a decorator that wraps a figure-generating function
    with image comparison code.
    """
    # 导入 pytest 模块，用于执行测试和断言
    import pytest

    # 定义一个常量，表示参数为仅限关键字参数
    KEYWORD_ONLY = inspect.Parameter.KEYWORD_ONLY
    # 定义一个装饰器函数，接受一个函数作为参数
    def decorator(func):
        # 获取被装饰函数的参数签名
        old_sig = inspect.signature(func)
    
        # 定义装饰器函数的实现
        @functools.wraps(func)
        @pytest.mark.parametrize('extension', extensions)
        @matplotlib.style.context(style)
        @_checked_on_freetype_version(freetype_version)
        @functools.wraps(func)
        # 定义装饰器函数的包装器
        def wrapper(*args, extension, request, **kwargs):
            # 设置 traceback 隐藏标志为 True
            __tracebackhide__ = True
            
            # 如果被装饰函数中有 'extension' 参数，则将其传递给装饰器的参数
            if 'extension' in old_sig.parameters:
                kwargs['extension'] = extension
            
            # 如果被装饰函数中有 'request' 参数，则将其传递给装饰器的参数
            if 'request' in old_sig.parameters:
                kwargs['request'] = request
            
            # 检查 extension 是否在可比较格式列表中
            if extension not in comparable_formats():
                # 如果不在列表中，则根据 extension 给出跳过测试的原因
                reason = {
                    'pdf': 'because Ghostscript is not installed',
                    'eps': 'because Ghostscript is not installed',
                    'svg': 'because Inkscape is not installed',
                }.get(extension, 'on this system')
                # 使用 pytest 跳过测试并给出原因
                pytest.skip(f"Cannot compare {extension} files {reason}")
            
            # 创建 _ImageComparisonBase 类的实例 img
            img = _ImageComparisonBase(func, tol=tol, remove_text=remove_text,
                                       savefig_kwargs=savefig_kwargs)
            
            # 设置 matplotlib 测试字体设置用于测试
            matplotlib.testing.set_font_settings_for_testing()
    
            # 使用 _collect_new_figures 上下文管理器收集新的图形
            with _collect_new_figures() as figs:
                # 调用被装饰的函数，传入参数和关键字参数
                func(*args, **kwargs)
    
            # 如果测试以除此装饰器之外的任何方式参数化，需要使用锁来防止两个进程同时访问同一输出文件
            needs_lock = any(
                marker.args[0] != 'extension'
                for marker in request.node.iter_markers('parametrize'))
    
            # 如果 baseline_images 不为空，则使用其作为我们的基准图像
            if baseline_images is not None:
                our_baseline_images = baseline_images
            else:
                # 否则，基于当前参数化，使用 request 获取 baseline_images
                our_baseline_images = request.getfixturevalue('baseline_images')
    
            # 断言生成的图形数量与基准图像数量相同
            assert len(figs) == len(our_baseline_images), (
                f"Test generated {len(figs)} images but there are "
                f"{len(our_baseline_images)} baseline images")
            
            # 遍历 figs 和 our_baseline_images，比较图形
            for fig, baseline in zip(figs, our_baseline_images):
                img.compare(fig, baseline, extension, _lock=needs_lock)
    
        # 将装饰器函数的参数列表作为 parameters
        parameters = list(old_sig.parameters.values())
        
        # 如果装饰器函数中没有 'extension' 参数，则将其添加到参数列表中
        if 'extension' not in old_sig.parameters:
            parameters += [inspect.Parameter('extension', KEYWORD_ONLY)]
        
        # 如果装饰器函数中没有 'request' 参数，则将其添加到参数列表中
        if 'request' not in old_sig.parameters:
            parameters += [inspect.Parameter("request", KEYWORD_ONLY)]
        
        # 使用新的参数列表创建新的函数签名
        new_sig = old_sig.replace(parameters=parameters)
        # 将新的函数签名设置给包装器函数的签名
        wrapper.__signature__ = new_sig
    
        # 获取被装饰函数的 pytest 标记，并将其与装饰器函数的标记合并
        new_marks = getattr(func, 'pytestmark', []) + wrapper.pytestmark
        wrapper.pytestmark = new_marks
    
        # 返回包装器函数
        return wrapper
# 定义函数 image_comparison，用于比较生成的图像与指定的基准图像是否一致，否则会引发 ImageComparisonFailure 异常
def image_comparison(baseline_images, extensions=None, tol=0,
                     freetype_version=None, remove_text=False,
                     savefig_kwarg=None,
                     # 默认的 mpl_test_settings fixture 和 cleanup 也要使用此值。
                     style=("classic", "_classic_test_patch")):
    """
    Compare images generated by the test with those specified in
    *baseline_images*, which must correspond, else an `ImageComparisonFailure`
    exception will be raised.

    Parameters
    ----------
    baseline_images : list or None
        A list of strings specifying the names of the images generated by
        calls to `.Figure.savefig`.

        If *None*, the test function must use the ``baseline_images`` fixture,
        either as a parameter or with `pytest.mark.usefixtures`. This value is
        only allowed when using pytest.

    extensions : None or list of str
        The list of extensions to test, e.g. ``['png', 'pdf']``.

        If *None*, defaults to all supported extensions: png, pdf, and svg.

        When testing a single extension, it can be directly included in the
        names passed to *baseline_images*.  In that case, *extensions* must not
        be set.

        In order to keep the size of the test suite from ballooning, we only
        include the ``svg`` or ``pdf`` outputs if the test is explicitly
        exercising a feature dependent on that backend (see also the
        `check_figures_equal` decorator for that purpose).

    tol : float, default: 0
        The RMS threshold above which the test is considered failed.

        Due to expected small differences in floating-point calculations, on
        32-bit systems an additional 0.06 is added to this threshold.

    freetype_version : str or tuple
        The expected freetype version or range of versions for this test to
        pass.

    remove_text : bool
        Remove the title and tick text from the figure before comparison.  This
        is useful to make the baseline images independent of variations in text
        rendering between different versions of FreeType.

        This does not remove other, more deliberate, text, such as legends and
        annotations.

    savefig_kwarg : dict
        Optional arguments that are passed to the savefig method.

    style : str, dict, or list
        The optional style(s) to apply to the image test. The test itself
        can also apply additional styles if desired. Defaults to ``["classic",
        "_classic_test_patch"]``.
    """
    # 如果 baseline_images 不为空，则处理基线图像文件名的扩展名
    if baseline_images is not None:
        # 创建基线图像文件的非空扩展名列表
        baseline_exts = [*filter(None, {Path(baseline).suffix[1:]
                                        for baseline in baseline_images})]
        # 如果存在基线图像文件的扩展名
        if baseline_exts:
            # 如果同时设置了 extensions 参数，则抛出数值错误
            if extensions is not None:
                raise ValueError(
                    "When including extensions directly in 'baseline_images', "
                    "'extensions' cannot be set as well")
            # 如果基线图像文件具有多个不同的扩展名，则抛出数值错误
            if len(baseline_exts) > 1:
                raise ValueError(
                    "When including extensions directly in 'baseline_images', "
                    "all baselines must share the same suffix")
            # 将 extensions 设置为基线图像文件的扩展名列表
            extensions = baseline_exts
            # 从 baseline_images 中删除扩展名
            baseline_images = [
                Path(baseline).stem for baseline in baseline_images]
    
    # 如果 extensions 参数未设置，则使用默认的扩展名列表进行测试
    if extensions is None:
        extensions = ['png', 'pdf', 'svg']
    
    # 如果 savefig_kwarg 参数未设置，则将其设置为空字典
    if savefig_kwarg is None:
        savefig_kwarg = dict()  # default no kwargs to savefig
    
    # 如果系统最大整数值小于等于 2^32，则增加 tol 值 0.06
    if sys.maxsize <= 2**32:
        tol += 0.06
    
    # 调用 _pytest_image_comparison 函数进行图像比较测试
    return _pytest_image_comparison(
        baseline_images=baseline_images, extensions=extensions, tol=tol,
        freetype_version=freetype_version, remove_text=remove_text,
        savefig_kwargs=savefig_kwarg, style=style)
# 定义装饰器函数，用于测试生成并比较两个图形是否相等
def check_figures_equal(*, extensions=("png", "pdf", "svg"), tol=0):
    """
    Decorator for test cases that generate and compare two figures.

    The decorated function must take two keyword arguments, *fig_test*
    and *fig_ref*, and draw the test and reference images on them.
    After the function returns, the figures are saved and compared.

    This decorator should be preferred over `image_comparison` when possible in
    order to keep the size of the test suite from ballooning.

    Parameters
    ----------
    extensions : list, default: ["png", "pdf", "svg"]
        The extensions to test.
    tol : float
        The RMS threshold above which the test is considered failed.

    Raises
    ------
    RuntimeError
        If any new figures are created (and not subsequently closed) inside
        the test function.

    Examples
    --------
    Check that calling `.Axes.plot` with a single argument plots it against
    ``[0, 1, 2, ...]``::

        @check_figures_equal()
        def test_plot(fig_test, fig_ref):
            fig_test.subplots().plot([1, 3, 5])
            fig_ref.subplots().plot([0, 1, 2], [1, 3, 5])

    """
    # 允许的字符集合，用于检查参数名是否合法
    ALLOWED_CHARS = set(string.digits + string.ascii_letters + '_-[]()')
    # 关键字参数类型，仅允许在参数列表中使用的参数
    KEYWORD_ONLY = inspect.Parameter.KEYWORD_ONLY
    def decorator(func):
        # 导入 pytest 模块，用于参数化测试
        import pytest

        # 获取函数的路径和结果目录
        _, result_dir = _image_directories(func)
        
        # 获取原始函数的参数签名
        old_sig = inspect.signature(func)

        # 检查函数签名是否包含必需的参数 'fig_test' 和 'fig_ref'
        if not {"fig_test", "fig_ref"}.issubset(old_sig.parameters):
            raise ValueError("The decorated function must have at least the "
                             "parameters 'fig_test' and 'fig_ref', but your "
                             f"function has the signature {old_sig}")

        # 定义装饰函数的包装器，参数化测试用例
        @pytest.mark.parametrize("ext", extensions)
        def wrapper(*args, ext, request, **kwargs):
            # 如果原始函数有 'ext' 参数，则将其传递给 kwargs
            if 'ext' in old_sig.parameters:
                kwargs['ext'] = ext
            # 如果原始函数有 'request' 参数，则将其传递给 kwargs
            if 'request' in old_sig.parameters:
                kwargs['request'] = request

            # 根据 pytest 的 node 名称生成文件名，仅保留 ALLOWED_CHARS 中的字符
            file_name = "".join(c for c in request.node.name
                                if c in ALLOWED_CHARS)
            
            try:
                # 创建名为 "test" 和 "reference" 的图形对象
                fig_test = plt.figure("test")
                fig_ref = plt.figure("reference")
                
                # 使用 _collect_new_figures 捕获新创建的图形对象
                with _collect_new_figures() as figs:
                    func(*args, fig_test=fig_test, fig_ref=fig_ref, **kwargs)
                
                # 如果有新创建的图形对象，则抛出异常
                if figs:
                    raise RuntimeError('Number of open figures changed during '
                                       'test. Make sure you are plotting to '
                                       'fig_test or fig_ref, or if this is '
                                       'deliberate explicitly close the '
                                       'new figure(s) inside the test.')
                
                # 设置测试结果图像路径和参考图像路径
                test_image_path = result_dir / (file_name + "." + ext)
                ref_image_path = result_dir / (file_name + "-expected." + ext)
                
                # 将图形保存为文件
                fig_test.savefig(test_image_path)
                fig_ref.savefig(ref_image_path)
                
                # 检查测试图像和参考图像的差异
                _raise_on_image_difference(
                    ref_image_path, test_image_path, tol=tol
                )
            
            finally:
                # 关闭图形对象
                plt.close(fig_test)
                plt.close(fig_ref)

        # 从原始参数中筛选出不是 'fig_test' 和 'fig_ref' 的参数
        parameters = [
            param
            for param in old_sig.parameters.values()
            if param.name not in {"fig_test", "fig_ref"}
        ]

        # 如果原始函数没有 'ext' 参数，则添加该参数
        if 'ext' not in old_sig.parameters:
            parameters += [inspect.Parameter("ext", KEYWORD_ONLY)]

        # 如果原始函数没有 'request' 参数，则添加该参数
        if 'request' not in old_sig.parameters:
            parameters += [inspect.Parameter("request", KEYWORD_ONLY)]

        # 创建新的函数签名
        new_sig = old_sig.replace(parameters=parameters)

        # 将新的函数签名赋给包装器函数
        wrapper.__signature__ = new_sig

        # 从被装饰函数获取 pytest 标记，并添加到包装器函数的 pytestmark 属性中
        new_marks = getattr(func, "pytestmark", []) + wrapper.pytestmark
        wrapper.pytestmark = new_marks

        # 返回包装器函数
        return wrapper

    # 返回装饰器函数
    return decorator
def _image_directories(func):
    """
    Compute the baseline and result image directories for testing *func*.

    For test module ``foo.bar.test_baz``, the baseline directory is at
    ``foo/bar/baseline_images/test_baz`` and the result directory at
    ``$(pwd)/result_images/test_baz``.  The result directory is created if it
    doesn't exist.
    """
    # 获取函数 *func* 所在模块的路径
    module_path = Path(inspect.getfile(func))
    # 根据模块路径确定基准图像目录，例如 foo/bar/baseline_images/test_baz
    baseline_dir = module_path.parent / "baseline_images" / module_path.stem
    # 确定结果图像目录，当前工作目录下的 result_images/test_baz
    result_dir = Path().resolve() / "result_images" / module_path.stem
    # 如果结果目录不存在，则创建
    result_dir.mkdir(parents=True, exist_ok=True)
    # 返回基准目录和结果目录的元组
    return baseline_dir, result_dir
```