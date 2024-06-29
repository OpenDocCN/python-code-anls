# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_animation.py`

```
# 导入标准库和第三方库
import os  # 操作系统相关功能
from pathlib import Path  # 处理路径相关操作
import platform  # 获取平台信息
import re  # 正则表达式库
import shutil  # 文件操作相关功能
import subprocess  # 执行外部命令
import sys  # 系统相关功能
import weakref  # 弱引用支持

import numpy as np  # 数值计算库
import pytest  # 测试框架

import matplotlib as mpl  # 绘图库
from matplotlib import pyplot as plt  # 绘图功能
from matplotlib import animation  # 动画支持
from matplotlib.testing.decorators import check_figures_equal  # 测试绘图相关功能

@pytest.fixture()
def anim(request):
    """创建一个简单的动画（带有选项）。"""
    # 创建图形和轴对象
    fig, ax = plt.subplots()
    # 创建一个空的线条对象
    line, = ax.plot([], [])

    # 设置轴的范围
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 1)

    # 初始化函数，清空线条数据
    def init():
        line.set_data([], [])
        return line,

    # 动画函数，更新线条的数据
    def animate(i):
        x = np.linspace(0, 10, 100)
        y = np.sin(x + i)
        line.set_data(x, y)
        return line,

    # 如果请求中包含参数，复制一份
    kwargs = dict(getattr(request, 'param', {}))  # make a copy
    # 获取或默认使用的动画类
    klass = kwargs.pop('klass', animation.FuncAnimation)
    # 如果参数中没有指定帧数，设定默认值为5
    if 'frames' not in kwargs:
        kwargs['frames'] = 5
    # 返回动画对象
    return klass(fig=fig, func=animate, init_func=init, **kwargs)


class NullMovieWriter(animation.AbstractMovieWriter):
    """
    一个最小的 MovieWriter。它并不实际写入任何东西。
    它只是保存给 setup() 和 grab_frame() 方法传递的参数作为属性，
    并计算 grab_frame() 方法被调用的次数。

    这个类没有带有正确签名的 __init__ 方法，并且没有定义 isAvailable() 方法，
    因此无法被添加到 'writers' 注册表中。
    """

    def setup(self, fig, outfile, dpi, *args):
        # 设置要保存的图形、输出文件名、DPI 和其他参数
        self.fig = fig
        self.outfile = outfile
        self.dpi = dpi
        self.args = args
        self._count = 0

    def grab_frame(self, **savefig_kwargs):
        # 验证并保存 grab_frame() 方法的参数
        from matplotlib.animation import _validate_grabframe_kwargs
        _validate_grabframe_kwargs(savefig_kwargs)
        self.savefig_kwargs = savefig_kwargs
        self._count += 1

    def finish(self):
        # 完成保存动作，这里为空实现
        pass


def test_null_movie_writer(anim):
    # 使用 NullMovieWriter 测试运行动画
    plt.rcParams["savefig.facecolor"] = "auto"
    filename = "unused.null"
    dpi = 50
    savefig_kwargs = dict(foo=0)
    writer = NullMovieWriter()

    # 保存动画
    anim.save(filename, dpi=dpi, writer=writer,
              savefig_kwargs=savefig_kwargs)

    # 断言动画的属性与预期值相等
    assert writer.fig == plt.figure(1)  # The figure used by anim fixture
    assert writer.outfile == filename
    assert writer.dpi == dpi
    assert writer.args == ()
    # 检查保存参数的关键字
    for k, v in savefig_kwargs.items():
        assert writer.savefig_kwargs[k] == v
    assert writer._count == anim._save_count


@pytest.mark.parametrize('anim', [dict(klass=dict)], indirect=['anim'])
def test_animation_delete(anim):
    # 测试删除动画
    # 检查当前 Python 解释器是否为 PyPy
    if platform.python_implementation() == 'PyPy':
        # 在 PyPy 上，测试设置夹具中的某些内容会残留到测试中，
        # 导致 pytest.warns 失效。这里进行垃圾回收以修复该问题。
        # 参考：https://foss.heptapod.net/pypy/pypy/-/issues/3536
        np.testing.break_cycles()

    # 创建动画对象，并使用 animation.FuncAnimation 初始化
    anim = animation.FuncAnimation(**anim)

    # 使用 pytest.warns 捕获 Warning 类型的异常，匹配字符串 'Animation was deleted'
    with pytest.warns(Warning, match='Animation was deleted'):
        # 删除动画对象以触发警告
        del anim
        # 手动调用 np.testing.break_cycles() 以确保对象循环引用被正确处理
        np.testing.break_cycles()
# 定义一个测试函数，用于测试默认设置下的电影写入器行为
def test_movie_writer_dpi_default():
    # 定义一个虚拟的电影写入器类，继承自animation.MovieWriter，但实际上没有实现_run方法
    class DummyMovieWriter(animation.MovieWriter):
        def _run(self):
            pass

    # 创建一个新的绘图窗口对象
    fig = plt.figure()

    # 定义文件名，但实际上未使用
    filename = "unused.null"
    # 设定帧率为5帧每秒
    fps = 5
    # 定义编解码器名称，但实际上未使用
    codec = "unused"
    # 设定比特率为1
    bitrate = 1
    # 定义额外参数列表，但实际上未使用
    extra_args = ["unused"]

    # 创建一个DummyMovieWriter实例对象，传入帧率、编解码器、比特率和额外参数
    writer = DummyMovieWriter(fps, codec, bitrate, extra_args)
    # 设置绘图对象fig为电影写入器的配置
    writer.setup(fig, filename)
    # 断言电影写入器的dpi设置与绘图窗口的dpi设置相同
    assert writer.dpi == fig.dpi


# 使用装饰器注册一个自定义的NullMovieWriter类，以便将其添加到'writers'注册表中
@animation.writers.register('null')
class RegisteredNullMovieWriter(NullMovieWriter):

    # 为了能够将NullMovieWriter添加到'writers'注册表中，
    # 我们必须定义一个带有特定签名的__init__方法，
    # 并且必须定义类方法isAvailable()。
    # （实际上，这些方法并不是使用此类作为Animation.save()的'writer'参数所必需的。）

    # 定义构造方法__init__，接受帧率、编解码器、比特率、额外参数和元数据作为参数
    def __init__(self, fps=None, codec=None, bitrate=None,
                 extra_args=None, metadata=None):
        pass

    # 定义类方法isAvailable()，始终返回True，表示该写入器始终可用
    @classmethod
    def isAvailable(cls):
        return True


# 定义一个包含多个元组的常量列表WRITER_OUTPUT，每个元组包含写入器名称和输出文件名
WRITER_OUTPUT = [
    ('ffmpeg', 'movie.mp4'),
    ('ffmpeg_file', 'movie.mp4'),
    ('imagemagick', 'movie.gif'),
    ('imagemagick_file', 'movie.gif'),
    ('pillow', 'movie.gif'),
    ('html', 'movie.html'),
    ('null', 'movie.null')
]


# 定义一个生成器函数gen_writers()，用于生成写入器名称和输出格式的参数组合
def gen_writers():
    # 遍历WRITER_OUTPUT列表中的每个元组
    for writer, output in WRITER_OUTPUT:
        # 如果当前写入器不可用，则跳过当前循环
        if not animation.writers.is_available(writer):
            # 创建一个pytest.mark.skip标记，说明当前写入器在此系统上不可用
            mark = pytest.mark.skip(
                f"writer '{writer}' not available on this system")
            # 生成一个带有pytest.mark.skip标记的参数组合，包含写入器名称、None和输出文件名
            yield pytest.param(writer, None, output, marks=[mark])
            # 生成一个带有pytest.mark.skip标记的参数组合，包含写入器名称、None和输出路径
            yield pytest.param(writer, None, Path(output), marks=[mark])
            # 继续下一次循环
            continue

        # 获取当前写入器对应的写入器类
        writer_class = animation.writers[writer]
        # 遍历当前写入器类的支持格式列表或者默认值[None]
        for frame_format in getattr(writer_class, 'supported_formats', [None]):
            # 生成写入器名称、帧格式和输出文件名的参数组合
            yield writer, frame_format, output
            # 生成写入器名称、帧格式和输出路径的参数组合
            yield writer, frame_format, Path(output)


# 定义一个用于保存动画的简单测试函数test_save_animation_smoketest。
# 未来，我们可能需要设计更复杂的测试，例如通过比较生成的帧来进行测试。
@pytest.mark.parametrize('writer, frame_format, output', gen_writers())
@pytest.mark.parametrize('anim', [dict(klass=dict)], indirect=['anim'])
def test_save_animation_smoketest(tmpdir, writer, frame_format, output, anim):
    # 如果指定了帧格式，则设置全局绘图参数中的动画帧格式
    if frame_format is not None:
        plt.rcParams["animation.frame_format"] = frame_format
    # 创建一个FuncAnimation对象anim，根据传入的参数字典创建动画
    anim = animation.FuncAnimation(**anim)
    # 初始化dpi和codec为None
    dpi = None
    codec = None
    # 如果当前写入器为'ffmpeg'
    if writer == 'ffmpeg':
        # 修复问题#8253
        anim._fig.set_size_inches((10.85, 9.21))
        # 设置dpi为100.0
        dpi = 100.
        # 设置codec为'h264'
        codec = 'h264'

    # 使用临时目录tmpdir作为当前工作目录
    with tmpdir.as_cwd():
        # 将动画anim保存到指定的输出文件中，设定帧率为30，写入器为writer，比特率为500，dpi为dpi，编解码器为codec
        anim.save(output, fps=30, writer=writer, bitrate=500, dpi=dpi,
                  codec=codec)

    # 删除anim对象，释放资源
    del anim


# 通过参数化测试，遍历所有的写入器、帧格式和输出格式组合，对动画保存功能进行测试
@pytest.mark.parametrize('writer, frame_format, output', gen_writers())
# 测试抓取帧函数，用于生成动画帧并保存为指定格式
def test_grabframe(tmpdir, writer, frame_format, output):
    # 根据 writer 参数选择相应的动画写入器类
    WriterClass = animation.writers[writer]

    # 如果指定了帧格式，设置 matplotlib 的动画帧格式参数
    if frame_format is not None:
        plt.rcParams["animation.frame_format"] = frame_format

    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()

    dpi = None
    codec = None
    # 如果使用 'ffmpeg' 写入器，设置特定的图形尺寸、分辨率和编解码器
    if writer == 'ffmpeg':
        # 修复问题 #8253
        fig.set_size_inches((10.85, 9.21))
        dpi = 100.
        codec = 'h264'

    # 创建测试用的动画写入器对象
    test_writer = WriterClass()
    
    # 使用临时目录作为文件型写入器的当前工作目录，每帧生成一个已知名称的文件
    with tmpdir.as_cwd():
        with test_writer.saving(fig, output, dpi):
            # 确认抓取帧函数正常工作
            test_writer.grab_frame()
            # 针对抓取帧函数的几个关键字参数进行异常测试
            for k in {'dpi', 'bbox_inches', 'format'}:
                with pytest.raises(
                        TypeError,
                        match=f"grab_frame got an unexpected keyword argument {k!r}"
                ):
                    test_writer.grab_frame(**{k: object()})


# 参数化测试：验证动画在不同输出格式下的 HTML 表示
@pytest.mark.parametrize('writer', [
    pytest.param(
        'ffmpeg', marks=pytest.mark.skipif(
            not animation.FFMpegWriter.isAvailable(),
            reason='Requires FFMpeg')),
    pytest.param(
        'imagemagick', marks=pytest.mark.skipif(
            not animation.ImageMagickWriter.isAvailable(),
            reason='Requires ImageMagick')),
])
@pytest.mark.parametrize('html, want', [
    ('none', None),
    ('html5', '<video width'),
    ('jshtml', '<script ')
])
@pytest.mark.parametrize('anim', [dict(klass=dict)], indirect=['anim'])
def test_animation_repr_html(writer, html, want, anim):
    # 在 PyPy 环境下修复 pytest.warns 的问题
    if platform.python_implementation() == 'PyPy':
        # 某些测试设置装置在测试过程中持续存在，并且会破坏 PyPy 上的 pytest.warns
        # 这里进行垃圾回收以修复问题
        np.testing.break_cycles()
    
    # 对于 'imagemagick' 写入器且输出格式为 'html5'，需要检查是否有 FFMpeg 支持
    if (writer == 'imagemagick' and html == 'html5'
            and not animation.FFMpegWriter.isAvailable()):
        pytest.skip('Requires FFMpeg')

    # 在此处创建动画对象，而不是在 fixture 中创建，以避免 __del__ 警告
    anim = animation.FuncAnimation(**anim)
    
    # 设置 matplotlib 的动画写入器和 HTML 表示参数，并获取 HTML 表示
    with plt.rc_context({'animation.writer': writer,
                         'animation.html': html}):
        html = anim._repr_html_()
    
    # 验证 HTML 表示是否符合预期
    if want is None:
        assert html is None
        with pytest.warns(UserWarning):
            del anim  # 由于动画未运行，所以清理时会警告
            np.testing.break_cycles()
    else:
        assert want in html


# 参数化测试：验证在没有帧长度的情况下保存动画
@pytest.mark.parametrize(
    'anim',
    [{'save_count': 10, 'frames': iter(range(5))}],
    indirect=['anim']
)
def test_no_length_frames(anim):
    # 测试在没有帧长度的情况下保存动画，使用 NullMovieWriter 作为写入器
    anim.save('unused.null', writer=NullMovieWriter())


# 测试动画写入器注册表的状态
def test_movie_writer_registry():
    # 验证动画写入器注册表中至少有一个写入器已注册
    assert len(animation.writers._registered) > 0
    
    # 设置 matplotlib 的动画 'ffmpeg_path' 参数为无效路径
    mpl.rcParams['animation.ffmpeg_path'] = "not_available_ever_xxxx"
    # 检查动画库中的ffmpeg是否不可用，如果不可用则触发断言错误
    assert not animation.writers.is_available("ffmpeg")
    
    # 根据操作系统选择一个合适的二进制文件路径
    bin = "true" if sys.platform != 'win32' else "where"
    
    # 设置Matplotlib配置以使用选择的二进制文件路径作为动画的ffmpeg路径
    mpl.rcParams['animation.ffmpeg_path'] = bin
    
    # 再次检查动画库中的ffmpeg是否可用，确保路径设置成功
    assert animation.writers.is_available("ffmpeg")
@pytest.mark.parametrize(
    "method_name",
    [pytest.param("to_html5_video", marks=pytest.mark.skipif(
        not animation.writers.is_available(mpl.rcParams["animation.writer"]),
        reason="animation writer not installed")),
     "to_jshtml"])
@pytest.mark.parametrize('anim', [dict(frames=1)], indirect=['anim'])
def test_embed_limit(method_name, caplog, tmpdir, anim):
    caplog.set_level("WARNING")
    # 设置当前工作目录为 tmpdir
    with tmpdir.as_cwd():
        # 在 mpl 的上下文中设置动画嵌入限制为极小值，大约为1字节
        with mpl.rc_context({"animation.embed_limit": 1e-6}):  # ~1 byte.
            # 调用动态获取的动画对象的指定方法名
            getattr(anim, method_name)()
    # 断言日志记录条数为1
    assert len(caplog.records) == 1
    record, = caplog.records
    # 断言记录的日志名称和级别为 WARNING
    assert (record.name == "matplotlib.animation"
            and record.levelname == "WARNING")


@pytest.mark.parametrize(
    "method_name",
    [pytest.param("to_html5_video", marks=pytest.mark.skipif(
        not animation.writers.is_available(mpl.rcParams["animation.writer"]),
        reason="animation writer not installed")),
     "to_jshtml"])
@pytest.mark.parametrize('anim', [dict(frames=1)], indirect=['anim'])
def test_cleanup_temporaries(method_name, tmpdir, anim):
    # 设置当前工作目录为 tmpdir
    with tmpdir.as_cwd():
        # 调用动态获取的动画对象的指定方法名
        getattr(anim, method_name)()
        # 断言临时目录中的文件列表为空
        assert list(Path(str(tmpdir)).iterdir()) == []


@pytest.mark.skipif(shutil.which("/bin/sh") is None, reason="requires a POSIX OS")
def test_failing_ffmpeg(tmpdir, monkeypatch, anim):
    """
    Test that we correctly raise a CalledProcessError when ffmpeg fails.

    To do so, mock ffmpeg using a simple executable shell script that
    succeeds when called with no arguments (so that it gets registered by
    `isAvailable`), but fails otherwise, and add it to the $PATH.
    """
    # 设置当前工作目录为 tmpdir
    with tmpdir.as_cwd():
        # 设置环境变量 PATH 包含当前目录和系统 PATH
        monkeypatch.setenv("PATH", ".:" + os.environ["PATH"])
        # 创建临时目录中的 ffmpeg 可执行文件
        exe_path = Path(str(tmpdir), "ffmpeg")
        exe_path.write_bytes(b"#!/bin/sh\n[[ $@ -eq 0 ]]\n")
        os.chmod(exe_path, 0o755)
        # 使用 pytest 的断言检查是否抛出 subprocess.CalledProcessError 异常
        with pytest.raises(subprocess.CalledProcessError):
            anim.save("test.mpeg")


@pytest.mark.parametrize("cache_frame_data", [False, True])
def test_funcanimation_cache_frame_data(cache_frame_data):
    fig, ax = plt.subplots()
    line, = ax.plot([], [])

    class Frame(dict):
        # this subclassing enables to use weakref.ref()
        pass

    def init():
        line.set_data([], [])
        return line,

    def animate(frame):
        line.set_data(frame['x'], frame['y'])
        return line,

    frames_generated = []

    def frames_generator():
        for _ in range(5):
            x = np.linspace(0, 10, 100)
            y = np.random.rand(100)

            frame = Frame(x=x, y=y)

            # collect weak references to frames
            # to validate their references later
            frames_generated.append(weakref.ref(frame))

            yield frame

    MAX_FRAMES = 100
    # 使用 FuncAnimation 创建动画对象 anim，指定以下参数：
    # - fig: 动画绘制的图形对象
    # - animate: 动画每帧更新的函数
    # - init_func: 初始化函数
    # - frames: 帧生成器函数，用于生成动画的每一帧
    # - cache_frame_data: 控制是否缓存帧数据的布尔值
    # - save_count: 最大帧数
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=frames_generator,
                                   cache_frame_data=cache_frame_data,
                                   save_count=MAX_FRAMES)

    # 创建一个 NullMovieWriter 对象 writer，用于保存动画，实际上并不会保存
    writer = NullMovieWriter()
    # 保存动画，输出到名为 'unused.null' 的文件中，使用 NullMovieWriter 写入
    anim.save('unused.null', writer=writer)

    # 断言生成的帧数为 5
    assert len(frames_generated) == 5

    # 执行 NumPy 测试，确保没有循环引用问题
    np.testing.break_cycles()

    # 遍历每个生成的帧 f
    for f in frames_generated:
        # 断言：如果 cache_frame_data 为 True，则弱引用应该存在；
        # 如果 cache_frame_data 为 False，则弱引用应该为 None。
        assert (f() is None) != cache_frame_data
@pytest.mark.parametrize('return_value', [
    # 参数化测试的返回值为 None，测试场景中用户忘记了返回值（返回 None）。
    None,
    # 参数化测试的返回值为字符串 'string'。
    'string',
    # 参数化测试的返回值为整数 1。
    1,
    # 参数化测试的返回值为元组 ('string', )，其中包含一个字符串，而非预期的序列。
    ('string', ),
    # 参数化测试的返回值为字符串 'artist'，但实际上并不是一个序列。
    # 在 `animate` 方法中会处理这种情况。
    'artist',
])
def test_draw_frame(return_value):
    # 测试 _draw_frame 方法

    # 创建一个新的图形和轴
    fig, ax = plt.subplots()
    # 在轴上绘制一个空的线条
    line, = ax.plot([])

    def animate(i):
        # 更新函数，每次更新将线条的数据设为 [0, 1] 和 [0, i]
        line.set_data([0, 1], [0, i])
        if return_value == 'artist':
            # 如果返回值是 'artist'，则返回线条对象 line
            return line
        else:
            # 否则返回参数化测试的返回值
            return return_value

    # 断言运行时错误，确保在调用 animation.FuncAnimation 时会抛出 RuntimeError 异常
    with pytest.raises(RuntimeError):
        animation.FuncAnimation(
            fig, animate, blit=True, cache_frame_data=False
        )


def test_exhausted_animation(tmpdir):
    # 测试在动画帧迭代完后会触发警告的情况
    fig, ax = plt.subplots()

    def update(frame):
        return []

    # 创建动画对象
    anim = animation.FuncAnimation(
        fig, update, frames=iter(range(10)), repeat=False,
        cache_frame_data=False
    )

    # 切换到临时目录进行操作
    with tmpdir.as_cwd():
        # 将动画保存为 GIF
        anim.save("test.gif", writer='pillow')

    # 使用 pytest.warns 断言确保会触发 UserWarning 警告，匹配警告消息中包含 'exhausted' 字样
    with pytest.warns(UserWarning, match="exhausted"):
        anim._start()


def test_no_frame_warning(tmpdir):
    # 测试当帧列表为空时会触发警告的情况
    fig, ax = plt.subplots()

    def update(frame):
        return []

    # 创建动画对象，帧列表为空
    anim = animation.FuncAnimation(
        fig, update, frames=[], repeat=False,
        cache_frame_data=False
    )

    # 使用 pytest.warns 断言确保会触发 UserWarning 警告，匹配警告消息中包含 'exhausted' 字样
    with pytest.warns(UserWarning, match="exhausted"):
        anim._start()


@check_figures_equal(extensions=["png"])
def test_animation_frame(tmpdir, fig_test, fig_ref):
    # 测试动画帧迭代后的预期图像

    # 在测试图形上添加子图
    ax = fig_test.add_subplot()
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1, 1)
    x = np.linspace(0, 2 * np.pi, 100)
    # 在子图上绘制空线条
    line, = ax.plot([], [])

    def init():
        # 初始化函数，将线条的数据设为空
        line.set_data([], [])
        return line,

    def animate(i):
        # 动画更新函数，更新线条的数据为 sin 曲线
        line.set_data(x, np.sin(x + i / 100))
        return line,

    # 创建 FuncAnimation 对象来生成动画，迭代 5 帧
    anim = animation.FuncAnimation(
        fig_test, animate, init_func=init, frames=5,
        blit=True, repeat=False)
    
    # 在临时目录下保存动画为 GIF
    with tmpdir.as_cwd():
        anim.save("test.gif")

    # 参考图像中不包含动画，仅绘制第 5 帧的数据
    ax = fig_ref.add_subplot()
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1, 1)
    ax.plot(x, np.sin(x + 4 / 100))


@pytest.mark.parametrize('anim', [dict(klass=dict)], indirect=['anim'])
def test_save_count_override_warnings_has_length(anim):
    # 测试覆盖保存计数时会触发警告

    save_count = 5
    frames = list(range(2))
    match_target = (
        f'You passed in an explicit {save_count=} '
        "which is being ignored in favor of "
        f"{len(frames)=}."
    )
    # 使用 pytest 来检查是否有特定类型的警告，并且警告消息需要与 match_target 相匹配
    with pytest.warns(UserWarning, match=re.escape(match_target)):
        # 创建一个 FuncAnimation 动画对象，使用指定的参数来初始化
        anim = animation.FuncAnimation(
            **{**anim, 'frames': frames, 'save_count': save_count}
        )
    # 断言动画对象的 _save_count 属性等于 frames 列表的长度
    assert anim._save_count == len(frames)
    # 执行动画的初始化绘制操作
    anim._init_draw()
# 使用 pytest 框架的 parametrize 装饰器标记这个测试函数，其中 anim 参数是通过间接方式传递的字典对象
@pytest.mark.parametrize('anim', [dict(klass=dict)], indirect=['anim'])
# 定义一个测试函数，测试保存计数的覆盖警告和缩放器
def test_save_count_override_warnings_scaler(anim):
    # 设置保存计数和帧数
    save_count = 5
    frames = 7
    # 构造警告匹配的目标字符串，用于检查警告消息内容
    match_target = (
        f'You passed in an explicit {save_count=} ' +
        "which is being ignored in favor of " +
        f"{frames=}."
    )

    # 使用 pytest 的 warn 方法检查是否有特定警告，并匹配给定的目标字符串
    with pytest.warns(UserWarning, match=re.escape(match_target)):
        # 创建动画对象 animation.FuncAnimation，传入参数 anim 字典以及 frames 和 save_count
        anim = animation.FuncAnimation(
            **{**anim, 'frames': frames, 'save_count': save_count}
        )

    # 断言动画对象的保存计数属性等于帧数
    assert anim._save_count == frames
    # 初始化绘图
    anim._init_draw()


# 使用 pytest 框架的 parametrize 装饰器标记这个测试函数，其中 anim 参数是通过间接方式传递的字典对象
@pytest.mark.parametrize('anim', [dict(klass=dict)], indirect=['anim'])
# 定义一个测试函数，测试禁用缓存警告
def test_disable_cache_warning(anim):
    # 设置是否缓存帧数据和帧迭代器
    cache_frame_data = True
    frames = iter(range(5))
    # 构造警告匹配的目标字符串，用于检查警告消息内容
    match_target = (
        f"{frames=!r} which we can infer the length of, "
        "did not pass an explicit *save_count* "
        f"and passed {cache_frame_data=}.  To avoid a possibly "
        "unbounded cache, frame data caching has been disabled. "
        "To suppress this warning either pass "
        "`cache_frame_data=False` or `save_count=MAX_FRAMES`."
    )
    # 使用 pytest 的 warn 方法检查是否有特定警告，并匹配给定的目标字符串
    with pytest.warns(UserWarning, match=re.escape(match_target)):
        # 创建动画对象 animation.FuncAnimation，传入参数 anim 字典以及 cache_frame_data 和 frames
        anim = animation.FuncAnimation(
            **{**anim, 'cache_frame_data': cache_frame_data, 'frames': frames}
        )
    # 断言动画对象的缓存帧数据属性为 False
    assert anim._cache_frame_data is False
    # 初始化绘图
    anim._init_draw()


# 定义一个测试函数，测试无效路径的影片写入器
def test_movie_writer_invalid_path(anim):
    # 如果操作系统是 Windows，设置匹配字符串为 Windows 环境下的错误信息
    if sys.platform == "win32":
        match_str = r"\[WinError 3] .*'\\\\foo\\\\bar\\\\aardvark'"
    else:
        match_str = r"\[Errno 2] .*'/foo"
    # 使用 pytest 的 raises 方法检查是否引发指定异常，并匹配给定的匹配字符串
    with pytest.raises(FileNotFoundError, match=match_str):
        # 调用动画对象的保存方法，尝试保存到一个不存在的路径，指定使用 FFMpegFileWriter 写入器
        anim.save("/foo/bar/aardvark/thiscannotreallyexist.mp4",
                  writer=animation.FFMpegFileWriter())
```