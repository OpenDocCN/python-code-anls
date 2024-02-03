# `stable-diffusion-webui\modules\errors.py`

```py
# 导入 sys 模块，用于访问与 Python 解释器交互的变量和函数
import sys
# 导入 textwrap 模块，用于格式化文本块
import textwrap
# 导入 traceback 模块，用于提取和格式化异常的回溯信息


# 用于存储异常记录的列表
exception_records = []


# 格式化回溯信息，返回包含文件名、行号、函数名和代码行的列表
def format_traceback(tb):
    return [[f"{x.filename}, line {x.lineno}, {x.name}", x.line] for x in traceback.extract_tb(tb)]


# 格式化异常信息，返回包含异常消息和回溯信息的字典
def format_exception(e, tb):
    return {"exception": str(e), "traceback": format_traceback(tb)}


# 获取异常记录列表
def get_exceptions():
    try:
        return list(reversed(exception_records))
    except Exception as e:
        return str(e)


# 记录异常信息
def record_exception():
    _, e, tb = sys.exc_info()
    if e is None:
        return

    if exception_records and exception_records[-1] == e:
        return

    exception_records.append(format_exception(e, tb))

    if len(exception_records) > 5:
        exception_records.pop(0)


# 打印错误消息到 stderr，可选择是否打印回溯信息
def report(message: str, *, exc_info: bool = False) -> None:
    """
    Print an error message to stderr, with optional traceback.
    """

    record_exception()

    for line in message.splitlines():
        print("***", line, file=sys.stderr)
    if exc_info:
        print(textwrap.indent(traceback.format_exc(), "    "), file=sys.stderr)
        print("---", file=sys.stderr)


# 打印错误解释信息到 stderr
def print_error_explanation(message):
    record_exception()

    lines = message.strip().split("\n")
    max_len = max([len(x) for x in lines])

    print('=' * max_len, file=sys.stderr)
    for line in lines:
        print(line, file=sys.stderr)
    print('=' * max_len, file=sys.stderr)


# 显示异常信息，包括任务名称、异常类型和完整回溯信息
def display(e: Exception, task, *, full_traceback=False):
    record_exception()

    print(f"{task or 'error'}: {type(e).__name__}", file=sys.stderr)
    te = traceback.TracebackException.from_exception(e)
    if full_traceback:
        # include frames leading up to the try-catch block
        te.stack = traceback.StackSummary(traceback.extract_stack()[:-2] + te.stack)
    print(*te.format(), sep="", file=sys.stderr)

    message = str(e)
    # 如果消息中包含指定的字符串
    if "copying a param with shape torch.Size([640, 1024]) from checkpoint, the shape in current model is torch.Size([640, 768])" in message:
        # 打印错误解释
        print_error_explanation("""
# 最有可能的原因是您尝试加载 Stable Diffusion 2.0 模型，但没有指定其配置文件。
# 请查看 https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#stable-diffusion-20 以了解如何解决此问题。
        """)


already_displayed = {}


def display_once(e: Exception, task):
    # 记录异常
    record_exception()

    # 如果任务已经显示过，则直接返回
    if task in already_displayed:
        return

    # 显示异常和任务
    display(e, task)

    # 标记任务已经显示过
    already_displayed[task] = 1


def run(code, task):
    try:
        # 执行传入的代码块
        code()
    except Exception as e:
        # 显示任务和异常信息
        display(task, e)


def check_versions():
    from packaging import version
    from modules import shared

    import torch
    import gradio

    expected_torch_version = "2.0.0"
    expected_xformers_version = "0.0.20"
    expected_gradio_version = "3.41.2"

    # 检查 Torch 版本是否符合预期
    if version.parse(torch.__version__) < version.parse(expected_torch_version):
        print_error_explanation(f"""
You are running torch {torch.__version__}.
The program is tested to work with torch {expected_torch_version}.
To reinstall the desired version, run with commandline flag --reinstall-torch.
Beware that this will cause a lot of large files to be downloaded, as well as
there are reports of issues with training tab on the latest version.

Use --skip-version-check commandline argument to disable this check.
        """.strip())

    if shared.xformers_available:
        import xformers

        # 检查 Xformers 版本是否符合预期
        if version.parse(xformers.__version__) < version.parse(expected_xformers_version):
            print_error_explanation(f"""
You are running xformers {xformers.__version__}.
The program is tested to work with xformers {expected_xformers_version}.
To reinstall the desired version, run with commandline flag --reinstall-xformers.

Use --skip-version-check commandline argument to disable this check.
            """.strip())

    # 检查 Gradio 版本是否符合预期
    if gradio.__version__ != expected_gradio_version:
        print_error_explanation(f"""
You are running gradio {gradio.__version__}.
# 程序设计用于与 gradio {expected_gradio_version} 版本配合使用。
# 使用不同版本的 gradio 非常可能会导致程序出错。
# 出现不匹配的 gradio 版本的原因可能是：
#   - 使用了 --skip-install 标志。
#   - 使用 webui.py 启动程序而不是 launch.py。
#   - 扩展安装了不兼容的 gradio 版本。
# 使用 --skip-version-check 命令行参数来禁用此检查。
```