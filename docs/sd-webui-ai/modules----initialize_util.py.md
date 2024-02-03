# `stable-diffusion-webui\modules\initialize_util.py`

```
import json
import os
import signal
import sys
import re

from modules.timer import startup_timer

# 定义函数用于获取 Gradio 服务器名称
def gradio_server_name():
    from modules.shared_cmd_options import cmd_opts

    # 如果命令行参数中指定了服务器名称，则返回该名称
    if cmd_opts.server_name:
        return cmd_opts.server_name
    # 否则，如果监听选项为真，则返回默认服务器名称 "0.0.0.0"
    else:
        return "0.0.0.0" if cmd_opts.listen else None

# 定义函数用于修复 Torch 版本号
def fix_torch_version():
    import torch

    # 如果 Torch 版本号为 nightly/local build，则截断版本号以避免与 CodeFormer 或 Safetensors 引起异常
    if ".dev" in torch.__version__ or "+git" in torch.__version__:
        torch.__long_version__ = torch.__version__
        torch.__version__ = re.search(r'[\d.]+[\d]', torch.__version__).group(0)

# 定义函数用于修复 asyncio 事件循环策略
def fix_asyncio_event_loop_policy():
    """
        默认的 `asyncio` 事件循环策略只会在主线程中自动创建事件循环。
        其他线程必须显式创建事件循环，否则 `asyncio.get_event_loop`（因此 `.IOLoop.current`）将失败。
        安装此策略允许在任何线程上自动创建事件循环，与 Tornado 5.0 之前的行为匹配（或 Python 2 上的 5.0）。
    """

    import asyncio

    # 如果运行平台为 win32 并且 asyncio 模块中存在 WindowsSelectorEventLoopPolicy 属性
    if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        # 选择正确的基类策略
        _BasePolicy = asyncio.WindowsSelectorEventLoopPolicy  # type: ignore
    else:
        _BasePolicy = asyncio.DefaultEventLoopPolicy
    # 定义一个自定义的事件循环策略类，继承自 _BasePolicy
    class AnyThreadEventLoopPolicy(_BasePolicy):  # type: ignore
        """Event loop policy that allows loop creation on any thread.
        Usage::

            asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())
        """

        # 重写获取事件循环的方法
        def get_event_loop(self) -> asyncio.AbstractEventLoop:
            try:
                # 尝试获取当前线程的事件循环
                return super().get_event_loop()
            except (RuntimeError, AssertionError):
                # 捕获可能出现的异常，根据不同版本的 Python 抛出不同的异常
                # 在 Python 3.4.2 中是 AssertionError，在 3.4.3 中是 RuntimeError
                # "There is no current event loop in thread %r"
                # 创建一个新的事件循环
                loop = self.new_event_loop()
                # 设置当前线程的事件循环
                self.set_event_loop(loop)
                # 返回新创建的事件循环
                return loop

    # 设置全局的事件循环策略为自定义的 AnyThreadEventLoopPolicy
    asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())
# 恢复配置状态文件
def restore_config_state_file():
    # 导入模块 shared 和 config_states
    from modules import shared, config_states

    # 获取恢复配置状态文件路径
    config_state_file = shared.opts.restore_config_state_file
    # 如果配置状态文件为空，则返回
    if config_state_file == "":
        return

    # 清空恢复配置状态文件路径
    shared.opts.restore_config_state_file = ""
    # 保存配置
    shared.opts.save(shared.config_filename)

    # 如果配置状态文件存在
    if os.path.isfile(config_state_file):
        # 打印提示信息
        print(f"*** About to restore extension state from file: {config_state_file}")
        # 打开配置状态文件，读取内容
        with open(config_state_file, "r", encoding="utf-8") as f:
            # 加载 JSON 数据
            config_state = json.load(f)
            # 恢复扩展配置
            config_states.restore_extension_config(config_state)
        # 记录启动时间
        startup_timer.record("restore extension config")
    # 如果配置状态文件不存在
    elif config_state_file:
        # 打印警告信息
        print(f"!!! Config state backup not found: {config_state_file}")


# 验证 TLS 选项
def validate_tls_options():
    # 导入模块 shared_cmd_options 中的 cmd_opts
    from modules.shared_cmd_options import cmd_opts

    # 如果 TLS 密钥文件和证书文件路径不存在，则返回
    if not (cmd_opts.tls_keyfile and cmd_opts.tls_certfile):
        return

    try:
        # 如果 TLS 密钥文件路径不存在
        if not os.path.exists(cmd_opts.tls_keyfile):
            print("Invalid path to TLS keyfile given")
        # 如果 TLS 证书文件路径不存在
        if not os.path.exists(cmd_opts.tls_certfile):
            print(f"Invalid path to TLS certfile: '{cmd_opts.tls_certfile}'")
    except TypeError:
        # 设置 TLS 密钥文件和证书文件路径为空
        cmd_opts.tls_keyfile = cmd_opts.tls_certfile = None
        # 打印警告信息
        print("TLS setup invalid, running webui without TLS")
    else:
        # 打印提示信息
        print("Running with TLS")
    # 记录启动时间
    startup_timer.record("TLS")


# 获取 Gradio 认证凭据
def get_gradio_auth_creds():
    """
    Convert the gradio_auth and gradio_auth_path commandline arguments into
    an iterable of (username, password) tuples.
    """
    # 导入模块 shared_cmd_options 中的 cmd_opts

    # 处理凭据行，将 gradio_auth 和 gradio_auth_path 命令行参数转换为 (用户名, 密码) 元组的可迭代对象
    def process_credential_line(s):
        s = s.strip()
        if not s:
            return None
        return tuple(s.split(':', 1))

    # 如果存在 Gradio 认证凭据
    if cmd_opts.gradio_auth:
        # 遍历 Gradio 认证凭据
        for cred in cmd_opts.gradio_auth.split(','):
            # 处理凭据行
            cred = process_credential_line(cred)
            if cred:
                # 返回凭据
                yield cred
    # 如果命令行参数中包含 gradio_auth_path，则执行以下操作
    if cmd_opts.gradio_auth_path:
        # 打开指定路径的文件，以只读模式打开，使用 utf8 编码
        with open(cmd_opts.gradio_auth_path, 'r', encoding="utf8") as file:
            # 逐行读取文件内容
            for line in file.readlines():
                # 对每一行去除首尾空格并按逗号分割成列表
                for cred in line.strip().split(','):
                    # 处理每个凭证行，返回处理后的凭证
                    cred = process_credential_line(cred)
                    # 如果凭证有效，则生成凭证
                    if cred:
                        # 生成凭证
                        yield cred
# 定义一个函数，用于输出当前线程的堆栈信息
def dumpstacks():
    # 导入必要的模块
    import threading
    import traceback

    # 创建线程 ID 到线程名称的映射字典
    id2name = {th.ident: th.name for th in threading.enumerate()}
    # 初始化一个空列表用于存储堆栈信息
    code = []
    # 遍历当前所有线程的堆栈信息
    for threadId, stack in sys._current_frames().items():
        # 添加线程信息到列表中
        code.append(f"\n# Thread: {id2name.get(threadId, '')}({threadId})")
        # 遍历堆栈信息中的文件名、行号、函数名和代码行
        for filename, lineno, name, line in traceback.extract_stack(stack):
            # 添加文件名、行号、函数名到列表中
            code.append(f"""File: "{filename}", line {lineno}, in {name}""")
            # 如果代码行存在，则添加到列表中
            if line:
                code.append("  " + line.strip())

    # 打印堆栈信息
    print("\n".join(code))


# 配置 SIGINT 信号处理函数
def configure_sigint_handler():
    # 导入模块
    from modules import shared

    # 定义 SIGINT 信号处理函数
    def sigint_handler(sig, frame):
        # 打印中断信号和帧信息
        print(f'Interrupted with signal {sig} in {frame}')

        # 如果设置了选项 dump_stacks_on_signal，则输出堆栈信息
        if shared.opts.dump_stacks_on_signal:
            dumpstacks()

        # 立即退出程序
        os._exit(0)

    # 如果不是在 COVERAGE_RUN 环境下运行，则安装 SIGINT 信号处理函数
    if not os.environ.get("COVERAGE_RUN"):
        # 不在 coverage 下运行时，安装 SIGINT 信号处理函数
        signal.signal(signal.SIGINT, sigint_handler)


# 配置选项变更时的回调函数
def configure_opts_onchange():
    # 导入必要的模块
    from modules import shared, sd_models, sd_vae, ui_tempdir, sd_hijack
    from modules.call_queue import wrap_queued_call

    # 配置选项变更时的回调函数
    shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: sd_models.reload_model_weights()), call=False)
    shared.opts.onchange("sd_vae", wrap_queued_call(lambda: sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("sd_vae_overrides_per_model_preferences", wrap_queued_call(lambda: sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("temp_dir", ui_tempdir.on_tmpdir_changed)
    shared.opts.onchange("gradio_theme", shared.reload_gradio_theme)
    shared.opts.onchange("cross_attention_optimization", wrap_queued_call(lambda: sd_hijack.model_hijack.redo_hijack(shared.sd_model)), call=False)
    # 记录选项变更的启动时间
    startup_timer.record("opts onchange")


# 设置中间件
def setup_middleware(app):
    # 导入 GZipMiddleware 模块
    from starlette.middleware.gzip import GZipMiddleware
    
    # 重置当前中间件堆栈以允许修改用户提供的中间件列表
    app.middleware_stack = None  
    # 向应用程序添加 GZipMiddleware 中间件，设置最小压缩大小为1000字节
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    # 配置跨域资源共享中间件
    configure_cors_middleware(app)
    # 在运行时重新构建中间件堆栈
    app.build_middleware_stack()  
# 配置跨域资源共享（CORS）中间件
def configure_cors_middleware(app):
    # 导入 CORS 中间件
    from starlette.middleware.cors import CORSMiddleware
    # 导入共享命令选项模块
    from modules.shared_cmd_options import cmd_opts

    # 设置 CORS 选项
    cors_options = {
        "allow_methods": ["*"],  # 允许的 HTTP 方法
        "allow_headers": ["*"],  # 允许的 HTTP 头部
        "allow_credentials": True,  # 是否允许凭证
    }
    # 如果存在跨域允许的源
    if cmd_opts.cors_allow_origins:
        # 将允许的源拆分为列表
        cors_options["allow_origins"] = cmd_opts.cors_allow_origins.split(',')
    # 如果存在跨域允许的源正则表达式
    if cmd_opts.cors_allow_origins_regex:
        # 设置允许的源正则表达式
        cors_options["allow_origin_regex"] = cmd_opts.cors_allow_origins_regex
    # 将 CORS 中间件添加到应用程序中，并传入设置的 CORS 选项
    app.add_middleware(CORSMiddleware, **cors_options)
```