# `.\chatglm4-finetune\composite_demo\src\tools\python.py`

```
# 导入用于打印美化的模块
from pprint import pprint
# 导入队列模块
import queue
# 导入正则表达式模块
import re
# 导入子进程模块的管道功能
from subprocess import PIPE
# 导入字面量类型
from typing import Literal

# 导入 Jupyter 客户端模块
import jupyter_client
# 导入 Streamlit 模块
import streamlit as st

# 定义正则表达式用于匹配 ANSI 转义序列
ANSI_ESCAPE = re.compile(r'(\x9B|\x1B\[|\u001b\[)[0-?]*[ -/]*[@-~]')
# 定义正则表达式用于匹配代码块
CODE = re.compile(r'```([^\n]*)\n(.*?)```')

# 定义 CodeKernel 类
class CodeKernel:
    # 初始化类的构造函数
    def __init__(self,
                 kernel_name='kernel',  # 设置内核名称，默认为 'kernel'
                 kernel_id=None,  # 可选内核 ID
                 kernel_config_path="",  # 内核配置文件路径
                 python_path=None,  # Python 路径
                 ipython_path=None,  # IPython 路径
                 init_file_path="./startup.py",  # 初始化文件路径
                 verbose=1):  # 是否打印详细信息

        # 初始化内核名称
        self.kernel_name = kernel_name
        # 初始化内核 ID
        self.kernel_id = kernel_id
        # 初始化内核配置文件路径
        self.kernel_config_path = kernel_config_path
        # 初始化 Python 路径
        self.python_path = python_path
        # 初始化 IPython 路径
        self.ipython_path = ipython_path
        # 初始化启动文件路径
        self.init_file_path = init_file_path
        # 初始化详细模式
        self.verbose = verbose

        # 如果没有提供 Python 和 IPython 路径，设置环境变量为 None
        if python_path is None and ipython_path is None:
            env = None
        else:
            # 设置环境变量包含 Python 路径
            env = {"PATH": self.python_path + ":$PATH", "PYTHONPATH": self.python_path}

        # 初始化后端内核管理器
        self.kernel_manager = jupyter_client.KernelManager(kernel_name=IPYKERNEL,
                                                           connection_file=self.kernel_config_path,
                                                           exec_files=[self.init_file_path],
                                                           env=env)
        # 如果有配置文件路径，加载连接文件并启动内核
        if self.kernel_config_path:
            self.kernel_manager.load_connection_file()
            self.kernel_manager.start_kernel(stdout=PIPE, stderr=PIPE)
            # 打印后端内核启动的信息
            print("Backend kernel started with the configuration: {}".format(
                self.kernel_config_path))
        else:
            # 否则直接启动内核
            self.kernel_manager.start_kernel(stdout=PIPE, stderr=PIPE)
            # 打印后端内核启动的信息
            print("Backend kernel started with the configuration: {}".format(
                self.kernel_manager.connection_file))

        # 如果 verbose 为真，打印连接信息
        if verbose:
            pprint(self.kernel_manager.get_connection_info())

        # 初始化代码内核
        self.kernel = self.kernel_manager.blocking_client()
        # 启动内核通道
        self.kernel.start_channels()
        # 打印代码内核启动的信息
        print("Code kernel started.")
    # 定义执行代码的方法
        def execute(self, code):
            # 执行给定的代码
            self.kernel.execute(code)
            try:
                # 获取 shell 消息，最多等待 30 秒
                shell_msg = self.kernel.get_shell_msg(timeout=30)
                # 获取 IOPub 消息内容，最多等待 30 秒
                io_msg_content = self.kernel.get_iopub_msg(timeout=30)['content']
                # 无限循环，直到执行状态变为 idle
                while True:
                    # 保存当前 IO 消息内容
                    msg_out = io_msg_content
                    ### 轮询消息
                    try:
                        # 获取新的 IOPub 消息内容，最多等待 30 秒
                        io_msg_content = self.kernel.get_iopub_msg(timeout=30)['content']
                        # 如果执行状态为 idle，则退出循环
                        if 'execution_state' in io_msg_content and io_msg_content['execution_state'] == 'idle':
                            break
                    except queue.Empty:
                        # 如果没有新消息，退出循环
                        break
    
                # 返回 shell 消息和最后的输出消息
                return shell_msg, msg_out
            except Exception as e:
                # 打印异常信息
                print(e)
                # 如果发生异常，返回 None
                return None
    
    # 定义交互式执行代码的方法
        def execute_interactive(self, code, verbose=False):
            # 交互式执行给定代码，获取 shell 消息
            shell_msg = self.kernel.execute_interactive(code)
            # 如果没有 shell 消息，则处理超时
            if shell_msg is queue.Empty:
                if verbose:
                    # 打印超时信息
                    print("Timeout waiting for shell message.")
            # 检查消息状态
            self.check_msg(shell_msg, verbose=verbose)
    
            # 返回 shell 消息
            return shell_msg
    
    # 定义检查代码的方法
        def inspect(self, code, verbose=False):
            # 发送代码检查请求，获取消息 ID
            msg_id = self.kernel.inspect(code)
            # 获取 shell 消息，最多等待 30 秒
            shell_msg = self.kernel.get_shell_msg(timeout=30)
            # 如果没有 shell 消息，则处理超时
            if shell_msg is queue.Empty:
                if verbose:
                    # 打印超时信息
                    print("Timeout waiting for shell message.")
            # 检查消息状态
            self.check_msg(shell_msg, verbose=verbose)
    
            # 返回 shell 消息
            return shell_msg
    
    # 定义获取错误消息的方法
        def get_error_msg(self, msg, verbose=False) -> str | None:
            # 检查消息状态是否为错误
            if msg['content']['status'] == 'error':
                try:
                    # 尝试获取完整的 traceback
                    error_msg = msg['content']['traceback']
                except:
                    try:
                        # 尝试获取最后一行的 traceback
                        error_msg = msg['content']['traceback'][-1].strip()
                    except:
                        # 如果都失败，返回默认错误信息
                        error_msg = "Traceback Error"
                if verbose:
                    # 打印错误信息
                    print("Error: ", error_msg)
                # 返回错误消息
                return error_msg
            # 如果没有错误，返回 None
            return None
    
    # 定义检查消息状态的方法
        def check_msg(self, msg, verbose=False):
            # 获取消息状态
            status = msg['content']['status']
            # 如果状态为 ok，表示执行成功
            if status == 'ok':
                if verbose:
                    # 打印执行成功信息
                    print("Execution succeeded.")
            # 如果状态为 error，打印 traceback
            elif status == 'error':
                for line in msg['content']['traceback']:
                    if verbose:
                        # 打印每行 traceback
                        print(line)
    
    # 定义关闭内核的方法
        def shutdown(self):
            # 关闭后端内核
            self.kernel_manager.shutdown_kernel()
            print("Backend kernel shutdown.")
            # 关闭代码内核
            self.kernel.shutdown()
            print("Code kernel shutdown.")
    
    # 定义重启内核的方法
        def restart(self):
            # 重启后端内核
            self.kernel_manager.restart_kernel()
            # print("Backend kernel restarted.")
    
    # 定义中断内核的方法
        def interrupt(self):
            # 中断后端内核
            self.kernel_manager.interrupt_kernel()
            # print("Backend kernel interrupted.")
    
    # 定义检查内核是否存活的方法
        def is_alive(self):
            # 返回内核存活状态
            return self.kernel.is_alive()
# 定义一个函数，用于清理输入字符串中的 ANSI 代码
def clean_ansi_codes(input_string):
    # 使用正则表达式去除输入字符串中的 ANSI 转义序列
    return ANSI_ESCAPE.sub('', input_string)

# 定义一个函数，从文本中提取代码段
def extract_code(text: str) -> str:
    # 查找文本中所有的代码段，返回匹配的列表
    matches = CODE.findall(text, re.DOTALL)
    # 返回最后一个匹配的代码段（假设代码段是元组，取第二个元素）
    return matches[-1][1]

# 定义一个执行代码的函数
def execute(
    code: str,
    kernel: CodeKernel
) -> tuple[Literal['text', 'image'] | None, str]:
    # 初始化结果和结果类型
    res = ""
    res_type = None
    # 清理代码中的特定 XML 标签
    code = code.replace("<|observation|>", "")
    code = code.replace("<|assistant|>python", "")
    code = code.replace("<|assistant|>", "")
    code = code.replace("<|user|>", "")
    code = code.replace("<|system|>", "")
    # 执行代码并获取消息和输出
    msg, output = kernel.execute(code)

    # 检查执行状态是否超时
    if msg['metadata']['status'] == "timeout":
        return res_type, 'Timed out'
    # 检查执行状态是否出错
    elif msg['metadata']['status'] == 'error':
        # 返回错误信息，清理 ANSI 代码
        return res_type, clean_ansi_codes('\n'.join(kernel.get_error_msg(msg, verbose=True)))

    # 检查输出中是否包含文本
    if 'text' in output:
        res_type = "text"  # 设置结果类型为文本
        res = output['text']  # 获取文本结果
    # 检查输出中是否包含数据
    elif 'data' in output:
        # 遍历输出数据的每一个键
        for key in output['data']:
            # 如果数据类型是文本，设置结果类型和结果
            if 'text/plain' in key:
                res_type = "text"
                res = output['data'][key]
            # 如果数据类型是图片，设置结果类型和结果
            elif 'image/png' in key:
                res_type = "image"
                res = output['data'][key]
                break  # 找到图片后退出循环

    # 返回结果类型和结果
    return res_type, res

# 使用 Streamlit 的缓存机制定义一个获取内核的函数
@st.cache_resource
def get_kernel() -> CodeKernel:
    # 创建并返回一个新的 CodeKernel 实例
    return CodeKernel()

# 定义一个工具调用的函数
def tool_call(code: str, session_id: str) -> list[ToolObservation]:
    # 获取内核
    kernel = get_kernel()
    # 执行代码并获取结果类型和结果
    res_type, res = execute(code, kernel)

    # 根据结果类型转换为数据 URI
    text = '[Image]' if res_type == 'image' else res  # 如果是图片，设置文本为 '[Image]'
    image = f'data:image/png;base64,{res}' if res_type == 'image' else None  # 如果是图片，生成数据 URI

    # 返回包含工具观察结果的列表
    return [ToolObservation(res_type, text, image)]
```