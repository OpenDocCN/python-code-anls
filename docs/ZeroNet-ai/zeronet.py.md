# `ZeroNet\zeronet.py`

```
#!/usr/bin/env python3
# 设置脚本的解释器为 Python 3

import os
# 导入操作系统模块
import sys
# 导入系统模块

def main():
    # 定义主函数

    if sys.version_info.major < 3:
        # 如果 Python 版本小于 3
        print("Error: Python 3.x is required")
        # 打印错误信息
        sys.exit(0)
        # 退出程序

    if "--silent" not in sys.argv:
        # 如果命令行参数中没有 "--silent"
        print("- Starting ZeroNet...")
        # 打印启动信息

    main = None
    # 初始化 main 变量为 None
    try:
        import main
        # 导入 main 模块
        main.start()
        # 调用 main 模块的 start 函数
    except Exception as err:  # Prevent closing
        # 捕获异常并命名为 err
        import traceback
        # 导入异常跟踪模块
        try:
            import logging
            # 导入日志模块
            logging.exception("Unhandled exception: %s" % err)
            # 记录异常信息
        except Exception as log_err:
            # 捕获异常并命名为 log_err
            print("Failed to log error:", log_err)
            # 打印错误信息
            traceback.print_exc()
            # 打印异常跟踪信息
        from Config import config
        # 从 Config 模块中导入 config 变量
        error_log_path = config.log_dir + "/error.log"
        # 设置错误日志路径
        traceback.print_exc(file=open(error_log_path, "w"))
        # 将异常跟踪信息写入错误日志文件
        print("---")
        # 打印分隔线
        print("Please report it: https://github.com/HelloZeroNet/ZeroNet/issues/new?assignees=&labels=&template=bug-report.md")
        # 提示用户报告错误
        if sys.platform.startswith("win") and "python.exe" not in sys.executable:
            # 如果是 Windows 平台且不是使用 python.exe 运行
            displayErrorMessage(err, error_log_path)
            # 调用 displayErrorMessage 函数处理错误信息

    if main and (main.update_after_shutdown or main.restart_after_shutdown):  # Updater
        # 如果 main 存在且需要更新或重启
        if main.update_after_shutdown:
            # 如果需要在关闭后更新
            print("Shutting down...")
            # 打印关闭信息
            prepareShutdown()
            # 调用准备关闭函数
            import update
            # 导入更新模块
            print("Updating...")
            # 打印更新信息
            update.update()
            # 调用更新函数
            if main.restart_after_shutdown:
                # 如果需要在关闭后重启
                print("Restarting...")
                # 打印重启信息
                restart()
                # 调用重启函数
        else:
            # 如果不需要在关闭后更新
            print("Shutting down...")
            # 打印关闭信息
            prepareShutdown()
            # 调用准备关闭函数
            print("Restarting...")
            # 打印重启信息
            restart()
            # 调用重启函数

def displayErrorMessage(err, error_log_path):
    # 定义显示错误信息函数，接受 err 和 error_log_path 两个参数
    import ctypes
    # 导入 ctypes 模块
    import urllib.parse
    # 导入 urllib.parse 模块
    import subprocess
    # 导入子进程模块

    MB_YESNOCANCEL = 0x3
    MB_ICONEXCLAIMATION = 0x30

    ID_YES = 0x6
    ID_NO = 0x7
    ID_CANCEL = 0x2
    # 定义消息框按钮的标识符

    err_message = "%s: %s" % (type(err).__name__, err)
    # 设置错误消息
    err_title = "Unhandled exception: %s\nReport error?" % err_message
    # 设置错误标题
    # 使用 ctypes 调用 Windows 用户界面库中的 MessageBoxW 函数，显示错误信息并返回用户的选择
    res = ctypes.windll.user32.MessageBoxW(0, err_title, "ZeroNet error", MB_YESNOCANCEL | MB_ICONEXCLAIMATION)
    # 如果用户选择了“是”，则打开 web 浏览器并跳转到报告问题的页面
    if res == ID_YES:
        import webbrowser
        report_url = "https://github.com/HelloZeroNet/ZeroNet/issues/new?assignees=&labels=&template=bug-report.md&title=%s"
        webbrowser.open(report_url % urllib.parse.quote("Unhandled exception: %s" % err_message))
    # 如果用户选择了“是”或“否”，则打开记事本并显示错误日志
    if res in [ID_YES, ID_NO]:
        subprocess.Popen(['notepad.exe', error_log_path])
# 准备关闭程序
def prepareShutdown():
    # 导入 atexit 模块，执行注册的退出函数
    import atexit
    atexit._run_exitfuncs()

    # 关闭日志文件
    if "main" in sys.modules:
        # 获取主模块的日志记录器
        logger = sys.modules["main"].logging.getLogger()

        # 遍历日志记录器的处理器，刷新并关闭每个处理器，然后从记录器中移除
        for handler in logger.handlers[:]:
            handler.flush()
            handler.close()
            logger.removeHandler(handler)

    import time
    time.sleep(1)  # 等待文件关闭

# 重新启动程序
def restart():
    args = sys.argv[:]

    # 替换可执行文件路径中的 ".pkg"，用于解冻 Mac 应用
    sys.executable = sys.executable.replace(".pkg", "")

    if not getattr(sys, 'frozen', False):
        args.insert(0, sys.executable)

    # 在重启后不打开浏览器
    if "--open_browser" in args:
        del args[args.index("--open_browser") + 1]  # 参数值
        del args[args.index("--open_browser")]  # 参数键

    if getattr(sys, 'frozen', False):
        pos_first_arg = 1  # 只有可执行文件
    else:
        pos_first_arg = 2  # 解释器和 .py 文件路径

    args.insert(pos_first_arg, "--open_browser")
    args.insert(pos_first_arg + 1, "False")

    if sys.platform == 'win32':
        args = ['"%s"' % arg for arg in args]

    try:
        print("执行 %s %s" % (sys.executable, args))
        os.execv(sys.executable, args)
    except Exception as err:
        print("Execv 错误: %s" % err)
    print("再见。")

# 启动程序
def start():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(app_dir)  # 将工作目录更改为 zeronet.py 目录
    sys.path.insert(0, os.path.join(app_dir, "src/lib"))  # 外部库目录
    sys.path.insert(0, os.path.join(app_dir, "src"))  # 相对于 src 的导入路径

    if "--update" in sys.argv:
        sys.argv.remove("--update")
        print("更新中...")
        import update
        update.update()
    else:
        main()

# 如果作为主程序运行，则调用 start() 函数
if __name__ == '__main__':
    start()
```