# `arknights-mower\webview_ui.py`

```
#!/usr/bin/env python3

# 导入 webview 模块
import webview
# 从 server 模块中导入 app 和 mower_process
from server import app, mower_process

# 导入 os 模块
import os
# 导入 multiprocessing 模块
import multiprocessing

# 从 arknights_mower.utils.conf 模块中导入 load_conf 和 save_conf
from arknights_mower.utils.conf import load_conf, save_conf
# 从 arknights_mower.__init__ 模块中导入 __version__
from arknights_mower.__init__ import __version__

# 从 threading 模块中导入 Thread
from threading import Thread
# 从 PIL 模块中导入 Image
from PIL import Image
# 从 pystray 模块中导入 Icon, Menu, MenuItem
from pystray import Icon, Menu, MenuItem

# 导入 socket 模块
import socket
# 导入 tkinter 模块
import tkinter
# 从 tkinter 模块中导入 messagebox
from tkinter import messagebox
# 导入 sleep 函数
from time import sleep
# 导入 sys 模块
import sys

# 初始化退出标志为 False
quit_app = False
# 初始化显示标志为 True
display = True

# 定义窗口大小变化的回调函数
def on_resized(w, h):
    global width
    global height
    width = w
    height = h

# 切换窗口显示状态的函数
def toggle_window():
    global window
    global display
    window.hide() if display else window.show()
    display = not display

# 窗口关闭时的回调函数
def on_closing():
    if not quit_app:
        # 创建线程来切换窗口显示状态
        Thread(target=toggle_window).start()
        return False

# 销毁窗口的函数
def destroy_window():
    global quit_app
    global window
    quit_app = True
    window.destroy()

# 检查端口是否被占用的函数
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0

# 主程序入口
if __name__ == "__main__":
    # 冻结多进程支持
    multiprocessing.freeze_support()

    # 加载配置文件
    conf = load_conf()

    # 从配置文件中获取端口和令牌
    port = conf["webview"]["port"]
    token = conf["webview"]["token"]
    host = "0.0.0.0" if token else "127.0.0.1"

    # 如果端口已被占用，则弹出错误消息框并退出程序
    if is_port_in_use(port):
        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror(
            "arknights-mower",
            f"端口{port}已被占用，无法启动！",
        )
        sys.exit()

    # 设置 app 对象的令牌
    app.token = token
    # 创建线程来运行 app
    Thread(
        target=app.run,
        kwargs={"host": host, "port": port},
        daemon=True,
    ).start()

    # 初始化窗口宽度和高度
    global width
    global height
    width = conf["webview"]["width"]
    height = conf["webview"]["height"]

    # 加载托盘图标
    tray_img = Image.open(os.path.join(os.getcwd(), "logo.png"))
    # 创建一个名为 "arknights-mower" 的图标对象，包括图标和菜单
    icon = Icon(
        "arknights-mower",
        icon=tray_img,
        menu=Menu(
            MenuItem(
                text="显示/隐藏窗口",
                action=toggle_window,
                default=True,
            ),
            MenuItem(
                text="退出",
                action=destroy_window,
            ),
        ),
    )
    # 运行图标对象，使其在后台独立运行
    icon.run_detached()

    # 创建一个名为 "window" 的全局变量，用于存储 webview 窗口对象
    global window
    # 创建一个 webview 窗口，显示指定的标题和链接，设置宽度、高度和文本选择功能
    window = webview.create_window(
        f"Mower {__version__} (http://{host}:{port})",
        f"http://127.0.0.1:{port}?token={token}",
        width=width,
        height=height,
        text_select=True,
    )

    # 监听窗口大小变化事件，并绑定相应的处理函数
    window.events.resized += on_resized
    # 监听窗口关闭事件，并绑定相应的处理函数
    window.events.closing += on_closing

    # 循环检查指定端口是否被占用，直到端口被占用为止
    while not is_port_in_use(port):
        sleep(0.1)
    # 启动 webview 窗口
    webview.start()

    # 创建一个名为 "mower_process" 的全局变量，用于存储进程对象
    global mower_process
    # 如果存在 "mower_process"，则终止该进程并将其置为 None
    if mower_process:
        mower_process.terminate()
        mower_process = None

    # 停止图标对象的运行
    icon.stop()

    # 加载配置文件，更新 webview 窗口的宽度和高度，并保存配置
    conf = load_conf()
    conf["webview"]["width"] = width
    conf["webview"]["height"] = height
    save_conf(conf)
```