# `arknights-mower\server.py`

```py
#!/usr/bin/env python3
# 设置脚本的解释器为 Python 3

from arknights_mower.solvers import record
from arknights_mower.utils.conf import load_conf, save_conf, load_plan, write_plan
from arknights_mower.__main__ import main
from arknights_mower.utils.asst import Asst
# 导入所需的模块和函数

from flask import Flask, send_from_directory, request, abort
from flask_cors import CORS
from flask_sock import Sock
# 导入 Flask 相关模块

from simple_websocket import ConnectionClosed
# 导入简单的 WebSocket 模块

import webview
# 导入 webview 模块

import os
import multiprocessing
import subprocess
from threading import Thread
import json
import time
import sys
import mimetypes
# 导入其他所需的模块

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
# 导入发送邮件相关的模块

from functools import wraps
# 导入装饰器相关的模块

mimetypes.add_type("text/html", ".html")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/javascript", ".js")
# 添加 MIME 类型

app = Flask(__name__, static_folder="dist", static_url_path="")
# 创建 Flask 应用实例，指定静态文件夹为 "dist"
sock = Sock(app)
CORS(app)
# 创建 WebSocket 实例和跨域资源共享

conf = {}
plan = {}
mower_process = None
read = None
operators = {}
log_lines = []
ws_connections = []
# 初始化全局变量

def require_token(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if hasattr(app, "token") and request.headers.get("token", "") != app.token:
            abort(403)
        return f(*args, **kwargs)
    return decorated_function
# 定义装饰器函数，用于验证 token

@app.route("/")
def serve_index():
    return send_from_directory("dist", "index.html")
# 定义路由，返回静态文件 "index.html"

@app.route("/conf", methods=["GET", "POST"])
@require_token
def load_config():
    global conf

    if request.method == "GET":
        conf = load_conf()
        return conf
    else:
        conf.update(request.json)
        save_conf(conf)
        return f"New config saved!"
# 定义路由，用于加载和保存配置信息

@app.route("/plan", methods=["GET", "POST"])
@require_token
def load_plan_from_json():
    global plan

    if request.method == "GET":
        global conf
        plan = load_plan(conf["planFile"])
        return plan
    else:
        plan = request.json
        write_plan(plan, conf["planFile"])
        return f"New plan saved at {conf['planFile']}"
# 定义路由，用于加载和保存计划信息
# 定义一个路由，用于获取操作员列表
@app.route("/operator")
def operator_list():
    # 检查是否是打包后的可执行文件，并且是否存在 _MEIPASS 属性
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        # 打开打包后的可执行文件中的 agent.json 文件，并以 utf8 编码读取内容
        with open(
            os.path.join(
                sys._MEIPASS,
                "arknights_mower",
                "__init__",
                "data",
                "agent.json",
            ),
            "r",
            encoding="utf8",
        ) as f:
            # 加载 JSON 文件内容并返回
            return json.load(f)
    else:
        # 打开当前工作目录下的 agent.json 文件，并以 utf8 编码读取内容
        with open(
            os.path.join(
                os.getcwd(),
                "arknights_mower",
                "data",
                "agent.json",
            ),
            "r",
            encoding="utf8",
        ) as f:
            # 加载 JSON 文件内容并返回
            return json.load(f)

# 定义一个路由，用于获取商店列表
@app.route("/shop")
def shop_list():
    # 检查是否是打包后的可执行文件，并且是否存在 _MEIPASS 属性
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        # 打开打包后的可执行文件中的 shop.json 文件，并以 utf8 编码读取内容
        with open(
            os.path.join(
                sys._MEIPASS,
                "arknights_mower",
                "__init__",
                "data",
                "shop.json",
            ),
            "r",
            encoding="utf8",
        ) as f:
            # 加载 JSON 文件内容并返回
            return json.load(f)
    else:
        # 打开当前工作目录下的 shop.json 文件，并以 utf8 编码读取内容
        with open(
            os.path.join(
                os.getcwd(),
                "arknights_mower",
                "data",
                "shop.json",
            ),
            "r",
            encoding="utf8",
        ) as f:
            # 加载 JSON 文件内容并返回
            return json.load(f)

# 定义一个函数，用于读取日志
def read_log(conn):
    # 声明全局变量
    global operators
    global mower_process
    global log_lines
    global ws_connections
    # 尝试循环接收消息，直到发生异常
    try:
        # 当连接存在时
        while True:
            # 接收消息
            msg = conn.recv()
            # 如果消息类型为"log"
            if msg["type"] == "log":
                # 创建新的日志行，包含时间和数据
                new_line = time.strftime("%m-%d %H:%M:%S ") + msg["data"]
                # 将新的日志行添加到日志列表中，并保留最后500行
                log_lines.append(new_line)
                log_lines = log_lines[-500:]
                # 遍历所有 WebSocket 连接，发送新的日志行
                for ws in ws_connections:
                    ws.send(new_line)
            # 如果消息类型为"operators"
            elif msg["type"] == "operators":
                # 更新操作员信息
                operators = msg["data"]
            # 如果消息类型为"update_conf"
            elif msg["type"] == "update_conf":
                # 全局变量 conf 被更新后，发送给连接方
                global conf
                conn.send(conf)
    # 捕获到文件结束异常时
    except EOFError:
        # 关闭连接
        conn.close()
# 定义一个路由，用于检查割草机是否正在运行
@app.route("/running")
def running():
    # 使用全局变量 mower_process 判断割草机是否正在运行，返回对应的字符串
    global mower_process
    return "false" if mower_process is None else "true"


# 定义一个路由，用于启动割草机
@app.route("/start")
@require_token
def start():
    # 使用全局变量定义需要使用的变量
    global conf
    global plan
    global mower_process
    global operators
    global log_lines

    # 如果割草机已经在运行，则返回提示信息
    if mower_process is not None:
        return "Mower is already running."

    # 创建一个双向管道，用于进程间通信
    read, write = multiprocessing.Pipe()
    # 创建一个新的割草机进程，并传入相应的参数
    mower_process = multiprocessing.Process(
        target=main,
        args=(
            conf,
            plan,
            operators,
            write,
        ),
        daemon=True,
    )
    mower_process.start()  # 启动割草机进程

    # 启动一个线程用于读取日志
    Thread(target=read_log, args=(read,)).start()

    log_lines = []  # 清空日志列表

    return "Mower started."  # 返回启动成功的提示信息


# 定义一个路由，用于停止割草机
@app.route("/stop")
@require_token
def stop():
    global mower_process  # 使用全局变量

    # 如果割草机未在运行，则返回提示信息
    if mower_process is None:
        return "Mower is not running."

    # 终止割草机进程
    mower_process.terminate()
    mower_process = None  # 将割草机进程置为 None

    return "Mower stopped."  # 返回停止成功的提示信息


# 定义一个 WebSocket 路由，用于发送日志信息
@sock.route("/log")
def log(ws):
    global ws_connections
    global log_lines

    ws.send("\n".join(log_lines))  # 将日志信息发送给 WebSocket 连接
    ws_connections.append(ws)  # 将 WebSocket 连接添加到连接列表

    try:
        while True:
            ws.receive()  # 接收 WebSocket 连接的消息
    except ConnectionClosed:
        ws_connections.remove(ws)  # 如果连接关闭，则从连接列表中移除


# 定义一个路由，用于打开文件对话框
@app.route("/dialog/file")
@require_token
def open_file_dialog():
    window = webview.active_window()
    file_path = window.create_file_dialog(dialog_type=webview.OPEN_DIALOG)
    if file_path:
        return file_path[0]
    else:
        return ""


# 定义一个路由，用于打开文件夹对话框
@app.route("/dialog/folder")
@require_token
def open_folder_dialog():
    window = webview.active_window()
    folder_path = window.create_file_dialog(dialog_type=webview.FOLDER_DIALOG)
    if folder_path:
        return folder_path[0]
    else:
        return ""


# 定义一个路由，用于获取 MAA ADB 版本信息
@app.route("/check-maa")
@require_token
def get_maa_adb_version():
    # 尝试加载配置文件中指定的 Maa 路径
    Asst.load(conf["maa_path"])
    # 创建 Asst 对象
    asst = Asst()
    # 获取 Maa 版本信息
    version = asst.get_version()
    # 设置实例选项
    asst.set_instance_option(2, conf["maa_touch_option"])
    # 尝试连接到指定的 Maa ADB 路径
    if asst.connect(conf["maa_adb_path"], conf["adb"]):
        # 如果连接成功，返回成功加载的消息
        maa_msg = f"Maa {version} 加载成功"
    else:
        # 如果连接失败，返回连接失败的消息
        maa_msg = "连接失败，请检查Maa日志！"
    # 捕获异常并返回异常消息
    except Exception as e:
    maa_msg = "Maa加载失败：" + str(e)
    # 返回 Maa 消息
    return maa_msg
# 定义路由，处理获取 MAA 连接预设的请求，需要验证 token
@app.route("/maa-conn-preset")
@require_token
def get_maa_conn_presets():
    try:
        # 打开配置文件，读取连接预设信息
        with open(
            os.path.join(conf["maa_path"], "resource", "config.json"),
            "r",
            encoding="utf-8",
        ) as f:
            # 从配置文件中获取连接预设的配置名称
            presets = [i["configName"] for i in json.load(f)["connection"]]
    except:
        # 如果出现异常，将预设列表置为空
        presets = []
    # 返回连接预设列表
    return presets


# 定义路由，处理获取心情比例的请求
@app.route("/record/getMoodRatios")
def get_mood_ratios():
    # 调用 record 模块的 get_mood_ratios 方法获取心情比例
    return record.get_mood_ratios()


# 定义路由，处理测试邮件发送的请求，需要验证 token
@app.route("/test-email")
@require_token
def test_email():
    # 创建邮件对象
    msg = MIMEMultipart()
    # 添加邮件内容
    msg.attach(MIMEText("arknights-mower测试邮件", "plain"))
    # 设置邮件主题
    msg["Subject"] = conf["mail_subject"] + "测试邮件"
    # 设置发件人
    msg["From"] = conf["account"]
    try:
        # 使用 SMTP_SSL 连接到邮件服务器
        s = smtplib.SMTP_SSL("smtp.qq.com", 465, timeout=10.0)
        # 登录邮箱
        s.login(conf["account"], conf["pass_code"])
        # 发送邮件
        s.sendmail(conf["account"], conf["account"], msg.as_string())
    except Exception as e:
        # 如果发送邮件出现异常，返回发送失败信息和异常信息
        return "邮件发送失败！\n" + str(e)
    # 发送成功则返回发送成功信息
    return "邮件发送成功！"
```