# ArknightMower源码解析 2

<div align="center">

![logo](https://github.com/ArkMowers/arknights-mower/raw/main/logo.png)

# arknights-mower

[![GitHub License](https://img.shields.io/github/license/ArkMowers/arknights-mower?style=flat-square)](https://github.com/ArkMowers/arknights-mower/blob/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/arknights-mower?style=flat-square)](https://pypi.org/project/arknights-mower/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/arknights-mower?style=flat-square)](https://pypi.org/project/arknights-mower/)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/ArkMowers/arknights-mower/Upload%20PyPI?style=flat-square)](https://github.com/ArkMowers/arknights-mower/actions/workflows/python-publish.yml)
![GitHub last commit (branch)](https://img.shields.io/github/last-commit/ArkMowers/arknights-mower/main?style=flat-square)
[![Code Climate maintainability](https://img.shields.io/codeclimate/maintainability/ArkMowers/arknights-mower?style=flat-square)](https://codeclimate.com/github/ArkMowers/arknights-mower)

7\*24 小时不间断长草，让你忘掉这个游戏！

</div>

## ⚠ 注意事项

- 本程序不支持国服以外的明日方舟区服，支持官服和 Bilibili 服。
- 原理上，使用本程序没有任何被判定为作弊并被封号的风险，但是作者不对使用此程序造成的任何损失负责。
- 开发组人数稀少，有时不太能及时反馈和修复 Bug，见谅一下。也欢迎更多有能力有意向的朋友参与。
- 本软件目前仅支持 1920\*1080 分辨率，使用夜神模拟器可以解决大部分问题

## 主要功能

- 自动打开《明日方舟》，支持官服和 Bilibili 服
- 自动登录
  - 账户密码需要手动输入
- 支持调用 maa 执行除基建外的长草活动(日常/肉鸽/保全...)
- 支持邮件提醒
- 读取基建心情，根据排班表**_动态换班_**力求最高工休比 :fire:
- 支持**_跑单操作_**，可设置全自动换班/仅跑单模式（不推荐使用）:fire:
- 自动使用菲亚梅塔恢复指定房间心情最低干员的心情并重回岗位 (N+1模式) :fire:
- 基建干员心情监控 :fire::fire:

## 软件界面
![246289794-97f7f9c6-6f7d-4504-bc45-48660282249b](https://github.com/ArkMowers/arknights-mower/assets/33809511/a6dd6f47-39df-41c4-b384-38c9efeefd6a)

## 赛博监工
![image](https://github.com/ArkMowers/arknights-mower/assets/33809511/61ad7eb4-bb93-4259-af4a-a28ea9b1f66c)

## 安装

在 [Releases](https://github.com/ArkMowers/arknights-mower/releases) 下载 Windows 可执行文件版本。
在 [action](https://github.com/ArkMowers/arknights-mower/actions/workflows/pyinstaller-win-shawn.yml) 产物下载最新测试版本
或进入下方 qq 群文件领取，目前软件迭代速度较快，进入群内可获取一手版本

## 使用教程

- 下载安装后，打开软件
- 设置 adb 地址（如 127.0.0.1:5555）
- 配置排班表[[配置方法](https://www.bilibili.com/video/BV1KT411s7Ar)]或前往 qq 群寻找现成的排班表导入
- 点击开始执行
- 详细配置内容请参考[[功能文档](https://arkmowers.github.io/arknights-mower/)]

欢迎大家提交 Pull requests 增加更多的功能！

## 遇到报错？想要更多功能？

如果你在使用过程中遇到问题，欢迎通过提交 Issue 的方式报错或者提问。报告 Issue 时建议附上调试信息以便定位问题。

也欢迎加入交流群讨论：

- [Telegram Group](https://t.me/ark_mover)
- [QQ Group](https://jq.qq.com/?_wv=1027&k=4gWboTVI): 239200680

## Star History

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=ArkMowers/arknights-mower&type=Date)](https://star-history.com/#ArkMowers/arknights-mower&Date)

</div>


# `/opt/arknights-mower/server.py`

这是一个使用 Flask 框架创建的一个 WebSocket 服务器应用程序。以下是该应用程序的主要功能和模块：

1. 配置文件模块：使用 Arknights Mower 的配置文件模块（配置文件存储在 `/path/to/config/files` 目录下）。
2. 记录游戏过程：通过记录游戏过程中的状态和操作，将游戏过程保存在一个游戏记录中。
3. 支持 Flask-CORS：自动处理跨域请求，允许来自任何域的请求访问服务器。
4. 创建一个简单的 WebSocket 客户端连接到服务器，然后通过发送消息来接收游戏状态和操作。
5. 显示游戏地图：通过在 WebSocket 连接上发送消息，在服务器端渲染游戏地图。
6. 创建 Asst：使用 Asst 库来处理游戏中的AI行为。
7. 加载和保存游戏计划：通过在应用程序中加载和保存游戏计划，使得游戏可以在没有重新启动时继续。
8. 导出游戏计划：通过应用程序中的 `write_plan` 函数，将游戏计划导出为 JSON 文件。
9. 处理游戏中的异常：当游戏中的某个组件出现异常时，通过 `abort` 函数来终止当前请求，并返回错误信息。
10. 启动应用程序：通过 `main` 函数来启动应用程序，然后进入游戏循环，接收和处理游戏中的消息。


```
#!/usr/bin/env python3
from arknights_mower.solvers import record
from arknights_mower.utils.conf import load_conf, save_conf, load_plan, write_plan
from arknights_mower.__main__ import main
from arknights_mower.utils.asst import Asst

from flask import Flask, send_from_directory, request, abort
from flask_cors import CORS
from flask_sock import Sock

from simple_websocket import ConnectionClosed

import webview

import os
```

这段代码使用了多个Python库，包括multiprocessing、subprocess、threading、json、time、sys、mimetypes和smtplib。它们各自的作用如下：

1. multiprocessing：用于编写多线程程序。
2. subprocess：用于执行命令行工具。
3. threading：用于创建和管理线程。
4. json：用于解析JSON格式的数据。
5. time：用于处理定时任务。
6. sys：用于访问Python标准库中的函数和模块。
7. mimetypes：用于解析电子邮件地址中的MIMEText类型数据。
8. smtplib：用于发送电子邮件。
9. email.mime.text：用于解析电子邮件中的文本MIME类型数据。
10. email.mime.multipart：用于解析电子邮件中的多部分MIME类型数据。

线程的作用是负责执行邮件发送操作，具体来说：

1. 导入短信服务，用于与指定手机号码的短信服务器建立连接；
2. 创建一个 multidom，用于发送指定短信内容；
3. 设置接口提示（设置 send_ai 的值），开启人工智能发送助手；
4. 解析收到的短信验证码，判断验证码是否正确；
5. 通过 python 的 mimetype 库解析短信内容的数据类型，以便调用合适的消息接口；
6. 通过 python 的 threading库创建一个新的线程，保证多线程的安全性；
7. 将解析得到的信息发送到指定手机号码，以实现短信发送功能。


```
import multiprocessing
import subprocess
from threading import Thread
import json
import time
import sys
import mimetypes

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from functools import wraps


```

这段代码使用了MimeType和Flask框架来实现一个Web应用程序。具体解释如下：

1. `mimetypes.add_type("text/html", ".html")`：向mimetypes库中添加了".html"格式的支持，以便在应用程序中能够正确解析HTML格式的请求。
2. `mimetypes.add_type("text/css", ".css")`：向mimetypes库中添加了".css"格式的支持，以便在应用程序中能够正确解析CSS格式的请求。
3. `mimetypes.add_type("application/javascript", ".js")`：向mimetypes库中添加了".js"格式的支持，以便在应用程序中能够正确解析JavaScript文件的请求。
4. `app = Flask(__name__, static_folder="dist", static_url_path="")`：创建了一个Flask应用程序实例，名为"app"，将应用程序根目录设置为"dist"，并且设置静态文件目录为"/"（默认为当前工作目录）。
5. `sock = Sock(app)`：创建了一个Socket对象，以便在应用程序中进行网络通信。
6. `CORS(app)`：在应用程序中设置跨源时区请求头，允许在不通过服务器时区（DST）的跨域访问。
7. `conf = {}`：创建了一个空字典，用于存储应用程序中的配置信息。
8. `plan = {}`：创建了一个空字典，用于存储应用程序中的路由规划信息。
9. `mower_process = None`：创建了一个空字符串，用于存储MongoDB数据库的连接信息。
10. `read = None`：创建了一个空字符串，用于存储MongoDB数据库中的查询操作。
11. `operators = {}`：创建了一个空字典，用于存储MongoDB数据库中的查询操作。
12. `from datetime import datetime, timedelta }`：从datetime模块中导入了一个名为"datetime"的类，用于日期和时间的处理。
13. `class Today(datetime.date):`：创建了一个名为"Today"的类，继承自datetime.date类，用于表示当前日期。
14. `def today_date(self):`：定义了一个名为"today_date"的方法，用于获取当前日期。
15. `today_date.strftime("%A %B, %Y-%m-%d")`：使用字符串格式化方法将当前日期格式化为"AAPRIL 01, 2023"。
16. `@app.route("/<date_time_object>")`：创建了一个名为"today_view"的路由，用于获取当前日期和时间的格式化后的字符串，并将其返回给客户端。
17. `@app.route("/")`：创建了一个名为"home_page"的路由，用于显示主页内容。
18. `@app.route("/plan/")`：创建了一个名为"plan_page"的路由，用于显示路由规划信息。
19. `@app.route("/conf/")`：创建了一个名为"conf_page"的路由，用于显示应用程序中的配置信息。
20. `@app.route("/api/")`：创建了一个名为"api_page"的路由，用于接口的访问。
21. `@app.route("/static/<path_string>")`：创建了一个名为"static_page"的路由，从"/static/"目录中获取静态文件，并将其返回给客户端。


```
mimetypes.add_type("text/html", ".html")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/javascript", ".js")


app = Flask(__name__, static_folder="dist", static_url_path="")
sock = Sock(app)
CORS(app)


conf = {}
plan = {}
mower_process = None
read = None
operators = {}
```

这段代码定义了一个函数 `require_token`，该函数接收一个参数 `f`，并返回一个经过装饰的函数 `decorated_function`。

装饰器函数 `require_token` 通过在函数内部创建一个名为 `app.token` 的属性，并在函数签名中使用 `@wraps(f)` 来告诉 Python 编译器，该函数的实现将会使用传递给 `require_token` 的函数 `f` 的内容。编译器将会创建一个链接，该链接将 `require_token` 函数与传递给 `decorated_function` 的参数和参数类型匹配。

在 `require_token` 函数中，如果给定的 `app.token` 存在且与调用 `require_token` 的请求头中的 `token` 不符，则会发生 `abort(403)` 错误，这意味着 HTTP 客户端将无法继续进行下一步操作。

另外，`require_token` 函数返回的 `decorated_function` 接收两个参数 `args` 和 `kwargs`，这些参数用于作为 `f` 函数的参数。


```
log_lines = []
ws_connections = []


def require_token(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if hasattr(app, "token") and request.headers.get("token", "") != app.token:
            abort(403)
        return f(*args, **kwargs)

    return decorated_function


@app.route("/")
```

这段代码定义了一个名为 `serve_index` 的函数，它接受一个参数 `dist` 和一个名为 `index.html` 的文件路径。它使用 `send_from_directory` 函数将 `dist` 目录下的 `index.html` 文件发送到客户端。

接下来，定义了一个名为 `load_config` 的函数，它接受一个名为 `conf` 的全局变量。这个函数使用 `require_token` 装饰，要求客户端发送一个令牌来授权进行此操作。如果客户端发送了一个有效的令牌，则加载现有的配置文件并返回；否则，更新配置文件并将更新后的配置保存到服务器上的一个文件中。最后，在客户端发送请求时，返回一条消息，表明新配置已成功保存。

该代码中的 `conf` 变量是一个全局变量，它的初始值是从服务器上下载的配置文件。在第一次调用 `load_config` 函数时，它会读取现有的配置文件并返回。在后续的调用中，函数将更新现有的配置文件以反映客户端的请求。


```
def serve_index():
    return send_from_directory("dist", "index.html")


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


```

这段代码是一个名为 `load_plan_from_json` 的函数，用于从 JSON 文件中加载计划数据并将其存储在 `plan` 变量中。

该函数有两个方法，分别是 GET 和 POST:

- GET: 如果是 GET 请求，函数先检查请求方法，如果是 GET，就从 `conf` 字典中读取计划文件，如果不是 GET，就从请求体中读取 JSON 数据。然后函数调用 `load_plan` 函数将计划文件加载到 `plan` 变量中，并将计划文件存储在 `conf` 字典中。最后，函数返回计划文件的内容。

- POST: 如果是 POST 请求，函数先读取请求 JSON 数据，然后调用 `write_plan` 函数将计划内容写入到 `conf` 字典中的计划文件中。最后，函数返回一个字符串，说明新计划已经成功保存到文件中。

函数中使用了两个全局变量，分别是 `conf` 和 `plan`，分别用于存储计划文件读取和写入时的配置信息。

另外，函数中还使用了 `require_token` 装饰器，用于验证 HTTP 令牌是否已经过期。


```
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


```

这段代码是一个用于获取机器人列表的 RESTful 路由，主要作用是读取并返回一个 JSON 格式的机器人列表。具体来说，它执行以下操作：

1. 检查是否获取到 "frozen" 属性的值，如果没有，则执行以下操作：

  
  if not hasattr(sys, 'frozen'):
      sys._MEIPASS = '/path/to/frozen/MEIPASS'
      with open('/path/to/frozen/MEIPASS/arknights_mower/data/agent.json', 'r', encoding='utf8') as f:
          return json.load(f)
      del sys._MEIPASS
  else:
      return json.load(os.path.join('/path/to/frozen/MEIPASS', 'arknights_mower', 'data', 'agent.json'))
  

  这一步的作用是，在没有 "frozen" 属性值的情况下，将系统中的 "MEIPASS" 目录下的 "arknights_mower" 目录下的 "data" 目录下的 "agent.json" 文件读取并返回。

2. 如果 "frozen" 属性值已经被获取到，则执行以下操作：

  
  if hasattr(sys, 'frozen'):
      return json.load(os.path.join('/path/to/frozen/MEIPASS', 'arknights_mower', 'data', 'agent.json'))
  

  这一步的作用是，如果已经获取到了 "frozen" 属性值，则直接返回从 "MEIPASS" 目录下的 "arknights_mower" 目录读取的 "data" 目录下的 "agent.json" 文件内容。

3. 如果上述两种情况中任何一种不成立，则执行以下操作：

  
  raise Exception('Failed to load agent.json')
  

  这一步的作用是，如果上述两种情况中任何一种不成立，则抛出一个自定义的异常。


```
@app.route("/operator")
def operator_list():
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
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
            return json.load(f)
    else:
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
            return json.load(f)


```

这段代码是一个用于处理 "shop" 路由的函数，主要作用是读取一个名为 "shop.json" 的 JSON 文件内容并返回给客户端。具体来说，代码可以做以下几件事情：

1. 检查是否使用了有效的库和模块。
2. 读取并返回 JSON 文件内容。
3. 如果使用了有效的库和模块，那么使用 "r" 模式打开文件，否则使用 "r+b" 模式打开文件。
4. 处理文件时使用 UTF-8 编码。
5. 检查是否使用了 '__init__' 包。如果是，则执行模块初始化操作。


```
@app.route("/shop")
def shop_list():
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
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
            return json.load(f)
    else:
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
            return json.load(f)


```

这段代码的作用是读取并处理服务器日志中的信息，包括向服务器发送新的日志行，将实时的日志信息存储到指定的日志文件中，并在需要时向指定的服务器客户端发送更新的配置信息。

具体来说，代码中的三个函数分别承担了以下任务：

1. `read_log`函数用于读取服务器日志。它使用服务器套接字连接到服务器，并在套接字中接收信息。如果接收到包含 "log" 消息的消息，函数会将新的日志行添加到 `log_lines` 列表中，并将其从数据库中删除超过 500 行后的内容从 `log_lines` 中删除。然后，它遍历所有可用的服务器客户端连接，将新的日志行发送到每个客户端。

2. `operators` 函数接收一个包含 "operators" 消息的消息，它用于更新服务器客户端的配置信息。

3. `update_conf` 函数用于接收服务器发送的 "update_conf" 消息。它用于更新服务器客户端的配置信息，并将其存储在 `conf` 变量中。


```
def read_log(conn):
    global operators
    global mower_process
    global log_lines
    global ws_connections

    try:
        while True:
            msg = conn.recv()
            if msg["type"] == "log":
                new_line = time.strftime("%m-%d %H:%M:%S ") + msg["data"]
                log_lines.append(new_line)
                log_lines = log_lines[-500:]
                for ws in ws_connections:
                    ws.send(new_line)
            elif msg["type"] == "operators":
                operators = msg["data"]
            elif msg["type"] == "update_conf":
                global conf
                conn.send(conf)
    except EOFError:
        conn.close()


```

这段代码是一个 Flask 应用的视图函数，定义了两个路由：

1. 根路由（/running）返回一个布尔值，表示割草进程是否正在运行。在函数内部，使用了全球变量 `mower_process`，如果 `mower_process` 变量不存在，则返回 `False`，否则返回 `True`。

2. 启动路由（/start）需要一个 JSON 密钥 `conf`、一个名为 `plan` 的计划文件和一个名为 `operators` 的操作员列表。函数内部读取 `conf` 和 `plan` 两个文件，并启动一个名为 `main` 的函数作为割草进程的入口。由于 `mower_process` 已经启动，所以该路由会返回 "Mower is already running."。在函数内部使用 `multiprocessing.Pipe()` 方法将读取和写入的管道建立起来，并启动 `main` 函数。最后，启动了 `mower_process` 进程，并将 `log_lines` 作为管道通道发送出去，接收方代码没有做任何处理，直接将 `read` 和 `plan` 两个文件发送过去。


```
@app.route("/running")
def running():
    global mower_process
    return "false" if mower_process is None else "true"


@app.route("/start")
@require_token
def start():
    global conf
    global plan
    global mower_process
    global operators
    global log_lines

    if mower_process is not None:
        return "Mower is already running."

    read, write = multiprocessing.Pipe()
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
    mower_process.start()

    Thread(target=read_log, args=(read,)).start()

    log_lines = []

    return "Mower started."


```

这段代码是一个 Flask Web 应用程序中的路由，用于处理 HTTP GET 请求到 /log 路径的请求。具体解释如下：

1. `@app.route("/stop")`是一个装饰器，用于告诉 Flask应用程序在路由 /stop 上处理 HTTP GET 请求。

2. `@require_token`是一个装饰器，用于告诉 Flask应用程序在路由 /stop 上的身份验证。如果 HTTP 请求头中携带了 token，则应用程序将检查该 token 是否与要求用户身份验证的系统所需的 token 匹配。

3. `def stop()`是一个函数，用于执行在 /stop 路由上执行的操作。在这个函数中，我们获取了一个名为 mower_process 的全局变量。

4. `if mower_process is None`是一个条件语句，用于检查 mower_process 是否为空。如果是，则我们通过返回字符串 "Mower is not running." 来告诉 HTTP 请求者。

5. `elif mower_process.terminate()`是一个条件语句，用于检查 mower_process 是否正在运行。如果是，则我们调用 mower_process.terminate() 方法来停止 mower 进程，并将 mower_process 设置为 None。

6. `mower_process = None`将 mower_process 变量设置为 None，意味着它将从根本上停止 mower 进程。

7. `return "Mower stopped."`将结果返回给 HTTP 请求者。

8. `@sock.route("/log")`是一个装饰器，用于告诉 Flask应用程序在路由 /log 上处理 HTTP POST 请求。

9. `@app.route("/log", methods=["POST"])`是一个装饰器，用于告诉 Flask应用程序在路由 /log 上处理 HTTP POST 请求，并使用 POST 方法发送请求。

10. `def log()`是一个函数，用于处理 HTTP POST 请求 /log 路径。在这个函数中，我们获取请求的数据并将其存储在名为 request_data 的变量中。

11. `if "username" in request_data:`是一个条件语句，用于检查请求中是否包含 "username"。

12. `if "password" in request_data:`是一个条件语句，用于检查请求中是否包含 "password"。

13. `if "message" in request_data:`是一个条件语句，用于检查请求中是否包含 "message"。

14. `print("message", request_data["message"])`将在满足上述三个条件的情况下输出请求中的 "message"。

15. `@app.route("/")`是一个装饰器，用于告诉 Flask应用程序在路由 / 路径根上处理 HTTP GET 请求。


```
@app.route("/stop")
@require_token
def stop():
    global mower_process

    if mower_process is None:
        return "Mower is not running."

    mower_process.terminate()
    mower_process = None

    return "Mower stopped."


@sock.route("/log")
```

这段代码是一个 Python 函数，名为 `log`，定义在名为 `app.route("/dialog/file")` 的路由上。

该函数的作用是将日志行发送到 WebSocket 连接，并将该连接添加到 `ws_connections` 列表中。如果连接已关闭，则从列表中删除连接。

具体来说，该函数通过以下步骤实现：

1. 定义了两个全局变量：`ws_connections` 和 `log_lines`。
2. 在函数中，通过 `ws.send()` 方法将日志行发送到 WebSocket 连接。
3. 通过 `ws_connections.append(ws)` 方法将连接添加到 `ws_connections` 列表中。
4. 在尝试获取数据时，使用 `ws.receive()` 方法尝试从 WebSocket 连接中读取数据。
5. 如果连接已关闭，使用 `ws_connections.remove(ws)` 方法从列表中删除连接。


```
def log(ws):
    global ws_connections
    global log_lines

    ws.send("\n".join(log_lines))
    ws_connections.append(ws)

    try:
        while True:
            ws.receive()
    except ConnectionClosed:
        ws_connections.remove(ws)


@app.route("/dialog/file")
```

这段代码使用了Python的webview库来实现文件对话框和文件夹对话框的功能。具体来说，这段代码定义了两个函数：`open_file_dialog` 和 `open_folder_dialog`。

这两个函数都会接受一个参数 `dialog_type`，表示对话框的类型，其中 `webview.OPEN_DIALOG` 表示打开文件对话框，而 `webview.FOLDER_DIALOG` 表示打开文件夹对话框。

函数内部首先获取当前窗口，然后调用 `window.create_file_dialog` 方法，传递 `dialog_type` 参数，得到一个 `file_path` 参数。如果文件路径不为空，则返回该文件路径，否则返回一个空字符串。

在另一个函数 `open_folder_dialog` 中，同样调用 `window.create_file_dialog` 方法，传递 `dialog_type` 参数，得到一个 `folder_path` 参数。如果文件夹路径不为空，则返回该文件夹路径，否则返回一个空字符串。


```
@require_token
def open_file_dialog():
    window = webview.active_window()
    file_path = window.create_file_dialog(dialog_type=webview.OPEN_DIALOG)
    if file_path:
        return file_path[0]
    else:
        return ""


@app.route("/dialog/folder")
@require_token
def open_folder_dialog():
    window = webview.active_window()
    folder_path = window.create_file_dialog(dialog_type=webview.FOLDER_DIALOG)
    if folder_path:
        return folder_path[0]
    else:
        return ""


```

这段代码的作用是获取一个名为 "maa" 的库的版本信息并返回。具体来说，它包含了以下几个步骤：

1. 通过 `require_token` 装饰器，使用已配置的 OAuth2 令牌进行身份验证。
2. 进入 `get_maa_adb_version` 函数内部，尝试从配置文件中读取 "maa_path" 和 "adb" 两个参数。
3. 如果已经配置了 "maa_path"，则加载对应的库，并获取其中的 "Asst" 对象。
4. 调用 "Asst.get_version()" 方法获取库的版本信息。
5. 调用 "Asst.set_instance_option(2, conf['maa_touch_option'])" 方法，设置 "maa_touch_option" 的值为 2。
6. 尝试连接到指定的 "maa_adb_path" 和 "adb" 路径，如果连接成功，则返回 "Maa {version} 加载成功" 的消息，否则返回 "连接失败，请检查Maa日志！" 的消息。
7. 如果出现异常，则返回 "Maa加载失败：" + str(e) 的消息。


```
@app.route("/check-maa")
@require_token
def get_maa_adb_version():
    try:
        Asst.load(conf["maa_path"])
        asst = Asst()
        version = asst.get_version()
        asst.set_instance_option(2, conf["maa_touch_option"])
        if asst.connect(conf["maa_adb_path"], conf["adb"]):
            maa_msg = f"Maa {version} 加载成功"
        else:
            maa_msg = "连接失败，请检查Maa日志！"
    except Exception as e:
        maa_msg = "Maa加载失败：" + str(e)
    return maa_msg


```

这段代码是一个用于获取Maa连接预置值的视图函数。视图函数的参数是一个字符串，通过调用 `get_maa_conn_presets` 并传递一个空字符串 `{}` 作为参数，来获取配置文件中所有Maa连接预置的名称列表。

具体来说，代码首先通过 `os.path.join` 组合多个文件路径，获取到一个名为 `conf["maa_path"]` 的配置文件路径，并在其中目录下查找名为 `"resource"` 目录下的一个名为 `"config.json"` 的文件。如果文件存在，则代码通过 `with` 语句打开该文件，并解码为Python可读的JSON格式，通过循环遍历文件中的每一行，提取出每一行的 `"configName"` 字段，并将它们存储到 `presets` 列表中。如果文件不存在，或者读取JSON文件时出现错误，则 `presets` 列表将包含一个空列表。

最后，代码通过 `return` 函数将 `presets` 列表返回给调用者，以便他们进行后续操作。


```
@app.route("/maa-conn-preset")
@require_token
def get_maa_conn_presets():
    try:
        with open(
            os.path.join(conf["maa_path"], "resource", "config.json"),
            "r",
            encoding="utf-8",
        ) as f:
            presets = [i["configName"] for i in json.load(f)["connection"]]
    except:
        presets = []
    return presets


```

这段代码定义了两个爱：

1. 定义了一个名为 `get_mood_ratios` 的路由，该路由返回一个名为 `record` 的对象的 `get_mood_ratios` 方法。这个方法调用了 `record` 应用程序的一个名为 `get_mood_ratios` 的函数。

2. 定义了一个名为 `test_email` 的路由，该路由返回一个名为 `MIMEMultipart` 对象的 `MIMEText` 类的一个实例。该实例包含一个主旨、发件人地址和一个消息主题，以及一个名为 `conf` 的应用程序配置对象的值。`conf` 应用程序配置对象包含用于发送邮件的主题、发件人邮箱和邮箱密码。

在这个例子中，假设 `conf` 中包含了电子邮件配置对象，比如 `SMTP_SSL_HOST`、`SMTP_SSL_PORT`、`SMTP_SSL_USERNAME`、`SMTP_SSL_PASSWORD` 等。


```
@app.route("/record/getMoodRatios")
def get_mood_ratios():
    return record.get_mood_ratios()


@app.route("/test-email")
@require_token
def test_email():
    msg = MIMEMultipart()
    msg.attach(MIMEText("arknights-mower测试邮件", "plain"))
    msg["Subject"] = conf["mail_subject"] + "测试邮件"
    msg["From"] = conf["account"]
    try:
        s = smtplib.SMTP_SSL("smtp.qq.com", 465, timeout=10.0)
        s.login(conf["account"], conf["pass_code"])
        s.sendmail(conf["account"], conf["account"], msg.as_string())
    except Exception as e:
        return "邮件发送失败！\n" + str(e)
    return "邮件发送成功！"

```

# `/opt/arknights-mower/setup.py`

该代码使用了Python的setuptools库来安装arknights_mower库。具体来说，它通过运行 `import setuptools` 来导入setup工具，并通过 `setuptools.setup(...)` 来设置arknights_mower库的元数据和版本信息。

然后，它通过读取README.md文件中的内容来设置arknights_mower库的描述。

最后，它通过循环遍历所有的类来定义arknights_mower库中的函数和类，并在setup.py文件中包含它们。


```
import setuptools
import arknights_mower
from pathlib import Path

LONG_DESC = Path('README.md').read_text('utf8')
VERSION = arknights_mower.__version__

setuptools.setup(
    name='arknights_mower',
    version=VERSION,
    author='Konano',
    author_email='w@nano.ac',
    description='Arknights Helper based on ADB and Python',
    long_description=LONG_DESC,
    long_description_content_type='text/markdown',
    url='https://github.com/Konano/arknights-mower',
    packages=setuptools.find_packages(),
    install_requires=[
        'colorlog', 'opencv_python', 'matplotlib', 'numpy', 'scikit_image==0.18.3', 'scikit_learn>=1',
        'onnxruntime', 'pyclipper', 'shapely', 'tornado', 'requests', 'ruamel.yaml', 'schedule'
    ],
    include_package_data=True,
    entry_points={'console_scripts': [
        'arknights-mower=arknights_mower.__main__:main'
    ]},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Games/Entertainment',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)

```

# `/opt/arknights-mower/webview_ui.py`

这段代码是一个Python脚本，它具有以下几个主要部分：

1. 导入webview模块，该模块可能需要从其他模块（如wxPython或webbrowser）导入。
2. 从名为“server”的类中导入名为“app”的函数，该函数可能是一个管理应用程序的类。
3. 从名为“mower_process”的类中导入名为“mower_process”的函数，该函数可能是一个处理草坪割草工作的函数。
4. 从名为“conf”的模块中导入名为“load_conf”和“save_conf”的函数，这些函数可能用于从配置文件中读取或写入配置信息。
5. 从名为“__init__”的函数中导入“arknights_mower”的类，该类可能是一个具有初始化方法的类。
6. 从PIL库中导入Image，该库可能用于在应用程序中显示图像。
7. 从pystray库中导入Icon、Menu和MenuItem，这些库可能用于创建通知栏和菜单。
8. 在脚本中创建一个Thread实例，该实例可能用于在应用程序中执行割草工作。
9. 在脚本中创建一个Image对象，该对象可能用于显示在应用程序中的图像。
10. 在脚本中创建一个Menu对象，该对象可能包含用于控制割草工作的菜单项。
11. 在脚本中创建一个MenuItem对象，该对象可能用于将菜单项添加到菜单中。


```
#!/usr/bin/env python3

import webview
from server import app, mower_process

import os
import multiprocessing

from arknights_mower.utils.conf import load_conf, save_conf
from arknights_mower.__init__ import __version__

from threading import Thread
from PIL import Image
from pystray import Icon, Menu, MenuItem

```

这段代码使用了Python的一些标准库函数，包括`socket`用于创建套接字并发送/接收数据，`tkinter`用于创建GUI界面，以及`messagebox`用于显示消息对话框。具体来说，这段代码的作用是创建一个基于TCP/IP的服务器，客户端连接到该服务器，可以在客户端发送任意类型的数据，并将所有接收到的数据在终端输出。

具体地，代码首先创建一个名为`server`的套接字，用于接收客户端发送的请求。然后代码创建一个名为`root`的Tkinter窗口，用于显示客户端发送的消息。接着，代码循环等待客户端连接，一旦连接建立，代码就从客户端接收数据，并显示在Tkinter的`textbox`中。

此外，代码还定义了一个`quit_app`变量，用于控制客户端是否通过按下`Quit`按钮而退出。如果这个变量为`True`，那么在客户端发送最后一个数据包后，程序将终止运行。


```
import socket
import tkinter
from tkinter import messagebox
from time import sleep
import sys


quit_app = False
display = True


def on_resized(w, h):
    global width
    global height

    width = w
    height = h


```

这段代码定义了一个名为 `toggle_window` 的函数，它会 global地修改 `window` 和 `display` 对象的值，并将 `display` 的值更改为相反的值，从而实现了一个窗口的显示与隐藏的切换。

接着，定义了一个名为 `on_closing` 的函数，当应用程序正在关闭时，它会创建一个新线程，目标为调用 `toggle_window` 函数，并将其作为参数传递。这样做可以确保在应用程序关闭时，将显示窗口并将其隐藏，以防止用户在关闭窗口时看到错误信息。

最后，定义了一个名为 `destroy_window` 的函数，它会全局地修改 `quit_app` 变量，将其设置为 `True`，并全局地修改 `window` 对象的值，将其销毁。这个函数通常在应用程序结束时被调用，以释放其资源并清理退出。


```
def toggle_window():
    global window
    global display
    window.hide() if display else window.show()
    display = not display


def on_closing():
    if not quit_app:
        Thread(target=toggle_window).start()
        return False


def destroy_window():
    global quit_app
    global window
    quit_app = True
    window.destroy()


```

这段代码是一个用于启动Mower游戏的脚本。它首先检查当前 port 是否已被占用，若占用则显示一个错误消息并退出游戏。然后创建一个 WebView 窗口，并在窗口加载游戏时启动游戏服务器。游戏服务器连接到指定的主机和端口，并在服务器运行时使用客户端提供的 token 进行身份验证。

脚本的主要功能包括：

1. 启动游戏服务器并连接到指定的主机和端口。
2. 创建一个 WebView 窗口，并在窗口加载游戏时启动游戏服务器。
3. 检查当前 port 是否已被占用，并在占用时显示一个错误消息并退出游戏。
4. 启动 Mower 游戏并运行时使用客户端提供的 token 进行身份验证。


```
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


if __name__ == "__main__":
    multiprocessing.freeze_support()

    conf = load_conf()

    port = conf["webview"]["port"]
    token = conf["webview"]["token"]
    host = "0.0.0.0" if token else "127.0.0.1"

    if is_port_in_use(port):
        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror(
            "arknights-mower",
            f"端口{port}已被占用，无法启动！",
        )
        sys.exit()

    app.token = token
    Thread(
        target=app.run,
        kwargs={"host": host, "port": port},
        daemon=True,
    ).start()

    global width
    global height

    width = conf["webview"]["width"]
    height = conf["webview"]["height"]

    tray_img = Image.open(os.path.join(os.getcwd(), "logo.png"))
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
    icon.run_detached()

    global window
    window = webview.create_window(
        f"Mower {__version__} (http://{host}:{port})",
        f"http://127.0.0.1:{port}?token={token}",
        width=width,
        height=height,
        text_select=True,
    )

    window.events.resized += on_resized
    window.events.closing += on_closing

    while not is_port_in_use(port):
        sleep(0.1)
    webview.start()

    global mower_process
    if mower_process:
        mower_process.terminate()
        mower_process = None

    icon.stop()

    conf = load_conf()
    conf["webview"]["width"] = width
    conf["webview"]["height"] = height
    save_conf(conf)

```

# `/opt/arknights-mower/纯跑单.py`

这段代码是一个 Python 程序，它导入了多个第三方库，如 `copy`、`ctypes`、`cv2`、`inspect`、`json`、`os`、`pystray`、`smtplib`、`sys`、`threading`、`time`、`warnings` 和 `datetime`。它还定义了一系列函数，用于从不同来源获取数据，并将它们组合在一起，形成一个完整的程序。

具体来说，这段代码的作用是：

1. 从 `__future__` 信使中导入了一些个性化的库，以提供更好的编程体验。
2. 导入 `copy` 库，以便在需要时复制数据对象。
3. 导入 `ctypes` 库，以便能够使用 `cstring` 和 `cdiv` 函数来对 C 语言文本进行操作。
4. 导入 `cv2` 库，以便使用 OpenCV 中的图像处理函数。
5. 导入 `inspect` 库，以便使用 `getmembers` 函数获取一个对象的属性列表，同时使用 `isinstance` 函数来检查某个对象是否属于某个特定的类。
6. 导入 `json` 库，以便能够将数据 JSON 编码。
7. 导入 `os` 库，以便能够处理文件和目录操作。
8. 导入 `pystray` 库，以便能够向屏幕发送消息。
9. 导入 `smtplib` 库，以便能够发送电子邮件。
10. 导入 `sys` 库，以便能够使用 `sys.exit` 函数来终止程序。
11. 导入 `threading` 库，以便能够创建和管理线程。
12. 导入 `time` 库，以便能够处理时间相关的操作。
13. 导入 `warnings` 库，以便能够捕获警告信息。
14. 导入 `datetime` 库，以便能够创建和处理日期和时间数据。
15. 导入 `timedelta` 库，以便能够创建和处理日期之间的差异。
16. 从 `copy` 库中导入 `copy` 函数，用于在需要时复制数据对象。

另外，这段代码还定义了一系列函数，用于从不同来源获取数据，并将它们组合在一起。具体来说，这些函数包括：

* `get_ip_address()`：从网络中获取 IP 地址。
* `generate_favicon()`：生成指定 URL 的书签图标。
* `to_dimensions()`：将一个图像从原始大小转换为指定大小的图像。
* `generate_png()`：生成 PNG 格式的图像。
* `create_message()`：根据指定的格式创建一个消息，以便使用 `pystray` 库发送到屏幕。
* `send_message()`：使用 `pystray` 库发送一条消息到屏幕。
* `format_message()`：格式化一条消息，以便能够通过 `pystray.Payload` 类发送到屏幕。
* `ctypes.windll_wait_for_event()`：使用 `ctypes` 库等待一个事件的发生。
* `os.path.join()`：将两个或多个文件或目录连接起来。
* `os.path.exists()`：检查文件或目录是否存在。
* `os.path.isfile()`：检查文件是否属于一个特定的文件类型。


```
from __future__ import annotations
import copy
import ctypes
import cv2
import inspect
import json
import os
import pystray
import smtplib
import sys
import threading
import time
import warnings
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
```

这段代码的作用是创建一个电子邮件文本，其中包含游戏中的角色名称和他们的移动速度。这些名称和速度用于在游戏界面上显示玩家的角色，并在玩家通过游戏内升级时更新。

具体来说，这段代码实现了以下功能：

1. 从 `email.mime.text` 模块中导入 `MIMEText` 类，用于创建和管理电子邮件文本。
2. 从 `enum` 模块中导入 `EnvelopeColor` 类，用于表示电子邮件的背景颜色。
3. 从 `Tkinter` 模块中导入 `*` 键，用于创建窗口和按钮。
4. 从 `typing` 模块中导入 `Optional` 类型，用于处理可能没有内容的 `Message` 对象。
5. 从 `PIL` 模块中导入 `Image` 类，用于显示游戏中的角色图片。
6. 从 `pystray` 模块中导入 `MenuItem`、`Menu` 和 `Asst` 类，用于创建菜单和助手功能。
7. 从 `arknights_mower.data` 模块中导入 `agent_list` 函数，用于获取游戏中的所有玩家及其相关信息。
8. 从 `arknights_mower.utils` 模块中导入 `character_recognize`、`config`、`detector` 和 `segment` 函数，用于处理字符识别、游戏状态检测和图像分割等任务。
9. 从 `arknights_mower.utils.asst` 模块中导入 `Asst` 和 `Message` 类，用于创建游戏内助手功能。
10. 从 `arknights_mower.utils.datetime` 模块中导入 `get_server_weekday` 和 `the_same_time` 函数，用于获取服务器当前的工作日和相同时间。
11. 从 `arknights_mower.utils.device.adb_client` 模块中导入 `ADBClient` 类，用于通过 Android Debug Bridge(ADB)设备控制 Android 设备。
12. 从 `arknights_mower.utils.device.minitouch` 模块中导入 `MiniTouch` 类，用于处理玩家通过触摸板进行操作。
13. 从 `arknights_mower.utils.device.scrcpy` 模块中导入 `Scrcpy` 类，用于读取和写入 Android 设备的屏幕截图。
14. 从 `arknights_mower.utils.digit_reader` 模块中导入 `DigitReader` 类，用于读取和写入玩家在游戏中点击的数字。
15. 从 `Pillow` 库中导入 `Image` 类，用于显示游戏中的角色图片。


```
from email.mime.text import MIMEText
from enum import Enum
from tkinter import *
from typing import Optional
from PIL import Image
from pystray import MenuItem, Menu
from arknights_mower.data import agent_list
from arknights_mower.utils import (character_recognize, config, detector, segment)
from arknights_mower.utils import typealias as tp
from arknights_mower.utils.asst import Asst, Message
from arknights_mower.utils.datetime import get_server_weekday, the_same_time
from arknights_mower.utils.device.adb_client import ADBClient
from arknights_mower.utils.device.minitouch import MiniTouch
from arknights_mower.utils.device.scrcpy import Scrcpy
from arknights_mower.utils.digit_reader import DigitReader
```

这段代码是一个Python程序，它的作用是执行以下操作：

1. 从 `arknights_mower.utils.log` 包中初始化 `init_fhlr`、`logger` 和 `save_screenshot` 函数。
2. 从 `arknights_mower.utils.operators` 包中导入 `Operator` 和 `Operators`。
3. 从 `arknights_mower.utils.pipe` 包中导入 `push_operators`。
4. 从 `arknights_mower.utils.scheduler_task` 包中导入 `SchedulerTask`。
5. 从 `arknights_mower.utils.solver` 包中导入 `BaseSolver`。
6. 从 `arknights_mower.utils.recognize` 包中导入 `Recognizer` 和 `RecognizeError`。
7. 定义了一个 `warn` 函数，它的参数是 `*args` 和 `**kwargs`。

`warn` 函数的作用是在运行任务时发出警告。在没有警告的情况下，代码会继续执行。警告可以通过运行 `python -m warnings.warn` 来开启。


```
from arknights_mower.utils.log import init_fhlr, logger, save_screenshot
from arknights_mower.utils.operators import Operator, Operators
from arknights_mower.utils.pipe import push_operators
from arknights_mower.utils.scheduler_task import SchedulerTask
from arknights_mower.utils.solver import BaseSolver
from arknights_mower.utils.recognize import Recognizer, RecognizeError


def warn(*args, **kwargs):
    pass


warnings.warn = warn

from paddleocr import PaddleOCR
```

这段代码是一个Python脚本，它使用了ArknightsMower游戏引擎中的Solver类来实现游戏中的AI。以下是该脚本的作用：

1. 创建一个GUI窗口，用于显示游戏中的信息。
2. 设置游戏服务器，官方服务器和Bilibili服务器。
3. 设置窗口标题和窗口大小。
4. 定义窗口中的变量，包括游戏中的服务器名称，跑单提前运行时间，更换干员前缓冲时间等。
5. 创建一个空白的窗口，并在窗口中显示游戏中的相关信息。
6. 循环运行游戏，直到用户关闭窗口。
7. 在游戏过程中，使用Solver类提供的算法来处理游戏逻辑，包括自动切换干员，加入或退出游戏联盟等等。


```
from arknights_mower.strategy import Solver

官方服务器 = 'com.hypergryph.arknights'
Bilibili服务器 = 'com.hypergryph.arknights.bilibili'

窗口 = Tk()
################################################################################################
# # # # # # # # # # # # # # # # # # # # # # 用户设置区 # # # # # # # # # # # # # # # # # # # # # #
################################################################################################

服务器 = 官方服务器  # 服务器选择 (官方服务器/Bilibili服务器)

跑单提前运行时间 = 300  # 秒
更换干员前缓冲时间 = 30  # 秒 需要严格大于一次跟服务器交换数据的时间 建议大于等于15秒

```

这段代码是一个 Python 程序，其作用是设置一个贸易站中两个跑单干员的位置以及他们在休息时间是否可用。

具体来说，它将每个跑单干员的位置存储在一个字典中，字典的键是不同的房间编号，而值则是一个列表，包含该房间的所有跑单干员。然后，它还设置了一个名为 "龙舌兰" 的布尔值，表示两个跑单干员是否在休息状态，以及一个名为 "但书" 的布尔值，表示两个跑单干员是否正在睡觉。

接下来，它创建了一个宿舍设置对象，其中包含一个字典，该字典的键是不同的房间编号，而值则是一个列表，包含该房间的所有干员。此外，它还设置了一个名为 "log" 的字符串变量，用于存储每个截图文件名的文件路径，以及一个名为 "screenshot" 的文件夹，用于存储截图文件。

最后，它还设置了一个名为 "每种截图的最大保存数量" 的整数变量，用于限制每个截图文件保存的数量。当两个跑单干员都处于休息状态时，程序将退出游戏。


```
# 设置贸易站的房间以及跑单干员的具体位置
# 请注意手动换班后记得重新运行程序
跑单位置设置 = {
    'B101': ['', '龙舌兰', '但书'],
    'B201': ['', '龙舌兰', '但书'],
}

# 龙舌兰、但书休息设置
龙舌兰和但书休息 = True
宿舍设置 = {'B401': ['当前休息干员', '当前休息干员', '当前休息干员', '龙舌兰', '但书']}

日志存储目录 = './log'
截图存储目录 = './screenshot'
每种截图的最大保存数量 = 10
任务结束后退出游戏 = True

```

这段代码包含了一些设置，主要是关于跑单弹窗提醒开关、悬浮字幕窗口的设置以及邮件设置。

首先，设置跑单弹窗提醒开关为True，即在程序运行时，会始终显示跑单弹窗提醒。

然后，设置悬浮字幕窗口的宽度为窗口width除以2，高度为窗口高度除以4，并且设置字幕字号为窗口高度除以18。同时，设置了字幕字体为“楷体”，并将字幕颜色设置为“#9966FF”。

接下来，设置邮件设置，包括发信邮箱、授权码和收件人邮箱等。发信邮箱的设置为“qqqqqqqqqq@qq.com”，授权码为从QQ邮箱“账户设置-账户-开启SMTP服务”中获得的授权码，而收件人邮箱则设置为多个邮箱地址，如“name@example.com”。


```
跑单弹窗提醒开关 = True

# 悬浮字幕窗口设置
# 双击字幕可关闭字幕 在托盘可重新打开
悬浮字幕开关 = True
窗口宽度 = 窗口.winfo_screenwidth() / 2
窗口高度 = 窗口.winfo_screenheight() / 4
字幕字号 = int(窗口.winfo_screenheight() / 18)  # '50'
字幕字体 = '楷体'
字幕颜色 = '#9966FF'  # 16进制颜色代码

邮件设置 = {
    '邮件提醒开关': True,
    '发信邮箱': "qqqqqqqqqqqqq@qq.com",
    '授权码': 'xxxxxxxxxxxxxxxx',  # 在QQ邮箱“账户设置-账户-开启SMTP服务”中，按照指示开启服务获得授权码
    '收件人邮箱': ['name@example.com']  # 收件人邮箱
}

```

这段代码是一个 JSON 格式创建了一个名为 "MAA" 的设置对象，包含了 MAA 路径、MAA_adb 路径和 MAA_adb 地址等信息，用于配置一个 Android 模拟器的设置。

具体来说，MAA 路径指定了包含 MAA 资源和 dll 文件的目录。MAA_adb 路径指定了模拟器连接到计算机的 adb 应用程序的路径。MAA_adb 地址指定了模拟器连接到 Android 设备的 adb 地址。

此处设置了四个开关的配置，第一个开关是集成战略，用于开启或关闭 MAA 的集成战略。第二个开关是生息演算，用于开启或关闭 MAA 的生息演算。第三个开关是保全派驻，用于开启或关闭 MAA 的保全派驻。第四个开关是周计划，用于指定每周的每天需要进行哪些关卡演算，如果开关处于打开状态，则会自动根据 MAA 路径遍历所有的关卡并进行演算。应急理智药用于指定在演算过程中如果出现异常需要手动干预的药物数量。


```
MAA设置 = {
    'MAA路径': 'K:/MAA',  # 请设置为存放 dll 文件及资源的路径
    'MAA_adb路径': 'C:/Program Files/BlueStacks_bgp64/./HD-Adb.exe',  # 请设置为模拟器的 adb 应用程序路径
    'MAA_adb地址': ['127.0.0.1:5555'],  # 请设置为模拟器的 adb 地址

    # 以下配置，第一个设置为开的首先生效
    '集成战略': False,  # 集成战略开关
    '生息演算': False,  # 生息演算开关
    '保全派驻': False,  # 保全派驻开关
    '周计划': [
        {'日子': '周一', '关卡': ['集成战略'], '应急理智药': 0},
        {'日子': '周二', '关卡': ['集成战略'], '应急理智药': 0},
        {'日子': '周三', '关卡': ['集成战略'], '应急理智药': 0},
        {'日子': '周四', '关卡': ['集成战略'], '应急理智药': 0},
        {'日子': '周五', '关卡': ['集成战略'], '应急理智药': 0},
        {'日子': '周六', '关卡': ['集成战略'], '应急理智药': 0},
        {'日子': '周日', '关卡': ['集成战略'], '应急理智药': 0}
    ]
}


```

It looks like this is a Python class that overrides some methods of the Android-ism�d UI library. This class似乎 extends the Android Application class and defines some custom methods for interacting with the Android device.

The `__init__` method initializes the device and sets up some defaults. The `display_frames` method appears to take a screenshot of the device's display and returns it as a tuple of the width, height, and frame color of the screenshot. The `tap` method appears to block the default tap gesture and calls the `control.tap` method instead. The `swipe` method appears to block the swipe gesture and calls the `control.swipe` method instead. The `swipe_ext` method appears to be a more powerful swipe gesture version with custom parameters.

The `check_current_focus` method appears to check if the application is currently in the foreground. If it's not, it launches the application.


```
################################################################################################
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
################################################################################################
ocr = None
任务提示 = str()
下个任务开始时间 = datetime.now()
字幕 = "Mower的准备阶段..."


class 设备控制(object):
    class Control(object):

        def __init__(self, device: 设备控制, client: ADBClient = None, touch_device: str = None) -> None:
            self.device = device
            self.minitouch = None
            self.scrcpy = None

            if config.ADB_CONTROL_CLIENT == 'minitouch':
                self.minitouch = MiniTouch(client, touch_device)
            elif config.ADB_CONTROL_CLIENT == 'scrcpy':
                self.scrcpy = Scrcpy(client)
            else:
                # MiniTouch does not support Android 10+
                if int(client.android_version().split('.')[0]) < 10:
                    self.minitouch = MiniTouch(client, touch_device)
                else:
                    self.scrcpy = Scrcpy(client)

        def tap(self, point: tuple[int, int]) -> None:
            if self.minitouch:
                self.minitouch.tap([point], self.device.display_frames())
            elif self.scrcpy:
                self.scrcpy.tap(point[0], point[1])
            else:
                raise NotImplementedError

        def swipe(self, start: tuple[int, int], end: tuple[int, int], duration: int) -> None:
            if self.minitouch:
                self.minitouch.swipe(
                    [start, end], self.device.display_frames(), duration=duration)
            elif self.scrcpy:
                self.scrcpy.swipe(
                    start[0], start[1], end[0], end[1], duration / 1000)
            else:
                raise NotImplementedError

        def swipe_ext(self, points: list[tuple[int, int]], durations: list[int], up_wait: int) -> None:
            if self.minitouch:
                self.minitouch.swipe(
                    points, self.device.display_frames(), duration=durations, up_wait=up_wait)
            elif self.scrcpy:
                total = len(durations)
                for idx, (S, E, D) in enumerate(zip(points[:-1], points[1:], durations)):
                    self.scrcpy.swipe(S[0], S[1], E[0], E[1], D / 1000,
                                      up_wait / 1000 if idx == total - 1 else 0,
                                      fall=idx == 0, lift=idx == total - 1)
            else:
                raise NotImplementedError

    def __init__(self, device_id: str = None, connect: str = None, touch_device: str = None) -> None:
        self.device_id = device_id
        self.connect = connect
        self.touch_device = touch_device
        self.client = None
        self.control = None
        self.start()

    def start(self) -> None:
        self.client = ADBClient(self.device_id, self.connect)
        self.control = 设备控制.Control(self, self.client)

    def run(self, cmd: str) -> Optional[bytes]:
        return self.client.run(cmd)

    def launch(self, app: str) -> None:
        """ launch the application """
        self.run(f'am start -n {app}')

    def exit(self, app: str) -> None:
        """ exit the application """
        self.run(f'am force-stop {app}')

    def send_keyevent(self, keycode: int) -> None:
        """ send a key event """
        logger.debug(f'keyevent: {keycode}')
        command = f'input keyevent {keycode}'
        self.run(command)

    def send_text(self, text: str) -> None:
        """ send a text """
        logger.debug(f'text: {repr(text)}')
        text = text.replace('"', '\\"')
        command = f'input text "{text}"'
        self.run(command)

    def screencap(self, save: bool = False) -> bytes:
        """ get a screencap """
        command = 'screencap -p 2>/dev/null'
        screencap = self.run(command)
        if save:
            save_screenshot(screencap)
        return screencap

    def current_focus(self) -> str:
        """ detect current focus app """
        command = 'dumpsys window | grep mCurrentFocus'
        line = self.run(command).decode('utf8')
        return line.strip()[:-1].split(' ')[-1]

    def display_frames(self) -> tuple[int, int, int]:
        """ get display frames if in compatibility mode"""
        if not config.MNT_COMPATIBILITY_MODE:
            return None

        command = 'dumpsys window | grep DisplayFrames'
        line = self.run(command).decode('utf8')
        """ eg. DisplayFrames w=1920 h=1080 r=3 """
        res = line.strip().replace('=', ' ').split(' ')
        return int(res[2]), int(res[4]), int(res[6])

    def tap(self, point: tuple[int, int]) -> None:
        """ tap """
        logger.debug(f'tap: {point}')
        self.control.tap(point)

    def swipe(self, start: tuple[int, int], end: tuple[int, int], duration: int = 100) -> None:
        """ swipe """
        logger.debug(f'swipe: {start} -> {end}, duration={duration}')
        self.control.swipe(start, end, duration)

    def swipe_ext(self, points: list[tuple[int, int]], durations: list[int], up_wait: int = 500) -> None:
        """ swipe_ext """
        logger.debug(
            f'swipe_ext: points={points}, durations={durations}, up_wait={up_wait}')
        self.control.swipe_ext(points, durations, up_wait)

    def check_current_focus(self):
        """ check if the application is in the foreground """
        if self.current_focus() != f"{config.APPNAME}/{config.APP_ACTIVITY_NAME}":
            self.launch(f"{config.APPNAME}/{config.APP_ACTIVITY_NAME}")
            # wait for app to finish launching
            time.sleep(10)


```

这段代码定义了一个名为“干员排序方式”的枚举类型，包括了工作状态、技能、心情和信赖值四个成员变量。

特别地，代码还定义了一个名为“干员排序方式位置”的字典，它将每个干员排序方式的成员变量值作为键，并将它们与一个嵌套的元组对应。这意味着，每个字典条目都包含了对应干员排序方式的所有成员变量，并且它们在字典中的位置是固定的，次序是确定的。

最后，代码还创建了一个名为“工作状态”的整型变量，并将其初始化为1，将其存储到了干员排序方式位置的键中，以便后续枚举类型的成员变量引用。


```
class 干员排序方式(Enum):
    工作状态 = 1
    技能 = 2
    心情 = 3
    信赖值 = 4


干员排序方式位置 = {
    干员排序方式.工作状态: (1560 / 2496, 96 / 1404),
    干员排序方式.技能: (1720 / 2496, 96 / 1404),
    干员排序方式.心情: (1880 / 2496, 96 / 1404),
    干员排序方式.信赖值: (2050 / 2496, 96 / 1404),
}


```

This is a Python class that implements a simple task scheduler. It has a function called `run_tasks`, which runs the tasks according to the schedule defined in the `scheduled_tasks` list.

The scheduler has several features:

* It supports multiple email accounts for sending notifications. When a task is added, the scheduler checks if the email account is set. If it is, and the email address is marked as "通知我"。scheduler会发送一封包含任务的电子邮件通知。
* It supports batch processing. When a task is added, the scheduler checks if there is a batch process. If there is, it is implemented.
* It supports automatic retries. If a task fails, the scheduler will automatically retry the task up to a maximum number of times. If the task still fails, the scheduler will stop trying to contact the person responsible for the failure.
* It supports the setting of a maximum number of retries. This feature is useful if the scheduler is configured to stop trying to contact the person responsible for the failure.

The scheduler is run in the background, and it is accessible through the `run_tasks` function.


```
def 调试输出():
    logger.handlers[0].setLevel('DEBUG')


def 日志设置():
    config.LOGFILE_PATH = 日志存储目录
    config.SCREENSHOT_PATH = 截图存储目录
    config.SCREENSHOT_MAXNUM = 每种截图的最大保存数量 - 1
    config.MAX_RETRYTIME = 10


class 项目经理(BaseSolver):
    服务器 = ''

    def __init__(self, device: 设备控制 = None, recog: Recognizer = None) -> None:
        super().__init__(device, recog)
        self.plan = None
        self.planned = None
        self.todo_task = None
        self.邮件设置 = None
        self.干员信息 = None
        self.跑单提前运行时间 = 300
        self.digit_reader = DigitReader()
        self.error = False
        self.任务列表 = []
        self.run_order_rooms = {}

    def 返回基主界面(self):
        logger.info('返回基建主界面')
        while self.get_infra_scene() != 201:
            if self.find('index_infrastructure') is not None:
                self.tap_element('index_infrastructure')
            else:
                self.back()
            self.recog.update()

    def run(self) -> None:
        self.error = False
        if len(self.任务列表) == 0:
            self.recog.update()
            time.sleep(1)
        self.handle_error(True)
        if len(self.任务列表) > 0:
            # 找到时间最近的一次单个任务
            self.任务 = self.任务列表[0]
        else:
            self.任务 = None
        self.todo_task = False
        self.collect_notification = False
        self.planned = False
        if self.干员信息 is None or self.干员信息.operators is None:
            self.initialize_operators()
        return super().run()

    def transition(self) -> None:
        self.recog.update()
        if self.scene() == 1:
            self.tap_element('index_infrastructure')
        elif self.scene() == 201:
            return self.infra_main()
        elif self.scene() == 202:
            return self.收获产物()
        elif self.scene() == 205:
            self.back()
        elif self.scene() == 9998:
            time.sleep(1)
        elif self.scene() == 9:
            time.sleep(1)
        elif self.get_navigation():
            self.tap_element('nav_infrastructure')
        elif self.scene() == 207:
            self.tap_element('arrange_blue_yes')
        elif self.scene() != -1:
            self.back_to_index()
            self.last_room = ''
        else:
            raise RecognizeError('Unknown scene')

    def find_next_task(self, compare_time=None, task_type='', compare_type='<'):
        if compare_type == '=':
            return next((e for e in self.任务列表 if the_same_time(e.time, compare_time) and (
                True if task_type == '' else task_type in e.type)), None)
        elif compare_type == '>':
            return next((e for e in self.任务列表 if (True if compare_time is None else e.time > compare_time) and (
                True if task_type == '' else task_type in e.type)), None)
        else:
            return next((e for e in self.任务列表 if (True if compare_time is None else e.time < compare_time) and (
                True if task_type == '' else task_type in e.type)), None)

    def handle_error(self, force=False):
        while self.scene() == -1:
            self.recog.update()
            logger.info('返回基建主界面')
            unknown_count = 0
            while self.get_infra_scene() != 201 and unknown_count < 5:
                if self.find('index_infrastructure') is not None:
                    self.tap_element('index_infrastructure')
                else:
                    self.back()
                self.recog.update()
                time.sleep(1)
                unknown_count += 1
            self.device.exit(self.服务器)
        if self.error or force:
            # 如果没有任何时间小于当前时间的任务才生成空任务
            if self.find_next_task(datetime.now()) is None:
                logger.debug("由于出现错误情况，生成一次空任务来执行纠错")
                self.任务列表.append(SchedulerTask())
            # 如果没有任何时间小于当前时间的任务-10分钟 则清空任务
            if self.find_next_task(datetime.now() - timedelta(seconds=900)) is not None:
                logger.info("检测到执行超过15分钟的任务，清空全部任务")
                self.任务列表 = []
        elif self.find_next_task(datetime.now() + timedelta(hours=2.5)) is None:
            logger.debug("2.5小时内没有其他任务，生成一个空任务")
            self.任务列表.append(SchedulerTask(time=datetime.now() + timedelta(hours=2.5)))
        return True

    def infra_main(self):
        """ 位于基建首页 """
        if self.find('control_central') is None:
            self.back()
            return
        if self.任务 is not None:
            try:
                if len(self.任务.plan.keys()) > 0:
                    get_time = False
                    if "Shift_Change" == self.任务.type:
                        get_time = True
                    self.跑单(self.任务.plan, get_time)
                    if get_time:
                        self.plan_metadata()
                # elif self.任务.type == 'impart':
                #     self.skip(['planned', 'collect_notification'])
                del self.任务列表[0]
            except Exception as e:
                logger.exception(e)
                self.skip()
                self.error = True
            self.任务 = None
        elif not self.planned:
            try:
                self.plan_solver()
            except Exception as e:
                # 重新扫描
                self.error = True
                logger.exception({e})
            self.planned = True
        elif not self.todo_task:
            self.todo_task = True
        elif not self.collect_notification:
            notification = detector.infra_notification(self.recog.img)
            if notification is None:
                time.sleep(1)
                notification = detector.infra_notification(self.recog.img)
            if notification is not None:
                self.tap(notification)
            self.collect_notification = True
        else:
            return self.handle_error()

    def plan_solver(self):
        plan = self.plan
        # 如果下个 普通任务 <10 分钟则跳过 plan
        if self.find_next_task(datetime.now() + timedelta(seconds=600)) is not None:
            return
        if len(self.run_order_rooms) > 0:
            for k, v in self.run_order_rooms.items():
                # 如果没有当前房间数据
                if 'plan' not in v.keys():
                    v['plan'] = self.干员信息.get_current_room(k)
                if self.find_next_task(task_type=k) is not None: continue;
                in_out_plan = {k: ['Current'] * len(plan[k])}
                for idx, x in enumerate(plan[k]):
                    if '但书' in x['replacement'] or '龙舌兰' in x['replacement']:
                        in_out_plan[k][idx] = x['replacement'][0]
                self.任务列表.append(
                    SchedulerTask(time=self.读取接单时间(k), task_plan=in_out_plan, task_type=k))
        # 准备数据
        logger.debug(self.干员信息.print())

    def initialize_operators(self):
        plan = self.plan
        self.干员信息 = Operators({}, 4, plan)
        for room in plan.keys():
            for idx, data in enumerate(plan[room]):
                self.干员信息.add(Operator(data["agent"], room, idx, data["group"], data["replacement"], 'high',
                                       operator_type="high"))
        added = []
        # 跑单
        for x, y in self.plan.items():
            if not x.startswith('room'): continue
            if any(('但书' in obj['replacement'] or '龙舌兰' in obj['replacement']) for obj in y):
                self.run_order_rooms[x] = {}

    def 读取接单时间(self, room):
        logger.info('读取接单时间')
        # 点击进入该房间
        self.进入房间(room)
        # 进入房间详情
        error_count = 0
        while self.find('bill_accelerate') is None:
            if error_count > 5:
                raise Exception('未成功进入订单界面')
            self.tap((self.recog.w * 0.05, self.recog.h * 0.95), interval=1)
            error_count += 1
        execute_time = self.double_read_time((int(self.recog.w * 650 / 2496), int(self.recog.h * 660 / 1404),
                                              int(self.recog.w * 815 / 2496), int(self.recog.h * 710 / 1404)),
                                             use_digit_reader=True)
        logger.warning('房间 B' + room[5] + '0' + room[7] + ' 接单时间为 ' + execute_time.strftime("%H:%M:%S"))
        execute_time = execute_time - timedelta(seconds=(self.跑单提前运行时间))
        self.recog.update()
        self.返回基主界面()
        return execute_time

    def double_read_time(self, cord, upperLimit=None, use_digit_reader=False):
        if upperLimit is not None and upperLimit < 36000:
            upperLimit = 36000
        self.recog.update()
        time_in_seconds = self.read_time(cord, upperLimit, use_digit_reader)
        if time_in_seconds is None:
            return datetime.now()
        execute_time = datetime.now() + timedelta(seconds=(time_in_seconds))
        return execute_time

    def initialize_paddle(self):
        global ocr
        if ocr is None:
            # mac 平台不支持 mkldnn 加速，关闭以修复 mac 运行时错误
            if sys.platform == 'darwin':
                ocr = PaddleOCR(enable_mkldnn=False, use_angle_cls=False, show_log=False)
            else:
                ocr = PaddleOCR(enable_mkldnn=True, use_angle_cls=False, show_log=False)

    def read_screen(self, img, type="mood", limit=24, cord=None):
        if cord is not None:
            img = img[cord[1]:cord[3], cord[0]:cord[2]]
        if 'mood' in type or type == "time":
            # 心情图片太小，复制8次提高准确率
            for x in range(0, 4):
                img = cv2.vconcat([img, img])
        try:
            self.initialize_paddle()
            rets = ocr.ocr(img, cls=True)
            line_conf = []
            for idx in range(len(rets[0])):
                res = rets[0][idx]
                if 'mood' in type:
                    # filter 掉不符合规范的结果
                    if ('/' + str(limit)) in res[1][0]:
                        line_conf.append(res[1])
                else:
                    line_conf.append(res[1])
            logger.debug(line_conf)
            if len(line_conf) == 0:
                if 'mood' in type:
                    return -1
                elif 'name' in type:
                    logger.debug("使用老版识别")
                    return character_recognize.agent_name(img, self.recog.h)
                else:
                    return ""
            x = [i[0] for i in line_conf]
            __str = max(set(x), key=x.count)
            if "mood" in type:
                if '.' in __str:
                    __str = __str.replace(".", "")
                number = int(__str[0:__str.index('/')])
                return number
            elif 'time' in type:
                if '.' in __str:
                    __str = __str.replace(".", ":")
            elif 'name' in type and __str not in agent_list:
                logger.debug("使用老版识别")
                __str = character_recognize.agent_name(img, self.recog.h)
            logger.debug(__str)
            return __str
        except Exception as e:
            logger.exception(e)
            return limit + 1

    def read_time(self, cord, upperlimit, error_count=0, use_digit_reader=False):
        # 刷新图片
        self.recog.update()
        if use_digit_reader:
            time_str = self.digit_reader.get_time(self.recog.gray)
        else:
            time_str = self.read_screen(self.recog.img, type='time', cord=cord)
        try:
            h, m, s = str(time_str).split(':')
            if int(m) > 60 or int(s) > 60:
                raise Exception(f"读取错误")
            res = int(h) * 3600 + int(m) * 60 + int(s)
            if upperlimit is not None and res > upperlimit:
                raise Exception(f"超过读取上限")
            else:
                return res
        except:
            logger.error("读取失败")
            if error_count > 3:
                logger.exception(f"读取失败{error_count}次超过上限")
                return None
            else:
                return self.read_time(cord, upperlimit, error_count + 1, use_digit_reader)

    def 收获产物(self) -> None:
        """ 处理基建收获产物列表 """
        tapped = False
        trust = self.find('infra_collect_trust')
        if trust is not None:
            logger.info('干员信赖')
            self.tap(trust)
            tapped = True
        bill = self.find('infra_collect_bill')
        if bill is not None:
            logger.info('订单交付')
            self.tap(bill)
            tapped = True
        factory = self.find('infra_collect_factory')
        if factory is not None:
            logger.info('可收获')
            self.tap(factory)
            tapped = True
        if not tapped:
            self.tap((self.recog.w * 0.05, self.recog.h * 0.95))
            self.todo_task = True

    def 进入房间(self, room: str) -> tp.Rectangle:
        """ 获取房间的位置并进入 """

        # 获取基建各个房间的位置
        base_room = segment.base(self.recog.img, self.find('control_central', strict=True))
        # 将画面外的部分删去
        _room = base_room[room]

        for i in range(4):
            _room[i, 0] = max(_room[i, 0], 0)
            _room[i, 0] = min(_room[i, 0], self.recog.w)
            _room[i, 1] = max(_room[i, 1], 0)
            _room[i, 1] = min(_room[i, 1], self.recog.h)

        # 点击进入
        self.tap(_room[0], interval=1)
        while self.find('control_central') is not None:
            self.tap(_room[0], interval=1)

    def 无人机加速(self, room: str, not_customize=False, not_return=False):
        logger.info('无人机加速')
        # 点击进入该房间
        self.进入房间(room)
        # 进入房间详情
        self.tap((self.recog.w * 0.05, self.recog.h * 0.95), interval=3)
        # 关闭掉房间总览
        error_count = 0
        while self.find('factory_accelerate') is None and self.find('bill_accelerate') is None:
            if error_count > 5:
                raise Exception('未成功进入无人机界面')
            self.tap((self.recog.w * 0.05, self.recog.h * 0.95), interval=3)
            error_count += 1
        accelerate = self.find('bill_accelerate')
        if accelerate:
            while (self.任务列表[1].time - self.任务列表[0].time).total_seconds() < self.跑单提前运行时间:
                logger.info(room + ' 加速')
                self.tap(accelerate)
                self.device.tap((1320, 502))
                time.sleep(1)
                self.tap((self.recog.w * 0.75, self.recog.h * 0.8))
                while self.get_infra_scene() == 9:
                    time.sleep(1)
                while self.find('bill_accelerate') is None:
                    if error_count > 5:
                        raise Exception('未成功进入订单界面')
                    self.tap((self.recog.w * 0.05, self.recog.h * 0.95), interval=1)
                    error_count += 1
                加速后接单时间 = self.double_read_time((int(self.recog.w * 650 / 2496), int(self.recog.h * 660 / 1404),
                                                        int(self.recog.w * 815 / 2496), int(self.recog.h * 710 / 1404)),
                                                       use_digit_reader=True)
                self.任务列表[0].time = 加速后接单时间 - timedelta(seconds=(self.跑单提前运行时间))
                logger.info(
                    '房间 B' + room[5] + '0' + room[7] + ' 加速后接单时间为 ' + 加速后接单时间.strftime("%H:%M:%S"))
                if not_customize:
                    drone_count = self.digit_reader.get_drone(self.recog.gray)
                    logger.info(f'当前无人机数量为：{drone_count}')
                while self.find('bill_accelerate') is None:
                    if error_count > 5:
                        raise Exception('未成功进入订单界面')
                    self.tap((self.recog.w * 0.05, self.recog.h * 0.95), interval=1)
                    error_count += 1
        if not_return: return
        self.recog.update()
        logger.info('返回基建主界面')
        self.back(interval=2, rebuild=False)
        self.back(interval=2)

    def get_arrange_order(self) -> 干员排序方式:
        best_score, best_order = 0, None
        for order in 干员排序方式:
            score = self.recog.score(干员排序方式位置[order][0])
            if score is not None and score[0] > best_score:
                best_score, best_order = score[0], order
        logger.debug((best_score, best_order))
        return best_order

    def switch_arrange_order(self, index: int, asc="false") -> None:
        self.tap((self.recog.w * 干员排序方式位置[干员排序方式(index)][0],
                  self.recog.h * 干员排序方式位置[干员排序方式(index)][1]), interval=0, rebuild=False)
        # 点个不需要的
        if index < 4:
            self.tap((self.recog.w * 干员排序方式位置[干员排序方式(index + 1)][0],
                      self.recog.h * 干员排序方式位置[干员排序方式(index)][1]), interval=0, rebuild=False)
        else:
            self.tap((self.recog.w * 干员排序方式位置[干员排序方式(index - 1)][0],
                      self.recog.h * 干员排序方式位置[干员排序方式(index)][1]), interval=0, rebuild=False)
        # 切回来
        self.tap((self.recog.w * 干员排序方式位置[干员排序方式(index)][0],
                  self.recog.h * 干员排序方式位置[干员排序方式(index)][1]), interval=0.2, rebuild=True)
        # 倒序
        if asc != "false":
            self.tap((self.recog.w * 干员排序方式位置[干员排序方式(index)][0],
                      self.recog.h * 干员排序方式位置[干员排序方式(index)][1]), interval=0.2, rebuild=True)

    def scan_agant(self, agent: list[str], error_count=0, max_agent_count=-1):
        try:
            # 识别干员
            self.recog.update()
            ret = character_recognize.agent(self.recog.img)  # 返回的顺序是从左往右从上往下
            # 提取识别出来的干员的名字
            select_name = []
            for y in ret:
                name = y[0]
                if name in agent:
                    select_name.append(name)
                    # self.get_agent_detail((y[1][0]))
                    self.tap((y[1][0]), interval=0)
                    agent.remove(name)
                    # 如果是按照个数选择 Free
                    if max_agent_count != -1:
                        if len(select_name) >= max_agent_count:
                            return select_name, ret
            return select_name, ret
        except Exception as e:
            error_count += 1
            if error_count < 3:
                logger.exception(e)
                time.sleep(1)
                return self.scan_agant(agent, error_count, max_agent_count)
            else:
                raise e

    def detail_filter(self, turn_on, type="not_in_dorm"):
        logger.info(f'开始 {("打开" if turn_on else "关闭")} {type} 筛选')
        self.tap((self.recog.w * 0.95, self.recog.h * 0.05), interval=1)
        if type == "not_in_dorm":
            not_in_dorm = self.find('arrange_non_check_in', score=0.9)
            if turn_on ^ (not_in_dorm is None):
                self.tap((self.recog.w * 0.3, self.recog.h * 0.5), interval=0.5)
        # 确认
        self.tap((self.recog.w * 0.8, self.recog.h * 0.8), interval=0.5)

    def 安排干员(self, agents: list[str], room: str) -> None:
        """
        :param order: 干员排序方式, 选择干员时右上角的排序功能
        """
        first_name = ''
        max_swipe = 50
        for idx, n in enumerate(agents):
            if n == '':
                agents[idx] = 'Free'
        agent = copy.deepcopy(agents)
        logger.info(f'安排干员 ：{agent}')
        if room.startswith('room'):
            logger.warning('房间 B' + room[5] + '0' + room[7] + ' 进驻时间为 ' + (self.任务列表[0].time + timedelta(
                seconds=(self.跑单提前运行时间 - self.更换干员前缓冲时间))).strftime("%H:%M:%S"))
        h, w = self.recog.h, self.recog.w
        first_time = True
        right_swipe = 0
        retry_count = 0
        # 如果重复进入宿舍则需要排序
        selected = []
        if room.startswith('room'):
            self.switch_arrange_order(2, "asc")
        else:
            self.switch_arrange_order(3, "asc")
        while len(agent) > 0:
            if retry_count > 3: raise Exception(f"到达最大尝试次数 3次")
            if right_swipe > max_swipe:
                # 到底了则返回再来一次
                for _ in range(right_swipe):
                    self.swipe_only((w // 2, h // 2), (w // 2, 0), interval=0.5)
                right_swipe = 0
                max_swipe = 50
                retry_count += 1
                self.detail_filter(False)
            if first_time:
                self.tap((self.recog.w * 0.38, self.recog.h * 0.95), interval=0.5)
                changed, ret = self.scan_agant(agent)
                if changed:
                    selected.extend(changed)
                    if len(agent) == 0: break
            first_time = False

            changed, ret = self.scan_agant(agent)
            if changed:
                selected.extend(changed)
            else:
                # 如果没找到 而且右移次数大于5
                if ret[0][0] == first_name and right_swipe > 5:
                    max_swipe = right_swipe
                else:
                    first_name = ret[0][0]
                st = ret[-2][1][2]  # 起点
                ed = ret[0][1][1]  # 终点
                self.swipe_noinertia(st, (ed[0] - st[0], 0))
                right_swipe += 1
            if len(agent) == 0: break;

        # 排序
        if len(agents) != 1:
            # 左移
            self.swipe_left(right_swipe, w, h)
            self.tap((self.recog.w * 干员排序方式位置[干员排序方式.技能][0],
                      self.recog.h * 干员排序方式位置[干员排序方式.技能][1]), interval=0.5, rebuild=False)
            position = [(0.35, 0.35), (0.35, 0.75), (0.45, 0.35), (0.45, 0.75), (0.55, 0.35)]
            not_match = False
            for idx, item in enumerate(agents):
                if agents[idx] != selected[idx] or not_match:
                    not_match = True
                    p_idx = selected.index(agents[idx])
                    self.tap((self.recog.w * position[p_idx][0], self.recog.h * position[p_idx][1]), interval=0,
                             rebuild=False)
                    self.tap((self.recog.w * position[p_idx][0], self.recog.h * position[p_idx][1]), interval=0,
                             rebuild=False)
        self.last_room = room

    def swipe_left(self, right_swipe, w, h):
        for _ in range(right_swipe):
            self.swipe_only((w // 2, h // 2), (w // 2, 0), interval=0.5)
        return 0

    @push_operators
    def get_agent_from_room(self, room, read_time_index=None):
        if read_time_index is None:
            read_time_index = []
        error_count = 0
        while self.find('room_detail') is None:
            if error_count > 3:
                raise Exception('未成功进入房间')
            self.tap((self.recog.w * 0.05, self.recog.h * 0.4), interval=0.5)
            error_count += 1
        length = len(self.plan[room])
        if length > 3: self.swipe((self.recog.w * 0.8, self.recog.h * 0.5), (0, self.recog.h * 0.45), duration=500,
                                  interval=1,
                                  rebuild=True)
        name_p = [((1460, 155), (1700, 210)), ((1460, 370), (1700, 420)), ((1460, 585), (1700, 630)),
                  ((1460, 560), (1700, 610)), ((1460, 775), (1700, 820))]
        result = []
        swiped = False
        for i in range(0, length):
            if i >= 3 and not swiped:
                self.swipe((self.recog.w * 0.8, self.recog.h * 0.5), (0, -self.recog.h * 0.45), duration=500,
                           interval=1, rebuild=True)
                swiped = True
            data = {}
            _name = self.read_screen(self.recog.img[name_p[i][0][1]:name_p[i][1][1], name_p[i][0][0]:name_p[i][1][0]],
                                     type="name")
            error_count = 0
            while i >= 3 and _name != '' and (
                    next((e for e in result if e['agent'] == _name), None)) is not None:
                logger.warning("检测到滑动可能失败")
                self.swipe((self.recog.w * 0.8, self.recog.h * 0.5), (0, -self.recog.h * 0.45), duration=500,
                           interval=1, rebuild=True)
                _name = self.read_screen(
                    self.recog.img[name_p[i][0][1]:name_p[i][1][1], name_p[i][0][0]:name_p[i][1][0]], type="name")
                error_count += 1
                if error_count > 1:
                    raise Exception("超过出错上限")
            # 如果房间不为空
            if _name != '':
                if _name not in self.干员信息.operators.keys() and _name in agent_list:
                    self.干员信息.add(Operator(_name, ""))
                update_time = False
                if self.干员信息.operators[_name].need_to_refresh(r=room):
                    update_time = True
                high_no_time = self.干员信息.update_detail(_name, 24, room, i, update_time)
                data['depletion_rate'] = self.干员信息.operators[_name].depletion_rate
            data['agent'] = _name
            if i in read_time_index:
                data['time'] = datetime.now()
                self.干员信息.refresh_dorm_time(room, i, data)
                logger.debug(f"停止记录时间:{str(data)}")
            result.append(data)
        for _operator in self.干员信息.operators.keys():
            if self.干员信息.operators[_operator].current_room == room and _operator not in [res['agent'] for res in
                                                                                         result]:
                self.干员信息.operators[_operator].current_room = ''
                self.干员信息.operators[_operator].current_index = -1
                logger.info(f'重设 {_operator} 至空闲')
        return result

    def refresh_current_room(self, room):
        _current_room = self.干员信息.get_current_room(room)
        if _current_room is None:
            self.get_agent_from_room(room)
            _current_room = self.干员信息.get_current_room(room, True)
        return _current_room

    def 跑单(self, plan: tp.BasePlan, get_time=False):
        rooms = list(plan.keys())
        new_plan = {}
        # 优先替换工作站再替换宿舍
        rooms.sort(key=lambda x: x.startswith('dorm'), reverse=False)
        for room in rooms:
            finished = False
            choose_error = 0
            while not finished:
                try:
                    error_count = 0
                    self.进入房间(room)
                    while self.find('room_detail') is None:
                        if error_count > 3:
                            raise Exception('未成功进入房间')
                        self.tap((self.recog.w * 0.05, self.recog.h * 0.4), interval=0.5)
                        error_count += 1
                    error_count = 0
                    if choose_error == 0:
                        if '但书' in plan[room] or '龙舌兰' in plan[room]:
                            new_plan[room] = self.refresh_current_room(room)
                        if 'Current' in plan[room]:
                            self.refresh_current_room(room)
                            for current_idx, _name in enumerate(plan[room]):
                                if _name == 'Current':
                                    plan[room][current_idx] = self.干员信息.get_current_room(room, True)[current_idx]
                        if room in self.run_order_rooms and len(new_plan) == 0:
                            if ('plan' in self.run_order_rooms[room] and
                                    plan[room] != self.run_order_rooms[room]['plan']):
                                run_order_task = self.find_next_task(
                                    compare_time=datetime.now() + timedelta(minutes=10),
                                    task_type=room, compare_type=">")
                                if run_order_task is not None:
                                    logger.info("检测到跑单房间人员变动！")
                                    self.任务列表.remove(run_order_task)
                                    del self.run_order_rooms[room]['plan']
                    while self.find('arrange_order_options') is None:
                        if error_count > 3:
                            raise Exception('未成功进入干员选择界面')
                        self.tap((self.recog.w * 0.82, self.recog.h * 0.2), interval=1)
                        error_count += 1
                    self.安排干员(plan[room], room)
                    self.recog.update()
                    if room.startswith('room'):
                        龙舌兰_但书进驻前的等待时间 = ((self.任务列表[0].time - datetime.now()).total_seconds() +
                                                       self.跑单提前运行时间 - self.更换干员前缓冲时间)
                        if 龙舌兰_但书进驻前的等待时间 > 0:
                            logger.info('龙舌兰、但书进驻前的等待时间为 ' + str(龙舌兰_但书进驻前的等待时间) + ' 秒')
                            time.sleep(龙舌兰_但书进驻前的等待时间)
                    self.tap_element('confirm_blue', detected=True, judge=False, interval=3)
                    self.recog.update()
                    if self.get_infra_scene() == 206:
                        x0 = self.recog.w // 3 * 2  # double confirm
                        y0 = self.recog.h - 10
                        self.tap((x0, y0), rebuild=True)
                    read_time_index = []
                    if get_time:
                        read_time_index = self.干员信息.get_refresh_index(room, plan[room])
                    current = self.get_agent_from_room(room, read_time_index)
                    for idx, name in enumerate(plan[room]):
                        if current[idx]['agent'] != name:
                            logger.error(f'检测到的干员{current[idx]["agent"]},需要安排的干员{name}')
                            raise Exception('检测到安排干员未成功')
                    finished = True
                    # 如果完成则移除该任务
                    del plan[room]
                    if room.startswith('room'):
                        # 截图
                        while self.find('bill_accelerate') is None:
                            if error_count > 5:
                                raise Exception('未成功进入订单界面')
                            self.tap((self.recog.w * 0.05, self.recog.h * 0.95), interval=1)
                            error_count += 1
                        修正后的接单时间 = self.double_read_time(
                            (int(self.recog.w * 650 / 2496), int(self.recog.h * 660 / 1404),
                             int(self.recog.w * 815 / 2496), int(self.recog.h * 710 / 1404)),
                            use_digit_reader=True)
                        logger.warning('房间 B' + room[5] + '0' + room[7] +
                                       ' 修正后的接单时间为 ' + 修正后的接单时间.strftime("%H:%M:%S"))
                        截图等待时间 = (修正后的接单时间 - datetime.now()).total_seconds()
                        if (截图等待时间 > 0) and (截图等待时间 < 1000):
                            logger.info("等待截图时间为 " + str(截图等待时间) + ' 秒')
                            time.sleep(截图等待时间)
                        self.recog.save_screencap('run_order')
                    while self.scene() == 9:
                        time.sleep(1)
                except Exception as e:
                    logger.exception(e)
                    choose_error += 1
                    self.recog.update()
                    back_count = 0
                    while self.get_infra_scene() != 201:
                        self.back()
                        self.recog.update()
                        back_count += 1
                        if back_count > 3:
                            raise e
                    if choose_error > 3:
                        raise e
                    else:
                        continue
            self.back(0.5)
        if len(new_plan) == 1 and self.任务列表[0].type.startswith('room'):
            # 防止由于意外导致的死循环
            run_order_room = next(iter(new_plan))
            if '但书' in new_plan[run_order_room] or '龙舌兰' in new_plan[run_order_room]:
                new_plan[run_order_room] = [data["agent"] for data in self.plan[room]]
            # 返回基建主界面
            self.recog.update()
            self.返回基主界面()
            self.任务列表.append(SchedulerTask(time=self.任务列表[0].time, task_plan=new_plan))
            if 龙舌兰和但书休息:
                宿舍 = {}
                宿舍[龙舌兰和但书休息宿舍] = [data["agent"] for data in self.plan[龙舌兰和但书休息宿舍]]
                self.任务列表.append(SchedulerTask(time=self.任务列表[0].time, task_plan=宿舍))
                self.skip(['planned', 'todo_task'])

    def skip(self, task_names='All'):
        if task_names == 'All':
            task_names = ['planned', 'collect_notification', 'todo_task']
        if 'planned' in task_names:
            self.planned = True
        if 'todo_task':
            self.todo_task = True
        if 'collect_notification':
            self.collect_notification = True

    @Asst.CallBackType
    def log_maa(msg, details, arg):
        m = Message(msg)
        d = json.loads(details.decode('utf-8'))
        logger.debug(d)
        logger.debug(m)
        logger.debug(arg)

    def MAA初始化(self):
        # 若需要获取详细执行信息，请传入 callback 参数
        # 例如 asst = Asst(callback=my_callback)
        Asst.load(path=self.MAA设置['MAA路径'])
        self.MAA = Asst(callback=self.log_maa)
        self.关卡列表 = []
        # self.MAA.set_instance_option(2, 'maatouch')
        # 请自行配置 adb 环境变量，或修改为 adb 可执行程序的路径
        # logger.info(self.device.client.device_id)
        if self.MAA.connect(self.MAA设置['MAA_adb路径'], self.device.client.device_id):
            logger.info("MAA 连接成功")
        else:
            logger.info("MAA 连接失败")
            raise Exception("MAA 连接失败")

    def append_maa_task(self, type):
        if type in ['StartUp', 'Visit', 'Award']:
            self.MAA.append_task(type)
        elif type == 'Fight':
            _plan = self.MAA设置['周计划'][get_server_weekday()]
            logger.info(f"现在服务器是{_plan['日子']}")
            for stage in _plan["关卡"]:
                logger.info(f"添加关卡:{stage}")
                self.MAA.append_task('Fight', {
                    # 空值表示上一次
                    # 'stage': '',
                    'stage': stage,
                    '应急理智药': _plan["应急理智药"],
                    'stone': 0,
                    'times': 999,
                    'report_to_penguin': True,
                    'client_type': '',
                    'penguin_id': '',
                    'DrGrandet': False,
                    'server': 'CN',
                    'expiring_medicine': 9999
                })
                self.关卡列表.append(stage)
        # elif type == 'Recruit':
        #     self.MAA.append_task('Recruit', {
        #         'select': [4],
        #         'confirm': [3, 4],
        #         'times': 4,
        #         'refresh': True,
        #         "recruitment_time": {
        #             "3": 460,
        #             "4": 540
        #         }
        #     })
        # elif type == 'Mall':
        #     credit_fight = False
        #     if len(self.关卡列表) > 0 and self.关卡列表[- 1] != '':
        #         credit_fight = True
        #     self.MAA.append_task('Mall', {
        #         'shopping': True,
        #         'buy_first': ['招聘许可'],
        #         'blacklist': ['家具', '碳', '加急许可'],
        #         'credit_fight': credit_fight
        #     })

    # def maa_plan_solver(self, 任务列表='All', one_time=False):
    def maa_plan_solver(self, 任务列表=['Fight'], one_time=False):
        try:
            self.send_email('启动MAA')
            self.back_to_index()
            # 任务及参数请参考 docs/集成文档.md
            self.MAA初始化()
            if 任务列表 == 'All':
                任务列表 = ['StartUp', 'Fight', 'Recruit', 'Visit', 'Mall', 'Award']
            for maa_task in 任务列表:
                self.append_maa_task(maa_task)
            # asst.append_task('Copilot', {
            #     'stage_name': '千层蛋糕',
            #     'filename': './GA-EX8-raid.json',
            #     'formation': False

            # })
            self.MAA.start()
            stop_time = None
            if one_time:
                stop_time = datetime.now() + timedelta(minutes=5)
            logger.info(f"MAA 启动")
            hard_stop = False
            while self.MAA.running():
                # 单次任务默认5分钟
                if one_time and stop_time < datetime.now():
                    self.MAA.stop()
                    hard_stop = True
                # 5分钟之前就停止
                elif not one_time and (self.任务列表[0].time - datetime.now()).total_seconds() < 300:
                    self.MAA.stop()
                    hard_stop = True
                else:
                    time.sleep(0)
            self.send_email('MAA停止')
            if hard_stop:
                logger.info(f"由于MAA任务并未完成，等待3分钟重启软件")
                time.sleep(180)
                self.device.exit(self.服务器)
            elif not one_time:
                logger.info(f"记录MAA 本次执行时间")
                self.MAA设置['上一次作战'] = datetime.now()
                logger.info(self.MAA设置['上一次作战'])
            if self.MAA设置['集成战略'] or self.MAA设置['生息演算'] or self.MAA设置['保全派驻']:
                while (self.任务列表[0].time - datetime.now()).total_seconds() > 30:
                    self.MAA = None
                    self.MAA初始化()
                    if self.MAA设置['集成战略']:
                        self.MAA.append_task('Roguelike', {
                            'mode': 1,
                            'starts_count': 9999999,
                            'investment_enabled': True,
                            'investments_count': 9999999,
                            'stop_when_investment_full': False,
                            'squad': '后勤分队',
                            'roles': '取长补短',
                            'theme': 'Sami',
                            'core_char': ''
                        })
                    elif self.MAA设置['生息演算']:
                        self.back_to_MAA设置['生息演算']()
                        self.MAA.append_task('ReclamationAlgorithm')
                    # elif self.MAA设置['保全派驻'] :
                    #     self.MAA.append_task('SSSCopilot', {
                    #         'filename': "F:\\MAA-v4.10.5-win-x64\\resource\\copilot\\SSS_阿卡胡拉丛林.json",
                    #         'formation': False,
                    #         'loop_times':99
                    #     })
                    self.MAA.start()
                    while self.MAA.running():
                        if (self.任务列表[0].time - datetime.now()).total_seconds() < 30:
                            self.MAA.stop()
                            break
                        else:
                            time.sleep(0)
                    self.device.exit(self.服务器)
            # 生息演算逻辑 结束
            if one_time:
                if len(self.任务列表) > 0:
                    del self.任务列表[0]
                self.MAA = None
                if self.find_next_task(datetime.now() + timedelta(seconds=900)) is None:
                    # 未来10分钟没有任务就新建
                    self.任务列表.append(SchedulerTask())
                return
            remaining_time = (self.任务列表[0].time - datetime.now()).total_seconds()
            subject = f"开始休息 {'%.2f' % (remaining_time / 60)} 分钟，到{self.任务列表[0].time.strftime('%H:%M:%S')}"
            context = f"下一次任务:{self.任务列表[0].plan if len(self.任务列表[0].plan) != 0 else '空任务' if self.任务列表[0].type == '' else self.任务列表[0].type}"
            logger.info(context)
            logger.info(subject)
            self.send_email(context, subject)
            if remaining_time > 0:
                time.sleep(remaining_time)
            self.MAA = None
        except Exception as e:
            logger.error(e)
            self.MAA = None
            remaining_time = (self.任务列表[0].time - datetime.now()).total_seconds()
            if remaining_time > 0:
                logger.info(
                    f"开始休息 {'%.2f' % (remaining_time / 60)} 分钟，到{self.任务列表[0].time.strftime('%H:%M:%S')}")
                time.sleep(remaining_time)
            self.device.exit(self.服务器)

    def send_email(self, context=None, subject='', retry_time=3):
        global 任务
        if '邮件提醒' in self.邮件设置.keys() and self.邮件设置['邮件提醒'] == 0:
            logger.info('邮件功能未开启')
            return
        while retry_time > 0:
            try:
                msg = MIMEMultipart()
                if context is None:
                    context = """
                    <html>
                        <body>
                        <table border="1">
                        <tr><th>时间</th><th>房间</th></tr>                    
                    """
                    for 任务 in self.任务列表:
                        context += f"""<tr><td>{任务.time.strftime('%Y-%m-%d %H:%M:%S')}</td>
                                            <td>{任务.type}</td></tr>    
                                    """
                    context += "</table></body></html>"
                    msg.attach(MIMEText(context, 'html'))
                else:
                    msg.attach(MIMEText(str(context), 'plain', 'utf-8'))
                msg['Subject'] = ('将在 ' + self.任务列表[0].time.strftime('%H:%M:%S') +
                                  ' 于房间 B' + self.任务列表[0].type[5] + '0' + self.任务列表[0].type[7] + ' 进行跑单')
                msg['From'] = self.邮件设置['发信邮箱']
                s = smtplib.SMTP_SSL("smtp.qq.com", 465, timeout=10.0)
                # 登录邮箱
                s.login(self.邮件设置['发信邮箱'], self.邮件设置['授权码'])
                # 开始发送
                s.sendmail(self.邮件设置['发信邮箱'], self.邮件设置['收件人邮箱'], msg.as_string())
                break
            except Exception as e:
                logger.error("邮件发送失败")
                logger.exception(e)
                retry_time -= 1
                time.sleep(1)


```

This is a Python implementation of a role-based game scheduler. It can handle various aspects of the game, such as connecting to a MAA, sending messages to other players, and managing certain game features.

It first sets up a database for storing game information, such as the current base, the task list, and the room settings. It then initializes the scheduler, which is responsible for managing the game loop and handling user input.

The scheduler then handles the game loop by constantly updating the game state and handling user input. It can also send messages to other players, set their agent to a specific player, and send a notification of a task that is about to be run.

Finally, the scheduler can send a notification of an error, a runout of tasks, or a failure when a task cannot be completed. It can also enable debug mode, which will send debug messages to log.


```
def 初始化(任务列表, scheduler=None):
    config.ADB_DEVICE = MAA设置['MAA_adb地址']
    config.ADB_CONNECT = MAA设置['MAA_adb地址']
    config.APPNAME = 服务器
    config.TAP_TO_LAUNCH = [{"enable": "false", "x": "0", "y": "0"}]
    init_fhlr()
    device = 设备控制()
    cli = Solver(device)
    if scheduler is None:
        当前项目 = 项目经理(cli.device, cli.recog)
        当前项目.服务器 = 服务器
        当前项目.operators = {}
        当前项目.plan = {}
        当前项目.current_base = {}
        for 房间 in 跑单位置设置:
            当前项目.plan['room_' + 房间[1] + '_' + 房间[3]] = []
            for 干员 in 跑单位置设置[房间]:
                当前项目.plan['room_' + 房间[1] + '_' + 房间[3]].append(
                    {'agent': '', 'group': '', 'replacement': [干员]})
        if 龙舌兰和但书休息:
            global 龙舌兰和但书休息宿舍
            for 宿舍 in 宿舍设置:
                if 宿舍 == 'B401':
                    龙舌兰和但书休息宿舍 = 'dormitory_4'
                else:
                    龙舌兰和但书休息宿舍 = 'dormitory_' + 宿舍[1]
                当前项目.plan[龙舌兰和但书休息宿舍] = []
                for 干员 in 宿舍设置[宿舍]:
                    if 干员 == '当前休息干员':  干员 = 'Current'
                    当前项目.plan[龙舌兰和但书休息宿舍].append({'agent': 干员, 'group': '', 'replacement': ''})
        当前项目.任务列表 = 任务列表
        当前项目.last_room = ''
        当前项目.MAA = None
        当前项目.邮件设置 = 邮件设置
        当前项目.ADB_CONNECT = config.ADB_CONNECT[0]
        当前项目.MAA设置 = MAA设置
        当前项目.error = False
        当前项目.跑单提前运行时间 = 跑单提前运行时间
        当前项目.更换干员前缓冲时间 = 更换干员前缓冲时间
        return 当前项目
    else:
        scheduler.device = cli.device
        scheduler.recog = cli.recog
        scheduler.handle_error(True)
        return scheduler


```

这是一段 Python 语言的代码，定义了一个名为 `main` 的函数，是该程序的核心部分。主要实现了 Strength 爬虫的 MAA 集成和自动化运行。

首先，定义了 MAA 的一些参数设置，然后判断是否启用了生息演算，如果不是，则不进行 MAA 集成。接着，定义了一个任务间隔的判断条件，如果满足条件，则启动 MAA 并调用 `current_project.maa_plan_solver()` 方法进行任务规划。

如果任务间隔超过 10 分钟，则 MAA 启动后自动调用 `current_project.device.exit()` 方法退出游戏，并等待任务结束后退出游戏的时间，如果这个时间大于跑单提前运行时间，则关闭弹窗提醒并重置 MAA。

如果跑单设置已开启，且当前任务结束后存在 Mower 跑单，则关闭弹窗提醒并等待 Mower 跑单完成。如果跑单设置已开启，但是当前任务结束后不存在 Mower 跑单，则等待 MAA 调用任务规划后，当前项目将重新进入索引。

如果当前任务结束后 Mower 仍然存在，但是 MAA 没有启动，则当前项目将重新进入索引。如果当前任务结束后 Mower 仍然存在，但是 MAA 已经启动，则 MAA 调用任务规划后，当前项目将重新进入索引。

此外，定义了一个重连次数的判断条件，如果重连次数未达到最大值，则继续尝试连接服务器。如果连接出现异常，则重连次数会增加。如果已经达到最大重连次数，则强制关闭连接并抛出异常。

最后，如果当前任务结束后 Mower 仍然存在，但是 MAA 已经启动，则 MAA 调用任务规划后，当前项目将重新进入索引。


```
class 线程(threading.Thread):

    def __init__(self, *args, **kwargs):
        super(线程, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def run(self):
        global ope_list, 当前项目, 任务提示, 下个任务开始时间
        # 第一次执行任务
        任务列表 = []
        for t in 任务列表:
            t.time = datetime.strptime(str(t.time), '%Y-%m-%d %H:%M:%S.%f')
        reconnect_max_tries = 10
        reconnect_tries = 0
        当前项目 = 初始化(任务列表)
        当前项目.device.launch(f"{服务器}/{config.APP_ACTIVITY_NAME}")
        当前项目.initialize_operators()
        while True:
            try:
                if len(当前项目.任务列表) > 0:
                    当前项目.任务列表.sort(key=lambda x: x.time, reverse=False)    # 任务按时间排序
                    # 如果订单间的时间差距小，无人机加速拉开订单间的时间差距
                    if (len(任务列表) > 1 and (任务列表[0].time - datetime.now()).total_seconds()
                            > 当前项目.跑单提前运行时间 > (任务列表[1].time - 任务列表[0].time).total_seconds()):
                        logger.warning("无人机加速拉开订单间的时间差距")
                        当前项目.无人机加速(任务列表[0].type, True, True)
                    下个任务开始时间 = 任务列表[0].time
                    当前项目.recog.update()
                    当前项目.返回基主界面()
                    任务间隔 = (当前项目.任务列表[0].time - datetime.now()).total_seconds()
                    if (当前项目.任务列表[0].time - datetime.now()).total_seconds() > 0:
                        当前项目.send_email()
                        任务提示 = str()
                        for i in range(len(任务列表)):
                            任务提示 += ('房间 B' + 任务列表[i].type[5] + '0' + 任务列表[i].type[7]
                                         + ' 开始跑单的时间为 ' + 任务列表[i].time.strftime("%H:%M:%S") + '\n')
                        if 跑单弹窗提醒开关:    托盘.notify(任务提示, "Mower跑单提醒")
                        托盘.notify(任务提示, "Mower跑单提醒")
                        for i in range(len(任务列表)):
                            logger.warning('房间 B' + 任务列表[i].type[5] + '0' + 任务列表[i].type[7] +
                                           ' 开始跑单的时间为 ' + 任务列表[i].time.strftime("%H:%M:%S"))

                    # 如果有高强度重复MAA任务,任务间隔超过10分钟则启动MAA
                    if (MAA设置['集成战略'] or MAA设置['生息演算']) and (任务间隔 > 600):
                        当前项目.maa_plan_solver()
                    elif 任务间隔 > 0:
                        if 任务结束后退出游戏 and 任务间隔 > 跑单提前运行时间:
                            当前项目.device.exit(当前项目.服务器)
                            time.sleep(任务间隔 - 跑单提前运行时间)
                            if 跑单弹窗提醒开关:
                                托盘.notify("跑单时间快到了喔，请放下游戏中正在做的事，或者手动关闭Mower", "Mower跑单提醒")
                            time.sleep(跑单提前运行时间)
                            if 跑单弹窗提醒开关:    托盘.notify("开始跑单！", "Mower跑单提醒")
                        else:
                            time.sleep(任务间隔)
                            当前项目.back_to_index()

                if len(当前项目.任务列表) > 0 and 当前项目.任务列表[0].type.split('_')[0] == 'maa':
                    当前项目.maa_plan_solver((当前项目.任务列表[0].type.split('_')[1]).split(','), one_time=True)
                    continue
                当前项目.run()
                reconnect_tries = 0
            except ConnectionError as e:
                reconnect_tries += 1
                if reconnect_tries < reconnect_max_tries:
                    logger.warning(f'连接端口断开...正在重连...')
                    connected = False
                    while not connected:
                        try:
                            当前项目 = 初始化([], 当前项目)
                            break
                        except Exception as ce:
                            logger.error(ce)
                            time.sleep(1)
                            continue
                    continue
                else:
                    raise Exception(e)
            except Exception as E:
                logger.exception(f"程序出错--->{E}")


```

这段代码是一个 Python 函数，名为“终止线程报错”，它的参数包括两个整数类型的变量：tid 和 exctype。

该函数的主要作用是报告由于线程终止导致的错误，并在需要时进行清理。

具体来说，函数首先获取线程 ID，然后检查要报告的异常类型是否为可迭代类型。如果不是，函数将异常类型转换为对象类型。

接下来，函数使用 ctypes.pythonapi.PyThreadState_SetAsyncExc 函数设置线程的异常类型。如果设置成功，函数将返回 0；如果设置失败，函数将抛出一个 ValueError。

如果设置成功，函数将执行清理操作，并尝试再次设置线程的异常类型。如果清洁过程中出现错误，函数将抛出一个 SystemError。

总之，该函数的主要目的是在需要时报告线程终止导致的错误，并在尝试重新设置线程异常类型时进行清理。


```
def 终止线程报错(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


```

这段代码定义了三个函数，用于显示窗口中的字幕、选中窗口以及拖动窗口。

第一个函数 `显示字幕()` 的作用是恢复窗口的原始大小和位置，并使窗口重新显示包含显示的文本。

第二个函数 `选中窗口(event)` 获取当前鼠标在窗口内的位置，并将这些位置记录在两个全局变量 `鼠标水平初始位置` 和 `鼠标竖直初始位置` 中。然后，它通过 `geometry` 方法调整窗口的大小和位置，使得窗口能够左上角和右下角拖动。

第三个函数 `拖动窗口(event)` 接受一个 `event` 参数，其中包括 `event.x_root` 和 `event.y_root` 两个坐标值，这些值表示鼠标相对于窗口左上角边的 X 和 Y 坐标。函数通过 `geometry` 方法调整窗口的位置，将窗口从左上角拖动到指定位置。


```
def 显示字幕():
    窗口.deiconify()


def 选中窗口(event):
    global 鼠标水平初始位置, 鼠标竖直初始位置

    鼠标水平初始位置 = event.x  # 获取鼠标相对于窗体左上角的X坐标
    鼠标竖直初始位置 = event.y  # 获取鼠标相对于窗左上角体的Y坐标


def 拖动窗口(event):
    窗口.geometry(f'+{event.x_root - 鼠标水平初始位置}+{event.y_root - 鼠标竖直初始位置}')


```

以上代码定义了两个函数，其中第一个函数名为“关闭窗口”(CloseWindow)，第二个函数名为“缩放字幕”(ZoomSubtitle)。

“关闭窗口”函数的作用是关闭操作系统中的窗口。具体来说，它使用PyStray库中的pystray.Icon参数作为窗口的图标，然后使用withdraw()方法从窗口中移除该图标，使得窗口不再可见。

“缩放字幕”函数的作用是根据用户与字幕窗口交互的时间间隔来控制字幕的字号。具体来说，它使用event参数中的delta值来计算用户与窗口之间的距离，然后根据这个距离变化来调整字幕的字号。如果用户与窗口的交互时间间隔大于0，那么字幕字号会增加1；如果交互时间间隔小于0，那么字号会减少1。此外，函数还设置了一个默认值，即当拼音字号小于1时将字号设置为1，当拼音字号大于90时将字号设置为90。


```
def 关闭窗口(icon: pystray.Icon):
    窗口.withdraw()


def 缩放字幕(event):
    global 字幕字号
    if event.delta > 0:
        字幕字号 += 1
    else:
        字幕字号 -= 1
    if 字幕字号 < 1:
        字幕字号 = 1
    elif 字幕字号 > 90:
        字幕字号 = 90


```

这段代码定义了两个函数，其中第一个函数名为 `跑单任务查询`，第二个函数名为 `重新运行Mower`。

第一个函数 `跑单任务查询`，接收一个参数 `icon`，该参数是一个名为 `pystray.Icon` 的类实例，用于在界面上显示跑单任务列表。该函数的作用是通过调用实例的 `notify` 方法，向用户提供一个提示消息，表示有新的跑单任务可用。

第二个函数 `重新运行Mower`，与第一个函数的作用相反，该函数的作用是重新启动名为 `Mower` 的机器人，并让它开始运行。它使用了一个名为 `Mower` 的全局变量，该变量在代码的其他部分中被定义为来自 `Mower.ident` 的线程实例。函数尝试通过调用 `Mower._stop_event.set` 方法来停止运行中的机器人，并处理可能的异常。如果这个方法不能停止机器人，则函数会尝试通过调用 `Mower.start` 方法来启动机器人。


```
def 跑单任务查询(icon: pystray.Icon):
    icon.notify(任务提示, "Mower跑单任务列表")


def 重新运行Mower():
    global Mower
    try:
        Mower._stop_event.set()
        终止线程报错(Mower.ident, SystemExit)
    except:
        pass
    Mower = 线程()
    Mower.start()


```

这段代码是一个Python定义，定义了两个函数：停止运行Mower方法和退出程序函数。

停止运行Mower函数首先设置Mower对象的一个名为“_stop_event”的事件为真，然后调用了一个名为“终止线程报错”的函数，并传递了Mower对象的“ident”和“SystemExit”参数。接下来，代码通过“logger.info”函数输出了一句话“Mower已停止”。

退出程序函数接收两个参数：一个“icon”对象和一个“item”参数。首先，调用“stop”方法来停止Mower对象的运行。然后，获取当前进程ID，并使用“taskkill”命令来杀死后台进程。不过，这个命令可能无法杀死所有进程，因为有些程序会隐藏其进程。


```
def 停止运行Mower():
    Mower._stop_event.set()
    终止线程报错(Mower.ident, SystemExit)
    logger.info('Mower已停止')


def 退出程序(icon, item):
    icon.stop()  # 对象停止方法
    pid = os.getpid()  # 获取当前进程ID
    try:  # 杀掉后台进程
        if 悬浮字幕开关:  窗口.destroy()
        os.system('taskkill -f -pid %s' % pid)
    except:
        pass


```

这段代码定义了一个函数，名为“更新字幕”。在这个函数中，使用了Python中的一些全局变量，例如“datetime.now()”来获取当前的日期和时间，以及“global 字幕”变量来存储一个全局变量“字幕”。

接下来，函数首先计算出“下个任务开始时间”和当前时间的差值，然后将其除以60，得到一个表示秒数的整数。接着，将“Mower的回合！”作为全局变量“字幕”来存储。

接下来，判断“任务倒计时”是否大于等于0。如果是，将“Mower将在” + str(任务倒计时) + “分钟后开始跑单”作为全局变量“字幕”来存储。如果是，将“跑单即将开始！”作为全局变量“字幕”来存储。最后，如果“任务倒计时”小于等于5，将“跑单即将开始！”作为全局变量“字幕”来存储。

然后，函数使用“label.config(text=字幕， font=(字幕字体 + ' ' + 字幕字号), bg=字幕颜色， fg=处分颜色[:6] + str(int(字幕颜色[5] == '0'))）”设置了一个标签的显示文本、字体、背景和前景颜色，并在文本上添加了一个“？”符号。最后，函数使用了“window.after(100, 更新字幕)”来定期更新标签的文本，即每100毫秒更新一次标签的文本。


```
def 更新字幕():
    global 字幕
    任务倒计时 = int((下个任务开始时间 - datetime.now()).total_seconds() / 60)
    字幕 = 'Mower的回合！'
    if 任务倒计时 >= 0:
        字幕 = 'Mower将在' + str(任务倒计时) + '分钟后开始跑单'
        if 任务倒计时 <= 5:
            字幕 += '\n跑单即将开始！'
    label.config(text=字幕, font=(字幕字体 + ' ' + 字幕字号), bg=字幕颜色,
                 fg=字幕颜色[:6] + str(int(字幕颜色[5] == '0')))
    窗口.after(100, 更新字幕)


托盘菜单 = (MenuItem(任务提示, 跑单任务查询, default=True, visible=False),
            MenuItem('显示字幕', 显示字幕, visible=悬浮字幕开关),
            MenuItem('重新运行Mower', 重新运行Mower, visible=True),
            MenuItem('停止运行Mower', 停止运行Mower, visible=True),
            Menu.SEPARATOR, MenuItem('退出', 退出程序))
```

这段代码是一个Python程序，用于创建一个窗口并添加一个悬浮的标签小部件，用于表示窗口的标签。

首先，创建一个托盘(Icon)并将其设置为窗口的图标，同时从文件中加载并显示一个logo.png图像。然后，创建一个名为"Mower 纯跑单"的菜单并将它设置为托盘菜单项。

接下来，添加一个悬浮的标签小部件，设置标签在窗口的顶部，宽度为窗口宽度的一半，高度为窗口高度的75%，同时使用窗口的属性来设置标签的颜色。最后，添加三个事件处理程序，用于绑定窗口的关闭、拖动和鼠标旋转事件。


```
托盘 = pystray.Icon("Mower 纯跑单", Image.open("logo.png"), "Mower 纯跑单", 托盘菜单)
if 悬浮字幕开关:
    窗口.geometry("%dx%d+%d+%d" % (窗口宽度, 窗口高度,
                                   (窗口.winfo_screenwidth() - 窗口宽度) / 2,
                                   窗口.winfo_screenheight() * 3 / 4 - 窗口高度/ 2))
    窗口.overrideredirect(True)
    窗口.title("窗口")
    窗口.attributes("-topmost", 1)
    窗口.wm_attributes("-transparentcolor", 字幕颜色)

    # 添加一个标签小部件
    label = Label(窗口)
    label.pack(side="top", fill="both", expand=True)
    label.bind("<Button-1>", 选中窗口)
    label.bind("<B1-Motion>", 拖动窗口)
    label.bind("<Double-Button-1>", 关闭窗口)
    label.bind("<MouseWheel>", 缩放字幕)

```

这段代码是一个Python程序，其主要作用是创建并启动一个名为“日志设置”的交互式窗口。该窗口包含一个名为“悬浮字幕开关”的复选按钮，一个运行“run”函数的线程，以及一个名为“Mower”的线程。

具体来说，代码的作用如下：

1. 创建并启动一个名为“日志设置”的窗口。
2. 创建一个名为“Mower”的线程，并启动该线程。
3. 开启悬浮字幕开关。
4. 在窗口中显示悬浮的字幕内容，并在屏幕上显示更新字幕的进度。
5. 启动线程“run”。

“Mower”线程的行为是通过调用“run”函数来实现的，该函数的具体实现不在代码中给出。


```
if __name__ == "__main__":
    日志设置()
    threading.Thread(target=托盘.run, daemon=False).start()
    Mower = 线程()
    Mower.start()
    if 悬浮字幕开关:
        窗口.after(100, 更新字幕)
        窗口.mainloop()

```