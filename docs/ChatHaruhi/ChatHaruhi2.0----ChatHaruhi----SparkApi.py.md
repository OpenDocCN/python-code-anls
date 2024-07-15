# `.\Chat-Haruhi-Suzumiya\ChatHaruhi2.0\ChatHaruhi\SparkApi.py`

```py
# 导入必要的模块和库
import _thread as thread  # 使用_thread模块来进行多线程操作
import base64  # base64编解码模块
import datetime  # 处理日期和时间的模块
import hashlib  # 提供常见的哈希算法，如SHA-1、SHA-224、SHA-256
import hmac  # 提供消息认证码算法的模块
import json  # 处理JSON数据的模块
from urllib.parse import urlparse, urlencode  # 处理URL的解析和编码模块
import ssl  # 提供SSL通信的支持
from datetime import datetime  # 导入datetime模块中的datetime类
from time import mktime  # 将时间转换为UNIX时间戳
from urllib.parse import urlencode  # 编码URL参数
from wsgiref.handlers import format_date_time  # 格式化日期时间

import websocket  # 导入websocket库，用于WebSocket通信

answer = ""  # 初始化一个空字符串用于存储答案

class Ws_Param(object):
    # 初始化WebSocket参数类
    def __init__(self, APPID, APIKey, APISecret, Spark_url):
        self.APPID = APPID  # 设置APPID
        self.APIKey = APIKey  # 设置APIKey
        self.APISecret = APISecret  # 设置APISecret
        self.host = urlparse(Spark_url).netloc  # 解析出WebSocket URL的主机部分
        self.path = urlparse(Spark_url).path  # 解析出WebSocket URL的路径部分
        self.Spark_url = Spark_url  # 存储WebSocket的URL

    # 生成包含鉴权信息的WebSocket连接URL
    def create_url(self):
        # 获取当前时间并生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 构建待签名的原始字符串
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        # 使用HMAC-SHA256算法对原始字符串进行签名
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        # 构建授权信息的原始字符串
        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 将鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        # 拼接WebSocket连接的URL
        url = self.Spark_url + '?' + urlencode(v)
        # 返回构建好的WebSocket连接URL
        return url

# 处理WebSocket错误信息的回调函数
def on_error(ws, error):
    print("### error:", error)

# 处理WebSocket关闭事件的回调函数
def on_close(ws,one,two):
    print(" ")

# 处理WebSocket连接建立事件的回调函数
def on_open(ws):
    # 在新线程中运行WebSocket消息处理函数
    thread.start_new_thread(run, (ws,))

# 在新线程中运行的WebSocket消息处理函数
def run(ws, *args):
    # 生成参数并将其转换为JSON格式发送给WebSocket服务器
    data = json.dumps(gen_params(appid=ws.appid, domain= ws.domain,question=ws.question))
    ws.send(data)

# 处理WebSocket接收到消息事件的回调函数
def on_message(ws, message):
    # 解析接收到的JSON消息
    data = json.loads(message)
    code = data['header']['code']
    if code != 0:
        # 打印错误信息并关闭WebSocket连接
        print(f'请求错误: {code}, {data}')
        ws.close()
    else:
        # 获取选择项和状态信息，并将答案内容追加到全局变量中
        choices = data["payload"]["choices"]
        status = choices["status"]
        content = choices["text"][0]["content"]
        global answer
        answer += content
        # 如果状态为2，即处理完成，关闭WebSocket连接
        if status == 2:
            ws.close()

# 生成请求参数的函数，通过APPID和用户提问生成参数
def gen_params(appid, domain,question):
    """
    通过appid和用户的提问来生成请参数
    """
    # 构建一个包含请求数据的字典
    data = {
        "header": {
            "app_id": appid,  # 设置应用程序 ID
            "uid": "1234"     # 设置用户 ID
        },
        "parameter": {
            "chat": {
                "domain": domain,                 # 设置聊天领域
                "random_threshold": 0.5,          # 设置随机阈值
                "max_tokens": 2048,               # 设置最大生成标记数
                "auditing": "default"             # 设置审计策略为默认
            }
        },
        "payload": {
            "message": {
                "text": question  # 将问题文本作为消息的文本字段
            }
        }
    }
    # 返回构建的请求数据字典
    return data
# 主程序入口函数，接受五个参数：应用ID，API密钥，API密钥密钥，Spark URL，域名和问题
def main(appid, api_key, api_secret, Spark_url, domain, question):
    # 创建一个 WebSocket 参数对象，用于配置 WebSocket 连接所需的参数
    wsParam = Ws_Param(appid, api_key, api_secret, Spark_url)
    # 启用 WebSocket 的跟踪信息（关闭为 False）
    websocket.enableTrace(False)
    # 根据给定的 WebSocket URL 创建 WebSocketApp 对象，设置回调函数
    wsUrl = wsParam.create_url()
    ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
    # 将应用ID、问题和域名附加到 WebSocket 对象上，以备后续处理
    ws.appid = appid
    ws.question = question
    ws.domain = domain
    # 运行 WebSocket 连接，设置 SSL 参数以忽略证书请求
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
```