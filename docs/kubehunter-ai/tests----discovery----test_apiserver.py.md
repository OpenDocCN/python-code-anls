# `kubehunter\tests\discovery\test_apiserver.py`

```
# 导入 requests_mock 和 time 模块
import requests_mock
import time

# 导入 ApiServer 和 ApiServiceDiscovery 模块
from kube_hunter.modules.discovery.apiserver import ApiServer, ApiServiceDiscovery
# 导入 Event 类和 handler 函数
from kube_hunter.core.events.types import Event
from kube_hunter.core.events import handler

# 定义全局变量 counter，并初始化为 0
counter = 0

# 定义测试函数 test_ApiServer
def test_ApiServer():
    # 使用 global 关键字声明在函数内部使用全局变量 counter
    global counter
    # 将 counter 初始化为 0
    counter = 0
    # 使用 requests_mock 模块创建一个 Mock 对象，并使用 with 语句进行上下文管理
    with requests_mock.Mocker() as m:
        # 模拟发送 GET 请求，返回文本 "elephant"
        m.get("https://mockOther:443", text="elephant")
        # 模拟发送 GET 请求，返回 JSON 数据 {"code":403}，状态码为 403
        m.get("https://mockKubernetes:443", text='{"code":403}', status_code=403)
        # 模拟发送 GET 请求，返回 JSON 数据 {"major": "1.14.10"}，状态码为 200
        m.get(
            "https://mockKubernetes:443/version", text='{"major": "1.14.10"}', status_code=200,
        )
# 创建一个事件对象
e = Event()
# 设置事件的协议为 HTTPS
e.protocol = "https"
# 设置事件的端口为 443
e.port = 443
# 设置事件的主机为 mockOther
e.host = "mockOther"

# 创建一个 API 服务发现对象，并执行服务发现
a = ApiServiceDiscovery(e)
a.execute()

# 修改事件的主机为 mockKubernetes，并再次执行服务发现
e.host = "mockKubernetes"
a.execute()

# 等待一段时间，以便事件被处理。只有针对 mockKubernetes 的事件应该触发一个事件
time.sleep(1)
# 断言计数器的值为 1
assert counter == 1

# 测试使用服务账户令牌的 API 服务器
def test_ApiServerWithServiceAccountToken():
    # 设置全局计数器为 0
    global counter
    counter = 0
    # 使用 requests_mock 创建一个模拟器
    with requests_mock.Mocker() as m:
# 发送带有请求头的 GET 请求到指定 URL，返回状态码为 200 的响应
m.get(
    "https://mockKubernetes:443", request_headers={"Authorization": "Bearer very_secret"}, text='{"code":200}',
)
# 发送不带请求头的 GET 请求到指定 URL，返回状态码为 403 的响应
m.get("https://mockKubernetes:443", text='{"code":403}', status_code=403)
# 发送 GET 请求到指定 URL，返回状态码为 200 的响应，响应内容为版本信息
m.get(
    "https://mockKubernetes:443/version", text='{"major": "1.14.10"}', status_code=200,
)
# 发送 GET 请求到指定 URL，返回响应内容为 "elephant"
m.get("https://mockOther:443", text="elephant")

# 创建一个事件对象
e = Event()
# 设置事件对象的协议为 HTTPS
e.protocol = "https"
# 设置事件对象的端口为 443
e.port = 443

# 设置事件对象的主机为 "mockKubernetes"
e.host = "mockKubernetes"
# 创建一个 API 服务发现对象
a = ApiServiceDiscovery(e)
# 执行 API 服务发现
a.execute()
# 等待 0.1 秒
time.sleep(0.1)
# 断言计数器的值为 1
assert counter == 1
        # 设置认证令牌
        e.auth_token = "very_secret"
        # 创建 ApiServiceDiscovery 对象
        a = ApiServiceDiscovery(e)
        # 执行 API 服务发现
        a.execute()
        # 等待0.1秒
        time.sleep(0.1)
        # 断言计数器的值为2
        assert counter == 2

        # 但是如果我们没有看到错误代码或在 /version 中找到 'major'，就不应该生成事件
        # 设置主机名为 "mockOther"
        e.host = "mockOther"
        # 创建 ApiServiceDiscovery 对象
        a = ApiServiceDiscovery(e)
        # 执行 API 服务发现
        a.execute()
        # 等待0.1秒
        time.sleep(0.1)
        # 断言计数器的值为2
        assert counter == 2


def test_InsecureApiServer():
    # 设置全局计数器为0
    global counter
    counter = 0
    # 使用 requests_mock 创建一个模拟器
    with requests_mock.Mocker() as m:
        # 模拟发送 GET 请求，返回文本 "elephant"
        m.get("http://mockOther:8080", text="elephant")
        # 模拟发送 GET 请求
        m.get(
# 使用 requests-mock 库创建一个模拟的 HTTP 请求
m = requests_mock.Mocker()
# 模拟一个 GET 请求，返回指定的 JSON 数据
m.get(
    "http://mockKubernetes:8080",
    text="""{
  "paths": [
    "/api",
    "/api/v1",
    "/apis",
    "/apis/",
    "/apis/admissionregistration.k8s.io",
    "/apis/admissionregistration.k8s.io/v1beta1",
    "/apis/apiextensions.k8s.io"
  ]}""",
)
# 模拟一个 GET 请求，返回指定的 JSON 数据
m.get("http://mockKubernetes:8080/version", text='{"major": "1.14.10"}')
# 模拟一个 GET 请求，返回 404 错误
m.get("http://mockOther:8080/version", status_code=404)

# 创建一个事件对象
e = Event()
# 设置事件对象的协议为 HTTP
e.protocol = "http"
# 设置事件对象的端口号为 8080
e.port = 8080
# 设置事件对象的主机名为 mockOther
e.host = "mockOther"
# 创建一个 ApiServiceDiscovery 对象，并执行其方法
a = ApiServiceDiscovery(e)
a.execute()

# 修改 e 对象的 host 属性为 "mockKubernetes"，然后再次执行 ApiServiceDiscovery 对象的方法
e.host = "mockKubernetes"
a.execute()

# 等待一段时间，让事件被处理。只有 host 为 "mockKubernetes" 的事件才会触发
time.sleep(0.1)
assert counter == 1

# 定义一个 testApiServer 类，订阅 ApiServer 事件
# 当事件发生时，会执行 __init__ 方法
# 打印事件信息，并断言事件的 host 属性为 "mockKubernetes"
# 增加全局计数器 counter 的值
@handler.subscribe(ApiServer)
class testApiServer(object):
    def __init__(self, event):
        print("Event")
        assert event.host == "mockKubernetes"
        global counter
        counter += 1
抱歉，我无法为您提供代码注释，因为您没有提供任何代码。如果您有任何需要帮助的代码，请随时告诉我。我会尽力帮助您。
```