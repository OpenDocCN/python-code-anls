# `kubehunter\tests\discovery\test_apiserver.py`

```py
# 导入requests_mock和time模块
import requests_mock
import time

# 从kube_hunter.modules.discovery.apiserver模块中导入ApiServer和ApiServiceDiscovery类
from kube_hunter.modules.discovery.apiserver import ApiServer, ApiServiceDiscovery
# 从kube_hunter.core.events.types模块中导入Event类
from kube_hunter.core.events.types import Event
# 从kube_hunter.core.events模块中导入handler函数
from kube_hunter.core.events import handler

# 定义全局变量counter并初始化为0
counter = 0

# 定义测试函数test_ApiServer
def test_ApiServer():
    # 使用全局变量counter
    global counter
    # 将counter重置为0
    counter = 0
    # 使用requests_mock模拟HTTP请求
    with requests_mock.Mocker() as m:
        # 模拟GET请求返回文本"elephant"
        m.get("https://mockOther:443", text="elephant")
        # 模拟GET请求返回状态码403和JSON格式的数据'{"code":403}'
        m.get("https://mockKubernetes:443", text='{"code":403}', status_code=403)
        # 模拟GET请求返回状态码200和JSON格式的数据'{"major": "1.14.10"}'
        m.get(
            "https://mockKubernetes:443/version", text='{"major": "1.14.10"}', status_code=200,
        )

        # 创建Event对象e，并设置protocol、port和host属性
        e = Event()
        e.protocol = "https"
        e.port = 443
        e.host = "mockOther"

        # 创建ApiServiceDiscovery对象a，并执行execute方法
        a = ApiServiceDiscovery(e)
        a.execute()

        # 修改Event对象e的host属性
        e.host = "mockKubernetes"
        # 再次执行ApiServiceDiscovery对象a的execute方法
        a.execute()

    # 等待1秒，以便事件被处理。只有对mockKubernetes的请求应该触发一个事件
    time.sleep(1)
    # 断言counter的值为1
    assert counter == 1

# 定义测试函数test_ApiServerWithServiceAccountToken
def test_ApiServerWithServiceAccountToken():
    # 使用全局变量counter
    global counter
    # 将counter重置为0
    counter = 0
    # 使用 requests_mock 创建一个模拟器对象，并进入上下文管理器
    with requests_mock.Mocker() as m:
        # 模拟发送带有特定请求头的 GET 请求，并返回指定的文本内容和状态码
        m.get(
            "https://mockKubernetes:443", request_headers={"Authorization": "Bearer very_secret"}, text='{"code":200}',
        )
        # 模拟发送 GET 请求，并返回指定的文本内容和状态码
        m.get("https://mockKubernetes:443", text='{"code":403}', status_code=403)
        # 模拟发送 GET 请求，并返回指定的文本内容和状态码
        m.get(
            "https://mockKubernetes:443/version", text='{"major": "1.14.10"}', status_code=200,
        )
        # 模拟发送 GET 请求，并返回指定的文本内容
        m.get("https://mockOther:443", text="elephant")

        # 创建一个事件对象
        e = Event()
        # 设置事件对象的协议和端口
        e.protocol = "https"
        e.port = 443

        # 设置事件对象的主机名，创建一个 ApiServiceDiscovery 对象并执行
        e.host = "mockKubernetes"
        a = ApiServiceDiscovery(e)
        a.execute()
        # 等待 0.1 秒
        time.sleep(0.1)
        # 断言计数器的值为 1
        assert counter == 1

        # 设置事件对象的认证令牌，创建一个 ApiServiceDiscovery 对象并执行
        e.auth_token = "very_secret"
        a = ApiServiceDiscovery(e)
        a.execute()
        # 等待 0.1 秒
        time.sleep(0.1)
        # 断言计数器的值为 2
        assert counter == 2

        # 设置事件对象的主机名为 "mockOther"，创建一个 ApiServiceDiscovery 对象并执行
        e.host = "mockOther"
        a = ApiServiceDiscovery(e)
        a.execute()
        # 等待 0.1 秒
        time.sleep(0.1)
        # 断言计数器的值为 2
        assert counter == 2
# 定义一个名为 test_InsecureApiServer 的函数
def test_InsecureApiServer():
    # 声明 counter 变量为全局变量，并初始化为 0
    global counter
    counter = 0
    # 使用 requests_mock 创建一个 Mock 对象，并命名为 m
    with requests_mock.Mocker() as m:
        # 模拟对 "http://mockOther:8080" 的 GET 请求，返回文本 "elephant"
        m.get("http://mockOther:8080", text="elephant")
        # 模拟对 "http://mockKubernetes:8080" 的 GET 请求，返回指定的 JSON 文本
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
        # 模拟对 "http://mockKubernetes:8080/version" 的 GET 请求，返回指定的 JSON 文本
        m.get("http://mockKubernetes:8080/version", text='{"major": "1.14.10"}')
        # 模拟对 "http://mockOther:8080/version" 的 GET 请求，返回状态码 404
        m.get("http://mockOther:8080/version", status_code=404)
        # 创建一个 Event 对象
        e = Event()
        # 设置 Event 对象的 protocol 属性为 "http"
        e.protocol = "http"
        # 设置 Event 对象的 port 属性为 8080
        e.port = 8080
        # 设置 Event 对象的 host 属性为 "mockOther"
        e.host = "mockOther"
        # 创建一个 ApiServiceDiscovery 对象，并传入 Event 对象，执行相关操作
        a = ApiServiceDiscovery(e)
        a.execute()
        # 修改 Event 对象的 host 属性为 "mockKubernetes"
        e.host = "mockKubernetes"
        # 再次执行 ApiServiceDiscovery 对象的相关操作
        a.execute()
    # 等待一段时间，以便事件被处理。只有针对 mockKubernetes 的事件应该触发一个事件
    time.sleep(0.1)
    # 断言 counter 的值为 1
    assert counter == 1

# 使用 handler.subscribe 装饰器，订阅 ApiServer 事件，并定义一个名为 testApiServer 的类
@handler.subscribe(ApiServer)
class testApiServer(object):
    # 初始化方法，接收一个 event 参数
    def __init__(self, event):
        # 打印 "Event" 字符串
        print("Event")
        # 断言 event 的 host 属性为 "mockKubernetes"
        assert event.host == "mockKubernetes"
        # 声明 counter 变量为全局变量，并增加 1
        global counter
        counter += 1
```