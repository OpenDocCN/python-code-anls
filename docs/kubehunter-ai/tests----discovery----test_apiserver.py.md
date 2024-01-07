# `.\kubehunter\tests\discovery\test_apiserver.py`

```

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
    # 在函数内部声明counter为全局变量
    global counter
    # 重置counter为0
    counter = 0
    # 使用requests_mock创建一个Mocker对象，并使用with语句进行上下文管理
    with requests_mock.Mocker() as m:
        # 对"https://mockOther:443"发起GET请求，返回文本"elephant"
        m.get("https://mockOther:443", text="elephant")
        # 对"https://mockKubernetes:443"发起GET请求，返回文本'{"code":403}'，状态码为403
        m.get("https://mockKubernetes:443", text='{"code":403}', status_code=403)
        # 对"https://mockKubernetes:443/version"发起GET请求，返回文本'{"major": "1.14.10"}'，状态码为200
        m.get(
            "https://mockKubernetes:443/version", text='{"major": "1.14.10"}', status_code=200,
        )

        # 创建一个Event对象e，并设置其protocol为"https"，port为443，host为"mockOther"
        e = Event()
        e.protocol = "https"
        e.port = 443
        e.host = "mockOther"

        # 创建一个ApiServiceDiscovery对象a，并执行其execute方法
        a = ApiServiceDiscovery(e)
        a.execute()

        # 修改Event对象e的host为"mockKubernetes"，并再次执行ApiServiceDiscovery对象a的execute方法
        e.host = "mockKubernetes"
        a.execute()

    # 等待一段时间以便事件被处理，只有对mockKubernetes的请求应该触发一个事件
    time.sleep(1)
    # 断言counter的值为1
    assert counter == 1

# 定义测试函数test_ApiServerWithServiceAccountToken
def test_ApiServerWithServiceAccountToken():
    # 在函数内部声明counter为全局变量
    global counter
    # 重置counter为0
    counter = 0
    # 使用requests_mock创建一个Mocker对象，并使用with语句进行上下文管理
    with requests_mock.Mocker() as m:
        # 对"https://mockKubernetes:443"发起GET请求，请求头包含Authorization字段，返回文本'{"code":200}'
        m.get(
            "https://mockKubernetes:443", request_headers={"Authorization": "Bearer very_secret"}, text='{"code":200}',
        )
        # 对"https://mockKubernetes:443"发起GET请求，返回文本'{"code":403}'，状态码为403
        m.get("https://mockKubernetes:443", text='{"code":403}', status_code=403)
        # 对"https://mockKubernetes:443/version"发起GET请求，返回文本'{"major": "1.14.10"}'，状态码为200
        m.get(
            "https://mockKubernetes:443/version", text='{"major": "1.14.10"}', status_code=200,
        )
        # 对"https://mockOther:443"发起GET请求，返回文本"elephant"
        m.get("https://mockOther:443", text="elephant")

        # 创建一个Event对象e，并设置其protocol为"https"，port为443
        e = Event()
        e.protocol = "https"
        e.port = 443

        # 无论是否有token，我们都应该发现一个API Server
        # 设置Event对象e的host为"mockKubernetes"，创建一个ApiServiceDiscovery对象a，并执行其execute方法
        e.host = "mockKubernetes"
        a = ApiServiceDiscovery(e)
        a.execute()
        # 等待一段时间
        time.sleep(0.1)
        # 断言counter的值为1
        assert counter == 1

        # 设置Event对象e的auth_token为"very_secret"，创建一个ApiServiceDiscovery对象a，并执行其execute方法
        e.auth_token = "very_secret"
        a = ApiServiceDiscovery(e)
        a.execute()
        # 等待一段时间
        time.sleep(0.1)
        # 断言counter的值为2
        assert counter == 2

        # 但是如果我们没有看到错误代码或在/version中找到'major'，我们不应该生成事件
        # 设置Event对象e的host为"mockOther"，创建一个ApiServiceDiscovery对象a，并执行其execute方法
        e.host = "mockOther"
        a = ApiServiceDiscovery(e)
        a.execute()
        # 等待一段时间
        time.sleep(0.1)
        # 断言counter的值为2
        assert counter == 2

# 定义测试函数test_InsecureApiServer
def test_InsecureApiServer():
    # 在函数内部声明counter为全局变量
    global counter
    # 重置counter为0
    counter = 0
    # 使用requests_mock创建一个Mocker对象，并使用with语句进行上下文管理
    with requests_mock.Mocker() as m:
        # 对"http://mockOther:8080"发起GET请求，返回文本"elephant"
        m.get("http://mockOther:8080", text="elephant")
        # 对"http://mockKubernetes:8080"发起GET请求，返回指定的JSON文本
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
        # 对"http://mockKubernetes:8080/version"发起GET请求，返回文本'{"major": "1.14.10"}'
        m.get("http://mockKubernetes:8080/version", text='{"major": "1.14.10"}')
        # 对"http://mockOther:8080/version"发起GET请求，返回状态码404
        m.get("http://mockOther:8080/version", status_code=404)

        # 创建一个Event对象e，并设置其protocol为"http"，port为8080，host为"mockOther"
        e = Event()
        e.protocol = "http"
        e.port = 8080
        e.host = "mockOther"

        # 创建一个ApiServiceDiscovery对象a，并执行其execute方法
        a = ApiServiceDiscovery(e)
        a.execute()

        # 修改Event对象e的host为"mockKubernetes"，并再次执行ApiServiceDiscovery对象a的execute方法
        e.host = "mockKubernetes"
        a.execute()

    # 等待一段时间以便事件被处理，只有对mockKubernetes的请求应该触发一个事件
    time.sleep(0.1)
    # 断言counter的值为1

# 使用handler.subscribe装饰器订阅ApiServer事件
@handler.subscribe(ApiServer)
# 定义testApiServer类
class testApiServer(object):
    # 初始化方法，接受一个event参数
    def __init__(self, event):
        # 打印"Event"
        print("Event")
        # 断言event的host属性为"mockKubernetes"
        assert event.host == "mockKubernetes"
        # 全局变量counter加1
        global counter
        counter += 1

```