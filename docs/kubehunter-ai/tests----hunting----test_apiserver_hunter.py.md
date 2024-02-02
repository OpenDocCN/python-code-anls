# `kubehunter\tests\hunting\test_apiserver_hunter.py`

```py
# 导入requests_mock和time模块
import requests_mock
import time

# 从kube_hunter.modules.hunting.apiserver模块中导入多个类
from kube_hunter.modules.hunting.apiserver import (
    AccessApiServer,
    AccessApiServerWithToken,
    ServerApiAccess,
    AccessApiServerActive,
)

# 从kube_hunter.modules.hunting.apiserver模块中导入多个类
from kube_hunter.modules.hunting.apiserver import (
    ListNamespaces,
    ListPodsAndNamespaces,
    ListRoles,
    ListClusterRoles,
)

# 从kube_hunter.modules.hunting.apiserver模块中导入ApiServerPassiveHunterFinished类
from kube_hunter.modules.hunting.apiserver import ApiServerPassiveHunterFinished

# 从kube_hunter.modules.hunting.apiserver模块中导入CreateANamespace和DeleteANamespace类
from kube_hunter.modules.hunting.apiserver import CreateANamespace, DeleteANamespace

# 从kube_hunter.modules.discovery.apiserver模块中导入ApiServer类
from kube_hunter.modules.discovery.apiserver import ApiServer

# 从kube_hunter.core.types模块中导入UnauthenticatedAccess和InformationDisclosure类
from kube_hunter.core.types import UnauthenticatedAccess, InformationDisclosure

# 从kube_hunter.core.events模块中导入handler函数
from kube_hunter.core.events import handler

# 初始化计数器为0
counter = 0

# 定义测试函数test_ApiServerToken
def test_ApiServerToken():
    global counter
    counter = 0

    # 创建ApiServer对象e，设置host和auth_token属性
    e = ApiServer()
    e.host = "1.2.3.4"
    e.auth_token = "my-secret-token"

    # 创建AccessApiServerWithToken对象h，验证事件中的auth_token属性是否等于"my-secret-token"
    h = AccessApiServerWithToken(e)
    assert h.event.auth_token == "my-secret-token"

    # 休眠0.01秒，验证计数器是否为0
    time.sleep(0.01)
    assert counter == 0

# 定义测试函数test_AccessApiServer
def test_AccessApiServer():
    global counter
    counter = 0

    # 创建ApiServer对象e，设置host、port和protocol属性
    e = ApiServer()
    e.host = "mockKubernetes"
    e.port = 443
    e.protocol = "https"
    # 使用 requests_mock 模拟 HTTP 请求
    with requests_mock.Mocker() as m:
        # 模拟 GET 请求返回空对象
        m.get("https://mockKubernetes:443/api", text="{}")
        # 模拟 GET 请求返回包含一个命名空间的对象
        m.get(
            "https://mockKubernetes:443/api/v1/namespaces", text='{"items":[{"metadata":{"name":"hello"}}]}',
        )
        # 模拟 GET 请求返回包含两个 Pod 的对象
        m.get(
            "https://mockKubernetes:443/api/v1/pods",
            text='{"items":[{"metadata":{"name":"podA", "namespace":"namespaceA"}}, \
                            {"metadata":{"name":"podB", "namespace":"namespaceB"}}]}',
        )
        # 模拟 GET 请求返回 403 错误
        m.get(
            "https://mockkubernetes:443/apis/rbac.authorization.k8s.io/v1/roles", status_code=403,
        )
        # 模拟 GET 请求返回空的 ClusterRoles 对象
        m.get(
            "https://mockkubernetes:443/apis/rbac.authorization.k8s.io/v1/clusterroles", text='{"items":[]}',
        )
        # 模拟 GET 请求返回 Kubernetes 版本信息
        m.get(
            "https://mockkubernetes:443/version",
            text='{"major": "1","minor": "13+", "gitVersion": "v1.13.6-gke.13", \
                   "gitCommit": "fcbc1d20b6bca1936c0317743055ac75aef608ce", \
                   "gitTreeState": "clean", "buildDate": "2019-06-19T20:50:07Z", \
                   "goVersion": "go1.11.5b4", "compiler": "gc", \
                   "platform": "linux/amd64"}',
        )
    
        # 创建 AccessApiServer 对象
        h = AccessApiServer(e)
        # 执行 AccessApiServer 对象的方法
        h.execute()
    
        # 等待一小段时间
        time.sleep(0.01)
        # 断言计数器的值为 4
        assert counter == 4
    
    # 重置计数器
    counter = 0
    # 使用 requests_mock 创建一个模拟器，用于模拟 HTTP 请求和响应
    with requests_mock.Mocker() as m:
        # 发送一个 GET 请求，返回空的 JSON 对象
        m.get("https://mocktoken:443/api", text="{}")
        # 发送一个 GET 请求，返回包含一个命名空间的 JSON 对象
        m.get(
            "https://mocktoken:443/api/v1/namespaces", text='{"items":[{"metadata":{"name":"hello"}}]}',
        )
        # 发送一个 GET 请求，返回包含两个 Pod 的 JSON 对象
        m.get(
            "https://mocktoken:443/api/v1/pods",
            text='{"items":[{"metadata":{"name":"podA", "namespace":"namespaceA"}}, \
                            {"metadata":{"name":"podB", "namespace":"namespaceB"}}]}',
        )
        # 发送一个 GET 请求，返回状态码为 403
        m.get(
            "https://mocktoken:443/apis/rbac.authorization.k8s.io/v1/roles", status_code=403,
        )
        # 发送一个 GET 请求，返回包含一个 Cluster Role 的 JSON 对象
        m.get(
            "https://mocktoken:443/apis/rbac.authorization.k8s.io/v1/clusterroles",
            text='{"items":[{"metadata":{"name":"my-role"}}]}',
        )

        # 设置 e 对象的 auth_token 属性为 "so-secret"
        e.auth_token = "so-secret"
        # 设置 e 对象的 host 属性为 "mocktoken"
        e.host = "mocktoken"
        # 创建 AccessApiServerWithToken 对象 h
        h = AccessApiServerWithToken(e)
        # 执行 h 对象的 execute 方法
        h.execute()

        # 等待 0.01 秒
        time.sleep(0.01)
        # 断言 counter 的值为 5
        assert counter == 5
# 订阅 ListNamespaces 事件的处理器
@handler.subscribe(ListNamespaces)
class test_ListNamespaces(object):
    # 初始化方法，打印信息
    def __init__(self, event):
        print("ListNamespaces")
        # 断言事件的证据为 ["hello"]
        assert event.evidence == ["hello"]
        # 如果事件的主机为 "mocktoken"，则断言事件的认证令牌为 "so-secret"
        if event.host == "mocktoken":
            assert event.auth_token == "so-secret"
        else:
            # 否则断言事件的认证令牌为 None
            assert event.auth_token is None
        # 声明全局变量 counter，并自增 1
        global counter
        counter += 1


# 订阅 ListPodsAndNamespaces 事件的处理器
@handler.subscribe(ListPodsAndNamespaces)
class test_ListPodsAndNamespaces(object):
    # 初始化方法，打印信息
    def __init__(self, event):
        print("ListPodsAndNamespaces")
        # 断言事件的证据长度为 2
        assert len(event.evidence) == 2
        # 遍历事件的证据列表
        for pod in event.evidence:
            # 如果证据中的 pod 名称为 "podA"，则断言其所属的命名空间为 "namespaceA"
            if pod["name"] == "podA":
                assert pod["namespace"] == "namespaceA"
            # 如果证据中的 pod 名称为 "podB"，则断言其所属的命名空间为 "namespaceB"
            if pod["name"] == "podB":
                assert pod["namespace"] == "namespaceB"
        # 如果事件的主机为 "mocktoken"
        if event.host == "mocktoken":
            # 断言事件的认证令牌为 "so-secret"
            assert event.auth_token == "so-secret"
            # 断言事件的名称中包含 "token"，且不包含 "anon"
            assert "token" in event.name
            assert "anon" not in event.name
        else:
            # 否则断言事件的认证令牌为 None
            assert event.auth_token is None
            # 断言事件的名称中不包含 "token"，且包含 "anon"
            assert "token" not in event.name
            assert "anon" in event.name
        # 声明全局变量 counter，并自增 1
        global counter
        counter += 1


# 不应该看到这个，因为测试中的 API 调用返回 403 状态码
@handler.subscribe(ListRoles)
class test_ListRoles(object):
    # 初始化方法，打印信息，然后断言 0
    def __init__(self, event):
        print("ListRoles")
        assert 0
        # 声明全局变量 counter，并自增 1
        global counter
        counter += 1


# 只有在有令牌时才能看到这个，因为测试中的 API 调用返回一个空的项目列表
# 在没有令牌的测试中
@handler.subscribe(ListClusterRoles)
class test_ListClusterRoles(object):
    # 初始化方法，打印信息，然后断言事件的认证令牌为 "so-secret"
    def __init__(self, event):
        print("ListClusterRoles")
        assert event.auth_token == "so-secret"
        # 声明全局变量 counter，并自增 1
        global counter
        counter += 1


# 订阅 ServerApiAccess 事件的处理器
@handler.subscribe(ServerApiAccess)
class test_ServerApiAccess(object):
    # 初始化方法，接受一个事件对象作为参数
    def __init__(self, event):
        # 打印信息，表示服务器 API 访问
        print("ServerApiAccess")
        # 如果事件类别为未经身份验证的访问
        if event.category == UnauthenticatedAccess:
            # 断言事件的身份验证令牌为空
            assert event.auth_token is None
        # 如果事件类别为信息泄露
        else:
            # 断言事件的类别为信息泄露
            assert event.category == InformationDisclosure
            # 断言事件的身份验证令牌为 "so-secret"
            assert event.auth_token == "so-secret"
        # 声明全局变量 counter，并增加其值
        global counter
        counter += 1
# 订阅 ApiServerPassiveHunterFinished 事件的处理器类
@handler.subscribe(ApiServerPassiveHunterFinished)
class test_PassiveHunterFinished(object):
    # 初始化方法，打印信息，断言事件的命名空间为 ["hello"]，并使用全局变量 counter
    def __init__(self, event):
        print("PassiveHunterFinished")
        assert event.namespaces == ["hello"]
        global counter
        counter += 1


# 测试访问 ApiServerActive 的方法
def test_AccessApiServerActive():
    # 创建 ApiServerPassiveHunterFinished 事件对象，并设置相关属性
    e = ApiServerPassiveHunterFinished(namespaces=["hello-namespace"])
    e.host = "mockKubernetes"
    e.port = 443
    e.protocol = "https"

    # 使用 requests_mock 创建一个 Mock 对象，并进行接口测试
    with requests_mock.Mocker() as m:
        # 发送 POST 请求，模拟返回的 JSON 数据
        m.post(
            "https://mockKubernetes:443/api/v1/namespaces",
            text="""
{
  "kind": "Namespace",
  "apiVersion": "v1",
  "metadata": {
    "name": "abcde",
    "selfLink": "/api/v1/namespaces/abcde",
    "uid": "4a7aa47c-39ba-11e9-ab46-08002781145e",
    "resourceVersion": "694180",
    "creationTimestamp": "2019-02-26T11:33:08Z"
  },
  "spec": {
    "finalizers": [
      "kubernetes"
    ]
  },
  "status": {
    "phase": "Active"
  }
}
""",
        )
        # 发送 POST 请求，模拟返回空 JSON 数据
        m.post("https://mockKubernetes:443/api/v1/clusterroles", text="{}")
        # 发送 POST 请求，模拟返回空 JSON 数据
        m.post(
            "https://mockkubernetes:443/apis/rbac.authorization.k8s.io/v1/clusterroles", text="{}",
        )
        # 发送 POST 请求，模拟返回空 JSON 数据
        m.post(
            "https://mockkubernetes:443/api/v1/namespaces/hello-namespace/pods", text="{}",
        )
        # 发送 POST 请求，模拟返回空 JSON 数据
        m.post(
            "https://mockkubernetes:443" "/apis/rbac.authorization.k8s.io/v1/namespaces/hello-namespace/roles",
            text="{}",
        )

        # 发送 DELETE 请求，模拟返回的 JSON 数据
        m.delete(
            "https://mockKubernetes:443/api/v1/namespaces/abcde",
            text="""
{
  "kind": "Namespace",
  "apiVersion": "v1",
  "metadata": {
    "name": "abcde",
    "selfLink": "/api/v1/namespaces/abcde",
    "uid": "4a7aa47c-39ba-11e9-ab46-08002781145e",
    "resourceVersion": "694780",
    "creationTimestamp": "2019-02-26T11:33:08Z",
    "deletionTimestamp": "2019-02-26T11:40:58Z"
  },
  "spec": {
    "finalizers": [
      "kubernetes"
    ]
  },
  "status": {
    # 定义一个键为"phase"，值为"Terminating"的字典
    "phase": "Terminating"
  }
# 定义一个类，用于处理 AccessApiServerActive 事件
class AccessApiServerActive(object):
    # 初始化方法，接收一个事件对象作为参数
    def __init__(self, event):
        # 断言事件对象的 evidence 属性中包含 "abcde" 字符串
        assert "abcde" in event.evidence

# 定义一个类，用于处理 CreateANamespace 事件
@handler.subscribe(CreateANamespace)
class test_CreateANamespace(object):
    # 初始化方法，接收一个事件对象作为参数
    def __init__(self, event):
        # 断言事件对象的 evidence 属性中包含 "abcde" 字符串
        assert "abcde" in event.evidence

# 定义一个类，用于处理 DeleteANamespace 事件
@handler.subscribe(DeleteANamespace)
class test_DeleteANamespace(object):
    # 初始化方法，接收一个事件对象作为参数
    def __init__(self, event):
        # 断言事件对象的 evidence 属性中包含 "2019-02-26" 字符串
        assert "2019-02-26" in event.evidence
```