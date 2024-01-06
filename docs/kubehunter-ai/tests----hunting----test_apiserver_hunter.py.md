# `kubehunter\tests\hunting\test_apiserver_hunter.py`

```
# 导入 requests_mock 模块，用于模拟 HTTP 请求
import requests_mock
# 导入 time 模块，用于处理时间相关的操作

# 导入 kube_hunter.modules.hunting.apiserver 模块中的相关类和函数
from kube_hunter.modules.hunting.apiserver import (
    AccessApiServer,  # 访问 API 服务器
    AccessApiServerWithToken,  # 使用令牌访问 API 服务器
    ServerApiAccess,  # 服务器 API 访问
    AccessApiServerActive,  # 主动访问 API 服务器
)

# 导入 kube_hunter.modules.hunting.apiserver 模块中的相关类和函数
from kube_hunter.modules.hunting.apiserver import (
    ListNamespaces,  # 列出命名空间
    ListPodsAndNamespaces,  # 列出 Pod 和命名空间
    ListRoles,  # 列出角色
    ListClusterRoles,  # 列出集群角色
)

# 导入 kube_hunter.modules.hunting.apiserver 模块中的相关类和函数
from kube_hunter.modules.hunting.apiserver import ApiServerPassiveHunterFinished  # API 服务器被动猎手完成
from kube_hunter.modules.hunting.apiserver import CreateANamespace, DeleteANamespace  # 创建/删除命名空间

# 导入 kube_hunter.modules.discovery.apiserver 模块中的 ApiServer 类
from kube_hunter.modules.discovery.apiserver import ApiServer  # API 服务器发现

# 导入 kube_hunter.core.types 模块中的相关类
from kube_hunter.core.types import UnauthenticatedAccess, InformationDisclosure  # 未经身份验证的访问、信息泄露

# 导入 kube_hunter.core.events 模块中的 handler 函数
from kube_hunter.core.events import handler  # 事件处理程序
# 初始化一个全局计数器
counter = 0

# 测试 ApiServerToken 函数
def test_ApiServerToken():
    # 使用全局计数器
    global counter
    # 重置计数器
    counter = 0

    # 创建 ApiServer 对象
    e = ApiServer()
    # 设置 ApiServer 对象的主机地址
    e.host = "1.2.3.4"
    # 设置 ApiServer 对象的认证令牌
    e.auth_token = "my-secret-token"

    # 测试事件通过令牌传递给访问 ApiServer 的函数
    h = AccessApiServerWithToken(e)
    # 断言事件的认证令牌与设置的认证令牌相同
    assert h.event.auth_token == "my-secret-token"

    # 这个测试不会生成任何事件
    time.sleep(0.01)
    # 断言计数器为 0
    assert counter == 0
# 定义一个测试访问 API 服务器的函数
def test_AccessApiServer():
    # 声明全局变量 counter
    global counter
    counter = 0

    # 创建一个 ApiServer 对象
    e = ApiServer()
    # 设置 ApiServer 对象的主机名、端口和协议
    e.host = "mockKubernetes"
    e.port = 443
    e.protocol = "https"

    # 使用 requests_mock 创建一个 HTTP 请求的模拟器
    with requests_mock.Mocker() as m:
        # 模拟一个 GET 请求，返回空字符串
        m.get("https://mockKubernetes:443/api", text="{}")
        # 模拟一个 GET 请求，返回包含一个名为 "hello" 的项目的 JSON 字符串
        m.get(
            "https://mockKubernetes:443/api/v1/namespaces", text='{"items":[{"metadata":{"name":"hello"}}]}',
        )
        # 模拟一个 GET 请求，返回包含两个名为 "podA" 和 "podB" 的项目的 JSON 字符串
        m.get(
            "https://mockKubernetes:443/api/v1/pods",
            text='{"items":[{"metadata":{"name":"podA", "namespace":"namespaceA"}}, \
                            {"metadata":{"name":"podB", "namespace":"namespaceB"}}]}',
        )
# 发送 GET 请求获取指定 URL 的资源，返回状态码为 403
m.get(
    "https://mockkubernetes:443/apis/rbac.authorization.k8s.io/v1/roles", status_code=403,
)
# 发送 GET 请求获取指定 URL 的资源，返回内容为一个空数组
m.get(
    "https://mockkubernetes:443/apis/rbac.authorization.k8s.io/v1/clusterroles", text='{"items":[]}',
)
# 发送 GET 请求获取指定 URL 的资源，返回内容为 Kubernetes 版本信息
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

# 等待 0.01 秒
# 我们应该看到关于服务器 API 访问、命名空间、Pods 的事件，以及被动猎手的完成
time.sleep(0.01)
    # 断言计数器的值是否等于4
    assert counter == 4

    # 使用带有身份验证令牌的请求模拟器
    counter = 0
    with requests_mock.Mocker() as m:
        # 发送获取API的请求，并返回空的JSON对象
        m.get("https://mocktoken:443/api", text="{}")
        # 发送获取命名空间的请求，并返回包含一个名为"hello"的项目的JSON对象
        m.get(
            "https://mocktoken:443/api/v1/namespaces", text='{"items":[{"metadata":{"name":"hello"}}]}',
        )
        # 发送获取Pods的请求，并返回包含两个名为"podA"和"podB"的项目的JSON对象
        m.get(
            "https://mocktoken:443/api/v1/pods",
            text='{"items":[{"metadata":{"name":"podA", "namespace":"namespaceA"}}, \
                            {"metadata":{"name":"podB", "namespace":"namespaceB"}}]}',
        )
        # 发送获取角色的请求，并返回403状态码
        m.get(
            "https://mocktoken:443/apis/rbac.authorization.k8s.io/v1/roles", status_code=403,
        )
        # 发送获取集群角色的请求
        m.get(
            "https://mocktoken:443/apis/rbac.authorization.k8s.io/v1/clusterroles",
# 设置一个 JSON 格式的文本作为参数传递给 AccessApiServerWithToken 方法
text='{"items":[{"metadata":{"name":"my-role"}}]}',
)

# 设置访问令牌
e.auth_token = "so-secret"
# 设置主机地址
e.host = "mocktoken"
# 创建 AccessApiServerWithToken 对象
h = AccessApiServerWithToken(e)
# 执行对象的方法
h.execute()

# 等待一段时间
time.sleep(0.01)
# 断言计数器的值为 5
assert counter == 5

# 订阅 ListNamespaces 事件
@handler.subscribe(ListNamespaces)
class test_ListNamespaces(object):
    # 初始化方法，接收事件作为参数
    def __init__(self, event):
        # 打印事件名称
        print("ListNamespaces")
        # 断言事件的证据为 ["hello"]
        assert event.evidence == ["hello"]
        # 如果事件的主机地址为 "mocktoken"，则断言事件的访问令牌为 "so-secret"
        if event.host == "mocktoken":
            assert event.auth_token == "so-secret"
        else:
            # 如果条件不成立，即 event.auth_token 为 None，则触发断言错误
            assert event.auth_token is None
        # 声明全局变量 counter，并自增1
        global counter
        counter += 1


@handler.subscribe(ListPodsAndNamespaces)
class test_ListPodsAndNamespaces(object):
    def __init__(self, event):
        # 初始化方法，打印输出 "ListPodsAndNamespaces"
        print("ListPodsAndNamespaces")
        # 断言 event.evidence 的长度为2
        assert len(event.evidence) == 2
        # 遍历 event.evidence 中的每个 pod
        for pod in event.evidence:
            # 如果 pod 的名称为 "podA"，则断言其命名空间为 "namespaceA"
            if pod["name"] == "podA":
                assert pod["namespace"] == "namespaceA"
            # 如果 pod 的名称为 "podB"，则断言其命名空间为 "namespaceB"
            if pod["name"] == "podB":
                assert pod["namespace"] == "namespaceB"
        # 如果 event.host 为 "mocktoken"，则进行以下断言
        if event.host == "mocktoken":
            assert event.auth_token == "so-secret"
            assert "token" in event.name
            assert "anon" not in event.name
# 如果条件不成立，即事件的 auth_token 为 None，事件的名称中不包含 "token"，但包含 "anon"，则断言失败
        else:
            assert event.auth_token is None
            assert "token" not in event.name
            assert "anon" in event.name
        # 声明全局变量 counter，并自增1
        global counter
        counter += 1


# 由于测试中的 API 调用返回 403 状态码，不应该看到这个事件
@handler.subscribe(ListRoles)
class test_ListRoles(object):
    def __init__(self, event):
        print("ListRoles")
        # 断言失败
        assert 0
        # 声明全局变量 counter，并自增1
        global counter
        counter += 1


# 只有在有令牌的情况下才会看到这个事件，因为在没有令牌的测试中，API 调用返回一个空的项目列表
# 订阅 ListClusterRoles 事件，并定义处理函数 test_ListClusterRoles
@handler.subscribe(ListClusterRoles)
class test_ListClusterRoles(object):
    # 初始化函数，打印事件名称，并断言事件的认证令牌为 "so-secret"
    def __init__(self, event):
        print("ListClusterRoles")
        assert event.auth_token == "so-secret"
        # 使用全局变量 counter 计数加一
        global counter
        counter += 1

# 订阅 ServerApiAccess 事件，并定义处理函数 test_ServerApiAccess
@handler.subscribe(ServerApiAccess)
class test_ServerApiAccess(object):
    # 初始化函数，打印事件名称，并根据事件的类别进行断言
    def __init__(self, event):
        print("ServerApiAccess")
        # 如果事件类别为 UnauthenticatedAccess，则断言认证令牌为 None
        if event.category == UnauthenticatedAccess:
            assert event.auth_token is None
        # 如果事件类别为 InformationDisclosure，则断言认证令牌为 "so-secret"
        else:
            assert event.category == InformationDisclosure
            assert event.auth_token == "so-secret"
        # 使用全局变量 counter 计数加一
        global counter
        counter += 1
# 订阅 ApiServerPassiveHunterFinished 事件
@handler.subscribe(ApiServerPassiveHunterFinished)
class test_PassiveHunterFinished(object):
    # 初始化方法
    def __init__(self, event):
        # 打印信息
        print("PassiveHunterFinished")
        # 断言事件的命名空间为 ["hello"]
        assert event.namespaces == ["hello"]
        # 声明全局变量 counter
        global counter
        # 增加计数器
        counter += 1

# 测试访问 ApiServerActive
def test_AccessApiServerActive():
    # 创建 ApiServerPassiveHunterFinished 事件
    e = ApiServerPassiveHunterFinished(namespaces=["hello-namespace"])
    e.host = "mockKubernetes"
    e.port = 443
    e.protocol = "https"

    # 使用 requests_mock 创建一个模拟器
    with requests_mock.Mocker() as m:
        # TODO 更多的测试用例，使用真实的响应
        # 模拟 POST 请求
        m.post(
# 发送一个 HTTP GET 请求到指定的 URL，获取命名空间的信息
requests.get(
    "https://mockKubernetes:443/api/v1/namespaces",
    # 传递一个 JSON 格式的字符串作为请求体
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
"""
)
# 发送 POST 请求，创建集群角色
m.post("https://mockKubernetes:443/api/v1/clusterroles", text="{}")

# 发送 POST 请求，创建集群角色
m.post("https://mockkubernetes:443/apis/rbac.authorization.k8s.io/v1/clusterroles", text="{}")

# 发送 POST 请求，创建指定命名空间下的 Pod
m.post("https://mockkubernetes:443/api/v1/namespaces/hello-namespace/pods", text="{}")

# 发送 POST 请求，创建指定命名空间下的角色
m.post("https://mockkubernetes:443" "/apis/rbac.authorization.k8s.io/v1/namespaces/hello-namespace/roles", text="{}")

# 发送 DELETE 请求，删除指定命名空间
m.delete("https://mockKubernetes:443/api/v1/namespaces/abcde", text="""
{
  "kind": "Namespace",
# 定义 API 版本
"apiVersion": "v1",
# 元数据信息
"metadata": {
    # 对象名称
    "name": "abcde",
    # 对象自身链接
    "selfLink": "/api/v1/namespaces/abcde",
    # 对象唯一标识符
    "uid": "4a7aa47c-39ba-11e9-ab46-08002781145e",
    # 资源版本
    "resourceVersion": "694780",
    # 创建时间戳
    "creationTimestamp": "2019-02-26T11:33:08Z",
    # 删除时间戳
    "deletionTimestamp": "2019-02-26T11:40:58Z"
  },
  # 对象规范
  "spec": {
    # 最终处理器
    "finalizers": [
      "kubernetes"
    ]
  },
  # 对象状态
  "status": {
    # 状态阶段
    "phase": "Terminating"
  }
}
# 创建一个 AccessApiServerActive 对象，并执行其方法
h = AccessApiServerActive(e)
h.execute()

# 当接收到 CreateANamespace 事件时，执行 test_CreateANamespace 类
@handler.subscribe(CreateANamespace)
class test_CreateANamespace(object):
    # 初始化方法，检查事件的证据中是否包含 "abcde"
    def __init__(self, event):
        assert "abcde" in event.evidence

# 当接收到 DeleteANamespace 事件时，执行 test_DeleteANamespace 类
@handler.subscribe(DeleteANamespace)
class test_DeleteANamespace(object):
    # 初始化方法，检查事件的证据中是否包含 "2019-02-26"
    def __init__(self, event):
        assert "2019-02-26" in event.evidence
```