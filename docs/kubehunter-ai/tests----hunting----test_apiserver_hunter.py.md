# `.\kubehunter\tests\hunting\test_apiserver_hunter.py`

```

# 导入requests_mock和time模块
import requests_mock
import time

# 导入kube_hunter.modules.hunting.apiserver模块中的相关类和函数
from kube_hunter.modules.hunting.apiserver import (
    AccessApiServer,
    AccessApiServerWithToken,
    ServerApiAccess,
    AccessApiServerActive,
)
# 导入kube_hunter.modules.hunting.apiserver模块中的相关类和函数
from kube_hunter.modules.hunting.apiserver import (
    ListNamespaces,
    ListPodsAndNamespaces,
    ListRoles,
    ListClusterRoles,
)
# 导入kube_hunter.modules.hunting.apiserver模块中的相关类和函数
from kube_hunter.modules.hunting.apiserver import ApiServerPassiveHunterFinished
# 导入kube_hunter.modules.hunting.apiserver模块中的相关类和函数
from kube_hunter.modules.hunting.apiserver import CreateANamespace, DeleteANamespace
# 导入kube_hunter.modules.discovery.apiserver模块中的相关类和函数
from kube_hunter.modules.discovery.apiserver import ApiServer
# 导入kube_hunter.core.types模块中的相关类
from kube_hunter.core.types import UnauthenticatedAccess, InformationDisclosure
# 导入kube_hunter.core.events模块中的handler函数
from kube_hunter.core.events import handler

# 初始化计数器
counter = 0

# 测试函数，测试访问API服务器的token
def test_ApiServerToken():
    global counter
    counter = 0

    # 创建ApiServer对象
    e = ApiServer()
    e.host = "1.2.3.4"
    e.auth_token = "my-secret-token"

    # 测试事件中是否传递了pod的token
    h = AccessApiServerWithToken(e)
    assert h.event.auth_token == "my-secret-token"

    # 该测试不会生成任何事件
    time.sleep(0.01)
    assert counter == 0

# 测试函数，测试访问API服务器
def test_AccessApiServer():
    global counter
    counter = 0

    # 创建ApiServer对象
    e = ApiServer()
    e.host = "mockKubernetes"
    e.port = 443
    e.protocol = "https"

    # 使用requests_mock模拟API服务器的响应
    with requests_mock.Mocker() as m:
        # 模拟API服务器的各种响应
        # ...

        # 执行访问API服务器的操作
        h = AccessApiServer(e)
        h.execute()

        # 应该会看到Server API Access、Namespaces、Pods和被动猎手完成的事件
        time.sleep(0.01)
        assert counter == 4

    # 使用auth token进行测试
    counter = 0
    with requests_mock.Mocker() as m:
        # 模拟API服务器的各种响应
        # ...

        e.auth_token = "so-secret"
        e.host = "mocktoken"
        h = AccessApiServerWithToken(e)
        h.execute()

        # 应该会看到相同的一组事件，但是还会有Cluster Roles的事件
        time.sleep(0.01)
        assert counter == 5

# 订阅ListNamespaces事件的处理函数
@handler.subscribe(ListNamespaces)
class test_ListNamespaces(object):
    def __init__(self, event):
        # 打印事件名称
        print("ListNamespaces")
        # 断言事件的evidence属性
        assert event.evidence == ["hello"]
        # 断言事件的host和auth_token属性
        # ...

# 订阅ListPodsAndNamespaces事件的处理函数
@handler.subscribe(ListPodsAndNamespaces)
class test_ListPodsAndNamespaces(object):
    def __init__(self, event):
        # 打印事件名称
        print("ListPodsAndNamespaces")
        # 断言事件的evidence属性
        # ...
        # 断言事件的host和auth_token属性
        # ...

# 订阅ListRoles事件的处理函数
@handler.subscribe(ListRoles)
class test_ListRoles(object):
    def __init__(self, event):
        # 打印事件名称
        print("ListRoles")
        # 断言事件
        assert 0
        # ...

# 订阅ListClusterRoles事件的处理函数
@handler.subscribe(ListClusterRoles)
class test_ListClusterRoles(object):
    def __init__(self, event):
        # 打印事件名称
        print("ListClusterRoles")
        # 断言事件的auth_token属性
        # ...

# 订阅ServerApiAccess事件的处理函数
@handler.subscribe(ServerApiAccess)
class test_ServerApiAccess(object):
    def __init__(self, event):
        # 打印事件名称
        print("ServerApiAccess")
        # 断言事件
        # ...

# 订阅ApiServerPassiveHunterFinished事件的处理函数
@handler.subscribe(ApiServerPassiveHunterFinished)
class test_PassiveHunterFinished(object):
    def __init__(self, event):
        # 打印事件名称
        print("PassiveHunterFinished")
        # 断言事件的namespaces属性
        # ...

# 测试函数，测试访问活动的API服务器
def test_AccessApiServerActive():
    e = ApiServerPassiveHunterFinished(namespaces=["hello-namespace"])
    e.host = "mockKubernetes"
    e.port = 443
    e.protocol = "https"

    with requests_mock.Mocker() as m:
        # 模拟API服务器的各种响应
        # ...

        h = AccessApiServerActive(e)
        h.execute()

# 订阅CreateANamespace事件的处理函数
@handler.subscribe(CreateANamespace)
class test_CreateANamespace(object):
    def __init__(self, event):
        # 断言事件的evidence属性
        assert "abcde" in event.evidence

# 订阅DeleteANamespace事件的处理函数
@handler.subscribe(DeleteANamespace)
class test_DeleteANamespace(object):
    def __init__(self, event):
        # 断言事件的evidence属性
        assert "2019-02-26" in event.evidence

```