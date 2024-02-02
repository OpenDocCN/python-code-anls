# `kubehunter\kube_hunter\modules\hunting\apiserver.py`

```py
# 导入日志、JSON、请求和唯一标识模块
import logging
import json
import requests
import uuid

# 从 kube_hunter.conf 模块中导入配置
from kube_hunter.conf import config
# 从 kube_hunter.modules.discovery.apiserver 模块中导入 ApiServer 类
from kube_hunter.modules.discovery.apiserver import ApiServer
# 从 kube_hunter.core.events 模块中导入 handler 函数
from kube_hunter.core.events import handler
# 从 kube_hunter.core.events.types 模块中导入 Vulnerability、Event 和 K8sVersionDisclosure 类
from kube_hunter.core.events.types import Vulnerability, Event, K8sVersionDisclosure
# 从 kube_hunter.core.types 模块中导入 Hunter、ActiveHunter 和 KubernetesCluster 类
from kube_hunter.core.types import Hunter, ActiveHunter, KubernetesCluster
# 从 kube_hunter.core.types 模块中导入 AccessRisk、InformationDisclosure 和 UnauthenticatedAccess 类
from kube_hunter.core.types import AccessRisk, InformationDisclosure, UnauthenticatedAccess

# 获取 logger 对象
logger = logging.getLogger(__name__)

# 定义 ServerApiAccess 类，继承自 Vulnerability 和 Event 类
class ServerApiAccess(Vulnerability, Event):
    """The API Server port is accessible.
    Depending on your RBAC settings this could expose access to or control of your cluster."""

    # 初始化方法
    def __init__(self, evidence, using_token):
        # 如果使用 token，则设置名称和类别
        if using_token:
            name = "Access to API using service account token"
            category = InformationDisclosure
        # 否则设置名称和类别
        else:
            name = "Unauthenticated access to API"
            category = UnauthenticatedAccess
        # 调用父类的初始化方法
        Vulnerability.__init__(
            self, KubernetesCluster, name=name, category=category, vid="KHV005",
        )
        # 设置 evidence 属性
        self.evidence = evidence

# 定义 ServerApiHTTPAccess 类，继承自 Vulnerability 和 Event 类
class ServerApiHTTPAccess(Vulnerability, Event):
    """The API Server port is accessible over HTTP, and therefore unencrypted.
    Depending on your RBAC settings this could expose access to or control of your cluster."""

    # 初始化方法
    def __init__(self, evidence):
        # 设置名称和类别
        name = "Insecure (HTTP) access to API"
        category = UnauthenticatedAccess
        # 调用父类的初始化方法
        Vulnerability.__init__(
            self, KubernetesCluster, name=name, category=category, vid="KHV006",
        )
        # 设置 evidence 属性
        self.evidence = evidence

# 定义 ApiInfoDisclosure 类，继承自 Vulnerability 和 Event 类
    # 初始化方法，接受证据、使用令牌标志和名称作为参数
    def __init__(self, evidence, using_token, name):
        # 如果使用令牌，则在名称后添加相应的描述
        if using_token:
            name += " using service account token"
        # 否则，在名称后添加匿名用户的描述
        else:
            name += " as anonymous user"
        # 调用父类Vulnerability的初始化方法，传入KubernetesCluster、名称、类别和漏洞ID作为参数
        Vulnerability.__init__(
            self, KubernetesCluster, name=name, category=InformationDisclosure, vid="KHV007",
        )
        # 设置对象的证据属性
        self.evidence = evidence
# 定义一个类ListPodsAndNamespaces，继承自ApiInfoDisclosure类，用于访问pods可能给攻击者提供有价值的信息
class ListPodsAndNamespaces(ApiInfoDisclosure):
    """ Accessing pods might give an attacker valuable information"""

    # 初始化方法，接收evidence和using_token参数，调用父类的初始化方法，并传入特定的信息
    def __init__(self, evidence, using_token):
        ApiInfoDisclosure.__init__(self, evidence, using_token, "Listing pods")

# 定义一个类ListNamespaces，继承自ApiInfoDisclosure类，用于访问namespaces可能给攻击者提供有价值的信息
class ListNamespaces(ApiInfoDisclosure):
    """ Accessing namespaces might give an attacker valuable information """

    # 初始化方法，接收evidence和using_token参数，调用父类的初始化方法，并传入特定的信息
    def __init__(self, evidence, using_token):
        ApiInfoDisclosure.__init__(self, evidence, using_token, "Listing namespaces")

# 定义一个类ListRoles，继承自ApiInfoDisclosure类，用于访问roles可能给攻击者提供有价值的信息
class ListRoles(ApiInfoDisclosure):
    """ Accessing roles might give an attacker valuable information """

    # 初始化方法，接收evidence和using_token参数，调用父类的初始化方法，并传入特定的信息
    def __init__(self, evidence, using_token):
        ApiInfoDisclosure.__init__(self, evidence, using_token, "Listing roles")

# 定义一个类ListClusterRoles，继承自ApiInfoDisclosure类，用于访问cluster roles可能给攻击者提供有价值的信息
class ListClusterRoles(ApiInfoDisclosure):
    """ Accessing cluster roles might give an attacker valuable information """

    # 初始化方法，接收evidence和using_token参数，调用父类的初始化方法，并传入特定的信息
    def __init__(self, evidence, using_token):
        ApiInfoDisclosure.__init__(self, evidence, using_token, "Listing cluster roles")

# 定义一个类CreateANamespace，继承自Vulnerability和Event类，用于创建一个namespace可能给攻击者提供有价值的信息
class CreateANamespace(Vulnerability, Event):

    """ Creating a namespace might give an attacker an area with default (exploitable) permissions to run pods in.
    """

    # 初始化方法，接收evidence参数，调用父类的初始化方法，并传入特定的信息
    def __init__(self, evidence):
        Vulnerability.__init__(
            self, KubernetesCluster, name="Created a namespace", category=AccessRisk,
        )
        self.evidence = evidence

# 定义一个类DeleteANamespace，继承自Vulnerability和Event类，用于删除一个namespace可能给攻击者提供有价值的信息
class DeleteANamespace(Vulnerability, Event):

    """ Deleting a namespace might give an attacker the option to affect application behavior """

    # 初始化方法，接收evidence参数，调用父类的初始化方法，并传入特定的信息
    def __init__(self, evidence):
        Vulnerability.__init__(
            self, KubernetesCluster, name="Delete a namespace", category=AccessRisk,
        )
        self.evidence = evidence

# 定义一个类CreateARole，继承自Vulnerability和Event类，用于创建一个role可能给攻击者提供有价值的信息
class CreateARole(Vulnerability, Event):
    """ Creating a role might give an attacker the option to harm the normal behavior of newly created pods
     within the specified namespaces.
    """
    # 初始化方法，用于创建对象实例
    def __init__(self, evidence):
        # 调用父类Vulnerability的初始化方法，传入KubernetesCluster、name和category参数
        Vulnerability.__init__(self, KubernetesCluster, name="Created a role", category=AccessRisk)
        # 设置对象实例的evidence属性为传入的evidence参数
        self.evidence = evidence
# 创建一个名为 CreateAClusterRole 的类，继承自 Vulnerability 和 Event 类
class CreateAClusterRole(Vulnerability, Event):
    """ 创建一个集群角色可能会给攻击者提供在整个集群范围内破坏新创建的 pod 的正常行为的选项 """

    # 初始化方法，接受 evidence 参数
    def __init__(self, evidence):
        # 调用 Vulnerability 类的初始化方法，传入 KubernetesCluster、name 和 category 参数
        Vulnerability.__init__(
            self, KubernetesCluster, name="Created a cluster role", category=AccessRisk,
        )
        # 设置 self.evidence 属性为传入的 evidence 参数
        self.evidence = evidence


# 创建一个名为 PatchARole 的类，继承自 Vulnerability 和 Event 类
class PatchARole(Vulnerability, Event):
    """ 对角色进行修补可能会给攻击者在特定角色的命名空间范围内创建具有自定义角色的新 pod 的选项 """

    # 初始化方法，接受 evidence 参数
    def __init__(self, evidence):
        # 调用 Vulnerability 类的初始化方法，传入 KubernetesCluster、name 和 category 参数
        Vulnerability.__init__(
            self, KubernetesCluster, name="Patched a role", category=AccessRisk,
        )
        # 设置 self.evidence 属性为传入的 evidence 参数
        self.evidence = evidence


# 创建一个名为 PatchAClusterRole 的类，继承自 Vulnerability 和 Event 类
class PatchAClusterRole(Vulnerability, Event):
    """ 对集群角色进行修补可能会给攻击者在整个集群范围内创建具有自定义角色的新 pod 的选项 """

    # 初始化方法，接受 evidence 参数
    def __init__(self, evidence):
        # 调用 Vulnerability 类的初始化方法，传入 KubernetesCluster、name 和 category 参数
        Vulnerability.__init__(
            self, KubernetesCluster, name="Patched a cluster role", category=AccessRisk,
        )
        # 设置 self.evidence 属性为传入的 evidence 参数
        self.evidence = evidence


# 创建一个名为 DeleteARole 的类，继承自 Vulnerability 和 Event 类
class DeleteARole(Vulnerability, Event):
    """ 删除角色可能允许攻击者影响命名空间中资源的访问 """

    # 初始化方法，接受 evidence 参数
    def __init__(self, evidence):
        # 调用 Vulnerability 类的初始化方法，传入 KubernetesCluster、name 和 category 参数
        Vulnerability.__init__(
            self, KubernetesCluster, name="Deleted a role", category=AccessRisk,
        )
        # 设置 self.evidence 属性为传入的 evidence 参数
        self.evidence = evidence


# 创建一个名为 DeleteAClusterRole 的类，继承自 Vulnerability 和 Event 类
class DeleteAClusterRole(Vulnerability, Event):
    """ 删除集群角色可能允许攻击者影响集群中资源的访问 """

    # 初始化方法，接受 evidence 参数
    def __init__(self, evidence):
        # 调用 Vulnerability 类的初始化方法，传入 KubernetesCluster、name 和 category 参数
        Vulnerability.__init__(
            self, KubernetesCluster, name="Deleted a cluster role", category=AccessRisk,
        )
        # 设置 self.evidence 属性为传入的 evidence 参数
        self.evidence = evidence


# 创建一个名为 CreateAPod 的类，继承自 Vulnerability 和 Event 类
class CreateAPod(Vulnerability, Event):
    """ 创建一个新的 pod 允许攻击者运行自定义代码 """
    # 定义初始化方法，接受一个参数 evidence
    def __init__(self, evidence):
        # 调用父类Vulnerability的初始化方法，传入参数KubernetesCluster, name="Created A Pod", category=AccessRisk
        Vulnerability.__init__(
            self, KubernetesCluster, name="Created A Pod", category=AccessRisk,
        )
        # 设置实例属性evidence为传入的参数
        self.evidence = evidence
# 创建一个特权的 Pod 会使攻击者获得对集群的完全控制
class CreateAPrivilegedPod(Vulnerability, Event):
    def __init__(self, evidence):
        # 初始化漏洞对象，指定 KubernetesCluster 作为受影响的实体，设置名称和类别
        Vulnerability.__init__(
            self, KubernetesCluster, name="Created A PRIVILEGED Pod", category=AccessRisk,
        )
        # 保存证据
        self.evidence = evidence


# 对 Pod 进行补丁操作允许攻击者对其进行破坏和控制
class PatchAPod(Vulnerability, Event):
    def __init__(self, evidence):
        # 初始化漏洞对象，指定 KubernetesCluster 作为受影响的实体，设置名称和类别
        Vulnerability.__init__(
            self, KubernetesCluster, name="Patched A Pod", category=AccessRisk,
        )
        # 保存证据
        self.evidence = evidence


# 删除 Pod 允许攻击者干扰集群上的应用程序
class DeleteAPod(Vulnerability, Event):
    def __init__(self, evidence):
        # 初始化漏洞对象，指定 KubernetesCluster 作为受影响的实体，设置名称和类别
        Vulnerability.__init__(
            self, KubernetesCluster, name="Deleted A Pod", category=AccessRisk,
        )
        # 保存证据
        self.evidence = evidence


# ApiServerPassiveHunterFinished 事件类
class ApiServerPassiveHunterFinished(Event):
    def __init__(self, namespaces):
        # 保存命名空间列表
        self.namespaces = namespaces


# 这个 Hunter 检查在没有服务账户令牌的情况下尝试访问 API Server 会发生什么
# 如果有服务账户令牌，我们还会触发下面的 AccessApiServerWithToken
@handler.subscribe(ApiServer)
class AccessApiServer(Hunter):
    """ API Server Hunter
    检查 API 服务器是否可访问
    """

    def __init__(self, event):
        # 保存事件对象
        self.event = event
        # 构建 API Server 的访问路径
        self.path = f"{self.event.protocol}://{self.event.host}:{self.event.port}"
        # 初始化请求头
        self.headers = {}
        # 是否使用令牌进行访问
        self.with_token = False
    # 访问 API 服务器的方法
    def access_api_server(self):
        # 记录调试信息，尝试访问指定路径的 API
        logger.debug(f"Passive Hunter is attempting to access the API at {self.path}")
        try:
            # 发起 GET 请求，获取 API 数据，设置超时时间
            r = requests.get(f"{self.path}/api", headers=self.headers, verify=False, timeout=config.network_timeout)
            # 如果响应状态码为 200 并且有内容，则返回响应内容
            if r.status_code == 200 and r.content:
                return r.content
        # 捕获连接错误异常
        except requests.exceptions.ConnectionError:
            pass
        # 返回 False
        return False

    # 获取项目列表的方法
    def get_items(self, path):
        try:
            # 初始化项目列表
            items = []
            # 发起 GET 请求，获取项目数据，设置超时时间
            r = requests.get(path, headers=self.headers, verify=False, timeout=config.network_timeout)
            # 如果响应状态码为 200
            if r.status_code == 200:
                # 解析 JSON 格式的响应内容
                resp = json.loads(r.content)
                # 遍历响应内容中的项目列表，将项目名称添加到 items 列表中
                for item in resp["items"]:
                    items.append(item["metadata"]["name"])
                # 返回项目列表
                return items
            # 记录调试信息，输出 HTTP 响应状态码和响应内容
            logger.debug(f"Got HTTP {r.status_code} respone: {r.text}")
        # 捕获连接错误异常和键错误异常
        except (requests.exceptions.ConnectionError, KeyError):
            # 记录调试信息，输出访问 API 服务器失败的路径
            logger.debug(f"Failed retrieving items from API server at {path}")

        # 返回 None
        return None
    # 获取指定命名空间下的所有 Pod 列表
    def get_pods(self, namespace=None):
        # 初始化一个空的 Pod 列表
        pods = []
        try:
            # 如果未指定命名空间，则发送请求获取所有 Pod 列表
            if not namespace:
                r = requests.get(
                    f"{self.path}/api/v1/pods", headers=self.headers, verify=False, timeout=config.network_timeout,
                )
            # 如果指定了命名空间，则发送请求获取该命名空间下的 Pod 列表
            else:
                r = requests.get(
                    f"{self.path}/api/v1/namespaces/{namespace}/pods",
                    headers=self.headers,
                    verify=False,
                    timeout=config.network_timeout,
                )
            # 如果请求成功
            if r.status_code == 200:
                # 解析响应内容为 JSON 格式
                resp = json.loads(r.content)
                # 遍历响应内容中的每个 Pod
                for item in resp["items"]:
                    # 获取 Pod 的名称并转换为 ASCII 编码
                    name = item["metadata"]["name"].encode("ascii", "ignore")
                    # 获取 Pod 的命名空间并转换为 ASCII 编码
                    namespace = item["metadata"]["namespace"].encode("ascii", "ignore")
                    # 将 Pod 的名称和命名空间添加到 Pod 列表中
                    pods.append({"name": name, "namespace": namespace})
                # 返回获取到的 Pod 列表
                return pods
        # 捕获请求连接错误和键错误的异常
        except (requests.exceptions.ConnectionError, KeyError):
            pass
        # 如果发生异常或者请求失败，则返回空值
        return None
    # 执行函数，用于执行一系列操作
    def execute(self):
        # 访问 API 服务器，获取 API 对象
        api = self.access_api_server()
        # 如果成功获取到 API 对象
        if api:
            # 如果事件协议是 HTTP
            if self.event.protocol == "http":
                # 发布 ServerApiHTTPAccess 事件
                self.publish_event(ServerApiHTTPAccess(api))
            # 如果事件协议不是 HTTP
            else:
                # 发布 ServerApiAccess 事件，带有 token
                self.publish_event(ServerApiAccess(api, self.with_token))

        # 获取命名空间列表
        namespaces = self.get_items("{path}/api/v1/namespaces".format(path=self.path))
        # 如果成功获取到命名空间列表
        if namespaces:
            # 发布 ListNamespaces 事件，带有 token
            self.publish_event(ListNamespaces(namespaces, self.with_token))

        # 获取角色列表
        roles = self.get_items(f"{self.path}/apis/rbac.authorization.k8s.io/v1/roles")
        # 如果成功获取到角色列表
        if roles:
            # 发布 ListRoles 事件，带有 token
            self.publish_event(ListRoles(roles, self.with_token))

        # 获取集群角色列表
        cluster_roles = self.get_items(f"{self.path}/apis/rbac.authorization.k8s.io/v1/clusterroles")
        # 如果成功获取到集群角色列表
        if cluster_roles:
            # 发布 ListClusterRoles 事件，带有 token
            self.publish_event(ListClusterRoles(cluster_roles, self.with_token))

        # 获取 Pod 列表
        pods = self.get_pods()
        # 如果成功获取到 Pod 列表
        if pods:
            # 发布 ListPodsAndNamespaces 事件，带有 token
            self.publish_event(ListPodsAndNamespaces(pods, self.with_token))

        # 如果有服务账户令牌，此事件应该触发两次 - 一次带有令牌，一次不带令牌
        self.publish_event(ApiServerPassiveHunterFinished(namespaces))
# 订阅 ApiServer 事件，并使用谓词函数对事件进行过滤
@handler.subscribe(ApiServer, predicate=lambda x: x.auth_token)
class AccessApiServerWithToken(AccessApiServer):
    """ API Server Hunter
    Accessing the API server using the service account token obtained from a compromised pod
    """

    # 初始化方法，接收事件参数
    def __init__(self, event):
        # 调用父类的初始化方法
        super(AccessApiServerWithToken, self).__init__(event)
        # 断言事件中存在 auth_token
        assert self.event.auth_token
        # 设置请求头，包含 Authorization 字段，使用事件中的 auth_token
        self.headers = {"Authorization": f"Bearer {self.event.auth_token}"}
        # 设置类别为 InformationDisclosure
        self.category = InformationDisclosure
        # 设置 with_token 为 True
        self.with_token = True


# Active Hunter
# 订阅 ApiServerPassiveHunterFinished 事件
@handler.subscribe(ApiServerPassiveHunterFinished)
class AccessApiServerActive(ActiveHunter):
    """API server hunter
    Accessing the api server might grant an attacker full control over the cluster
    """

    # 初始化方法，接收事件参数
    def __init__(self, event):
        # 将事件参数赋值给实例变量
        self.event = event
        # 设置请求路径，使用事件中的 protocol、host 和 port
        self.path = f"{self.event.protocol}://{self.event.host}:{self.event.port}"

    # 创建项目的方法，接收路径和数据参数
    def create_item(self, path, data):
        # 设置请求头，包含 Content-Type 字段
        headers = {"Content-Type": "application/json"}
        # 如果事件中存在 auth_token，则在请求头中添加 Authorization 字段
        if self.event.auth_token:
            headers["Authorization"] = f"Bearer {self.event.auth_token}"

        # 发送 POST 请求
        try:
            # 使用 requests.post 方法发送请求，忽略 SSL 验证，设置数据和请求头，超时时间使用配置中的网络超时时间
            res = requests.post(path, verify=False, data=data, headers=headers, timeout=config.network_timeout)
            # 如果响应状态码为 200、201 或 202
            if res.status_code in [200, 201, 202]:
                # 解析响应内容为 JSON 格式
                parsed_content = json.loads(res.content)
                # 返回解析后的内容中的 metadata 字段中的 name 值
                return parsed_content["metadata"]["name"]
        # 捕获请求异常和键错误异常
        except (requests.exceptions.ConnectionError, KeyError):
            pass
        # 返回 None
        return None
    # 对指定路径的项目进行局部更新
    def patch_item(self, path, data):
        # 设置请求头，指定内容类型为 JSON 补丁
        headers = {"Content-Type": "application/json-patch+json"}
        # 如果存在授权令牌，则添加授权信息到请求头
        if self.event.auth_token:
            headers["Authorization"] = f"Bearer {self.event.auth_token}"
        try:
            # 发起 PATCH 请求，禁用 SSL 验证，设置超时时间
            res = requests.patch(path, headers=headers, verify=False, data=data, timeout=config.network_timeout)
            # 如果响应状态码不在 200、201、202 中，则返回空值
            if res.status_code not in [200, 201, 202]:
                return None
            # 解析响应内容为 JSON 格式
            parsed_content = json.loads(res.content)
            # TODO 是否有补丁时间戳可用？
            # 返回解析后的内容中的命名空间信息
            return parsed_content["metadata"]["namespace"]
        except (requests.exceptions.ConnectionError, KeyError):
            pass
        # 发生异常时返回空值
        return None

    # 删除指定路径的项目
    def delete_item(self, path):
        headers = {}
        # 如果存在授权令牌，则添加授权信息到请求头
        if self.event.auth_token:
            headers["Authorization"] = f"Bearer {self.event.auth_token}"
        try:
            # 发起 DELETE 请求，禁用 SSL 验证，设置超时时间
            res = requests.delete(path, headers=headers, verify=False, timeout=config.network_timeout)
            # 如果响应状态码在 200、201、202 中，则解析响应内容并返回删除时间戳
            if res.status_code in [200, 201, 202]:
                parsed_content = json.loads(res.content)
                return parsed_content["metadata"]["deletionTimestamp"]
        except (requests.exceptions.ConnectionError, KeyError):
            pass
        # 发生异常时返回空值
        return None

    # 创建一个 Pod
    def create_a_pod(self, namespace, is_privileged):
        # 根据是否特权模式设置安全上下文
        privileged_value = {"securityContext": {"privileged": True}} if is_privileged else {}
        # 生成随机名称
        random_name = str(uuid.uuid4())[0:5]
        # 构建 Pod 对象
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": random_name},
            "spec": {
                "containers": [
                    {"name": random_name, "image": "nginx:1.7.9", "ports": [{"containerPort": 80}], **privileged_value}
                ]
            },
        }
        # 调用 create_item 方法创建 Pod
        return self.create_item(path=f"{self.path}/api/v1/namespaces/{namespace}/pods", data=json.dumps(pod))
    # 删除指定命名空间下的指定 Pod，并返回删除时间戳
    def delete_a_pod(self, namespace, pod_name):
        # 调用 delete_item 方法删除指定路径下的资源，并获取删除时间戳
        delete_timestamp = self.delete_item(f"{self.path}/api/v1/namespaces/{namespace}/pods/{pod_name}")
        # 如果删除时间戳为空，则记录错误日志
        if not delete_timestamp:
            logger.error(f"Created pod {pod_name} in namespace {namespace} but unable to delete it")
        # 返回删除时间戳
        return delete_timestamp

    # 对指定命名空间下的指定 Pod 进行部分更新，并返回更新结果
    def patch_a_pod(self, namespace, pod_name):
        # 定义部分更新的数据
        data = [{"op": "add", "path": "/hello", "value": ["world"]}]
        # 调用 patch_item 方法对指定路径下的资源进行部分更新，并返回更新结果
        return self.patch_item(
            path=f"{self.path}/api/v1/namespaces/{namespace}/pods/{pod_name}", data=json.dumps(data),
        )

    # 创建一个命名空间，并返回创建结果
    def create_namespace(self):
        # 生成一个随机的命名空间名称
        random_name = (str(uuid.uuid4()))[0:5]
        # 定义命名空间的数据
        data = {
            "kind": "Namespace",
            "apiVersion": "v1",
            "metadata": {"name": random_name, "labels": {"name": random_name}},
        }
        # 调用 create_item 方法创建命名空间，并返回创建结果
        return self.create_item(path=f"{self.path}/api/v1/namespaces", data=json.dumps(data))

    # 删除指定命名空间，并返回删除时间戳
    def delete_namespace(self, namespace):
        # 调用 delete_item 方法删除指定路径下的资源，并获取删除时间戳
        delete_timestamp = self.delete_item(f"{self.path}/api/v1/namespaces/{namespace}")
        # 如果删除时间戳为空，则记录错误日志
        if delete_timestamp is None:
            logger.error(f"Created namespace {namespace} but failed to delete it")
        # 返回删除时间戳
        return delete_timestamp

    # 在指定命名空间下创建一个角色，并返回创建结果
    def create_a_role(self, namespace):
        # 生成一个随机的角色名称
        name = str(uuid.uuid4())[0:5]
        # 定义角色的数据
        role = {
            "kind": "Role",
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "metadata": {"namespace": namespace, "name": name},
            "rules": [{"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "watch", "list"]}],
        }
        # 调用 create_item 方法创建角色，并返回创建结果
        return self.create_item(
            path=f"{self.path}/apis/rbac.authorization.k8s.io/v1/namespaces/{namespace}/roles", data=json.dumps(role),
        )
    # 创建一个集群角色
    def create_a_cluster_role(self):
        # 生成一个随机的名称
        name = str(uuid.uuid4())[0:5]
        # 定义集群角色的数据结构
        cluster_role = {
            "kind": "ClusterRole",
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "metadata": {"name": name},
            "rules": [{"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "watch", "list"]}],
        }
        # 调用 create_item 方法创建集群角色
        return self.create_item(
            path=f"{self.path}/apis/rbac.authorization.k8s.io/v1/clusterroles", data=json.dumps(cluster_role),
        )

    # 删除一个角色
    def delete_a_role(self, namespace, name):
        # 调用 delete_item 方法删除指定命名空间下的角色
        delete_timestamp = self.delete_item(
            f"{self.path}/apis/rbac.authorization.k8s.io/v1/namespaces/{namespace}/roles/{name}"
        )
        # 如果删除时间戳为空，则记录错误日志
        if delete_timestamp is None:
            logger.error(f"Created role {name} in namespace {namespace} but unable to delete it")
        return delete_timestamp

    # 删除一个集群角色
    def delete_a_cluster_role(self, name):
        # 调用 delete_item 方法删除指定名称的集群角色
        delete_timestamp = self.delete_item(f"{self.path}/apis/rbac.authorization.k8s.io/v1/clusterroles/{name}")
        # 如果删除时间戳为空，则记录错误日志
        if delete_timestamp is None:
            logger.error(f"Created cluster role {name} but unable to delete it")
        return delete_timestamp

    # 修改一个角色
    def patch_a_role(self, namespace, role):
        # 定义要修改的数据
        data = [{"op": "add", "path": "/hello", "value": ["world"]}]
        # 调用 patch_item 方法修改指定命名空间下的角色
        return self.patch_item(
            path=f"{self.path}/apis/rbac.authorization.k8s.io/v1/namespaces/{namespace}/roles/{role}",
            data=json.dumps(data),
        )

    # 修改一个集群角色
    def patch_a_cluster_role(self, cluster_role):
        # 定义要修改的数据
        data = [{"op": "add", "path": "/hello", "value": ["world"]}]
        # 调用 patch_item 方法修改指定集群角色
        return self.patch_item(
            path=f"{self.path}/apis/rbac.authorization.k8s.io/v1/clusterroles/{cluster_role}", data=json.dumps(data),
        )
# 订阅 ApiServer 事件，将 ApiVersionHunter 类注册为处理程序
@handler.subscribe(ApiServer)
class ApiVersionHunter(Hunter):
    """Api Version Hunter
    Tries to obtain the Api Server's version directly from /version endpoint
    """

    # 初始化方法，接收 event 参数
    def __init__(self, event):
        # 保存 event 参数到实例变量
        self.event = event
        # 构建请求路径
        self.path = f"{self.event.protocol}://{self.event.host}:{self.event.port}"
        # 创建会话对象
        self.session = requests.Session()
        # 禁用 SSL 证书验证
        self.session.verify = False
        # 如果存在认证令牌，添加认证头部
        if self.event.auth_token:
            self.session.headers.update({"Authorization": f"Bearer {self.event.auth_token}"})

    # 执行方法
    def execute(self):
        # 如果存在认证令牌，记录尝试访问 API 服务器版本端点的日志
        if self.event.auth_token:
            logger.debug(
                "Trying to access the API server version endpoint using pod's"
                f" service account token on {self.event.host}:{self.event.port} \t"
            )
        # 否则，记录尝试匿名访问 API 服务器版本端点的日志
        else:
            logger.debug("Trying to access the API server version endpoint anonymously")
        # 发送 GET 请求获取 API 服务器版本信息，并解析 JSON 数据获取 gitVersion 字段
        version = self.session.get(f"{self.path}/version", timeout=config.network_timeout).json()["gitVersion"]
        # 记录发现的 API 服务器版本日志
        logger.debug(f"Discovered version of api server {version}")
        # 发布 K8sVersionDisclosure 事件，包含版本信息和端点信息
        self.publish_event(K8sVersionDisclosure(version=version, from_endpoint="/version"))
```