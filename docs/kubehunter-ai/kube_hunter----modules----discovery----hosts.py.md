# `kubehunter\kube_hunter\modules\discovery\hosts.py`

```
# 导入所需的模块
import os  # 操作系统模块
import logging  # 日志模块
import requests  # 发送 HTTP 请求模块
import itertools  # 迭代工具模块
from enum import Enum  # 枚举模块
from netaddr import IPNetwork, IPAddress, AddrFormatError  # IP 地址操作模块
from netifaces import AF_INET, ifaddresses, interfaces  # 网络接口信息模块
from scapy.all import ICMP, IP, Ether, srp1  # 网络数据包操作模块

# 导入自定义模块
from kube_hunter.conf import config  # 导入 kube_hunter 配置模块
from kube_hunter.core.events import handler  # 导入事件处理模块
from kube_hunter.core.events.types import Event, NewHostEvent, Vulnerability  # 导入事件类型模块
from kube_hunter.core.types import Discovery, InformationDisclosure, Azure  # 导入自定义类型模块

# 获取 logger 对象
logger = logging.getLogger(__name__)

# 定义 RunningAsPodEvent 类，继承自 Event 类
class RunningAsPodEvent(Event):
    # 初始化方法
    def __init__(self):
# 设置属性 name 为 "Running from within a pod"
# 从服务账户文件中获取认证令牌，赋值给属性 auth_token
# 从服务账户文件中获取客户端证书，赋值给属性 client_cert
# 从服务账户文件中获取命名空间，赋值给属性 namespace
# 从环境变量中获取 Kubernetes 服务主机地址，赋值给属性 kubeservicehost

# 返回事件的逻辑位置，主要用于报告
def location(self):
    # 设置初始位置为 "Local to Pod"
    location = "Local to Pod"
    # 获取主机名
    hostname = os.getenv("HOSTNAME")
    # 如果主机名存在，则在位置后面加上主机名
    if hostname:
        location += f" ({hostname})"
    # 返回位置
    return location

# 从服务账户文件中获取指定文件的内容
def get_service_account_file(self, file):
    try:
        # 打开服务账户文件，读取内容并返回
        with open(f"/var/run/secrets/kubernetes.io/serviceaccount/{file}") as f:
            return f.read()
    # 如果发生 IO 错误，则捕获并处理
    except IOError:
# 定义一个空的代码块，不执行任何操作
pass

# 定义一个名为AzureMetadataApi的类，继承自Vulnerability和Event类
# 该类用于访问Azure Metadata API，获取与集群关联的机器信息
class AzureMetadataApi(Vulnerability, Event):
    """Access to the Azure Metadata API exposes information about the machines associated with the cluster"""

    # 初始化方法，接受cidr参数
    def __init__(self, cidr):
        # 调用Vulnerability类的初始化方法，设置相关属性
        Vulnerability.__init__(
            self, Azure, "Azure Metadata Exposure", category=InformationDisclosure, vid="KHV003",
        )
        # 设置cidr属性
        self.cidr = cidr
        # 设置evidence属性，包含cidr信息
        self.evidence = "cidr: {}".format(cidr)

# 定义一个名为HostScanEvent的类，继承自Event类
class HostScanEvent(Event):
    # 初始化方法，接受pod、active和predefined_hosts参数
    def __init__(self, pod=False, active=False, predefined_hosts=None):
        # 用于指定是否从漏洞中获取实际数据的标志
        self.active = active
        # 预定义的主机列表
        self.predefined_hosts = predefined_hosts or []
class HostDiscoveryHelpers:
    # 主机发现辅助类

    # 生成器，根据给定的 CIDR 生成子网
    @staticmethod
    def filter_subnet(subnet, ignore=None):
        # 遍历子网中的 IP 地址
        for ip in subnet:
            # 如果存在要忽略的 IP 地址，并且当前 IP 在要忽略的范围内，则记录日志并忽略
            if ignore and any(ip in s for s in ignore):
                logger.debug(f"HostDiscoveryHelpers.filter_subnet ignoring {ip}")
            # 否则，返回当前 IP 地址
            else:
                yield ip

    @staticmethod
    def generate_hosts(cidrs):
        ignore = list()
        scan = list()
        # 遍历给定的 CIDR 列表
        for cidr in cidrs:
            try:
                # 如果 CIDR 以 "!" 开头，将其添加到忽略列表中
                if cidr.startswith("!"):
                    ignore.append(IPNetwork(cidr[1:]))
                # 否则，将其添加到扫描列表中
                else:
# 将 IP 地址范围添加到扫描列表中
scan.append(IPNetwork(cidr))
# 如果无法解析 CIDR，则抛出异常
except AddrFormatError as e:
    raise ValueError(f"Unable to parse CIDR {cidr}") from e

# 返回一个迭代器，该迭代器将多个子网的主机发现结果合并在一起
return itertools.chain.from_iterable(HostDiscoveryHelpers.filter_subnet(sb, ignore=ignore) for sb in scan)


@handler.subscribe(RunningAsPodEvent)
class FromPodHostDiscovery(Discovery):
    """当作为 pod 运行时的主机发现
    根据集群/扫描类型生成要扫描的 IP 地址
    """

    def __init__(self, event):
        self.event = event

    def execute(self):
        # 扫描用户指定的任何主机
        if config.remote or config.cidr:
            self.publish_event(HostScanEvent())
        else:
            # 发现集群子网，我们将扫描所有这些主机
            cloud = None
            # 如果是 Azure Pod，则进行 Azure 元数据发现
            if self.is_azure_pod():
                subnets, cloud = self.azure_metadata_discovery()
            else:
                subnets, ext_ip = self.traceroute_discovery()

            should_scan_apiserver = False
            # 如果存在 kubeservicehost，则应扫描 API 服务器
            if self.event.kubeservicehost:
                should_scan_apiserver = True
            for ip, mask in subnets:
                # 如果 kubeservicehost 存在并且在当前子网中
                if self.event.kubeservicehost and self.event.kubeservicehost in IPNetwork(f"{ip}/{mask}"):
                    should_scan_apiserver = False
                logger.debug(f"From pod scanning subnet {ip}/{mask}")
                # 遍历子网中的 IP 地址，发布新主机事件
                for ip in IPNetwork(f"{ip}/{mask}"):
                    self.publish_event(NewHostEvent(host=ip, cloud=cloud))
            # 如果需要扫描 API 服务器，则发布新主机事件
            if should_scan_apiserver:
                self.publish_event(NewHostEvent(host=IPAddress(self.event.kubeservicehost), cloud=cloud))
# 检查当前环境是否是 Azure 的 Pod
def is_azure_pod(self):
    # 记录日志，表示从 Pod 尝试访问 Azure 元数据 API
    logger.debug("From pod attempting to access Azure Metadata API")
    # 发送请求到 Azure 元数据 API，检查返回状态码是否为 200
    if (
        requests.get(
            "http://169.254.169.254/metadata/instance?api-version=2017-08-01",
            headers={"Metadata": "true"},
            timeout=config.network_timeout,
        ).status_code
        == 200
    ):
        # 如果状态码为 200，表示是 Azure 的 Pod，返回 True
        return True
    # 如果连接错误，记录日志并返回 False
    except requests.exceptions.ConnectionError:
        logger.debug("Failed to connect Azure metadata server")
        return False

# 用于 Pod 扫描
def traceroute_discovery(self):
    # 获取外部 IP，以确定是否是云集群
    external_ip = requests.get("https://canhazip.com", timeout=config.network_timeout).text
# 使用 scapy 库发送一个 ICMP 包，以获取内部节点的 IP 地址
node_internal_ip = srp1(
    Ether() / IP(dst="1.1.1.1", ttl=1) / ICMP(), verbose=0, timeout=config.network_timeout,
)[IP].src
# 返回内部节点的 IP 地址和外部 IP 地址
return [[node_internal_ip, "24"]], external_ip

# 从 Pod 中查询 Azure 的接口元数据 API
def azure_metadata_discovery(self):
    logger.debug("From pod attempting to access azure's metadata")
    # 发送 GET 请求获取 Azure 的机器元数据
    machine_metadata = requests.get(
        "http://169.254.169.254/metadata/instance?api-version=2017-08-01",
        headers={"Metadata": "true"},
        timeout=config.network_timeout,
    ).json()
    address, subnet = "", ""
    subnets = list()
    # 遍历机器元数据中的网络接口信息
    for interface in machine_metadata["network"]["interface"]:
        # 获取 IPv4 地址和子网信息
        address, subnet = (
            interface["ipv4"]["subnet"][0]["address"],
            interface["ipv4"]["subnet"][0]["prefix"],
            )
            # 如果配置中的快速扫描标志为假，则使用子网掩码，否则使用默认的“24”
            subnet = subnet if not config.quick else "24"
            # 记录从 pod 发现的子网
            logger.debug(f"From pod discovered subnet {address}/{subnet}")
            # 将地址和子网添加到子网列表中
            subnets.append([address, subnet if not config.quick else "24"])

            # 发布 AzureMetadataApi 事件
            self.publish_event(AzureMetadataApi(cidr=f"{address}/{subnet}"))

        # 返回子网列表和字符串“Azure”
        return subnets, "Azure"


@handler.subscribe(HostScanEvent)
class HostDiscovery(Discovery):
    """Host Discovery
    Generates ip adresses to scan, based on cluster/scan type
    """

    def __init__(self, event):
        self.event = event

    def execute(self):
```

# 如果配置文件中有指定CIDR，则根据CIDR生成主机列表，并发布新主机事件
if config.cidr:
    for ip in HostDiscoveryHelpers.generate_hosts(config.cidr):
        self.publish_event(NewHostEvent(host=ip))
# 如果配置文件中有指定接口，则扫描接口
elif config.interface:
    self.scan_interfaces()
# 如果配置文件中有指定远程主机，则根据远程主机列表发布新主机事件
elif len(config.remote) > 0:
    for host in config.remote:
        self.publish_event(NewHostEvent(host=host))

# 用于正常扫描接口
def scan_interfaces(self):
    # 生成所有内部网络接口的子网
    for ip in self.generate_interfaces_subnet():
        handler.publish_event(NewHostEvent(host=ip))

# 从所有内部网络接口生成所有子网
def generate_interfaces_subnet(self, sn="24"):
    for ifaceName in interfaces():
        for ip in [i["addr"] for i in ifaddresses(ifaceName).setdefault(AF_INET, [])]:
            # 如果不是本地主机，并且IP地址中包含localhost，则跳过
            if not self.event.localhost and InterfaceTypes.LOCALHOST.value in ip.__str__():
                continue
# 遍历指定 IP 地址范围内的所有 IP 地址，并逐个返回
for ip in IPNetwork(f"{ip}/{sn}"):
    yield ip

# 枚举类，用于比较不同接口类型的前缀
class InterfaceTypes(Enum):
    LOCALHOST = "127"
```