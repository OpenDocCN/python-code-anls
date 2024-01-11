# `kubehunter\kube_hunter\modules\discovery\hosts.py`

```
# 导入所需的模块
import os  # 导入操作系统模块
import logging  # 导入日志记录模块
import requests  # 导入发送 HTTP 请求的模块
import itertools  # 导入迭代工具模块

from enum import Enum  # 从枚举模块中导入 Enum 类
from netaddr import IPNetwork, IPAddress, AddrFormatError  # 从 netaddr 模块中导入 IPNetwork、IPAddress 和 AddrFormatError 类
from netifaces import AF_INET, ifaddresses, interfaces  # 从 netifaces 模块中导入 AF_INET、ifaddresses 和 interfaces 函数
from scapy.all import ICMP, IP, Ether, srp1  # 从 scapy.all 模块中导入 ICMP、IP、Ether 和 srp1 函数

from kube_hunter.conf import config  # 从 kube_hunter.conf 模块中导入 config 对象
from kube_hunter.core.events import handler  # 从 kube_hunter.core.events 模块中导入 handler 函数
from kube_hunter.core.events.types import Event, NewHostEvent, Vulnerability  # 从 kube_hunter.core.events.types 模块中导入 Event、NewHostEvent 和 Vulnerability 类
from kube_hunter.core.types import Discovery, InformationDisclosure, Azure  # 从 kube_hunter.core.types 模块中导入 Discovery、InformationDisclosure 和 Azure 类

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class RunningAsPodEvent(Event):
    # 表示正在从 Pod 内部运行的事件
    def __init__(self):
        self.name = "Running from within a pod"  # 设置事件名称
        self.auth_token = self.get_service_account_file("token")  # 获取服务账户文件中的认证令牌
        self.client_cert = self.get_service_account_file("ca.crt")  # 获取服务账户文件中的客户端证书
        self.namespace = self.get_service_account_file("namespace")  # 获取服务账户文件中的命名空间
        self.kubeservicehost = os.environ.get("KUBERNETES_SERVICE_HOST", None)  # 获取环境变量中的 Kubernetes 服务主机地址

    # 事件的逻辑位置，主要用于报告
    def location(self):
        location = "Local to Pod"  # 设置默认位置
        hostname = os.getenv("HOSTNAME")  # 获取主机名
        if hostname:
            location += f" ({hostname})"  # 如果存在主机名，则在默认位置后面添加主机名

        return location  # 返回位置信息

    def get_service_account_file(self, file):
        try:
            with open(f"/var/run/secrets/kubernetes.io/serviceaccount/{file}") as f:  # 打开服务账户文件
                return f.read()  # 读取文件内容并返回
        except IOError:
            pass  # 如果发生 IO 错误，则忽略


class AzureMetadataApi(Vulnerability, Event):
    """访问 Azure 元数据 API 暴露了与集群关联的机器的信息"""

    def __init__(self, cidr):
        Vulnerability.__init__(
            self, Azure, "Azure Metadata Exposure", category=InformationDisclosure, vid="KHV003",
        )  # 初始化漏洞对象
        self.cidr = cidr  # 设置 CIDR 地址
        self.evidence = "cidr: {}".format(cidr)  # 设置证据信息


class HostScanEvent(Event):
    # 主机扫描事件
    # 初始化函数，用于创建对象实例
    def __init__(self, pod=False, active=False, predefined_hosts=None):
        # 用于指定是否从漏洞中获取实际数据的标志
        self.active = active
        # 预定义主机列表，如果没有提供则为空列表
        self.predefined_hosts = predefined_hosts or []
class HostDiscoveryHelpers:
    # generator, generating a subnet by given a cidr
    @staticmethod
    def filter_subnet(subnet, ignore=None):
        # 遍历子网中的每个 IP 地址
        for ip in subnet:
            # 如果存在要忽略的子网，并且当前 IP 在要忽略的子网中
            if ignore and any(ip in s for s in ignore):
                # 记录 debug 日志，表示忽略当前 IP
                logger.debug(f"HostDiscoveryHelpers.filter_subnet ignoring {ip}")
            else:
                # 否则，生成当前 IP
                yield ip

    @staticmethod
    def generate_hosts(cidrs):
        ignore = list()
        scan = list()
        # 遍历每个 CIDR
        for cidr in cidrs:
            try:
                # 如果 CIDR 以 "!" 开头，将其解析为要忽略的子网
                if cidr.startswith("!"):
                    ignore.append(IPNetwork(cidr[1:]))
                else:
                    # 否则，将其解析为要扫描的子网
                    scan.append(IPNetwork(cidr))
            except AddrFormatError as e:
                # 如果解析出错，抛出异常
                raise ValueError(f"Unable to parse CIDR {cidr}") from e

        # 生成要扫描的 IP 地址
        return itertools.chain.from_iterable(HostDiscoveryHelpers.filter_subnet(sb, ignore=ignore) for sb in scan)


@handler.subscribe(RunningAsPodEvent)
class FromPodHostDiscovery(Discovery):
    """Host Discovery when running as pod
    Generates ip adresses to scan, based on cluster/scan type
    """

    def __init__(self, event):
        # 初始化方法，接收事件对象
        self.event = event
    # 执行函数，用于执行扫描操作
    def execute(self):
        # 如果用户指定了远程主机或者CIDR，则扫描任何用户指定的主机
        if config.remote or config.cidr:
            # 发布主机扫描事件
            self.publish_event(HostScanEvent())
        else:
            # 发现集群子网，将扫描所有这些主机
            cloud = None
            # 如果是 Azure 容器，则进行 Azure 元数据发现
            if self.is_azure_pod():
                subnets, cloud = self.azure_metadata_discovery()
            else:
                subnets, ext_ip = self.traceroute_discovery()

            should_scan_apiserver = False
            # 如果存在 kubeservicehost，则应该扫描 API 服务器
            if self.event.kubeservicehost:
                should_scan_apiserver = True
            for ip, mask in subnets:
                # 如果 kubeservicehost 存在并且在当前子网中
                if self.event.kubeservicehost and self.event.kubeservicehost in IPNetwork(f"{ip}/{mask}"):
                    should_scan_apiserver = False
                # 记录日志，扫描当前子网的 IP 地址
                logger.debug(f"From pod scanning subnet {ip}/{mask}")
                for ip in IPNetwork(f"{ip}/{mask}"):
                    # 发布新主机事件，包括主机 IP 和云信息
                    self.publish_event(NewHostEvent(host=ip, cloud=cloud))
            # 如果需要扫描 API 服务器，则发布新主机事件
            if should_scan_apiserver:
                self.publish_event(NewHostEvent(host=IPAddress(self.event.kubeservicehost), cloud=cloud))

    # 判断是否为 Azure 容器
    def is_azure_pod(self):
        try:
            # 记录日志，尝试访问 Azure 元数据 API
            logger.debug("From pod attempting to access Azure Metadata API")
            # 发送请求，检查是否能够连接到 Azure 元数据服务器
            if (
                requests.get(
                    "http://169.254.169.254/metadata/instance?api-version=2017-08-01",
                    headers={"Metadata": "true"},
                    timeout=config.network_timeout,
                ).status_code
                == 200
            ):
                # 如果能够连接，则返回 True
                return True
        except requests.exceptions.ConnectionError:
            # 记录日志，连接 Azure 元数据服务器失败
            logger.debug("Failed to connect Azure metadata server")
            # 如果连接失败，则返回 False
            return False

    # 用于容器扫描
    # 进行路由跟踪发现，获取外部 IP 地址，以确定是否为云集群
    external_ip = requests.get("https://canhazip.com", timeout=config.network_timeout).text

    # 发送 ICMP 报文，获取节点内部 IP 地址
    node_internal_ip = srp1(
        Ether() / IP(dst="1.1.1.1", ttl=1) / ICMP(), verbose=0, timeout=config.network_timeout,
    )[IP].src
    # 返回节点内部 IP 地址和外部 IP 地址
    return [[node_internal_ip, "24"]], external_ip

    # 查询 Azure 的接口元数据 API，仅在 Pod 中有效
    def azure_metadata_discovery(self):
        # 记录调试信息
        logger.debug("From pod attempting to access azure's metadata")
        # 从 Azure 的元数据 API 获取机器元数据
        machine_metadata = requests.get(
            "http://169.254.169.254/metadata/instance?api-version=2017-08-01",
            headers={"Metadata": "true"},
            timeout=config.network_timeout,
        ).json()
        address, subnet = "", ""
        subnets = list()
        # 遍历机器元数据中的网络接口
        for interface in machine_metadata["network"]["interface"]:
            # 获取 IP 地址和子网信息
            address, subnet = (
                interface["ipv4"]["subnet"][0]["address"],
                interface["ipv4"]["subnet"][0]["prefix"],
            )
            # 如果是快速模式，则使用默认子网掩码 24
            subnet = subnet if not config.quick else "24"
            # 记录调试信息
            logger.debug(f"From pod discovered subnet {address}/{subnet}")
            # 将 IP 地址和子网信息添加到子网列表中
            subnets.append([address, subnet if not config.quick else "24"])

            # 发布 Azure 元数据 API 事件
            self.publish_event(AzureMetadataApi(cidr=f"{address}/{subnet}"))

        # 返回子网列表和 "Azure" 字符串
        return subnets, "Azure"
# 订阅 HostScanEvent 事件，并定义 HostDiscovery 类作为其处理程序
@handler.subscribe(HostScanEvent)
class HostDiscovery(Discovery):
    """Host Discovery
    Generates ip adresses to scan, based on cluster/scan type
    """

    # 初始化方法，接收事件对象作为参数
    def __init__(self, event):
        self.event = event

    # 执行方法，根据配置生成要扫描的 IP 地址
    def execute(self):
        # 如果配置了 cidr
        if config.cidr:
            # 生成要扫描的主机 IP 地址，并发布 NewHostEvent 事件
            for ip in HostDiscoveryHelpers.generate_hosts(config.cidr):
                self.publish_event(NewHostEvent(host=ip))
        # 如果配置了 interface
        elif config.interface:
            # 扫描网络接口
            self.scan_interfaces()
        # 如果配置了 remote
        elif len(config.remote) > 0:
            # 遍历远程主机列表，并发布 NewHostEvent 事件
            for host in config.remote:
                self.publish_event(NewHostEvent(host=host))

    # 用于正常扫描的方法
    def scan_interfaces(self):
        # 生成所有内部网络接口的子网，并发布 NewHostEvent 事件
        for ip in self.generate_interfaces_subnet():
            handler.publish_event(NewHostEvent(host=ip))

    # 生成所有内部网络接口的子网
    def generate_interfaces_subnet(self, sn="24"):
        # 遍历所有网络接口
        for ifaceName in interfaces():
            # 获取每个接口的 IP 地址
            for ip in [i["addr"] for i in ifaddresses(ifaceName).setdefault(AF_INET, [])]:
                # 如果不是本地主机，并且 IP 地址不包含本地主机的前缀
                if not self.event.localhost and InterfaceTypes.LOCALHOST.value in ip.__str__():
                    continue
                # 生成 IP 地址的子网，并返回
                for ip in IPNetwork(f"{ip}/{sn}"):
                    yield ip


# 用于比较前缀的枚举类
class InterfaceTypes(Enum):
    LOCALHOST = "127"
```