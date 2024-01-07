# `.\kubehunter\kube_hunter\modules\discovery\hosts.py`

```

# 导入所需的模块
import os  # 用于操作系统相关的功能
import logging  # 用于记录日志
import requests  # 用于发送 HTTP 请求
import itertools  # 用于迭代工具

from enum import Enum  # 用于创建枚举类型
from netaddr import IPNetwork, IPAddress, AddrFormatError  # 用于处理 IP 地址和网络
from netifaces import AF_INET, ifaddresses, interfaces  # 用于获取网络接口信息
from scapy.all import ICMP, IP, Ether, srp1  # 用于网络数据包的构造和分析

from kube_hunter.conf import config  # 导入配置信息
from kube_hunter.core.events import handler  # 导入事件处理器
from kube_hunter.core.events.types import Event, NewHostEvent, Vulnerability  # 导入事件类型
from kube_hunter.core.types import Discovery, InformationDisclosure, Azure  # 导入自定义类型

logger = logging.getLogger(__name__)  # 创建日志记录器


# 定义一个事件类，表示在 Pod 内运行
class RunningAsPodEvent(Event):
    def __init__(self):
        self.name = "Running from within a pod"  # 事件名称
        self.auth_token = self.get_service_account_file("token")  # 获取服务账户文件中的认证令牌
        self.client_cert = self.get_service_account_file("ca.crt")  # 获取服务账户文件中的客户端证书
        self.namespace = self.get_service_account_file("namespace")  # 获取服务账户文件中的命名空间
        self.kubeservicehost = os.environ.get("KUBERNETES_SERVICE_HOST", None)  # 获取环境变量中的 Kubernetes 服务主机地址

    # 用于逻辑位置，主要用于报告
    def location(self):
        location = "Local to Pod"  # 逻辑位置
        hostname = os.getenv("HOSTNAME")  # 获取主机名
        if hostname:
            location += f" ({hostname})"  # 如果有主机名，则加上主机名
        return location  # 返回逻辑位置

    # 获取服务账户文件中的内容
    def get_service_account_file(self, file):
        try:
            with open(f"/var/run/secrets/kubernetes.io/serviceaccount/{file}") as f:
                return f.read()  # 读取文件内容
        except IOError:
            pass  # 如果出现异常，则忽略


# 定义一个事件类，表示访问 Azure Metadata API 暴露了与集群关联的机器的信息
class AzureMetadataApi(Vulnerability, Event):
    """Access to the Azure Metadata API exposes information about the machines associated with the cluster"""

    def __init__(self, cidr):
        Vulnerability.__init__(
            self, Azure, "Azure Metadata Exposure", category=InformationDisclosure, vid="KHV003",
        )  # 初始化漏洞信息
        self.cidr = cidr  # CIDR 地址
        self.evidence = "cidr: {}".format(cidr)  # 证据信息


# 定义一个事件类，表示主机扫描
class HostScanEvent(Event):
    def __init__(self, pod=False, active=False, predefined_hosts=None):
        # 用于指定是否从漏洞中获取实际数据的标志
        self.active = active  # 是否活跃
        self.predefined_hosts = predefined_hosts or []  # 预定义的主机列表


# 主机发现辅助类
class HostDiscoveryHelpers:
    # 生成给定 CIDR 的子网的生成器
    @staticmethod
    def filter_subnet(subnet, ignore=None):
        for ip in subnet:
            if ignore and any(ip in s for s in ignore):
                logger.debug(f"HostDiscoveryHelpers.filter_subnet ignoring {ip}")  # 记录调试信息
            else:
                yield ip  # 生成 IP 地址

    # 生成所有 CIDR 的主机
    @staticmethod
    def generate_hosts(cidrs):
        ignore = list()
        scan = list()
        for cidr in cidrs:
            try:
                if cidr.startswith("!"):
                    ignore.append(IPNetwork(cidr[1:]))  # 添加要忽略的 CIDR
                else:
                    scan.append(IPNetwork(cidr))  # 添加要扫描的 CIDR
            except AddrFormatError as e:
                raise ValueError(f"Unable to parse CIDR {cidr}") from e  # 抛出异常，无法解析 CIDR

        return itertools.chain.from_iterable(HostDiscoveryHelpers.filter_subnet(sb, ignore=ignore) for sb in scan)  # 生成所有主机


# 订阅 RunningAsPodEvent 事件
@handler.subscribe(RunningAsPodEvent)
class FromPodHostDiscovery(Discovery):
    """Host Discovery when running as pod
    Generates ip adresses to scan, based on cluster/scan type
    """

    def __init__(self, event):
        self.event = event  # 初始化事件

    def execute(self):
        # 扫描用户指定的任何主机
        if config.remote or config.cidr:
            self.publish_event(HostScanEvent())
        else:
            # 发现集群子网，我们将扫描所有这些主机
            cloud = None
            if self.is_azure_pod():
                subnets, cloud = self.azure_metadata_discovery()  # Azure 元数据发现
            else:
                subnets, ext_ip = self.traceroute_discovery()  # 跟踪路由发现

            should_scan_apiserver = False
            if self.event.kubeservicehost:
                should_scan_apiserver = True
            for ip, mask in subnets:
                if self.event.kubeservicehost and self.event.kubeservicehost in IPNetwork(f"{ip}/{mask}"):
                    should_scan_apiserver = False
                logger.debug(f"From pod scanning subnet {ip}/{mask}")  # 记录调试信息
                for ip in IPNetwork(f"{ip}/{mask}"):
                    self.publish_event(NewHostEvent(host=ip, cloud=cloud))  # 发布新主机事件
            if should_scan_apiserver:
                self.publish_event(NewHostEvent(host=IPAddress(self.event.kubeservicehost), cloud=cloud))  # 发布新主机事件

    def is_azure_pod(self):
        try:
            logger.debug("From pod attempting to access Azure Metadata API")  # 记录调试信息
            if (
                requests.get(
                    "http://169.254.169.254/metadata/instance?api-version=2017-08-01",
                    headers={"Metadata": "true"},
                    timeout=config.network_timeout,
                ).status_code
                == 200
            ):
                return True  # 如果能够访问 Azure Metadata API，则返回 True
        except requests.exceptions.ConnectionError:
            logger.debug("Failed to connect Azure metadata server")  # 记录调试信息
            return False  # 如果连接失败，则返回 False

    # 用于 Pod 扫描
    def traceroute_discovery(self):
        # 获取外部 IP，以确定是否为云集群
        external_ip = requests.get("https://canhazip.com", timeout=config.network_timeout).text  # 获取外部 IP
        node_internal_ip = srp1(
            Ether() / IP(dst="1.1.1.1", ttl=1) / ICMP(), verbose=0, timeout=config.network_timeout,
        )[IP].src  # 获取节点内部 IP
        return [[node_internal_ip, "24"]], external_ip  # 返回内部 IP 和外部 IP

    # 查询 Azure 的接口元数据 API | 仅从 Pod 中工作
    def azure_metadata_discovery(self):
        logger.debug("From pod attempting to access azure's metadata")  # 记录调试信息
        machine_metadata = requests.get(
            "http://169.254.169.254/metadata/instance?api-version=2017-08-01",
            headers={"Metadata": "true"},
            timeout=config.network_timeout,
        ).json()  # 获取机器元数据
        address, subnet = "", ""
        subnets = list()
        for interface in machine_metadata["network"]["interface"]:
            address, subnet = (
                interface["ipv4"]["subnet"][0]["address"],
                interface["ipv4"]["subnet"][0]["prefix"],
            )  # 获取地址和子网
            subnet = subnet if not config.quick else "24"  # 如果快速模式，则使用 24 位子网
            logger.debug(f"From pod discovered subnet {address}/{subnet}")  # 记录调试信息
            subnets.append([address, subnet if not config.quick else "24"])  # 添加子网

            self.publish_event(AzureMetadataApi(cidr=f"{address}/{subnet}"))  # 发布 AzureMetadataApi 事件

        return subnets, "Azure"  # 返回子网和云类型


# 订阅 HostScanEvent 事件
@handler.subscribe(HostScanEvent)
class HostDiscovery(Discovery):
    """Host Discovery
    Generates ip adresses to scan, based on cluster/scan type
    """

    def __init__(self, event):
        self.event = event  # 初始化事件

    def execute(self):
        if config.cidr:
            for ip in HostDiscoveryHelpers.generate_hosts(config.cidr):
                self.publish_event(NewHostEvent(host=ip))  # 发布新主机事件
        elif config.interface:
            self.scan_interfaces()  # 扫描接口
        elif len(config.remote) > 0:
            for host in config.remote:
                self.publish_event(NewHostEvent(host=host))  # 发布新主机事件

    # 用于正常扫描
    def scan_interfaces(self):
        for ip in self.generate_interfaces_subnet():
            handler.publish_event(NewHostEvent(host=ip))  # 发布新主机事件

    # 生成所有内部网络接口的子网
    def generate_interfaces_subnet(self, sn="24"):
        for ifaceName in interfaces():
            for ip in [i["addr"] for i in ifaddresses(ifaceName).setdefault(AF_INET, [])]:
                if not self.event.localhost and InterfaceTypes.LOCALHOST.value in ip.__str__():
                    continue
                for ip in IPNetwork(f"{ip}/{sn}"):
                    yield ip  # 生成 IP 地址


# 用于比较前缀
class InterfaceTypes(Enum):
    LOCALHOST = "127"  # 本地主机

```