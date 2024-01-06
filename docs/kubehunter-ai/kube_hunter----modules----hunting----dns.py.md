# `kubehunter\kube_hunter\modules\hunting\dns.py`

```
# 导入所需的模块
import re  # 正则表达式模块
import logging  # 日志记录模块

from scapy.all import IP, ICMP, UDP, DNS, DNSQR, ARP, Ether, sr1, srp1, srp  # 导入网络数据包操作模块

from kube_hunter.conf import config  # 导入配置模块
from kube_hunter.core.events import handler  # 导入事件处理模块
from kube_hunter.core.events.types import Event, Vulnerability  # 导入事件和漏洞类型
from kube_hunter.core.types import ActiveHunter, KubernetesCluster, IdentityTheft  # 导入活跃探测器、Kubernetes集群和身份盗用类型
from kube_hunter.modules.hunting.arp import PossibleArpSpoofing  # 导入ARP欺骗模块

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class PossibleDnsSpoofing(Vulnerability, Event):
    """A malicious pod running on the cluster could potentially run a DNS Spoof attack
    and perform a MITM attack on applications running in the cluster."""
    # 可能的DNS欺骗漏洞类，继承自漏洞和事件类

    def __init__(self, kubedns_pod_ip):
        # 初始化方法，接受参数kubedns_pod_ip
        Vulnerability.__init__(
# 定义一个名为DnsSpoofHunter的类，继承自ActiveHunter类，用于检测可能的DNS欺骗攻击
# 只有在RunningAsPod基础事件触发时才会触发
@handler.subscribe(PossibleArpSpoofing)
class DnsSpoofHunter(ActiveHunter):
    """DNS Spoof Hunter
    Checks for the possibility for a malicious pod to compromise DNS requests of the cluster
    (results are based on the running node)
    """
    # 初始化方法，接收一个事件对象作为参数
    def __init__(self, event):
        self.event = event

    # 获取cbr0接口的IP地址和MAC地址
    def get_cbr0_ip_mac(self):
        # 发送一个ICMP请求，获取cbr0接口的IP地址和MAC地址
        res = srp1(Ether() / IP(dst="1.1.1.1", ttl=1) / ICMP(), verbose=0, timeout=config.network_timeout)
        return res[IP].src, res.src
# 从 /etc/resolv.conf 文件中提取出第一个 nameserver 的 IP 地址
def extract_nameserver_ip(self):
    with open("/etc/resolv.conf") as f:
        # 在 /etc/resolv.conf 文件中查找第一个 nameserver 的 IP 地址
        match = re.search(r"nameserver (\d+.\d+.\d+.\d+)", f.read())
        if match:
            return match.group(1)

# 获取 kube-dns 服务的 IP 地址
def get_kube_dns_ip_mac(self):
    kubedns_svc_ip = self.extract_nameserver_ip()

    # 通过比较 DNS 响应的源 MAC 地址和 ARP 扫描来获取 kube-dns 服务的实际 pod IP 地址
    dns_info_res = srp1(
        Ether() / IP(dst=kubedns_svc_ip) / UDP(dport=53) / DNS(rd=1, qd=DNSQR()),
        verbose=0,
        timeout=config.network_timeout,
    )
    kubedns_pod_mac = dns_info_res.src
    self_ip = dns_info_res[IP].dst
# 使用 scapy 库发送 ARP 请求并接收响应
arp_responses, _ = srp(
    Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(op=1, pdst=f"{self_ip}/24"), timeout=config.network_timeout, verbose=0,
)
# 遍历 ARP 响应，如果响应的源 MAC 地址等于 kubedns_pod_mac，则返回响应的源 IP 地址和源 MAC 地址
for _, response in arp_responses:
    if response[Ether].src == kubedns_pod_mac:
        return response[ARP].psrc, response.src

# 定义一个 execute 方法
def execute(self):
    # 记录调试信息
    logger.debug("Attempting to get kube-dns pod ip")
    # 发送 ICMP 请求并接收响应，获取自身 IP 地址
    self_ip = sr1(IP(dst="1.1.1.1", ttl=1) / ICMP(), verbose=0, timeout=config.netork_timeout)[IP].dst
    # 获取 cbr0 的 IP 地址和 MAC 地址
    cbr0_ip, cbr0_mac = self.get_cbr0_ip_mac()

    # 获取 kube-dns 的 IP 地址和 MAC 地址
    kubedns = self.get_kube_dns_ip_mac()
    if kubedns:
        kubedns_ip, kubedns_mac = kubedns
        # 记录调试信息
        logger.debug(f"ip={self_ip} kubednsip={kubedns_ip} cbr0ip={cbr0_ip}")
        # 如果 kube-dns 的 MAC 地址不等于 cbr0 的 MAC 地址
        if kubedns_mac != cbr0_mac:
            # 发布可能的 DNS 欺骗事件
            self.publish_event(PossibleDnsSpoofing(kubedns_pod_ip=kubedns_ip))
    else:
# 使用日志记录器记录无法获取 kubedns 身份的信息。
```