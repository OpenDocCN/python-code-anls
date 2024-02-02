# `kubehunter\kube_hunter\modules\hunting\dns.py`

```py
# 导入所需的模块
import re
import logging
from scapy.all import IP, ICMP, UDP, DNS, DNSQR, ARP, Ether, sr1, srp1, srp
from kube_hunter.conf import config
from kube_hunter.core.events import handler
from kube_hunter.core.events.types import Event, Vulnerability
from kube_hunter.core.types import ActiveHunter, KubernetesCluster, IdentityTheft
from kube_hunter.modules.hunting.arp import PossibleArpSpoofing

# 获取 logger 对象
logger = logging.getLogger(__name__)

# 定义可能的 DNS 欺骗漏洞类，继承自 Vulnerability 和 Event 类
class PossibleDnsSpoofing(Vulnerability, Event):
    """A malicious pod running on the cluster could potentially run a DNS Spoof attack
    and perform a MITM attack on applications running in the cluster."""

    def __init__(self, kubedns_pod_ip):
        # 调用父类的初始化方法
        Vulnerability.__init__(
            self, KubernetesCluster, "Possible DNS Spoof", category=IdentityTheft, vid="KHV030",
        )
        # 设置 kubedns_pod_ip 属性
        self.kubedns_pod_ip = kubedns_pod_ip
        # 设置 evidence 属性
        self.evidence = "kube-dns at: {}".format(self.kubedns_pod_ip)

# 只有在 RunningAsPod 基础事件触发时才执行
@handler.subscribe(PossibleArpSpoofing)
class DnsSpoofHunter(ActiveHunter):
    """DNS Spoof Hunter
    Checks for the possibility for a malicious pod to compromise DNS requests of the cluster
    (results are based on the running node)
    """

    def __init__(self, event):
        # 设置 event 属性
        self.event = event

    # 获取 cbr0 接口的 IP 和 MAC 地址
    def get_cbr0_ip_mac(self):
        res = srp1(Ether() / IP(dst="1.1.1.1", ttl=1) / ICMP(), verbose=0, timeout=config.network_timeout)
        return res[IP].src, res.src

    # 从 /etc/resolv.conf 文件中提取 nameserver 的 IP 地址
    def extract_nameserver_ip(self):
        with open("/etc/resolv.conf") as f:
            # 查找 /etc/resolv.conf 中第一个 nameserver
            match = re.search(r"nameserver (\d+.\d+.\d+.\d+)", f.read())
            if match:
                return match.group(1)
    # 获取 kube-dns 服务的 IP 地址
    def get_kube_dns_ip_mac(self):
        # 从 DNS 服务器中提取 kube-dns 服务的 IP 地址
        kubedns_svc_ip = self.extract_nameserver_ip()

        # 通过比较 DNS 响应的源 MAC 地址和 ARP 扫描来获取 kube-dns 服务的实际 pod IP 地址
        dns_info_res = srp1(
            Ether() / IP(dst=kubedns_svc_ip) / UDP(dport=53) / DNS(rd=1, qd=DNSQR()),
            verbose=0,
            timeout=config.network_timeout,
        )
        kubedns_pod_mac = dns_info_res.src
        self_ip = dns_info_res[IP].dst

        # 执行 ARP 扫描，获取 ARP 响应
        arp_responses, _ = srp(
            Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(op=1, pdst=f"{self_ip}/24"), timeout=config.network_timeout, verbose=0,
        )
        for _, response in arp_responses:
            if response[Ether].src == kubedns_pod_mac:
                return response[ARP].psrc, response.src

    # 执行函数
    def execute(self):
        logger.debug("Attempting to get kube-dns pod ip")
        # 发送 ICMP 包，获取自身 IP 地址
        self_ip = sr1(IP(dst="1.1.1.1", ttl=1) / ICMP(), verbose=0, timeout=config.netork_timeout)[IP].dst
        cbr0_ip, cbr0_mac = self.get_cbr0_ip_mac()

        # 获取 kube-dns 服务的 IP 地址和 MAC 地址
        kubedns = self.get_kube_dns_ip_mac()
        if kubedns:
            kubedns_ip, kubedns_mac = kubedns
            logger.debug(f"ip={self_ip} kubednsip={kubedns_ip} cbr0ip={cbr0_ip}")
            if kubedns_mac != cbr0_mac:
                # 如果自身 pod 与 kube-dns pod 在同一子网中
                self.publish_event(PossibleDnsSpoofing(kubedns_pod_ip=kubedns_ip))
        else:
            logger.debug("Could not get kubedns identity")
```