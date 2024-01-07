# `.\kubehunter\kube_hunter\modules\hunting\arp.py`

```

# 导入日志模块
import logging
# 导入 scapy 库中的相关模块
from scapy.all import ARP, IP, ICMP, Ether, sr1, srp
# 导入 kube_hunter 中的配置和事件模块
from kube_hunter.conf import config
from kube_hunter.core.events import handler
from kube_hunter.core.events.types import Event, Vulnerability
from kube_hunter.core.types import ActiveHunter, KubernetesCluster, IdentityTheft
# 导入网络原始数据包发送能力模块
from kube_hunter.modules.hunting.capabilities import CapNetRawEnabled

# 获取 logger 对象
logger = logging.getLogger(__name__)

# 定义可能的 ARP 欺骗漏洞类
class PossibleArpSpoofing(Vulnerability, Event):
    """A malicious pod running on the cluster could potentially run an ARP Spoof attack
    and perform a MITM between pods on the node."""
    # 初始化方法
    def __init__(self):
        # 调用父类初始化方法
        Vulnerability.__init__(
            self, KubernetesCluster, "Possible Arp Spoof", category=IdentityTheft, vid="KHV020",
        )

# 订阅 CapNetRawEnabled 事件
@handler.subscribe(CapNetRawEnabled)
# 定义 ARP 欺骗猎手类
class ArpSpoofHunter(ActiveHunter):
    """Arp Spoof Hunter
    Checks for the possibility of running an ARP spoof
    attack from within a pod (results are based on the running node)
    """
    # 初始化方法
    def __init__(self, event):
        self.event = event

    # 尝试获取指定 IP 的 MAC 地址
    def try_getting_mac(self, ip):
        ans = sr1(ARP(op=1, pdst=ip), timeout=config.network_timeout, verbose=0)
        return ans[ARP].hwsrc if ans else None

    # 检测主机上的 L3 网络
    def detect_l3_on_host(self, arp_responses):
        """ returns True for an existence of an L3 network plugin """
        logger.debug("Attempting to detect L3 network plugin using ARP")
        unique_macs = list(set(response[ARP].hwsrc for _, response in arp_responses))

        # 如果 LAN 地址不唯一
        if len(unique_macs) == 1:
            # 如果一个 IP 地址在子网外获取了 MAC 地址
            outside_mac = self.try_getting_mac("1.1.1.1")
            # 如果外部 MAC 地址与 LAN MAC 地址相同
            if outside_mac == unique_macs[0]:
                return True
        # 只有一个 MAC 地址用于整个 LAN 和外部
        return False

    # 执行方法
    def execute(self):
        self_ip = sr1(IP(dst="1.1.1.1", ttl=1) / ICMP(), verbose=0, timeout=config.network_timeout)[IP].dst
        arp_responses, _ = srp(
            Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(op=1, pdst=f"{self_ip}/24"), timeout=config.netork_timeout, verbose=0,
        )

        # 如果 ARP 在集群中启用并且节点上有多个 pod
        if len(arp_responses) > 1:
            # 如果主机上未安装 L3 插件
            if not self.detect_l3_on_host(arp_responses):
                self.publish_event(PossibleArpSpoofing())

```