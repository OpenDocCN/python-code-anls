# `kubehunter\kube_hunter\modules\hunting\arp.py`

```
# 导入日志模块
import logging
# 导入 scapy 库中的相关模块
from scapy.all import ARP, IP, ICMP, Ether, sr1, srp
# 导入 kube_hunter.conf 配置模块
from kube_hunter.conf import config
# 导入事件处理模块
from kube_hunter.core.events import handler
# 导入事件类型模块
from kube_hunter.core.events.types import Event, Vulnerability
# 导入核心类型模块
from kube_hunter.core.types import ActiveHunter, KubernetesCluster, IdentityTheft
# 导入网络原始数据处理模块
from kube_hunter.modules.hunting.capabilities import CapNetRawEnabled

# 获取日志记录器
logger = logging.getLogger(__name__)

# 定义可能的 ARP 欺骗漏洞类，继承自漏洞和事件类
class PossibleArpSpoofing(Vulnerability, Event):
    """A malicious pod running on the cluster could potentially run an ARP Spoof attack
    and perform a MITM between pods on the node."""
    # 初始化方法
    def __init__(self):
        # 调用父类的初始化方法
        Vulnerability.__init__(
            self, KubernetesCluster, "Possible Arp Spoof", category=IdentityTheft, vid="KHV020",
        )

# 订阅 CapNetRawEnabled 事件，表示启用原始网络数据包捕获
@handler.subscribe(CapNetRawEnabled)
class ArpSpoofHunter(ActiveHunter):
    """Arp Spoof Hunter
    检查可能在 pod 内运行 ARP 欺骗攻击的可能性（结果基于运行节点）
    """

    def __init__(self, event):
        self.event = event

    def try_getting_mac(self, ip):
        # 尝试获取指定 IP 的 MAC 地址
        ans = sr1(ARP(op=1, pdst=ip), timeout=config.network_timeout, verbose=0)
        return ans[ARP].hwsrc if ans else None

    def detect_l3_on_host(self, arp_responses):
        """ 返回存在 L3 网络插件的情况下为 True """
        logger.debug("Attempting to detect L3 network plugin using ARP")
        # 从 ARP 响应中获取唯一的 MAC 地址列表
        unique_macs = list(set(response[ARP].hwsrc for _, response in arp_responses))

        # 如果局域网地址不唯一
        if len(unique_macs) == 1:
            # 如果一个 IP 地址在子网之外获取了一个 MAC 地址
            outside_mac = self.try_getting_mac("1.1.1.1")
            # 如果外部 MAC 地址与局域网 MAC 地址相同
            if outside_mac == unique_macs[0]:
                return True
        # 只有一个 MAC 地址用于整个局域网和外部
        return False

    def execute(self):
        # 获取本机 IP 地址
        self_ip = sr1(IP(dst="1.1.1.1", ttl=1) / ICMP(), verbose=0, timeout=config.network_timeout)[IP].dst
        # 发送 ARP 请求并接收响应
        arp_responses, _ = srp(
            Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(op=1, pdst=f"{self_ip}/24"), timeout=config.netork_timeout, verbose=0,
        )

        # 如果 ARP 在集群上启用并且节点上有多个 pod
        if len(arp_responses) > 1:
# 如果 L3 插件未安装
if not self.detect_l3_on_host(arp_responses):
    # 发布可能的 ARP 欺骗事件
    self.publish_event(PossibleArpSpoofing())
```