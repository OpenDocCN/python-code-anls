# `kubehunter\kube_hunter\modules\hunting\arp.py`

```
# 导入 logging 模块
import logging
# 从 scapy.all 模块中导入所需的类和函数
from scapy.all import ARP, IP, ICMP, Ether, sr1, srp
# 从 kube_hunter.conf 模块中导入 config 对象
from kube_hunter.conf import config
# 从 kube_hunter.core.events 模块中导入 handler 函数
from kube_hunter.core.events import handler
# 从 kube_hunter.core.events.types 模块中导入 Event 和 Vulnerability 类
from kube_hunter.core.events.types import Event, Vulnerability
# 从 kube_hunter.core.types 模块中导入 ActiveHunter, KubernetesCluster, IdentityTheft 类
from kube_hunter.core.types import ActiveHunter, KubernetesCluster, IdentityTheft
# 从 kube_hunter.modules.hunting.capabilities 模块中导入 CapNetRawEnabled 类
from kube_hunter.modules.hunting.capabilities import CapNetRawEnabled

# 获取 logger 对象
logger = logging.getLogger(__name__)

# 定义 PossibleArpSpoofing 类，继承自 Vulnerability 和 Event 类
class PossibleArpSpoofing(Vulnerability, Event):
    """A malicious pod running on the cluster could potentially run an ARP Spoof attack
    and perform a MITM between pods on the node."""
    # 初始化方法
    def __init__(self):
        # 调用父类的初始化方法
        Vulnerability.__init__(
            self, KubernetesCluster, "Possible Arp Spoof", category=IdentityTheft, vid="KHV020",
        )

# 使用 handler.subscribe 装饰器注册 CapNetRawEnabled 事件
@handler.subscribe(CapNetRawEnabled)
# 定义 ArpSpoofHunter 类，继承自 ActiveHunter 类
class ArpSpoofHunter(ActiveHunter):
    """Arp Spoof Hunter
    Checks for the possibility of running an ARP spoof
    attack from within a pod (results are based on the running node)
    """
    # 初始化方法
    def __init__(self, event):
        # 保存传入的 event 参数
        self.event = event

    # 定义 try_getting_mac 方法，用于获取指定 IP 对应的 MAC 地址
    def try_getting_mac(self, ip):
        # 发送 ARP 请求并等待响应
        ans = sr1(ARP(op=1, pdst=ip), timeout=config.network_timeout, verbose=0)
        # 如果有响应，则返回对应的 MAC 地址
        return ans[ARP].hwsrc if ans else None

    # 定义 detect_l3_on_host 方法，用于检测主机上是否存在 L3 网络插件
    def detect_l3_on_host(self, arp_responses):
        """ returns True for an existence of an L3 network plugin """
        # 打印调试信息
        logger.debug("Attempting to detect L3 network plugin using ARP")
        # 获取所有 ARP 响应中的唯一 MAC 地址列表
        unique_macs = list(set(response[ARP].hwsrc for _, response in arp_responses))

        # 如果 LAN 地址不唯一
        if len(unique_macs) == 1:
            # 如果外部 IP 地址获取到了 MAC 地址
            outside_mac = self.try_getting_mac("1.1.1.1")
            # 如果外部 MAC 地址与 LAN MAC 地址相同
            if outside_mac == unique_macs[0]:
                return True
        # 如果整个 LAN 只有一个 MAC 地址，并且外部也是相同的 MAC 地址
        return False
    # 执行函数，用于执行一系列网络操作
    def execute(self):
        # 发送 ICMP 报文到目标 IP 地址，获取自身 IP 地址
        self_ip = sr1(IP(dst="1.1.1.1", ttl=1) / ICMP(), verbose=0, timeout=config.network_timeout)[IP].dst
        # 发送 ARP 请求，获取 ARP 响应
        arp_responses, _ = srp(
            Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(op=1, pdst=f"{self_ip}/24"), timeout=config.netork_timeout, verbose=0,
        )

        # 如果收到多个 ARP 响应
        if len(arp_responses) > 1:
            # 如果主机上未安装 L3 插件
            if not self.detect_l3_on_host(arp_responses):
                # 发布可能的 ARP 欺骗事件
                self.publish_event(PossibleArpSpoofing())
```