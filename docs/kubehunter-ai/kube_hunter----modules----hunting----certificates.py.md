# `.\kubehunter\kube_hunter\modules\hunting\certificates.py`

```

# 导入所需的模块
import ssl  # 用于处理 SSL/TLS 连接
import logging  # 用于记录日志
import base64  # 用于处理 base64 编码
import re  # 用于处理正则表达式

# 导入自定义模块
from kube_hunter.core.types import Hunter, KubernetesCluster, InformationDisclosure  # 导入自定义类型
from kube_hunter.core.events import handler  # 导入事件处理器
from kube_hunter.core.events.types import Vulnerability, Event, Service  # 导入自定义事件类型

# 获取日志记录器
logger = logging.getLogger(__name__)

# 定义正则表达式模式，用于匹配邮箱地址
email_pattern = re.compile(r"([a-z0-9]+@[a-z0-9]+\.[a-z0-9]+)")

# 定义一个自定义事件类，表示证书中包含邮箱地址的漏洞
class CertificateEmail(Vulnerability, Event):
    """Certificate includes an email address"""

    def __init__(self, email):
        # 初始化漏洞信息
        Vulnerability.__init__(
            self, KubernetesCluster, "Certificate Includes Email Address", category=InformationDisclosure, khv="KHV021",
        )
        self.email = email  # 保存邮箱地址
        self.evidence = "email: {}".format(self.email)  # 保存证据信息

# 定义一个事件处理器类，用于检测证书中是否包含邮箱地址
@handler.subscribe(Service)
class CertificateDiscovery(Hunter):
    """Certificate Email Hunting
    Checks for email addresses in kubernetes ssl certificates
    """

    def __init__(self, event):
        self.event = event  # 保存事件信息

    def execute(self):
        try:
            logger.debug("Passive hunter is attempting to get server certificate")
            addr = (str(self.event.host), self.event.port)  # 获取服务器地址和端口
            cert = ssl.get_server_certificate(addr)  # 获取服务器证书
        except ssl.SSLError:
            # 如果服务器在该端口上不提供 SSL，则无法获取证书
            return
        c = cert.strip(ssl.PEM_HEADER).strip(ssl.PEM_FOOTER)  # 去除证书头尾部信息
        certdata = base64.decodebytes(c)  # 解码证书数据
        emails = re.findall(email_pattern, certdata)  # 查找证书数据中的邮箱地址
        for email in emails:
            self.publish_event(CertificateEmail(email=email))  # 发布证书中包含邮箱地址的事件

```