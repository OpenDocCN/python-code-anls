# `kubehunter\kube_hunter\modules\hunting\certificates.py`

```
# 导入所需的模块
import ssl
import logging
import base64
import re

# 从 kube_hunter.core.types 模块中导入 Hunter, KubernetesCluster, InformationDisclosure 类
from kube_hunter.core.types import Hunter, KubernetesCluster, InformationDisclosure
# 从 kube_hunter.core.events 模块中导入 handler
from kube_hunter.core.events import handler
# 从 kube_hunter.core.events.types 模块中导入 Vulnerability, Event, Service 类
from kube_hunter.core.events.types import Vulnerability, Event, Service

# 获取 logger 对象
logger = logging.getLogger(__name__)
# 定义邮箱地址的正则表达式模式
email_pattern = re.compile(r"([a-z0-9]+@[a-z0-9]+\.[a-z0-9]+)")


# 定义 CertificateEmail 类，继承自 Vulnerability 和 Event 类
class CertificateEmail(Vulnerability, Event):
    """Certificate includes an email address"""

    def __init__(self, email):
        # 调用父类的构造函数
        Vulnerability.__init__(
            self, KubernetesCluster, "Certificate Includes Email Address", category=InformationDisclosure, khv="KHV021",
        )
        # 设置邮箱地址
        self.email = email
        # 设置证据信息
        self.evidence = "email: {}".format(self.email)


# 使用 handler.subscribe 装饰器注册 Service 类
@handler.subscribe(Service)
# 定义 CertificateDiscovery 类，继承自 Hunter 类
class CertificateDiscovery(Hunter):
    """Certificate Email Hunting
    Checks for email addresses in kubernetes ssl certificates
    """

    def __init__(self, event):
        # 初始化函数，接收 event 参数
        self.event = event

    def execute(self):
        try:
            # 尝试获取服务器证书
            logger.debug("Passive hunter is attempting to get server certificate")
            addr = (str(self.event.host), self.event.port)
            cert = ssl.get_server_certificate(addr)
        except ssl.SSLError:
            # 如果服务器在该端口上不提供 SSL，则不会得到证书
            return
        # 去除证书的头部和尾部，并解码为字节数据
        c = cert.strip(ssl.PEM_HEADER).strip(ssl.PEM_FOOTER)
        certdata = base64.decodebytes(c)
        # 使用正则表达式在证书数据中查找邮箱地址
        emails = re.findall(email_pattern, certdata)
        # 遍历找到的邮箱地址，发布 CertificateEmail 事件
        for email in emails:
            self.publish_event(CertificateEmail(email=email))
```