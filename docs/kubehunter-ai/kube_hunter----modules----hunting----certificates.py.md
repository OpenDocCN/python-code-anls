# `kubehunter\kube_hunter\modules\hunting\certificates.py`

```
# 导入所需的模块
import ssl  # 用于处理 SSL/TLS 相关操作
import logging  # 用于记录日志
import base64  # 用于进行 base64 编解码
import re  # 用于进行正则表达式匹配

# 导入自定义模块
from kube_hunter.core.types import Hunter, KubernetesCluster, InformationDisclosure  # 导入自定义类型
from kube_hunter.core.events import handler  # 导入事件处理器
from kube_hunter.core.events.types import Vulnerability, Event, Service  # 导入自定义事件和服务类型

# 获取日志记录器
logger = logging.getLogger(__name__)

# 定义正则表达式模式，用于匹配邮箱地址
email_pattern = re.compile(r"([a-z0-9]+@[a-z0-9]+\.[a-z0-9]+)")

# 定义一个证书包含邮箱地址的漏洞类，继承自 Vulnerability 和 Event 类
class CertificateEmail(Vulnerability, Event):
    """Certificate includes an email address"""

    # 初始化方法，接受邮箱地址作为参数
    def __init__(self, email):
        # 调用父类的初始化方法，设置漏洞类型、描述和分类
        Vulnerability.__init__(
            self, KubernetesCluster, "Certificate Includes Email Address", category=InformationDisclosure, khv="KHV021",
        )
# 设置对象的 email 属性为传入的 email 地址
self.email = email
# 设置对象的 evidence 属性为包含 email 地址的字符串
self.evidence = "email: {}".format(self.email)

# 订阅 Service 事件的处理器
@handler.subscribe(Service)
class CertificateDiscovery(Hunter):
    """Certificate Email Hunting
    Checks for email addresses in kubernetes ssl certificates
    """

    # 初始化方法，接收 event 参数
    def __init__(self, event):
        self.event = event

    # 执行方法
    def execute(self):
        try:
            # 记录调试信息
            logger.debug("Passive hunter is attempting to get server certificate")
            # 获取服务器证书
            addr = (str(self.event.host), self.event.port)
            cert = ssl.get_server_certificate(addr)
        except ssl.SSLError:
            # 如果服务器在该端口上不提供 SSL，则不会得到证书
# 返回空值，结束函数的执行
return

# 去除证书中的头部和尾部信息，并去除空格
c = cert.strip(ssl.PEM_HEADER).strip(ssl.PEM_FOOTER)

# 将证书数据进行 base64 解码
certdata = base64.decodebytes(c)

# 使用正则表达式找出证书数据中的电子邮件地址
emails = re.findall(email_pattern, certdata)

# 遍历找到的电子邮件地址，并发布证书电子邮件事件
for email in emails:
    self.publish_event(CertificateEmail(email=email))
```