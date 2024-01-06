# `kubehunter\kube_hunter\modules\discovery\kubectl.py`

```
# 导入日志和子进程模块
import logging
import subprocess

# 导入自定义模块
from kube_hunter.core.types import Discovery
from kube_hunter.core.events import handler
from kube_hunter.core.events.types import HuntStarted, Event

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 定义一个名为KubectlClientEvent的事件类，继承自Event类
class KubectlClientEvent(Event):
    """The API server is in charge of all operations on the cluster."""

    # 初始化方法，接受版本参数
    def __init__(self, version):
        self.version = version

    # 返回事件发生的位置
    def location(self):
        return "local machine"
# 在每次猎取开始时触发
@handler.subscribe(HuntStarted)
class KubectlClientDiscovery(Discovery):
    """Kubectl客户端发现
    检查本地是否存在kubectl客户端
    """

    def __init__(self, event):
        self.event = event

    def get_kubectl_binary_version(self):
        version = None
        try:
            # kubectl version --client 不会连接到集群/互联网
            version_info = subprocess.check_output("kubectl version --client", stderr=subprocess.STDOUT)
            if b"GitVersion" in version_info:
                # 从kubectl输出中提取版本
                version_info = version_info.decode()
                start = version_info.find("GitVersion")
                version = version_info[start + len("GitVersion':\"") : version_info.find('",', start)]
# 捕获任何异常并记录日志，说明未找到 kubectl 客户端
        except Exception:
            logger.debug("Could not find kubectl client")
        # 返回版本信息
        return version

    # 执行方法
    def execute(self):
        # 记录调试日志，尝试发现本地 kubectl 客户端
        logger.debug("Attempting to discover a local kubectl client")
        # 获取 kubectl 二进制版本信息
        version = self.get_kubectl_binary_version()
        # 如果获取到版本信息，则发布 kubectl 客户端事件
        if version:
            self.publish_event(KubectlClientEvent(version=version))
```