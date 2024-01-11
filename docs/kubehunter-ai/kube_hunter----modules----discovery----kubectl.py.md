# `kubehunter\kube_hunter\modules\discovery\kubectl.py`

```
# 导入日志和子进程模块
import logging
import subprocess

# 导入自定义类型和事件
from kube_hunter.core.types import Discovery
from kube_hunter.core.events import handler
from kube_hunter.core.events.types import HuntStarted, Event

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


# 定义一个自定义事件类，表示 Kubectl 客户端事件
class KubectlClientEvent(Event):
    """The API server is in charge of all operations on the cluster."""

    def __init__(self, version):
        self.version = version

    def location(self):
        return "local machine"


# 在每次搜索开始时触发
@handler.subscribe(HuntStarted)
class KubectlClientDiscovery(Discovery):
    """Kubectl Client Discovery
    Checks for the existence of a local kubectl client
    """

    def __init__(self, event):
        self.event = event

    # 获取 kubectl 二进制版本信息
    def get_kubectl_binary_version(self):
        version = None
        try:
            # 使用子进程执行命令获取 kubectl 版本信息，不会连接到集群/互联网
            version_info = subprocess.check_output("kubectl version --client", stderr=subprocess.STDOUT)
            if b"GitVersion" in version_info:
                # 从 kubectl 输出中提取版本信息
                version_info = version_info.decode()
                start = version_info.find("GitVersion")
                version = version_info[start + len("GitVersion':\"") : version_info.find('",', start)]
        except Exception:
            logger.debug("Could not find kubectl client")
        return version

    # 执行 kubectl 客户端发现操作
    def execute(self):
        logger.debug("Attempting to discover a local kubectl client")
        version = self.get_kubectl_binary_version()
        if version:
            # 发布 KubectlClientEvent 事件
            self.publish_event(KubectlClientEvent(version=version))
```