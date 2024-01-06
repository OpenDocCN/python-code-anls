# `kubehunter\kube_hunter\__main__.py`

```
#!/usr/bin/env python3
# 指定使用 Python3 解释器

import logging
import threading
# 导入日志和线程模块

from kube_hunter.conf import config
from kube_hunter.core.events import handler
from kube_hunter.core.events.types import HuntFinished, HuntStarted
from kube_hunter.modules.discovery.hosts import RunningAsPodEvent, HostScanEvent
from kube_hunter.modules.report import get_reporter, get_dispatcher
# 导入 kube_hunter 相关模块

config.reporter = get_reporter(config.report)
config.dispatcher = get_dispatcher(config.dispatch)
# 设置报告和分发器

logger = logging.getLogger(__name__)
# 获取当前模块的日志记录器

import kube_hunter  # noqa
# 导入 kube_hunter 模块，忽略 Flake8 的警告

def interactive_set_config():
    """Sets config manually, returns True for success"""
    # 手动设置配置，成功返回 True
# 创建一个选项列表，每个选项包括名称和解释
options = [
    ("Remote scanning", "scans one or more specific IPs or DNS names"),
    ("Interface scanning", "scans subnets on all local network interfaces"),
    ("IP range scanning", "scans a given IP range"),
]

# 打印选项列表
print("Choose one of the options below:")
for i, (option, explanation) in enumerate(options):
    print("{}. {} ({})".format(i + 1, option.ljust(20), explanation))
# 用户输入选择
choice = input("Your choice: ")
# 根据用户选择进行相应的操作
if choice == "1":
    # 设置远程扫描的参数
    config.remote = input("Remotes (separated by a ','): ").replace(" ", "").split(",")
elif choice == "2":
    # 设置接口扫描的参数
    config.interface = True
elif choice == "3":
    # 设置IP范围扫描的参数
    config.cidr = (
        input("CIDR separated by a ',' (example - 192.168.0.0/16,!192.168.0.8/32,!192.168.1.0/24): ")
        .replace(" ", "")
        .split(",")
    )
# 如果条件不满足，则返回 False
    else:
        return False
    # 如果条件满足，则返回 True
    return True

# 打印 passvie hunters 的信息
def list_hunters():
    print("\nPassive Hunters:\n----------------")
    # 遍历 passvie hunters 字典，打印每个 hunter 的信息
    for hunter, docs in handler.passive_hunters.items():
        name, doc = hunter.parse_docs(docs)
        print("* {}\n  {}\n".format(name, doc))

    # 如果配置为 active，则打印 active hunters 的信息
    if config.active:
        print("\n\nActive Hunters:\n---------------")
        # 遍历 active hunters 字典，打印每个 hunter 的信息
        for hunter, docs in handler.active_hunters.items():
            name, doc = hunter.parse_docs(docs)
            print("* {}\n  {}\n".format(name, doc))

# 创建全局变量 hunt_started_lock，并初始化为 threading.Lock() 对象
global hunt_started_lock
hunt_started_lock = threading.Lock()
# 定义一个全局变量，表示猎取是否已经开始
hunt_started = False

# 主函数
def main():
    # 声明全局变量
    global hunt_started
    # 定义扫描选项
    scan_options = [config.pod, config.cidr, config.remote, config.interface]
    try:
        # 如果配置中有列表选项，则列出猎取者
        if config.list:
            list_hunters()
            return

        # 如果没有任何扫描选项，则尝试交互式设置配置
        if not any(scan_options):
            if not interactive_set_config():
                return

        # 使用线程锁确保猎取已经开始
        with hunt_started_lock:
            hunt_started = True
        # 发布猎取开始事件
        handler.publish_event(HuntStarted())
        # 如果配置中有 pod 选项，则发布运行作为 pod 事件
        if config.pod:
            handler.publish_event(RunningAsPodEvent())
        else:
            # 如果没有发现漏洞，则发布主机扫描事件
            handler.publish_event(HostScanEvent())

        # 阻塞等待发现输出
        handler.join()
    except KeyboardInterrupt:
        # 用户中断程序时的处理
        logger.debug("Kube-Hunter stopped by user")
    # 在没有交互选项的情况下运行容器时发生
    except EOFError:
        # 提示用户使用 -it 选项重新运行
        logger.error("\033[0;31mPlease run again with -it\033[0m")
    finally:
        # 获取互斥锁
        hunt_started_lock.acquire()
        if hunt_started:
            hunt_started_lock.release()
            # 发布猎手完成事件
            handler.publish_event(HuntFinished())
            handler.join()
            # 释放资源
            handler.free()
            logger.debug("Cleaned Queue")
        else:
            hunt_started_lock.release()
# 如果当前脚本被直接执行而不是被导入，那么执行 main() 函数。
```