# `kubehunter\kube_hunter\__main__.py`

```py
#!/usr/bin/env python3

import logging  # 导入日志模块
import threading  # 导入线程模块

from kube_hunter.conf import config  # 从 kube_hunter.conf 模块导入 config 配置
from kube_hunter.core.events import handler  # 从 kube_hunter.core.events 模块导入 handler 事件处理器
from kube_hunter.core.events.types import HuntFinished, HuntStarted  # 从 kube_hunter.core.events.types 模块导入 HuntFinished 和 HuntStarted 事件类型
from kube_hunter.modules.discovery.hosts import RunningAsPodEvent, HostScanEvent  # 从 kube_hunter.modules.discovery.hosts 模块导入 RunningAsPodEvent 和 HostScanEvent
from kube_hunter.modules.report import get_reporter, get_dispatcher  # 从 kube_hunter.modules.report 模块导入 get_reporter 和 get_dispatcher

config.reporter = get_reporter(config.report)  # 设置配置中的报告器
config.dispatcher = get_dispatcher(config.dispatch)  # 设置配置中的分发器
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

import kube_hunter  # noqa


def interactive_set_config():
    """Sets config manually, returns True for success"""
    options = [  # 定义选项列表
        ("Remote scanning", "scans one or more specific IPs or DNS names"),  # 远程扫描选项
        ("Interface scanning", "scans subnets on all local network interfaces"),  # 接口扫描选项
        ("IP range scanning", "scans a given IP range"),  # IP范围扫描选项
    ]

    print("Choose one of the options below:")  # 打印选择提示
    for i, (option, explanation) in enumerate(options):  # 遍历选项列表
        print("{}. {} ({})".format(i + 1, option.ljust(20), explanation))  # 打印选项和解释
    choice = input("Your choice: ")  # 获取用户选择
    if choice == "1":  # 如果选择了1
        config.remote = input("Remotes (separated by a ','): ").replace(" ", "").split(",")  # 设置远程扫描的目标
    elif choice == "2":  # 如果选择了2
        config.interface = True  # 设置为接口扫描
    elif choice == "3":  # 如果选择了3
        config.cidr = (
            input("CIDR separated by a ',' (example - 192.168.0.0/16,!192.168.0.8/32,!192.168.1.0/24): ")
            .replace(" ", "")
            .split(",")  # 设置CIDR范围
        )
    else:  # 如果选择了其他
        return False  # 返回失败
    return True  # 返回成功


def list_hunters():
    print("\nPassive Hunters:\n----------------")  # 打印 passvie hunters 标题
    for hunter, docs in handler.passive_hunters.items():  # 遍历 passvie hunters
        name, doc = hunter.parse_docs(docs)  # 解析文档
        print("* {}\n  {}\n".format(name, doc))  # 打印名称和文档

    if config.active:  # 如果配置为 active
        print("\n\nActive Hunters:\n---------------")  # 打印 active hunters 标题
        for hunter, docs in handler.active_hunters.items():  # 遍历 active hunters
            name, doc = hunter.parse_docs(docs)  # 解析文档
            print("* {}\n  {}\n".format(name, doc))  # 打印名称和文档


global hunt_started_lock  # 定义全局变量 hunt_started_lock
# 创建一个线程锁，用于控制 hunt_started 变量的访问
hunt_started_lock = threading.Lock()
# 初始化 hunt_started 变量为 False
hunt_started = False

# 主函数
def main():
    global hunt_started
    # 将配置选项存储在列表中
    scan_options = [config.pod, config.cidr, config.remote, config.interface]
    try:
        # 如果配置中包含 list 选项，则列出所有的 hunters 并返回
        if config.list:
            list_hunters()
            return

        # 如果没有任何扫描选项被设置，并且交互式设置配置失败，则返回
        if not any(scan_options):
            if not interactive_set_config():
                return

        # 使用 hunt_started_lock 锁定代码块，设置 hunt_started 为 True
        with hunt_started_lock:
            hunt_started = True
        # 发布 HuntStarted 事件
        handler.publish_event(HuntStarted())
        # 如果配置中包含 pod 选项，则发布 RunningAsPodEvent 事件，否则发布 HostScanEvent 事件
        if config.pod:
            handler.publish_event(RunningAsPodEvent())
        else:
            handler.publish_event(HostScanEvent())

        # 阻塞代码，等待发现输出
        handler.join()
    except KeyboardInterrupt:
        logger.debug("Kube-Hunter stopped by user")
    # 在运行容器时没有使用交互式选项时发生
    except EOFError:
        logger.error("\033[0;31mPlease run again with -it\033[0m")
    finally:
        # 获取 hunt_started_lock 锁
        hunt_started_lock.acquire()
        # 如果 hunt_started 为 True，则释放 hunt_started_lock 锁，发布 HuntFinished 事件，清理队列，并打印日志
        if hunt_started:
            hunt_started_lock.release()
            handler.publish_event(HuntFinished())
            handler.join()
            handler.free()
            logger.debug("Cleaned Queue")
        # 如果 hunt_started 为 False，则释放 hunt_started_lock 锁
        else:
            hunt_started_lock.release()

# 如果当前脚本被直接执行，则调用 main 函数
if __name__ == "__main__":
    main()
```