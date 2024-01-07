# `.\kubehunter\kube_hunter\__main__.py`

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
# 获取日志记录器

import kube_hunter  # noqa
# 导入 kube_hunter 模块

def interactive_set_config():
    """Sets config manually, returns True for success"""
    # 手动设置配置，成功返回 True
    options = [
        ("Remote scanning", "scans one or more specific IPs or DNS names"),
        ("Interface scanning", "scans subnets on all local network interfaces"),
        ("IP range scanning", "scans a given IP range"),
    ]
    # 选项列表

    print("Choose one of the options below:")
    for i, (option, explanation) in enumerate(options):
        print("{}. {} ({})".format(i + 1, option.ljust(20), explanation))
    # 打印选项

    choice = input("Your choice: ")
    # 获取用户选择
    if choice == "1":
        config.remote = input("Remotes (separated by a ','): ").replace(" ", "").split(",")
    elif choice == "2":
        config.interface = True
    elif choice == "3":
        config.cidr = (
            input("CIDR separated by a ',' (example - 192.168.0.0/16,!192.168.0.8/32,!192.168.1.0/24): ")
            .replace(" ", "")
            .split(",")
        )
    else:
        return False
    return True
    # 根据用户选择设置配置

def list_hunters():
    print("\nPassive Hunters:\n----------------")
    for hunter, docs in handler.passive_hunters.items():
        name, doc = hunter.parse_docs(docs)
        print("* {}\n  {}\n".format(name, doc))
    # 打印被动猎手列表

    if config.active:
        print("\n\nActive Hunters:\n---------------")
        for hunter, docs in handler.active_hunters.items():
            name, doc = hunter.parse_docs(docs)
            print("* {}\n  {}\n".format(name, doc))
    # 如果配置为活动状态，打印活动猎手列表

global hunt_started_lock
hunt_started_lock = threading.Lock()
hunt_started = False
# 定义全局变量

def main():
    global hunt_started
    scan_options = [config.pod, config.cidr, config.remote, config.interface]
    # 获取扫描选项
    try:
        if config.list:
            list_hunters()
            return
        # 如果配置为列表，打印猎手列表并返回

        if not any(scan_options):
            if not interactive_set_config():
                return
        # 如果没有扫描选项，且用户未手动设置配置，则交互式设置配置

        with hunt_started_lock:
            hunt_started = True
        handler.publish_event(HuntStarted())
        if config.pod:
            handler.publish_event(RunningAsPodEvent())
        else:
            handler.publish_event(HostScanEvent())
        # 发布事件

        # Blocking to see discovery output
        handler.join()
    except KeyboardInterrupt:
        logger.debug("Kube-Hunter stopped by user")
    # 捕获键盘中断异常
    except EOFError:
        logger.error("\033[0;31mPlease run again with -it\033[0m")
    # 捕获 EOF 异常
    finally:
        hunt_started_lock.acquire()
        if hunt_started:
            hunt_started_lock.release()
            handler.publish_event(HuntFinished())
            handler.join()
            handler.free()
            logger.debug("Cleaned Queue")
        else:
            hunt_started_lock.release()
    # 最终处理

if __name__ == "__main__":
    main()
# 如果是主程序入口，则执行 main() 函数

```