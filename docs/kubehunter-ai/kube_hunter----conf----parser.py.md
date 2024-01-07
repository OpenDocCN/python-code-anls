# `.\kubehunter\kube_hunter\conf\parser.py`

```

# 导入 ArgumentParser 类
from argparse import ArgumentParser

# 解析命令行参数
def parse_args():
    # 创建 ArgumentParser 对象，并设置描述信息
    parser = ArgumentParser(description="kube-hunter - hunt for security weaknesses in Kubernetes clusters")

    # 添加命令行参数，显示所有 kubehunter 中的测试（添加 --active 标志以查看活动测试）
    parser.add_argument("--list", action="store_true", help="Displays all tests in kubehunter (add --active flag to see active tests)")

    # 添加命令行参数，设置在所有网络接口上进行搜索
    parser.add_argument("--interface", action="store_true", help="Set hunting on all network interfaces")

    # 添加命令行参数，将 hunter 设置为内部 pod
    parser.add_argument("--pod", action="store_true", help="Set hunter as an insider pod")

    # 添加命令行参数，优先进行快速扫描（子网 24）
    parser.add_argument("--quick", action="store_true", help="Prefer quick scan (subnet 24)")

    # 添加命令行参数，包括已修补版本在内，不跳过已修补的版本进行扫描
    parser.add_argument("--include-patched-versions", action="store_true", help="Don't skip patched versions when scanning")

    # 添加命令行参数，设置要扫描/忽略的 IP 范围
    parser.add_argument("--cidr", type=str, help="Set an IP range to scan/ignore, example: '192.168.0.0/24,!192.168.0.8/32,!192.168.0.16/32'")

    # 添加命令行参数，仅输出集群节点的映射
    parser.add_argument("--mapping", action="store_true", help="Outputs only a mapping of the cluster's nodes")

    # 添加命令行参数，设置要进行搜索的一个或多个远程 IP/dns
    parser.add_argument("--remote", nargs="+", metavar="HOST", default=list(), help="One or more remote ip/dns to hunt")

    # 添加命令行参数，启用主动搜索
    parser.add_argument("--active", action="store_true", help="Enables active hunting")

    # 添加命令行参数，设置日志级别
    parser.add_argument("--log", type=str, metavar="LOGLEVEL", default="INFO", help="Set log level, options are: debug, info, warn, none")

    # 添加命令行参数，设置报告类型
    parser.add_argument("--report", type=str, default="plain", help="Set report type, options are: plain, yaml, json")

    # 添加命令行参数，设置报告发送位置
    parser.add_argument("--dispatch", type=str, default="stdout", help="Where to send the report to, options are: stdout, http (set KUBEHUNTER_HTTP_DISPATCH_URL and KUBEHUNTER_HTTP_DISPATCH_METHOD environment variables to configure)")

    # 添加命令行参数，显示搜索统计信息
    parser.add_argument("--statistics", action="store_true", help="Show hunting statistics")

    # 添加命令行参数，设置网络操作超时时间
    parser.add_argument("--network-timeout", type=float, default=5.0, help="network operations timeout")

    # 解析命令行参数
    args = parser.parse_args()
    # 如果存在 cidr 参数，则去除空格并以逗号分隔
    if args.cidr:
        args.cidr = args.cidr.replace(" ", "").split(",")
    return args

```