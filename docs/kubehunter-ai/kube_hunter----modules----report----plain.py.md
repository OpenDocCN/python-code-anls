# `kubehunter\kube_hunter\modules\report\plain.py`

```
# 从未来导入 print_function，确保代码在 Python 2 和 Python 3 中都能正常工作
from __future__ import print_function

# 导入 PrettyTable 库中的 ALL 和 PrettyTable
from prettytable import ALL, PrettyTable

# 从 kube_hunter.modules.report.base 模块中导入 BaseReporter 类
from kube_hunter.modules.report.base import BaseReporter

# 从 kube_hunter.modules.report.collector 模块中导入以下变量
from kube_hunter.modules.report.collector import (
    services,
    vulnerabilities,
    hunters,
    services_lock,
    vulnerabilities_lock,
)

# 设置常量 EVIDENCE_PREVIEW 为 40
EVIDENCE_PREVIEW = 40

# 设置常量 MAX_TABLE_WIDTH 为 20
MAX_TABLE_WIDTH = 20

# 设置常量 KB_LINK 为指向 GitHub 上 kube-hunter 项目文档的链接
KB_LINK = "https://github.com/aquasecurity/kube-hunter/tree/master/docs/_kb"

# 定义 PlainReporter 类，继承自 BaseReporter 类
class PlainReporter(BaseReporter):
    # 定义 get_report 方法，生成报告表格
    def get_report(self, *, statistics=None, mapping=None, **kwargs):
        """generates report tables"""
        # 初始化输出为空字符串
        output = ""

        # 使用 vulnerabilities_lock 锁定关键部分
        with vulnerabilities_lock:
            # 获取漏洞列表的长度
            vulnerabilities_len = len(vulnerabilities)

        # 获取猎人列表的长度
        hunters_len = len(hunters.items())

        # 使用 services_lock 锁定关键部分
        with services_lock:
            # 获取服务列表的长度
            services_len = len(services)

        # 如果服务列表不为空
        if services_len:
            # 添加节点表格到输出
            output += self.nodes_table()
            # 如果没有映射，则添加服务表格到输出
            if not mapping:
                output += self.services_table()
                # 如果存在漏洞，则添加漏洞表格到输出，否则添加提示信息到输出
                if vulnerabilities_len:
                    output += self.vulns_table()
                else:
                    output += "\nNo vulnerabilities were found"
                # 如果存在统计信息
                if statistics:
                    # 如果存在猎人，则添加猎人表格到输出，否则添加提示信息到输出
                    if hunters_len:
                        output += self.hunters_table()
                    else:
                        output += "\nNo hunters were found"
        # 如果服务列表为空
        else:
            # 如果存在漏洞，则添加漏洞表格到输出
            if vulnerabilities_len:
                output += self.vulns_table()
            # 添加提示信息到输出
            output += "\nKube Hunter couldn't find any clusters"
            # 输出提示信息
            # print("\nKube Hunter couldn't find any clusters. {}".format("Maybe try with --active?" if not config.active else ""))
        # 返回输出
        return output
    # 创建一个表格对象，包含节点类型和位置信息
    nodes_table = PrettyTable(["Type", "Location"], hrules=ALL)
    # 设置表格对齐方式
    nodes_table.align = "l"
    # 设置表格最大宽度
    nodes_table.max_width = MAX_TABLE_WIDTH
    # 设置表格内边距
    nodes_table.padding_width = 1
    # 设置表格排序字段
    nodes_table.sortby = "Type"
    # 设置表格排序方式为逆序
    nodes_table.reversesort = True
    # 设置表格头部样式
    nodes_table.header_style = "upper"
    # 创建一个集合，用于存储事件 ID
    id_memory = set()
    # 获取服务列表的锁
    services_lock.acquire()
    # 遍历服务列表
    for service in services:
        # 如果事件 ID 不在集合中
        if service.event_id not in id_memory:
            # 向节点表格中添加一行数据，包含节点类型和位置信息
            nodes_table.add_row(["Node/Master", service.host])
            # 将事件 ID 添加到集合中
            id_memory.add(service.event_id)
    # 格式化节点表格并返回
    nodes_ret = "\nNodes\n{}\n".format(nodes_table)
    # 释放服务列表的锁
    services_lock.release()
    # 返回节点表格
    return nodes_ret

    # 创建一个表格对象，包含服务名称、位置和描述信息
    services_table = PrettyTable(["Service", "Location", "Description"], hrules=ALL)
    # 设置表格对齐方式
    services_table.align = "l"
    # 设置表格最大宽度
    services_table.max_width = MAX_TABLE_WIDTH
    # 设置表格内边距
    services_table.padding_width = 1
    # 设置表格排序字段
    services_table.sortby = "Service"
    # 设置表格排序方式为逆序
    services_table.reversesort = True
    # 设置表格头部样式
    services_table.header_style = "upper"
    # 获取服务列表的锁
    with services_lock:
        # 遍历服务列表
        for service in services:
            # 向服务表格中添加一行数据，包含服务名称、位置和描述信息
            services_table.add_row(
                [service.get_name(), f"{service.host}:{service.port}{service.get_path()}", service.explain()]
            )
        # 格式化检测到的服务表格并返回
        detected_services_ret = f"\nDetected Services\n{services_table}\n"
    # 返回服务表格
    return detected_services_ret
    # 定义一个方法，用于生成漏洞表格
    def vulns_table(self):
        # 定义漏洞表格的列名
        column_names = [
            "ID",
            "Location",
            "Category",
            "Vulnerability",
            "Description",
            "Evidence",
        ]
        # 创建一个漂亮的表格对象，并设置表格样式
        vuln_table = PrettyTable(column_names, hrules=ALL)
        vuln_table.align = "l"
        vuln_table.max_width = MAX_TABLE_WIDTH
        vuln_table.sortby = "Category"
        vuln_table.reversesort = True
        vuln_table.padding_width = 1
        vuln_table.header_style = "upper"

        # 使用漏洞锁，遍历漏洞列表，填充表格数据
        with vulnerabilities_lock:
            for vuln in vulnerabilities:
                # 处理漏洞的证据，如果超过预览长度则截断
                evidence = str(vuln.evidence)
                if len(evidence) > EVIDENCE_PREVIEW:
                    evidence = evidence[:EVIDENCE_PREVIEW] + "..."
                # 组装表格的一行数据
                row = [
                    vuln.get_vid(),
                    vuln.location(),
                    vuln.category.name,
                    vuln.get_name(),
                    vuln.explain(),
                    evidence,
                ]
                # 将一行数据添加到表格中
                vuln_table.add_row(row)
        # 返回漏洞表格的字符串表示，包括漏洞的相关信息链接
        return (
            "\nVulnerabilities\n"
            "For further information about a vulnerability, search its ID in: \n"
            f"{KB_LINK}\n{vuln_table}\n"
        )

    # 定义一个方法，用于生成猎人统计表格
    def hunters_table(self):
        # 定义猎人统计表格的列名
        column_names = ["Name", "Description", "Vulnerabilities"]
        # 创建一个漂亮的表格对象，并设置表格样式
        hunters_table = PrettyTable(column_names, hrules=ALL)
        hunters_table.align = "l"
        hunters_table.max_width = MAX_TABLE_WIDTH
        hunters_table.sortby = "Name"
        hunters_table.reversesort = True
        hunters_table.padding_width = 1
        hunters_table.header_style = "upper"

        # 获取猎人统计数据，并填充表格数据
        hunter_statistics = self.get_hunter_statistics()
        for item in hunter_statistics:
            # 将猎人统计数据添加到表格中
            hunters_table.add_row([item.get("name"), item.get("description"), item.get("vulnerabilities")])
        # 返回猎人统计表格的字符串表示
        return f"\nHunter Statistics\n{hunters_table}\n"
```