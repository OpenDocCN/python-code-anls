# `kubehunter\kube_hunter\modules\report\plain.py`

```
# 导入未来的打印函数，以便在 Python 2 和 Python 3 中使用相同的打印函数
from __future__ import print_function

# 导入 PrettyTable 库中的 ALL 和 PrettyTable 对象
from prettytable import ALL, PrettyTable

# 导入基础报告类 BaseReporter 和相关的模块
from kube_hunter.modules.report.base import BaseReporter
from kube_hunter.modules.report.collector import (
    services,
    vulnerabilities,
    hunters,
    services_lock,
    vulnerabilities_lock,
)

# 设置预览证据的长度和最大表格宽度
EVIDENCE_PREVIEW = 40
MAX_TABLE_WIDTH = 20

# 设置知识库链接
KB_LINK = "https://github.com/aquasecurity/kube-hunter/tree/master/docs/_kb"

# 定义 PlainReporter 类，继承自 BaseReporter
class PlainReporter(BaseReporter):
    # 定义获取报告的方法，接受统计数据和映射数据作为参数
    def get_report(self, *, statistics=None, mapping=None, **kwargs):
# 生成报告表格
output = ""

# 使用漏洞锁定，获取漏洞列表的长度
with vulnerabilities_lock:
    vulnerabilities_len = len(vulnerabilities)

# 获取猎人列表的长度
hunters_len = len(hunters.items())

# 使用服务锁定，获取服务列表的长度
with services_lock:
    services_len = len(services)

# 如果服务列表不为空，生成节点表格
if services_len:
    output += self.nodes_table()
    # 如果没有映射，生成服务表格
    if not mapping:
        output += self.services_table()
        # 如果存在漏洞，生成漏洞表格；否则输出“未发现漏洞”
        if vulnerabilities_len:
            output += self.vulns_table()
        else:
            output += "\nNo vulnerabilities were found"
        # 如果需要统计信息
        if statistics:
    # 如果找到了 hunters，则将 hunters_table 的输出添加到 output 中
    if hunters_len:
        output += self.hunters_table()
    # 如果未找到 hunters，则将提示信息添加到 output 中
    else:
        output += "\nNo hunters were found"

# 如果未找到 clusters，则将 vulnerabilities_table 的输出添加到 output 中，并添加提示信息
else:
    if vulnerabilities_len:
        output += self.vulns_table()
    output += "\nKube Hunter couldn't find any clusters"
    # 如果未找到 clusters，则打印提示信息
    # print("\nKube Hunter couldn't find any clusters. {}".format("Maybe try with --active?" if not config.active else ""))

# 返回最终的 output
return output

# 创建 nodes_table 函数
def nodes_table(self):
    # 创建一个 PrettyTable 对象，指定列名和样式
    nodes_table = PrettyTable(["Type", "Location"], hrules=ALL)
    nodes_table.align = "l"
    nodes_table.max_width = MAX_TABLE_WIDTH
    nodes_table.padding_width = 1
    nodes_table.sortby = "Type"
    nodes_table.reversesort = True
    nodes_table.header_style = "upper"
    # 创建一个集合用于存储 id 和 memory
    id_memory = set()
# 获取服务锁，防止多线程访问冲突
services_lock.acquire()
# 遍历服务列表
for service in services:
    # 如果服务的事件ID不在内存中
    if service.event_id not in id_memory:
        # 向节点表中添加一行记录，包括节点/主机和服务的主机
        nodes_table.add_row(["Node/Master", service.host])
        # 将服务的事件ID添加到内存中
        id_memory.add(service.event_id)
# 生成节点表的字符串表示
nodes_ret = "\nNodes\n{}\n".format(nodes_table)
# 释放服务锁
services_lock.release()
# 返回节点表的字符串表示
return nodes_ret

# 定义服务表格的方法
def services_table(self):
    # 创建漂亮的表格对象，包括服务、位置和描述
    services_table = PrettyTable(["Service", "Location", "Description"], hrules=ALL)
    # 设置表格对齐方式
    services_table.align = "l"
    # 设置表格最大宽度
    services_table.max_width = MAX_TABLE_WIDTH
    # 设置表格内边距宽度
    services_table.padding_width = 1
    # 按照服务名称排序
    services_table.sortby = "Service"
    # 设置表格排序方式为倒序
    services_table.reversesort = True
    # 设置表头样式为大写
    services_table.header_style = "upper"
    # 获取服务锁，防止多线程访问冲突
    with services_lock:
        # 遍历服务列表
        for service in services:
            # 向服务表格中添加一行记录，包括服务、位置和描述
            services_table.add_row(
    # 创建一个包含服务信息的表格
    services_table = PrettyTable(
        ["Name", "Location", "Explanation"],
        hrules=ALL
    )
    # 设置表格对齐方式
    services_table.align = "l"
    # 设置表格最大宽度
    services_table.max_width = MAX_TABLE_WIDTH
    # 遍历服务列表，将每个服务的名称、位置和解释添加到表格中
    for service in self.services:
        services_table.add_row(
            [service.get_name(), f"{service.host}:{service.port}{service.get_path()}", service.explain()]
        )
    # 生成包含检测到的服务信息的字符串
    detected_services_ret = f"\nDetected Services\n{services_table}\n"
    # 返回检测到的服务信息字符串
    return detected_services_ret

# 创建漏洞信息表格
def vulns_table(self):
    # 定义表格列名
    column_names = [
        "ID",
        "Location",
        "Category",
        "Vulnerability",
        "Description",
        "Evidence",
    ]
    # 创建漏洞信息表格
    vuln_table = PrettyTable(column_names, hrules=ALL)
    # 设置表格对齐方式
    vuln_table.align = "l"
    # 设置表格最大宽度
    vuln_table.max_width = MAX_TABLE_WIDTH
    # 设置表格按照"Category"列排序
    vuln_table.sortby = "Category"
    # 设置表格排序方式为逆序
    vuln_table.reversesort = True
    # 设置表格内边距
    vuln_table.padding_width = 1
# 设置漏洞表的标题样式为大写
vuln_table.header_style = "upper"

# 使用漏洞锁定，遍历漏洞列表
with vulnerabilities_lock:
    for vuln in vulnerabilities:
        # 将漏洞的证据转换为字符串，并进行长度限制处理
        evidence = str(vuln.evidence)
        if len(evidence) > EVIDENCE_PREVIEW:
            evidence = evidence[:EVIDENCE_PREVIEW] + "..."
        # 创建包含漏洞信息的行数据
        row = [
            vuln.get_vid(),
            vuln.location(),
            vuln.category.name,
            vuln.get_name(),
            vuln.explain(),
            evidence,
        ]
        # 将行数据添加到漏洞表中
        vuln_table.add_row(row)
# 返回包含漏洞信息的字符串
return (
    "\nVulnerabilities\n"
    "For further information about a vulnerability, search its ID in: \n"
    f"{KB_LINK}\n{vuln_table}\n"
# 定义一个方法用于生成猎人统计表
def hunters_table(self):
    # 定义表格的列名
    column_names = ["Name", "Description", "Vulnerabilities"]
    # 创建一个漂亮的表格对象，指定列名和水平线样式
    hunters_table = PrettyTable(column_names, hrules=ALL)
    # 设置表格对齐方式
    hunters_table.align = "l"
    # 设置表格最大宽度
    hunters_table.max_width = MAX_TABLE_WIDTH
    # 指定按照 "Name" 列进行排序
    hunters_table.sortby = "Name"
    # 设置排序方式为逆序
    hunters_table.reversesort = True
    # 设置表格的内边距宽度
    hunters_table.padding_width = 1
    # 设置表头的样式为大写
    hunters_table.header_style = "upper"

    # 获取猎人统计数据
    hunter_statistics = self.get_hunter_statistics()
    # 遍历统计数据，将每个猎人的信息添加到表格中
    for item in hunter_statistics:
        hunters_table.add_row([item.get("name"), item.get("description"), item.get("vulnerabilities")])
    # 返回包含猎人统计表的字符串
    return f"\nHunter Statistics\n{hunters_table}\n"
```