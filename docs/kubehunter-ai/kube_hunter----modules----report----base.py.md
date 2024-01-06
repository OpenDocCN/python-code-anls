# `kubehunter\kube_hunter\modules\report\base.py`

```
# 导入所需的模块和类
from kube_hunter.core.types import Discovery
from kube_hunter.modules.report.collector import (
    services,
    vulnerabilities,
    hunters,
    services_lock,
    vulnerabilities_lock,
)

# 定义一个基础的报告类
class BaseReporter(object):
    # 获取节点信息的方法
    def get_nodes(self):
        # 初始化节点列表和节点位置集合
        nodes = list()
        node_locations = set()
        # 使用服务锁定，遍历服务列表
        with services_lock:
            for service in services:
                # 获取服务的主机位置
                node_location = str(service.host)
                # 如果主机位置不在节点位置集合中，将其添加到节点列表中，并将主机位置加入节点位置集合
                if node_location not in node_locations:
                    nodes.append({"type": "Node/Master", "location": node_location})
                    node_locations.add(node_location)
    # 返回节点信息
    return nodes

    # 获取服务信息
    def get_services(self):
        # 使用服务锁保护临界区
        with services_lock:
            # 返回服务列表，包括服务名和位置信息
            return [
                {"service": service.get_name(), "location": f"{service.host}:{service.port}{service.get_path()}"}
                for service in services
            ]

    # 获取漏洞信息
    def get_vulnerabilities(self):
        # 使用漏洞锁保护临界区
        with vulnerabilities_lock:
            # 返回漏洞列表，包括位置、ID、类别、严重程度、漏洞名、描述和证据
            return [
                {
                    "location": vuln.location(),
                    "vid": vuln.get_vid(),
                    "category": vuln.category.name,
                    "severity": vuln.get_severity(),
                    "vulnerability": vuln.get_name(),
                    "description": vuln.explain(),
                    "evidence": str(vuln.evidence),
                }
                for vuln in vulnerabilities
            ]
    # 获取猎人的统计信息
    def get_hunter_statistics(self):
        # 初始化猎人数据列表
        hunters_data = []
        # 遍历猎人和其文档
        for hunter, docs in hunters.items():
            # 如果猎人不是Discovery类的子类
            if Discovery not in hunter.__mro__:
                # 解析文档，获取猎人名称和描述
                name, doc = hunter.parse_docs(docs)
                # 将猎人的统计信息添加到猎人数据列表
                hunters_data.append(
                    {"name": name, "description": doc, "vulnerabilities": hunter.publishedVulnerabilities}
                )
        # 返回猎人数据列表
        return hunters_data

    # 获取报告
    def get_report(self, *, statistics, **kwargs):
        # 初始化报告
        report = {
            "nodes": self.get_nodes(),  # 获取节点信息
            "services": self.get_services(),  # 获取服务信息
            "vulnerabilities": self.get_vulnerabilities(),  # 获取漏洞信息
            "hunters_statistics": self.get_hunter_statistics()  # 获取猎人统计信息
        }
        # 返回报告
        return report
        }

        # 如果需要统计信息，调用 get_hunter_statistics 方法并将结果存入报告中
        if statistics:
            report["hunter_statistics"] = self.get_hunter_statistics()

        # 设置报告中的 kburl 字段为固定链接加上漏洞 ID
        report["kburl"] = "https://aquasecurity.github.io/kube-hunter/kb/{vid}"

        # 返回报告
        return report
```