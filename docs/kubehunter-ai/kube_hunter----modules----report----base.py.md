# `kubehunter\kube_hunter\modules\report\base.py`

```
# 导入必要的模块和类
from kube_hunter.core.types import Discovery
from kube_hunter.modules.report.collector import (
    services,
    vulnerabilities,
    hunters,
    services_lock,
    vulnerabilities_lock,
)

# 定义基础报告类
class BaseReporter(object):
    # 获取节点信息
    def get_nodes(self):
        nodes = list()
        node_locations = set()
        # 使用服务锁定，遍历服务列表，获取节点位置信息
        with services_lock:
            for service in services:
                node_location = str(service.host)
                if node_location not in node_locations:
                    nodes.append({"type": "Node/Master", "location": node_location})
                    node_locations.add(node_location)
        return nodes

    # 获取服务信息
    def get_services(self):
        # 使用服务锁定，遍历服务列表，获取服务信息
        with services_lock:
            return [
                {"service": service.get_name(), "location": f"{service.host}:{service.port}{service.get_path()}"}
                for service in services
            ]

    # 获取漏洞信息
    def get_vulnerabilities(self):
        # 使用漏洞锁定，遍历漏洞列表，获取漏洞信息
        with vulnerabilities_lock:
            return [
                {
                    "location": vuln.location(),
                    "vid": vuln.get_vid(),
                    "category": vuln.category.name,
                    "severity": vuln.get_severity(),
                    "vulnerability": vuln.get_name(),
                    "description": vuln.explain(),
                    "evidence": str(vuln.evidence),
                    "hunter": vuln.hunter.get_name(),
                }
                for vuln in vulnerabilities
            ]

    # 获取猎人统计信息
    def get_hunter_statistics(self):
        hunters_data = []
        # 遍历猎人和文档，获取猎人统计信息
        for hunter, docs in hunters.items():
            if Discovery not in hunter.__mro__:
                name, doc = hunter.parse_docs(docs)
                hunters_data.append(
                    {"name": name, "description": doc, "vulnerabilities": hunter.publishedVulnerabilities}
                )
        return hunters_data
    # 定义一个方法，获取报告
    def get_report(self, *, statistics, **kwargs):
        # 创建一个报告字典，包括节点、服务和漏洞信息
        report = {
            "nodes": self.get_nodes(),  # 获取节点信息
            "services": self.get_services(),  # 获取服务信息
            "vulnerabilities": self.get_vulnerabilities(),  # 获取漏洞信息
        }

        # 如果需要统计信息
        if statistics:
            report["hunter_statistics"] = self.get_hunter_statistics()  # 获取猎人统计信息

        # 设置报告中的 KBURL
        report["kburl"] = "https://aquasecurity.github.io/kube-hunter/kb/{vid}"

        # 返回报告
        return report
```