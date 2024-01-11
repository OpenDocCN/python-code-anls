# `ZeroNet\src\Db\DbQuery.py`

```
# 导入正则表达式模块
import re

# 解析和修改 SQL 查询
class DbQuery:
    # 初始化方法，去除查询的首尾空格
    def __init__(self, query):
        self.setQuery(query.strip())

    # 分割查询的主要部分
    def parseParts(self, query):
        parts = re.split("(SELECT|FROM|WHERE|ORDER BY|LIMIT)", query)
        parts = [_f for _f in parts if _f]  # 移除空的部分
        parts = [s.strip() for s in parts]  # 移除空格
        return dict(list(zip(parts[0::2], parts[1::2])))

    # 解析选择的字段 SELECT ... FROM
    def parseFields(self, query_select):
        fields = re.findall("([^,]+) AS ([^,]+)", query_select)
        return {key: val.strip() for val, key in fields}

    # 解析查询条件 WHERE ...
    def parseWheres(self, query_where):
        if " AND " in query_where:
            return query_where.split(" AND ")
        elif query_where:
            return [query_where]
        else:
            return []

    # 设置查询
    def setQuery(self, query):
        self.parts = self.parseParts(query)
        self.fields = self.parseFields(self.parts["SELECT"])
        self.wheres = self.parseWheres(self.parts.get("WHERE", ""))

    # 将查询转换为字符串
    def __str__(self):
        query_parts = []
        for part_name in ["SELECT", "FROM", "WHERE", "ORDER BY", "LIMIT"]:
            if part_name == "WHERE" and self.wheres:
                query_parts.append("WHERE")
                query_parts.append(" AND ".join(self.wheres))
            elif part_name in self.parts:
                query_parts.append(part_name)
                query_parts.append(self.parts[part_name])
        return "\n".join(query_parts)
```