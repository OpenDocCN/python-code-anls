# `ZeroNet\src\Test\TestDbQuery.py`

```py
# 导入正则表达式模块
import re

# 从DbQuery模块中导入DbQuery类
from Db.DbQuery import DbQuery

# 定义测试类TestDbQuery
class TestDbQuery:
    # 定义测试方法testParse
    def testParse(self):
        # 定义查询文本
        query_text = """
            SELECT
             'comment' AS type,
             date_added, post.title AS title,
             keyvalue.value || ': ' || comment.body AS body,
             '?Post:' || comment.post_id || '#Comments' AS url
            FROM
             comment
             LEFT JOIN json USING (json_id)
             LEFT JOIN json AS json_content ON (json_content.directory = json.directory AND json_content.file_name='content.json')
             LEFT JOIN keyvalue ON (keyvalue.json_id = json_content.json_id AND key = 'cert_user_id')
             LEFT JOIN post ON (comment.post_id = post.post_id)
            WHERE
             post.date_added > 123
            ORDER BY
             date_added DESC
            LIMIT 20
        """
        # 创建DbQuery对象，传入查询文本
        query = DbQuery(query_text)
        # 断言查询结果中LIMIT部分为"20"
        assert query.parts["LIMIT"] == "20"
        # 断言查询结果中字段body的值为"keyvalue.value || ': ' || comment.body"
        assert query.fields["body"] == "keyvalue.value || ': ' || comment.body"
        # 断言查询结果转换成字符串后去除空格和换行符后与原查询文本相同
        assert re.sub("[ \r\n]", "", str(query)) == re.sub("[ \r\n]", "", query_text)
        # 向查询条件中添加新的条件"body LIKE '%hello%'"
        query.wheres.append("body LIKE '%hello%'")
        # 断言新添加的条件在查询结果中
        assert "body LIKE '%hello%'" in str(query)
```