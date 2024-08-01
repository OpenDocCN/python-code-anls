# `.\DB-GPT-src\dbgpt\app\scene\chat_db\auto_execute\example.py`

```py
# 导入必要的模块和类
from dbgpt.core._private.example_base import ExampleSelector, ExampleType

# 默认定义了两个示例
EXAMPLES = [
    {
        # 第一个示例包含两条消息
        "messages": [
            # 人类发送的消息，询问关于用户'test1'所在城市的信息
            {"type": "human", "data": {"content": "查询用户test1所在的城市", "example": True}},
            {
                # AI 的回复，包含了思考过程和 SQL 查询语句
                "type": "ai",
                "data": {
                    "content": """{\n\"thoughts\": \"直接查询用户表中用户名为'test1'的记录即可\",\n\"sql\": \"SELECT city FROM user where user_name='test1'\"}""",
                    "example": True,
                },
            },
        ]
    },
    {
        # 第二个示例也包含两条消息
        "messages": [
            # 人类发送的消息，询问成都用户的订单信息
            {"type": "human", "data": {"content": "查询成都的用户的订单信息", "example": True}},
            {
                # AI 的回复，包含了思考过程和复杂的 SQL 查询语句
                "type": "ai",
                "data": {
                    "content": """{\n\"thoughts\": \"根据订单表的用户名和用户表的用户名关联用户表和订单表，再通过用户表的城市为'成都'的过滤即可\",\n\"sql\": \"SELECT b.* FROM user a  LEFT JOIN tran_order b ON a.user_name=b.user_name  where a.city='成都'\"}""",
                    "example": True,
                },
            },
        ]
    },
]

# 创建一个示例选择器对象，用于选择示例和指定类型
sql_data_example = ExampleSelector(
    examples_record=EXAMPLES, use_example=True, type=ExampleType.ONE_SHOT.value
)
```