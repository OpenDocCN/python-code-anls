# `MetaGPT\tests\metagpt\serialize_deserialize\test_reasearcher.py`

```

# 设置文件编码为 utf-8
# @Desc    : 用于描述测试用例

# 导入 pytest 模块
import pytest

# 从 metagpt.actions 模块中导入 CollectLinks 类
# 从 metagpt.roles.researcher 模块中导入 Researcher 类
from metagpt.actions import CollectLinks
from metagpt.roles.researcher import Researcher

# 标记该测试函数为异步函数
@pytest.mark.asyncio
async def test_tutorial_assistant_deserialize():
    # 创建 Researcher 对象
    role = Researcher()
    # 将 role 对象序列化为字典
    ser_role_dict = role.model_dump()
    # 断言字典中包含 "name" 键
    assert "name" in ser_role_dict
    # 断言字典中包含 "language" 键
    assert "language" in ser_role_dict

    # 根据序列化的字典创建新的 Researcher 对象
    new_role = Researcher(**ser_role_dict)
    # 断言新对象的 language 属性为 "en-us"
    assert new_role.language == "en-us"
    # 断言新对象的 actions 列表长度为 3
    assert len(new_role.actions) == 3
    # 断言新对象的 actions 列表中的第一个元素为 CollectLinks 类的实例
    assert isinstance(new_role.actions[0], CollectLinks)

    # todo: 需要测试不同的action失败下，记忆是否正常保存
    # 待办事项：需要测试不同的 action 失败下，记忆是否正常保存

```