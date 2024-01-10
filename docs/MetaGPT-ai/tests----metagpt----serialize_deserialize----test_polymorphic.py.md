# `MetaGPT\tests\metagpt\serialize_deserialize\test_polymorphic.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : unittest of polymorphic conditions
# 导入需要的模块

from pydantic import BaseModel, ConfigDict, SerializeAsAny
# 从metagpt.actions模块中导入Action类
from metagpt.actions import Action
# 从tests.metagpt.serialize_deserialize.test_serdeser_base模块中导入ActionOKV2和ActionPass类

class ActionSubClasses(BaseModel):
    # 定义包含多个Action对象的列表，使用SerializeAsAny进行序列化
    actions: list[SerializeAsAny[Action]] = []

class ActionSubClassesNoSAA(BaseModel):
    """without SerializeAsAny"""
    # 定义模型配置，允许任意类型
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # 定义包含多个Action对象的列表，不使用SerializeAsAny进行序列化
    actions: list[Action] = []

def test_serialize_as_any():
    """test subclasses of action with different fields in ser&deser"""
    # 测试包含不同字段的Action子类的序列化和反序列化
    # 创建ActionSubClasses对象，包含ActionOKV2和ActionPass对象
    action_subcls = ActionSubClasses(actions=[ActionOKV2(), ActionPass()])
    # 将对象转换为字典
    action_subcls_dict = action_subcls.model_dump()
    # 断言包含额外字段的ActionOKV2对象的extra_field字段
    assert action_subcls_dict["actions"][0]["extra_field"] == ActionOKV2().extra_field

def test_no_serialize_as_any():
    # 测试不使用SerializeAsAny进行序列化的情况
    # 创建ActionSubClassesNoSAA对象，包含ActionOKV2和ActionPass对象
    action_subcls = ActionSubClassesNoSAA(actions=[ActionOKV2(), ActionPass()])
    # 将对象转换为字典
    action_subcls_dict = action_subcls.model_dump()
    # 断言不包含额外字段的ActionOKV2对象的extra_field字段
    assert "extra_field" not in action_subcls_dict["actions"][0]

def test_polymorphic():
    # 测试多态性
    # 创建ActionOKV2对象
    _ = ActionOKV2(
        **{"name": "ActionOKV2", "context": "", "prefix": "", "desc": "", "extra_field": "ActionOKV2 Extra Info"}
    )
    # 创建ActionSubClasses对象，包含ActionOKV2和ActionPass对象
    action_subcls = ActionSubClasses(actions=[ActionOKV2(), ActionPass()])
    # 将对象转换为字典
    action_subcls_dict = action_subcls.model_dump()
    # 断言序列化后的字典包含__module_class_name字段
    assert "__module_class_name" in action_subcls_dict["actions"][0]
    # 创建新的ActionSubClasses对象，从序列化后的字典中反序列化
    new_action_subcls = ActionSubClasses(**action_subcls_dict)
    # 断言新对象的第一个元素是ActionOKV2类的实例
    assert isinstance(new_action_subcls.actions[0], ActionOKV2)
    # 断言新对象的第二个元素是ActionPass类的实例
    assert isinstance(new_action_subcls.actions[1], ActionPass)
    # 通过model_validate方法创建新的ActionSubClasses对象
    new_action_subcls = ActionSubClasses.model_validate(action_subcls_dict)
    # 断言新对象的第一个元素是ActionOKV2类的实例
    assert isinstance(new_action_subcls.actions[0], ActionOKV2)
    # 断言新对象的第二个元素是ActionPass类的实例
    assert isinstance(new_action_subcls.actions[1], ActionPass)

```