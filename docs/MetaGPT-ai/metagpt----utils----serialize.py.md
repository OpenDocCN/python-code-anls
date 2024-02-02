# `MetaGPT\metagpt\utils\serialize.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : the implement of serialization and deserialization

import copy  # 导入copy模块，用于深拷贝对象
import pickle  # 导入pickle模块，用于序列化和反序列化对象

from metagpt.utils.common import import_class  # 从metagpt.utils.common模块中导入import_class函数

# 将schema转换为映射
def actionoutout_schema_to_mapping(schema: dict) -> dict:
    """
    直接遍历第一级的`properties`。
    schema结构如下
    ```
    {
        "title":"prd",
        "type":"object",
        "properties":{
            "Original Requirements":{
                "title":"Original Requirements",
                "type":"string"
            },
        },
        "required":[
            "Original Requirements",
        ]
    }
    """
    mapping = dict()  # 创建一个空字典
    for field, property in schema["properties"].items():  # 遍历schema中的properties字段
        if property["type"] == "string":  # 如果属性类型为字符串
            mapping[field] = (str, ...)  # 将字段名和类型信息添加到映射中
        elif property["type"] == "array" and property["items"]["type"] == "string":  # 如果属性类型为数组且元素类型为字符串
            mapping[field] = (list[str], ...)  # 将字段名和类型信息添加到映射中
        elif property["type"] == "array" and property["items"]["type"] == "array":  # 如果属性类型为数组且元素类型为数组
            # 这里只考虑`list[list[str]]`的情况
            mapping[field] = (list[list[str]], ...)  # 将字段名和类型信息添加到映射中
    return mapping  # 返回映射

# 将映射转换为字符串
def actionoutput_mapping_to_str(mapping: dict) -> dict:
    new_mapping = {}  # 创建一个空字典
    for key, value in mapping.items():  # 遍历映射中的键值对
        new_mapping[key] = str(value)  # 将键值对转换为字符串并添加到新字典中
    return new_mapping  # 返回新字典

# 将字符串转换为映射
def actionoutput_str_to_mapping(mapping: dict) -> dict:
    new_mapping = {}  # 创建一个空字典
    for key, value in mapping.items():  # 遍历映射中的键值对
        if value == "(<class 'str'>, Ellipsis)":  # 如果值为指定字符串
            new_mapping[key] = (str, ...)  # 将键值对添加到新字典中
        else:
            new_mapping[key] = eval(value)  # 将字符串转换为对应的类型信息，并添加到新字典中
    return new_mapping  # 返回新字典

# 序列化消息
def serialize_message(message: "Message"):
    message_cp = copy.deepcopy(message)  # 深拷贝消息对象，避免引用更新
    ic = message_cp.instruct_content  # 获取消息对象的instruct_content属性
    if ic:  # 如果instruct_content存在
        schema = ic.model_json_schema()  # 获取instruct_content的模型JSON schema
        mapping = actionoutout_schema_to_mapping(schema)  # 将schema转换为映射

        message_cp.instruct_content = {"class": schema["title"], "mapping": mapping, "value": ic.model_dump()}  # 更新instruct_content属性
    msg_ser = pickle.dumps(message_cp)  # 序列化消息对象

    return msg_ser  # 返回序列化后的消息对象

# 反序列化消息
def deserialize_message(message_ser: str) -> "Message":
    message = pickle.loads(message_ser)  # 反序列化消息对象
    if message.instruct_content:  # 如果消息对象的instruct_content存在
        ic = message.instruct_content  # 获取消息对象的instruct_content属性
        actionnode_class = import_class("ActionNode", "metagpt.actions.action_node")  # 导入ActionNode类
        ic_obj = actionnode_class.create_model_class(class_name=ic["class"], mapping=ic["mapping"])  # 创建模型类
        ic_new = ic_obj(**ic["value"])  # 使用value参数创建新的instruct_content对象
        message.instruct_content = ic_new  # 更新消息对象的instruct_content属性

    return message  # 返回反序列化后的消息对象

```