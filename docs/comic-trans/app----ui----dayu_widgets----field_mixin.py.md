# `.\comic-translate\app\ui\dayu_widgets\field_mixin.py`

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
###################################################################
# Author: Mu yanru
# Date  : 2018.9
# Email : muyanru345@163.com
###################################################################
# Import future modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import built-in modules
import functools


class MFieldMixin(object):
    computed_dict = None
    props_dict = None

    # 注册字段及其相关信息
    def register_field(self, name, getter=None, setter=None, required=False):
        # 如果计算字段字典为None，则初始化为空字典
        if self.computed_dict is None:
            self.computed_dict = {}
        # 如果属性字典为None，则初始化为空字典
        if self.props_dict is None:
            self.props_dict = {}
        
        # 如果getter是可调用的，则获取值并将相关信息存入计算字段字典中
        if callable(getter):
            value = getter()
            self.computed_dict[name] = {
                "value": value,
                "getter": getter,
                "setter": setter,
                "required": required,
                "bind": [],
            }
        else:
            # 否则将属性值及其相关信息存入属性字典中
            self.props_dict[name] = {"value": getter, "require": required, "bind": []}
        return

    # 将数据名、部件、信号等信息绑定到字段
    def bind(self, data_name, widget, qt_property, index=None, signal=None, callback=None):
        data_dict = {
            "data_name": data_name,
            "widget": widget,
            "widget_property": qt_property,
            "index": index,
            "callback": callback,
        }
        # 如果数据名存在于计算字段字典中，则将数据绑定信息添加到对应的列表中
        if data_name in self.computed_dict:
            self.computed_dict[data_name]["bind"].append(data_dict)
        else:
            # 否则将数据绑定信息添加到属性字典中
            self.props_dict[data_name]["bind"].append(data_dict)
        
        # 如果有信号，则连接到用户操作改变数据的槽函数
        if signal:
            getattr(widget, signal).connect(functools.partial(self._slot_changed_from_user, data_dict))
        
        # 更新UI以反映数据的当前状态
        self._data_update_ui(data_dict)
        return widget

    # 返回所有字段的名称列表
    def fields(self):
        return list(self.props_dict.keys()) + list(self.computed_dict.keys())

    # 返回指定字段的值，如果是计算字段则获取其getter返回的值
    def field(self, name):
        if name in self.props_dict:
            return self.props_dict[name]["value"]
        elif name in self.computed_dict:
            new_value = self.computed_dict[name]["getter"]()
            self.computed_dict[name]["value"] = new_value
            return new_value
        else:
            raise KeyError('There is no field named "{}"'.format(name))

    # 设置指定字段的值，如果是属性字段则触发属性变化的槽函数
    def set_field(self, name, value):
        if name in self.props_dict:
            self.props_dict[name]["value"] = value
            self._slot_prop_changed(name)
        elif name in self.computed_dict:
            self.computed_dict[name]["value"] = value
    # 更新 UI 元素的数据显示，根据传入的数据字典更新指定的 UI 元素
    def _data_update_ui(self, data_dict):
        # 从数据字典中获取数据名称
        data_name = data_dict.get("data_name")
        # 获取需要更新的 UI 控件对象
        widget = data_dict["widget"]
        # 获取数据在 UI 中的索引（如果有）
        index = data_dict["index"]
        # 获取需要更新的 UI 控件的属性名
        widget_property = data_dict["widget_property"]
        # 获取数据更新后的回调函数
        callback = data_dict["callback"]
        
        value = None
        # 根据索引情况获取数据值
        if index is None:
            value = self.field(data_name)
        elif isinstance(self.field(data_name), dict):
            value = self.field(data_name).get(index)
        elif isinstance(self.field(data_name), list):
            # 如果索引超出列表长度，则返回 None
            value = self.field(data_name)[index] if index < len(self.field(data_name)) else None
        
        # 检查 UI 控件是否有指定的属性名或动态属性名
        if widget.metaObject().indexOfProperty(widget_property) > -1 or widget_property in list(
            map(str, [b.data().decode() for b in widget.dynamicPropertyNames()])
        ):
            # 如果有，则设置 UI 控件的属性值为数据值
            widget.setProperty(widget_property, value)
        else:
            # 否则，通过自定义方法设置 UI 控件的属性值为数据值
            widget.set_field(widget_property, value)
        
        # 如果回调函数可调用，则执行回调函数
        if callable(callback):
            callback()

    # 处理属性变化的槽函数，根据属性名更新绑定的 UI 元素
    def _slot_prop_changed(self, property_name):
        # 遍历属性字典中的设置项
        for key, setting_dict in self.props_dict.items():
            # 如果属性名匹配当前处理的属性名
            if key == property_name:
                # 遍历与该属性绑定的数据字典列表，逐个更新 UI 元素
                for data_dict in setting_dict["bind"]:
                    self._data_update_ui(data_dict)

        # 遍历计算属性字典中的设置项
        for key, setting_dict in self.computed_dict.items():
            # 遍历与计算属性绑定的数据字典列表，逐个更新 UI 元素
            for data_dict in setting_dict["bind"]:
                self._data_update_ui(data_dict)

    # 用户操作导致数据变化的槽函数，更新 UI 元素显示的数据
    def _slot_changed_from_user(self, data_dict, ui_value):
        # 调用 UI 更新数据方法，更新指定数据在 UI 上的显示
        self._ui_update_data(data_dict, ui_value)

    # 更新 UI 元素显示数据的方法，根据用户界面传入的数据更新指定数据
    def _ui_update_data(self, data_dict, ui_value):
        # 获取数据名称和可能存在的索引
        data_name = data_dict.get("data_name")
        index = data_dict.get("index", None)
        
        # 根据索引情况更新数据值
        if index is None:
            # 如果没有索引，则直接设置数据字段的值为界面传入的值
            self.set_field(data_name, ui_value)
        else:
            # 如果有索引，则获取原始数据值，并更新索引位置的值为界面传入的值
            old_value = self.field(data_name)
            old_value[index] = ui_value
            self.set_field(data_name, old_value)
        
        # 如果数据名存在于属性字典中，则调用属性变化的槽函数进行进一步处理
        if data_name in self.props_dict.items():
            self._slot_prop_changed(data_name)

    # 检查是否完成数据填充的方法，检查所有必填字段是否已经填充
    def _is_complete(self):
        # 遍历计算属性字典和属性字典中的设置项
        for name, data_dict in self.computed_dict.items() + self.props_dict.items():
            # 如果当前数据项标记为必填，并且对应的字段数据为空，则返回 False
            if data_dict["required"]:
                if not self.field(name):
                    return False
        # 如果所有必填字段都已经填充，则返回 True
        return True
```