# `.\comic-translate\app\ui\dayu_widgets\utils.py`

```py
# -*- coding: utf-8 -*-
# 定义文件编码格式为 utf-8
####################################################################
# Author: Mu yanru
# Date  : 2018.5
# Email : muyanru345@163.com
####################################################################
# 作者信息和联系方式
"""
Some helper functions for handling color and formatter.
"""
# 用于处理颜色和格式的一些辅助函数
# Import future modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# 导入未来支持的模块

# Import built-in modules
import collections
import datetime as dt
import functools
import math
import os
# 导入内置模块

# Import third-party modules
from PySide6 import QtCore
from PySide6 import QtGui
from PySide6 import QtWidgets
import six
# 导入第三方模块

if hasattr(functools, "singledispatch"):
    # Import built-in modules
    from functools import singledispatch
else:
    from singledispatch import singledispatch
# 如果 functools 模块中有 singledispatch，则导入，否则从 singledispatch 模块导入

# Import local modules
from . import CUSTOM_STATIC_FOLDERS
from . import DEFAULT_STATIC_FOLDER
from .qt import MIcon
from .qt import get_scale_factor
# 导入本地模块

ItemViewMenuEvent = collections.namedtuple("ItemViewMenuEvent", ["view", "selection", "extra"])
# 定义一个命名元组 ItemViewMenuEvent

def get_static_file(path):
    """
    A convenient function to get the file in ./static,
    User just give the name of the file.
    eg. get_static_file('add_line.svg')
    :param path: file name
    :return: if input file found, return the full path, else return None
    """
    # 获取 static 文件夹中的文件的便捷函数
    # 用户只需提供文件名
    # 例如 get_static_file('add_line.svg')
    if not isinstance(path, six.string_types):
        raise TypeError("Input argument 'path' should be six.string_types type, " "but get {}".format(type(path)))
    # 如果输入的 path 不是字符串类型，抛出类型错误异常
    full_path = next(
        (
            os.path.join(prefix, path)
            for prefix in ["", DEFAULT_STATIC_FOLDER] + CUSTOM_STATIC_FOLDERS
            if os.path.isfile(os.path.join(prefix, path))
        ),
        path,
    )
    # 找到文件路径的函数，首先从默认路径，然后从自定义路径查找
    if os.path.isfile(full_path):
        return full_path
    return None
# 获取静态文件的函数

def from_list_to_nested_dict(input_arg, sep="/"):
    """
    A help function to convert the list of string to nested dict
    :param input_arg: a list/tuple/set of string
    :param sep: a separator to split input string
    :return: a list of nested dict
    """
    # 将字符串列表转换为嵌套字典的辅助函数
    # input_arg: 字符串列表/元组/集合
    # sep: 用于分割输入字符串的分隔符
    # 返回值: 嵌套字典的列表
    if not isinstance(input_arg, (list, tuple, set)):
        raise TypeError("Input argument 'input' should be list or tuple or set, " "but get {}".format(type(input_arg)))
    # 如果输入参数不是列表、元组或集合类型，抛出类型错误异常
    if not isinstance(sep, six.string_types):
        raise TypeError("Input argument 'sep' should be six.string_types, " "but get {}".format(type(sep)))
    # 如果分隔符不是字符串类型，抛出类型错误异常

    result = []
    # 初始化空列表 result
    for item in input_arg:
        components = item.strip(sep).split(sep)
        # 对每个项目进行处理，通过分隔符分割为组件
        component_count = len(components)
        current = result
        for i, comp in enumerate(components):
            atom = next((x for x in current if x["value"] == comp), None)
            # 查找当前组件在当前列表中是否存在，不存在则为 None
            if atom is None:
                atom = {"value": comp, "label": comp, "children": []}
                current.append(atom)
            current = atom["children"]
            if i == component_count - 1:
                atom.pop("children")
    return result
# 将列表转换为嵌套字典的函数
# 根据给定的颜色和透明度生成对应的 RGBA 格式的颜色字符串
def fade_color(color, alpha):
    q_color = QtGui.QColor(color)  # 将输入的十六进制颜色转换为 QColor 对象
    return "rgba({}, {}, {}, {})".format(q_color.red(), q_color.green(), q_color.blue(), alpha)


# 根据 Ant Design 颜色系统算法生成颜色
def generate_color(primary_color, index):
    hue_step = 2
    saturation_step = 16
    saturation_step2 = 5
    brightness_step1 = 5
    brightness_step2 = 15
    light_color_count = 5
    dark_color_count = 4

    def _get_hue(color, i, is_light):
        h_comp = color.hue()  # 获取颜色的色调
        if 60 <= h_comp <= 240:
            hue = h_comp - hue_step * i if is_light else h_comp + hue_step * i
        else:
            hue = h_comp + hue_step * i if is_light else h_comp - hue_step * i
        if hue < 0:
            hue += 359
        elif hue >= 359:
            hue -= 359
        return hue / 359.0

    def _get_saturation(color, i, is_light):
        s_comp = color.saturationF() * 100  # 获取颜色的饱和度
        if is_light:
            saturation = s_comp - saturation_step * i
        elif i == dark_color_count:
            saturation = s_comp + saturation_step
        else:
            saturation = s_comp + saturation_step2 * i
        saturation = min(100.0, saturation)
        if is_light and i == light_color_count and saturation > 10:
            saturation = 10
        saturation = max(6.0, saturation)
        return round(saturation * 10) / 1000.0

    def _get_value(color, i, is_light):
        v_comp = color.valueF()  # 获取颜色的亮度值
        if is_light:
            return min((v_comp * 100 + brightness_step1 * i) / 100, 1.0)
        return max((v_comp * 100 - brightness_step2 * i) / 100, 0.0)

    light = index <= 6
    hsv_color = QtGui.QColor(primary_color) if isinstance(primary_color, six.string_types) else primary_color
    index = light_color_count + 1 - index if light else index - light_color_count - 1
    # 根据色调、饱和度、亮度计算并返回 QColor 对象的颜色名称
    return QtGui.QColor.fromHsvF(
        _get_hue(hsv_color, index, light),
        _get_saturation(hsv_color, index, light),
        _get_value(hsv_color, index, light),
    ).name()


# 根据给定的模型对象返回源模型
@singledispatch
def real_model(source_model):
    """
    Get the source model whenever user give a source index or proxy index or proxy model.
    """
    return source_model


# 注册 QtCore.QSortFilterProxyModel 类型的 real_model 函数
@real_model.register(QtCore.QSortFilterProxyModel)
def _(proxy_model):
    return proxy_model.sourceModel()


# 注册 QtCore.QModelIndex 类型的 real_model 函数
@real_model.register(QtCore.QModelIndex)
def _(index):
    return real_model(index.model())
# 获取真实的索引，当用户提供源索引或代理索引时使用
def real_index(index):
    # 获取索引所属的模型对象
    model = index.model()
    # 如果模型是 QSortFilterProxyModel 类型，则将索引映射到源索引并返回
    if isinstance(model, QtCore.QSortFilterProxyModel):
        return model.mapToSource(index)
    # 否则直接返回索引本身
    return index


# 获取字典或对象的指定属性值，支持默认值
def get_obj_value(data_obj, attr, default=None):
    # 如果数据对象是字典，则尝试获取指定属性的值，获取不到返回默认值
    if isinstance(data_obj, dict):
        return data_obj.get(attr, default)
    # 如果数据对象是对象，则尝试获取指定属性的值，获取不到返回默认值
    return getattr(data_obj, attr, default)


# 设置字典或对象的指定属性值
def set_obj_value(data_obj, attr, value):
    # 如果数据对象是字典，则更新或添加指定的键值对
    if isinstance(data_obj, dict):
        return data_obj.update({attr: value})
    # 如果数据对象是对象，则设置指定属性的值
    return setattr(data_obj, attr, value)


# 检查字典是否包含指定键或对象是否具有指定属性
def has_obj_value(data_obj, attr):
    # 如果数据对象是字典，则判断指定的键是否存在于字典中
    if isinstance(data_obj, dict):
        return attr in data_obj.keys()
    # 如果数据对象是对象，则判断对象是否具有指定的属性
    return hasattr(data_obj, attr)


# 应用格式化器对数据进行格式化
def apply_formatter(formatter, *args, **kwargs):
    # 如果格式化器为 None，则直接返回第一个参数（通常是数据本身）
    if formatter is None:  # 压根就没有配置
        return args[0]
    # 如果格式化器是字典，则根据第一个参数作为键获取对应的值
    elif isinstance(formatter, dict):  # 字典选项型配置
        return formatter.get(args[0], None)
    # 如果格式化器是可调用对象，则调用该对象对数据进行格式化
    elif callable(formatter):  # 回调函数型配置
        return formatter(*args, **kwargs)
    # 否则直接返回格式化器的值（通常是一个直接的值型配置）
    # 这种情况一般不太常见，因为 formatter 可能是各种类型
    return formatter


# 用于处理各种数据类型的显示格式化器，特别用于 Qt.DisplayRole 的数据显示
@singledispatch
def display_formatter(input_other_type):
    # 将任意类型的输入值转换为字符串类型并返回
    return str(input_other_type)  # this function never reached


# 处理字典类型的数据显示格式化
@display_formatter.register(dict)
def _(input_dict):
    # 如果字典中包含 'name' 键，则对 'name' 对应的值进行递归格式化
    if "name" in input_dict.keys():
        return display_formatter(input_dict.get("name"))
    # 如果字典中包含 'code' 键，则对 'code' 对应的值进行递归格式化
    elif "code" in input_dict.keys():
        return display_formatter(input_dict.get("code"))
    # 否则将整个字典转换为字符串并返回
    return str(input_dict)


# 处理列表类型的数据显示格式化
@display_formatter.register(list)
def _(input_list):
    result = []
    # 对列表中的每个元素递归进行格式化，并以逗号分隔的字符串形式返回
    for i in input_list:
        result.append(str(display_formatter(i)))
    return ",".join(result)


# 处理字符串类型的数据显示格式化
@display_formatter.register(str)
def _(input_str):
    # 将输入字符串按 'windows-1252' 编码解码为 Unicode 字符串并返回
    return input_str.decode("windows-1252")
    # return obj.decode()


# 处理 Unicode 文本类型的数据显示格式化
@display_formatter.register(six.text_type)
def _(input_unicode):
    return input_unicode


# 处理 None 类型的数据显示格式化
@display_formatter.register(type(None))
def _(input_none):
    return "--"


# 处理整数类型的数据显示格式化
@display_formatter.register(int)
def _(input_int):
    # 直接返回整数值，不影响该列的排序
    return input_int


# 处理浮点数类型的数据显示格式化
@display_formatter.register(float)
def _(input_float):
    # 将浮点数保留两位小数后以字符串形式返回
    return "{:.2f}".format(round(input_float, 2))


# 处理其他对象类型的数据显示格式化
@display_formatter.register(object)
def _(input_object):
    # 没有具体的处理方式，可能需要根据实际情况扩展
    pass
    # 检查 input_object 对象是否有名为 "name" 的属性
    if hasattr(input_object, "name"):
        # 如果有 "name" 属性，则调用 display_formatter 函数处理其值并返回结果
        return display_formatter(getattr(input_object, "name"))
    
    # 检查 input_object 对象是否有名为 "code" 的属性
    if hasattr(input_object, "code"):
        # 如果有 "code" 属性，则调用 display_formatter 函数处理其值并返回结果
        return display_formatter(getattr(input_object, "code"))
    
    # 如果 input_object 对象既没有 "name" 属性也没有 "code" 属性，
    # 则将 input_object 转换成字符串并返回
    return str(input_object)
# 注册一个显示格式化器，用于处理 datetime 类型的数据
@display_formatter.register(dt.datetime)
def _(input_datetime):
    # 将输入的 datetime 对象格式化为指定格式的字符串
    return input_datetime.strftime("%Y-%m-%d %H:%M:%S")


# 定义一个字体格式化函数，用于生成 QFont 实例，用于 Qt 的字体角色
def font_formatter(setting_dict):
    """
    用于 QAbstractItemModel 的数据方法，用于 Qt.FontRole
    :param underline: 字体是否有下划线
    :param bold: 字体是否加粗
    :return: 一个具有给定样式的 QFont 实例
    """
    _font = QtGui.QFont()
    # 设置是否有下划线，如果未指定，默认为 False
    _font.setUnderline(setting_dict.get("underline") or False)
    # 设置是否加粗，如果未指定，默认为 False
    _font.setBold(setting_dict.get("bold") or False)
    return _font


# 单分派函数，用于处理不同类型的输入，生成 QIcon 实例
@singledispatch
def icon_formatter(input_other_type):
    """
    用于 QAbstractItemModel 的数据方法，用于 Qt.DecorationRole
    一个获取 QIcon 的辅助函数。
    输入可以是 dict/object, string, None, tuple(file_path, fill_color)
    :param input_other_type: 输入参数
    :return: 一个 QIcon 实例
    """
    return input_other_type  # 这个函数实际上不会被调用


# 注册处理 dict 类型输入的具体实现
@icon_formatter.register(dict)
def _(input_dict):
    # 尝试获取字典中的 "icon" 属性作为路径，并调用 icon_formatter 处理
    attr_list = ["icon"]
    path = next((get_obj_value(input_dict, attr) for attr in attr_list), None)
    return icon_formatter(path)


# 注册处理 QtGui.QIcon 类型输入的具体实现
@icon_formatter.register(QtGui.QIcon)
def _(input_dict):
    return input_dict


# 注册处理 object 类型输入的具体实现
@icon_formatter.register(object)
def _(input_object):
    # 尝试获取对象中的 "icon" 属性作为路径，并调用 icon_formatter 处理
    attr_list = ["icon"]
    path = next((get_obj_value(input_object, attr) for attr in attr_list), None)
    return icon_formatter(path)


# 注册处理 str 类型输入的具体实现
@icon_formatter.register(str)
def _(input_string):
    # 使用 MIcon 类处理字符串路径，并返回处理结果
    return MIcon(input_string)


# 注册处理 tuple 类型输入的具体实现
@icon_formatter.register(tuple)
def _(input_tuple):
    # 使用 MIcon 类处理元组（文件路径，填充颜色），并返回处理结果
    return MIcon(*input_tuple)


# 注册处理 type(None) 类型输入的具体实现
@icon_formatter.register(type(None))
def _(input_none):
    # 当输入为 None 时，默认返回指定的图标路径进行处理
    return icon_formatter("confirm_fill.svg")


# 溢出格式化函数，给定一个整数，返回相应的字符串表示
def overflow_format(num, overflow):
    """
    给定一个整数，返回相应的字符串。
    当该整数大于给定的溢出值时，返回 "溢出值+"
    """
    if not isinstance(num, int):
        raise ValueError("输入参数 'num' 应为整数类型，但得到 {}".format(type(num)))
    if not isinstance(overflow, int):
        raise ValueError("输入参数 'overflow' 应为整数类型，但得到 {}".format(type(overflow)))
    return str(num) if num <= overflow else "{}+".format(overflow)


# 获取给定值在范围内的百分比
def get_percent(value, minimum, maximum):
    """
    获取给定值在范围内的百分比。
    :param value: 值
    :param minimum: 范围的最小值
    :param maximum: 范围的最大值
    :return: 百分比浮点数
    """
    if minimum == maximum:
        # 引用自 qprogressbar.cpp
        # 如果最大值和最小值相等，并且到达此步骤，意味着进度条只有一个步骤，我们在这里返回 100%
        # 以避免进一步下面的除零错误。
        return 100
    return max(0, min(100, (value - minimum) * 100 / (maximum - minimum)))


# 获取总页数
def get_total_page(total, per):
    """
    获取总页数。
    :param total: 总数量
    :param per: 每页数量
    :return: 页数整数
    """
    return int(math.ceil(1.0 * total / per))
# 定义函数，生成用于显示当前页面内容的字符串，格式为 x - x of xx
# 根据参数 current（当前页）、per（每页数量）、total（总数）计算起始和结束位置，并返回格式化后的字符串
def get_page_display_string(current, per, total):
    return "{start} - {end} of {total}".format(
        start=((current - 1) * per + 1) if current else 0,  # 计算起始位置，如果当前页为零则从零开始
        end=min(total, current * per),  # 计算结束位置，不能超过总数
        total=total,  # 总数
    )


# 定义函数，读取指定组织和应用的设置信息
# 使用 QtCore.QSettings 类来读取 INI 格式的设置，属于用户范围
# 将所有的键值对存储在 result_dict 字典中返回
def read_settings(organization, app_name):
    settings = QtCore.QSettings(
        QtCore.QSettings.IniFormat,  # 使用 INI 格式存储
        QtCore.QSettings.UserScope,  # 用户范围
        organization,  # 组织名称
        app_name,  # 应用名称
    )
    # 通过遍历 settings 对象的 childKeys() 方法获取所有键，存储在 result_dict 中
    result_dict = {key: settings.value(key) for key in settings.childKeys()}
    # 遍历 settings 对象的 childGroups() 方法获取所有组名
    for grp_name in settings.childGroups():
        settings.beginGroup(grp_name)  # 开始处理指定组名的设置
        # 更新 result_dict 字典，将组名与键值对应的形式存储进去
        result_dict.update({grp_name + "/" + key: settings.value(key) for key in settings.childKeys()})
        settings.endGroup()  # 结束对当前组名的处理
    return result_dict  # 返回存储所有设置信息的字典


# 定义函数，添加指定组织和应用的设置信息，并绑定事件
# 内部定义了两个函数：_write_settings 和 trigger_event
def add_settings(organization, app_name, event_name="closeEvent"):
    # 内部函数，用于将窗口的属性写入设置
    def _write_settings(self):
        settings = QtCore.QSettings(
            QtCore.QSettings.IniFormat,  # 使用 INI 格式存储
            QtCore.QSettings.UserScope,  # 用户范围
            organization,  # 组织名称
            app_name,  # 应用名称
        )
        # 遍历绑定数据列表 self._bind_data，根据属性值的不同分别保存
        for attr, widget, property in self._bind_data:
            if property == "geometry":  # 如果是窗口位置大小属性
                settings.setValue(attr, widget.saveGeometry())  # 保存窗口位置大小
            elif property == "state":  # 如果是窗口状态属性
                settings.setValue(attr, widget.saveState())  # 保存窗口状态
            else:
                settings.setValue(attr, widget.property(property))  # 其他情况保存属性值

    # 内部函数，触发指定事件（一般是 closeEvent 或 hideEvent），并写入设置
    def trigger_event(self, event):
        # 提前保存设置
        self.write_settings()  # 调用内部函数 _write_settings 保存设置
        old_event = getattr(self, "old_trigger_event")  # 获取之前保存的旧事件处理方法
        return old_event(event)  # 调用旧事件处理方法处理当前事件

    # 内部函数，绑定设置属性到窗口部件
    def bind(self, attr, widget, property, default=None, formatter=None):
        old_setting_dict = read_settings(organization, app_name)  # 获取之前保存的设置字典
        value = old_setting_dict.get(attr, default)  # 根据属性获取设置的默认值或之前保存的值
        if callable(formatter):  # 如果指定了格式化函数，则对值进行格式化处理
            value = formatter(value)  # 使用格式化函数对值进行处理
        if property == "geometry":  # 如果是窗口位置大小属性
            if isinstance(value, QtCore.QRect):  # 如果值是 QRect 类型
                widget.setGeometry(value)  # 设置窗口位置大小
            elif isinstance(value, QtCore.QByteArray):  # 如果值是 QByteArray 类型
                widget.restoreGeometry(value)  # 恢复窗口位置大小
        elif property == "state":  # 如果是窗口状态属性
            if isinstance(value, QtCore.QByteArray):  # 如果值是 QByteArray 类型
                widget.restoreState(value)  # 恢复窗口状态
        else:
            widget.setProperty(property, value)  # 设置其他属性的值
        self._bind_data.append((attr, widget, property))  # 将绑定的属性、窗口部件、属性类型存入绑定数据列表中

    # 内部函数，解除绑定设置属性与窗口部件
    def unbind(self, attr, widget, property):
        self.write_settings()  # 调用内部函数 _write_settings 保存设置
        self._bind_data.remove((attr, widget, property))  # 从绑定数据列表中移除指定的属性、窗口部件、属性类型
    def wrapper(cls):
        # 定义装饰器函数 wrapper，接受一个类 cls 作为参数

        # 将 bind 函数绑定到类 cls 上
        cls.bind = bind
        # 将 unbind 函数绑定到类 cls 上
        cls.unbind = unbind
        # 将 _write_settings 函数绑定到类 cls 上
        cls.write_settings = _write_settings
        # 初始化一个空列表 _bind_data，绑定到类 cls 上
        cls._bind_data = []
        
        # 如果类 cls 中存在属性 event_name
        if hasattr(cls, event_name):
            # 获取类 cls 中的旧事件函数
            old_event = getattr(cls, event_name)
            # 将旧事件函数保存为类 cls 的属性 "old_trigger_event"
            setattr(cls, "old_trigger_event", old_event)
            # 将触发事件函数 trigger_event 绑定到类 cls 的属性 event_name 上
            setattr(cls, event_name, trigger_event)
        
        # 返回修改后的类 cls
        return cls

    # 返回装饰器函数 wrapper
    return wrapper
# 获取适合的屏幕几何形状
def get_fit_geometry():
    # 获取第一个屏幕的可用几何形状
    geo = next(
        (screen.availableGeometry() for screen in QtWidgets.QApplication.screens()),
        None,
    )
    # 返回一个矩形，其位置为屏幕宽度和高度的四分之一，尺寸为屏幕宽度和高度的一半
    return QtCore.QRect(geo.width() / 4, geo.height() / 4, geo.width() / 2, geo.height() / 2)


# 将原始 pixmap 转换为圆形 pixmap
def convert_to_round_pixmap(orig_pix):
    # 获取比例因子
    scale_x, _ = get_scale_factor()
    # 计算最小宽度
    w = min(orig_pix.width(), orig_pix.height())
    # 创建一个宽度为 w，高度为 w 的 QPixmap
    pix_map = QtGui.QPixmap(w, w)
    pix_map.fill(QtCore.Qt.transparent)

    painter = QtGui.QPainter(pix_map)
    painter.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

    # 创建圆形路径
    path = QtGui.QPainterPath()
    path.addEllipse(0, 0, w, w)
    painter.setClipPath(path)
    # 在圆形区域绘制原始 pixmap
    painter.drawPixmap(0, 0, w, w, orig_pix)
    return pix_map


# 生成包含文本的 QPixmap
def generate_text_pixmap(width, height, text, alignment=QtCore.Qt.AlignCenter, bg_color=None):
    # 导入本地模块
    from . import dayu_theme

    # 如果未提供背景色，则使用默认背景色
    bg_color = bg_color or dayu_theme.background_in_color
    # 创建指定宽度和高度的 QPixmap，并填充背景色
    pix_map = QtGui.QPixmap(width, height)
    pix_map.fill(QtGui.QColor(bg_color))
    painter = QtGui.QPainter(pix_map)
    painter.setRenderHints(QtGui.QPainter.TextAntialiasing)
    # 设置字体
    font = painter.font()
    font.setFamily(dayu_theme.font_family)
    painter.setFont(font)
    # 设置画笔颜色
    painter.setPen(QtGui.QPen(QtGui.QColor(dayu_theme.secondary_text_color)))

    font_metrics = painter.fontMetrics()
    text_width = font_metrics.horizontalAdvance(text)
    text_height = font_metrics.height()
    x = width / 2 - text_width / 2
    y = height / 2 - text_height / 2
    # 根据对齐方式调整文本绘制位置
    if alignment & QtCore.Qt.AlignLeft:
        x = 0
    elif alignment & QtCore.Qt.AlignRight:
        x = width - text_width
    elif alignment & QtCore.Qt.AlignTop:
        y = 0
    elif alignment & QtCore.Qt.AlignBottom:
        y = height - text_height

    # 绘制文本
    painter.drawText(x, y, text)
    painter.end()
    return pix_map


# 获取颜色图标 QIcon
def get_color_icon(color, size=24):
    # 获取比例因子
    scale_x, y = get_scale_factor()
    # 创建大小为 size * scale_x 的 QPixmap
    pix = QtGui.QPixmap(size * scale_x, size * scale_x)
    # 如果颜色是字符串
    q_color = color
    if isinstance(color, str):
        # 如果颜色以 "#" 开头，则创建 QColor 对象
        if color.startswith("#"):
            q_color = QtGui.QColor(color)
        # 如果颜色字符串包含两个逗号，则按 RGB 值创建 QColor 对象
        elif color.count(",") == 2:
            q_color = QtGui.QColor(*tuple(map(int, color.split(","))))
    # 用指定颜色填充 QPixmap
    pix.fill(q_color)
    return QtGui.QIcon(pix)
```