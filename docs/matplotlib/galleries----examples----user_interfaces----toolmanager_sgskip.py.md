# `D:\src\scipysrc\matplotlib\galleries\examples\user_interfaces\toolmanager_sgskip.py`

```py
"""
============
Tool Manager
============

This example demonstrates how to

* modify the Toolbar
* create tools
* add tools
* remove tools

using `matplotlib.backend_managers.ToolManager`.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块

from matplotlib.backend_tools import ToolBase, ToolToggleBase  # 导入工具基类和工具切换基类

plt.rcParams['toolbar'] = 'toolmanager'  # 设置绘图时使用的工具栏风格为 toolmanager

class ListTools(ToolBase):
    """List all the tools controlled by the `ToolManager`."""
    default_keymap = 'm'  # 默认的键盘快捷键为 'm'
    description = 'List Tools'  # 工具描述为 'List Tools'

    def trigger(self, *args, **kwargs):
        print('_' * 80)  # 打印一条分隔线
        fmt_tool = "{:12} {:45} {}".format  # 格式化输出工具信息的格式
        print(fmt_tool('Name (id)', 'Tool description', 'Keymap'))  # 打印表头
        print('-' * 80)  # 打印分隔线
        tools = self.toolmanager.tools  # 获取所有工具的字典
        for name in sorted(tools):  # 遍历所有工具名（按名称排序）
            if not tools[name].description:  # 如果工具没有描述则跳过
                continue
            keys = ', '.join(sorted(self.toolmanager.get_tool_keymap(name)))  # 获取工具的键盘快捷键
            print(fmt_tool(name, tools[name].description, keys))  # 输出工具的名称、描述和快捷键
        print('_' * 80)  # 打印一条分隔线
        fmt_active_toggle = "{!s:12} {!s:45}".format  # 格式化输出活动切换工具信息的格式
        print("Active Toggle tools")  # 打印标题
        print(fmt_active_toggle("Group", "Active"))  # 打印表头
        print('-' * 80)  # 打印分隔线
        for group, active in self.toolmanager.active_toggle.items():  # 遍历所有活动切换工具
            print(fmt_active_toggle(group, active))  # 输出活动切换工具的组名和状态


class GroupHideTool(ToolToggleBase):
    """Show lines with a given gid."""
    default_keymap = 'S'  # 默认的键盘快捷键为 'S'
    description = 'Show by gid'  # 工具描述为 'Show by gid'
    default_toggled = True  # 默认为开启状态

    def __init__(self, *args, gid, **kwargs):
        self.gid = gid  # 初始化时设置 gid 属性
        super().__init__(*args, **kwargs)

    def enable(self, *args):
        self.set_lines_visibility(True)  # 启用工具时显示指定 gid 的线条

    def disable(self, *args):
        self.set_lines_visibility(False)  # 禁用工具时隐藏指定 gid 的线条

    def set_lines_visibility(self, state):
        for ax in self.figure.get_axes():  # 遍历所有坐标轴
            for line in ax.get_lines():  # 遍历每个坐标轴的线条
                if line.get_gid() == self.gid:  # 如果线条的 gid 与工具的 gid 匹配
                    line.set_visible(state)  # 设置线条的可见性
        self.figure.canvas.draw()  # 重新绘制图形


fig = plt.figure()  # 创建一个新的图形对象

plt.plot([1, 2, 3], gid='mygroup')  # 绘制一条线并设置其 gid
plt.plot([2, 3, 4], gid='unknown')  # 绘制另一条线并设置其 gid
plt.plot([3, 2, 1], gid='mygroup')  # 绘制第三条线并设置其 gid

# 添加自定义工具
fig.canvas.manager.toolmanager.add_tool('List', ListTools)  # 添加 ListTools 工具
fig.canvas.manager.toolmanager.add_tool('Show', GroupHideTool, gid='mygroup')  # 添加 GroupHideTool 工具，并指定 gid

# 将现有工具添加到新组 'foo' 中，可以多次添加
fig.canvas.manager.toolbar.add_tool('zoom', 'foo')

# 移除向前按钮
fig.canvas.manager.toolmanager.remove_tool('forward')

# 将自定义工具添加到工具栏的导航组中特定位置
fig.canvas.manager.toolbar.add_tool('Show', 'navigation', 1)

plt.show()  # 显示图形界面
```