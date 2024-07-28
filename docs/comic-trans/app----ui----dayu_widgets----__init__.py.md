# `.\comic-translate\app\ui\dayu_widgets\__init__.py`

```py
# -*- coding: utf-8 -*-
# 导入未来的模块功能
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入内置模块
import os
import sys

# 默认静态文件夹路径为当前模块下的 "static" 文件夹
DEFAULT_STATIC_FOLDER = os.path.join(sys.modules[__name__].__path__[0], "static")
# 自定义静态文件夹列表为空
CUSTOM_STATIC_FOLDERS = []

# 导入本地模块
from .theme import MTheme

# 创建名为 "dayu_theme" 的 MTheme 主题对象，使用 'dark' 主题和自定义的橙色作为主色调
dayu_theme = MTheme("dark", primary_color=MTheme.orange)

# dayu_theme.default_size = dayu_theme.small
# dayu_theme = MTheme('light')

# 导入本地模块
from .alert import MAlert
from .avatar import MAvatar
from .badge import MBadge
from .breadcrumb import MBreadcrumb
from .browser import MClickBrowserFilePushButton
from .browser import MClickBrowserFileToolButton
from .browser import MClickBrowserFolderPushButton
from .browser import MClickBrowserFolderToolButton
from .browser import MDragFileButton
from .browser import MDragFolderButton
from .button_group import MCheckBoxGroup
from .button_group import MPushButtonGroup
from .button_group import MRadioButtonGroup
from .button_group import MToolButtonGroup
from .card import MCard
from .card import MMeta
from .carousel import MCarousel
from .check_box import MCheckBox
from .collapse import MCollapse
from .combo_box import MComboBox
from .divider import MDivider
from .field_mixin import MFieldMixin
from .flow_layout import MFlowLayout
from .item_model import MSortFilterModel
from .item_model import MTableModel
from .item_view import MBigView
from .item_view import MListView
from .item_view import MTableView
from .item_view import MTreeView
from .item_view_full_set import MItemViewFullSet
from .item_view_set import MItemViewSet
from .label import MLabel
from .line_edit import MLineEdit
from .line_tab_widget import MLineTabWidget
from .loading import MLoading
from .loading import MLoadingWrapper
from .menu import MMenu
from .menu_tab_widget import MMenuTabWidget
from .message import MMessage
from .page import MPage
from .progress_bar import MProgressBar
from .progress_circle import MProgressCircle
from .push_button import MPushButton
from .radio_button import MRadioButton
from .sequence_file import MSequenceFile
from .slider import MSlider
from .spin_box import MDateEdit
from .spin_box import MDateTimeEdit
from .spin_box import MDoubleSpinBox
from .spin_box import MSpinBox
from .spin_box import MTimeEdit
from .switch import MSwitch
from .tab_widget import MTabWidget
from .text_edit import MTextEdit
from .toast import MToast
from .tool_button import MToolButton

# 指定导出的模块列表
__all__ = [
    "MAlert",
    "MAvatar",
    "MBadge",
    "MBreadcrumb",
    "MClickBrowserFilePushButton",
    "MClickBrowserFileToolButton",
    "MClickBrowserFolderPushButton",
    "MClickBrowserFolderToolButton",
    "MDragFileButton",
    "MDragFolderButton",
    "MCheckBoxGroup",
    "MPushButtonGroup",
    "MRadioButtonGroup",
    "MToolButtonGroup",
    "MCard",
    "MMeta",
    "MCarousel",
    "MCheckBox",
    "MCollapse",
    "MComboBox",
    "MDivider",
    # 导入以下模块，每个模块代表一个自定义的控件或者视图部件
    "MFieldMixin",           # 控件混合字段
    "MFlowLayout",           # 流式布局
    "MSortFilterModel",      # 排序和过滤模型
    "MTableModel",           # 表格模型
    "MBigView",              # 大视图
    "MListView",             # 列表视图
    "MTableView",            # 表格视图
    "MTreeView",             # 树视图
    "MItemViewFullSet",      # 项目视图全集
    "MItemViewSet",          # 项目视图集
    "MLabel",                # 标签控件
    "MLineEdit",             # 行编辑器
    "MLineTabWidget",        # 行选项卡部件
    "MLoading",              # 加载状态
    "MLoadingWrapper",       # 加载包装器
    "MMenu",                 # 菜单控件
    "MMenuTabWidget",        # 菜单选项卡部件
    "MMessage",              # 消息控件
    "MPage",                 # 页面控件
    "MProgressBar",          # 进度条控件
    "MProgressCircle",       # 进度圆控件
    "MPushButton",           # 按钮控件
    "MRadioButton",          # 单选按钮控件
    "MSequenceFile",         # 序列文件
    "MSlider",               # 滑动条控件
    "MDateEdit",             # 日期编辑器
    "MDateTimeEdit",         # 日期时间编辑器
    "MDoubleSpinBox",        # 双精度浮点数微调器
    "MSpinBox",              # 微调器
    "MTimeEdit",             # 时间编辑器
    "MSwitch",               # 开关控件
    "MTabWidget",            # 选项卡部件
    "MTextEdit",             # 文本编辑器
    "MToast",                # 提示控件
    "MToolButton",           # 工具按钮控件
]
```