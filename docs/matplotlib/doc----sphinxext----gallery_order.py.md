# `D:\src\scipysrc\matplotlib\doc\sphinxext\gallery_order.py`

```
"""
Configuration for the order of gallery sections and examples.
Paths are relative to the conf.py file.
"""

from sphinx_gallery.sorting import ExplicitOrder

# Import ExplicitOrder class from sphinx_gallery.sorting module

# Gallery sections shall be displayed in the following order.
# Non-matching sections are inserted at the unsorted position

UNSORTED = "unsorted"

examples_order = [
    '../galleries/examples/lines_bars_and_markers',
    '../galleries/examples/images_contours_and_fields',
    '../galleries/examples/subplots_axes_and_figures',
    '../galleries/examples/statistics',
    '../galleries/examples/pie_and_polar_charts',
    '../galleries/examples/text_labels_and_annotations',
    '../galleries/examples/color',
    '../galleries/examples/shapes_and_collections',
    '../galleries/examples/style_sheets',
    '../galleries/examples/pyplots',
    '../galleries/examples/axes_grid1',
    '../galleries/examples/axisartist',
    '../galleries/examples/showcase',
    UNSORTED,
    '../galleries/examples/userdemo',
]

tutorials_order = [
    '../galleries/tutorials/introductory',
    '../galleries/tutorials/intermediate',
    '../galleries/tutorials/advanced',
    UNSORTED,
    '../galleries/tutorials/provisional'
]

plot_types_order = [
    '../galleries/plot_types/basic',
    '../galleries/plot_types/stats',
    '../galleries/plot_types/arrays',
    '../galleries/plot_types/unstructured',
    '../galleries/plot_types/3D',
    UNSORTED
]

folder_lists = [examples_order, tutorials_order, plot_types_order]

# Create a list of folders in explicit order by iterating through folder_lists
# and excluding UNSORTED placeholders
explicit_order_folders = [fd for folders in folder_lists
                          for fd in folders[:folders.index(UNSORTED)]]
# Add UNSORTED placeholder to the end of explicit_order_folders
explicit_order_folders.append(UNSORTED)
# Extend explicit_order_folders with folders that appear after UNSORTED in each list
explicit_order_folders.extend([fd for folders in folder_lists
                               for fd in folders[folders.index(UNSORTED):]])


class MplExplicitOrder(ExplicitOrder):
    """For use within the 'subsection_order' key."""
    
    # Override __call__ method from ExplicitOrder
    def __call__(self, item):
        """Return a string determining the sort order."""
        # If item is in ordered_list, return its index formatted as a four-digit string
        if item in self.ordered_list:
            return f"{self.ordered_list.index(item):04d}"
        else:
            # If item is not in ordered_list, return UNSORTED index formatted as a four-digit string followed by item
            return f"{self.ordered_list.index(UNSORTED):04d}{item}"

# Subsection order:
# Subsections are ordered by filename, unless they appear in the following
# lists in which case the list order determines the order within the section.
# Examples/tutorials that do not appear in a list will be appended.

list_all = [
    #  **Tutorials**
    #  introductory
    "quick_start", "pyplot", "images", "lifecycle", "customizing",
    #  intermediate
    "artists", "legend_guide", "color_cycle",
    "constrainedlayout_guide", "tight_layout_guide",
    #  advanced
    #  text
    "text_intro", "text_props",
    #  colors
    "colors",

    #  **Examples**
    #  color
    "color_demo",
    #  pies
    "pie_features", "pie_demo2",

    # **Plot Types
    # Basic
    "plot", "scatter_plot", "bar", "stem", "step", "fill_between",
    # Arrays
    "imshow", "pcolormesh", "contour", "contourf",
    "barbs", "quiver", "streamplot",
    # 以下是 Matplotlib 中用于绘制矢量场和流场的函数名
    "hist_plot", "boxplot_plot", "errorbar_plot", "violin",
    "eventplot", "hist2d", "hexbin", "pie",
    # 以下是 Matplotlib 中用于处理非结构化数据的函数名
    "tricontour", "tricontourf", "tripcolor", "triplot",
    # 以下是 Matplotlib 中与轴脊相关的演示函数名
    "spines", "spine_placement_demo", "spines_dropped",
    "multiple_yaxis_with_spines", "centered_spines_with_arrows",
]
# 创建一个显式指定顺序的列表，列表中的每个项是以 '.py' 结尾的文件名
explicit_subsection_order = [item + ".py" for item in list_all]

# 定义一个类 MplExplicitSubOrder，继承自 ExplicitOrder 类
class MplExplicitSubOrder(ExplicitOrder):
    """For use within the 'within_subsection_order' key."""
    
    def __init__(self, src_dir):
        # 初始化方法，保存传入的 src_dir 参数，但在此处未被使用
        self.src_dir = src_dir  # src_dir is unused here
        
        # 设置类的属性 ordered_list 为之前定义的 explicit_subsection_order 列表
        self.ordered_list = explicit_subsection_order

    def __call__(self, item):
        """Return a string determining the sort order."""
        # 如果 item 在 ordered_list 中，则返回其在列表中的索引值，格式化为四位数的字符串
        if item in self.ordered_list:
            return f"{self.ordered_list.index(item):04d}"
        else:
            # 如果 item 不在 ordered_list 中，则返回一个以 'zzz' 开头的字符串，后跟 item
            # 这样可以确保未显式列出的项排在最后
            return "zzz" + item


# 为 conf.py 提供上述定义的类以供使用
# sectionorder 使用 MplExplicitOrder 类，传入 explicit_order_folders 参数
sectionorder = MplExplicitOrder(explicit_order_folders)

# subsectionorder 使用 MplExplicitSubOrder 类，但未传入参数，应该传入一个 src_dir 参数
subsectionorder = MplExplicitSubOrder
```