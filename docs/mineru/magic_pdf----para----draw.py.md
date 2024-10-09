# `.\MinerU\magic_pdf\para\draw.py`

```
# 从 magic_pdf.libs.commons 模块导入 fitz 类
from magic_pdf.libs.commons import fitz

# 从 magic_pdf.para.commons 模块导入所有内容
from magic_pdf.para.commons import *


# 检查 Python 版本是否为 3 或更高
if sys.version_info[0] >= 3:
    # 如果是 Python 3，重新配置标准输出的编码为 UTF-8，忽略类型检查
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore


class DrawAnnos:
    """
    此类在 PDF 文件上绘制注释

    ----------------------------------------
                颜色代码
    ----------------------------------------
        红色: (1, 0, 0)
        绿色: (0, 1, 0)
        蓝色: (0, 0, 1)
        黄色: (1, 1, 0) - 红色和绿色的混合
        青色: (0, 1, 1) - 绿色和蓝色的混合
        品红: (1, 0, 1) - 红色和蓝色的混合
        白色: (1, 1, 1) - 红色、绿色和蓝色的全强度
        黑色: (0, 0, 0) - 完全没有颜色成分
        灰色: (0.5, 0.5, 0.5) - 红色、绿色和蓝色成分的均等和中等强度
        橙色: (1, 0.65, 0) - 红色的最大强度，绿色的中等强度，没有蓝色成分
    """

    def __init__(self) -> None:
        # 初始化方法，暂时不做任何事情
        pass

    def __is_nested_list(self, lst):
        """
        此函数返回 True，如果给定的列表是任何程度的嵌套列表。
        """
        if isinstance(lst, list):
            # 检查列表中的任何元素是否为嵌套列表，或列表本身是否包含列表
            return any(self.__is_nested_list(i) for i in lst) or any(isinstance(i, list) for i in lst)
        return False

    def __valid_rect(self, bbox):
        # 确保矩形不为空或无效
        if isinstance(bbox[0], list):
            # 如果第一个元素是列表，则不能是有效矩形
            return False  # 这是嵌套列表，因此不能是有效矩形
        else:
            # 检查矩形的左下角是否小于右上角
            return bbox[0] < bbox[2] and bbox[1] < bbox[3]

    def __draw_nested_boxes(self, page, nested_bbox, color=(0, 1, 1)):
        """
        此函数绘制嵌套的框

        参数
        ----------
        page : fitz.Page
            当前页面
        nested_bbox : list
            嵌套边界框
        color : tuple
            颜色，默认为 (0, 1, 1)    # 使用青色绘制组合段落
        """
        # 如果是嵌套列表
        if self.__is_nested_list(nested_bbox):
            # 遍历嵌套边界框
            for bbox in nested_bbox:
                # 递归调用该函数
                self.__draw_nested_boxes(page, bbox, color)
        # 如果是有效矩形
        elif self.__valid_rect(nested_bbox):
            # 创建 fitz.Rect 对象表示矩形
            para_rect = fitz.Rect(nested_bbox)
            # 在页面上添加矩形注释
            para_anno = page.add_rect_annot(para_rect)
            # 设置边框颜色为青色
            para_anno.set_colors(stroke=color)  # 使用青色绘制组合段落
            # 设置边框宽度为 1
            para_anno.set_border(width=1)
            # 更新注释以应用更改
            para_anno.update()
```