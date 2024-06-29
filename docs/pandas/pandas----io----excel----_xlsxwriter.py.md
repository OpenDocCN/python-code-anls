# `D:\src\scipysrc\pandas\pandas\io\excel\_xlsxwriter.py`

```
from __future__ import annotations
# 导入用于支持类型提示的未来特性

import json
# 导入用于处理 JSON 格式数据的模块

from typing import (
    TYPE_CHECKING,
    Any,
)
# 导入类型提示相关的模块和类

from pandas.io.excel._base import ExcelWriter
# 从 pandas 的 Excel 写入基础模块中导入 ExcelWriter 类

from pandas.io.excel._util import (
    combine_kwargs,
    validate_freeze_panes,
)
# 从 pandas 的 Excel 写入工具模块中导入 combine_kwargs 和 validate_freeze_panes 函数

if TYPE_CHECKING:
    from pandas._typing import (
        ExcelWriterIfSheetExists,
        FilePath,
        StorageOptions,
        WriteExcelBuffer,
    )
# 如果在类型检查模式下，导入 pandas 的类型定义以支持类型检查

class _XlsxStyler:
    # Map from openpyxl-oriented styles to flatter xlsxwriter representation
    # Ordering necessary for both determinism and because some are keyed by
    # prefixes of others.
    # 从面向 openpyxl 的样式映射到更扁平的 xlsxwriter 表示的映射
    # 排序对于确定性和一些按前缀键入的样式是必要的
    # 定义一个常量字典，用于将样式属性映射到Excel单元格样式配置的相关字段
    STYLE_MAPPING: dict[str, list[tuple[tuple[str, ...], str]]] = {
        "font": [  # 字体样式映射
            (("name",), "font_name"),  # 映射字体名称
            (("sz",), "font_size"),  # 映射字体大小
            (("size",), "font_size"),  # 映射字体大小（另一种可能的属性名）
            (("color", "rgb"), "font_color"),  # 映射字体颜色（带RGB属性）
            (("color",), "font_color"),  # 映射字体颜色（另一种可能的属性名）
            (("b",), "bold"),  # 映射粗体
            (("bold",), "bold"),  # 映射粗体（另一种可能的属性名）
            (("i",), "italic"),  # 映射斜体
            (("italic",), "italic"),  # 映射斜体（另一种可能的属性名）
            (("u",), "underline"),  # 映射下划线
            (("underline",), "underline"),  # 映射下划线（另一种可能的属性名）
            (("strike",), "font_strikeout"),  # 映射删除线
            (("vertAlign",), "font_script"),  # 映射垂直对齐方式
            (("vertalign",), "font_script"),  # 映射垂直对齐方式（另一种可能的属性名）
        ],
        "number_format": [  # 数字格式映射
            (("format_code",), "num_format"),  # 映射数字格式
            ((), "num_format"),  # 映射数字格式（默认情况）
        ],
        "protection": [  # 保护选项映射
            (("locked",), "locked"),  # 映射锁定
            (("hidden",), "hidden"),  # 映射隐藏
        ],
        "alignment": [  # 对齐方式映射
            (("horizontal",), "align"),  # 映射水平对齐方式
            (("vertical",), "valign"),  # 映射垂直对齐方式
            (("text_rotation",), "rotation"),  # 映射文本旋转角度
            (("wrap_text",), "text_wrap"),  # 映射自动换行
            (("indent",), "indent"),  # 映射缩进
            (("shrink_to_fit",), "shrink"),  # 映射缩小以适应
        ],
        "fill": [  # 填充样式映射
            (("patternType",), "pattern"),  # 映射填充类型
            (("patterntype",), "pattern"),  # 映射填充类型（另一种可能的属性名）
            (("fill_type",), "pattern"),  # 映射填充类型（另一种可能的属性名）
            (("start_color", "rgb"), "fg_color"),  # 映射前景色（带RGB属性）
            (("fgColor", "rgb"), "fg_color"),  # 映射前景色（带RGB属性）
            (("fgcolor", "rgb"), "fg_color"),  # 映射前景色（带RGB属性）
            (("start_color",), "fg_color"),  # 映射前景色
            (("fgColor",), "fg_color"),  # 映射前景色
            (("fgcolor",), "fg_color"),  # 映射前景色
            (("end_color", "rgb"), "bg_color"),  # 映射背景色（带RGB属性）
            (("bgColor", "rgb"), "bg_color"),  # 映射背景色（带RGB属性）
            (("bgcolor", "rgb"), "bg_color"),  # 映射背景色（带RGB属性）
            (("end_color",), "bg_color"),  # 映射背景色
            (("bgColor",), "bg_color"),  # 映射背景色
            (("bgcolor",), "bg_color"),  # 映射背景色
        ],
        "border": [  # 边框样式映射
            (("color", "rgb"), "border_color"),  # 映射边框颜色（带RGB属性）
            (("color",), "border_color"),  # 映射边框颜色
            (("style",), "border"),  # 映射边框样式
            (("top", "color", "rgb"), "top_color"),  # 映射顶部边框颜色（带RGB属性）
            (("top", "color"), "top_color"),  # 映射顶部边框颜色
            (("top", "style"), "top"),  # 映射顶部边框样式
            (("top",), "top"),  # 映射顶部边框样式
            (("right", "color", "rgb"), "right_color"),  # 映射右侧边框颜色（带RGB属性）
            (("right", "color"), "right_color"),  # 映射右侧边框颜色
            (("right", "style"), "right"),  # 映射右侧边框样式
            (("right",), "right"),  # 映射右侧边框样式
            (("bottom", "color", "rgb"), "bottom_color"),  # 映射底部边框颜色（带RGB属性）
            (("bottom", "color"), "bottom_color"),  # 映射底部边框颜色
            (("bottom", "style"), "bottom"),  # 映射底部边框样式
            (("bottom",), "bottom"),  # 映射底部边框样式
            (("left", "color", "rgb"), "left_color"),  # 映射左侧边框颜色（带RGB属性）
            (("left", "color"), "left_color"),  # 映射左侧边框颜色
            (("left", "style"), "left"),  # 映射左侧边框样式
            (("left",), "left"),  # 映射左侧边框样式
        ],
    }
    
    @classmethod
    def convert(cls, style_dict, num_format_str=None) -> dict[str, Any]:
        """
        将 style_dict 转换为 xlsxwriter 格式的字典

        Parameters
        ----------
        style_dict : 要转换的样式字典
        num_format_str : 可选的数字格式字符串
        """
        # 创建一个 XlsxWriter 格式对象
        props = {}

        if num_format_str is not None:
            # 如果提供了数字格式字符串，则设置 num_format 属性
            props["num_format"] = num_format_str

        if style_dict is None:
            # 如果 style_dict 为空，则直接返回空属性字典
            return props

        if "borders" in style_dict:
            # 处理 borders 键，将其改为 border，并复制 style_dict 以避免原地修改
            style_dict = style_dict.copy()
            style_dict["border"] = style_dict.pop("borders")

        for style_group_key, style_group in style_dict.items():
            for src, dst in cls.STYLE_MAPPING.get(style_group_key, []):
                # 遍历 STYLE_MAPPING 中指定的键值对映射关系
                # src 是一个用于访问嵌套字典的键序列，dst 是一个扁平的目标键
                if dst in props:
                    # 如果目标键已存在于 props 中，则跳过
                    continue
                v = style_group
                for k in src:
                    try:
                        v = v[k]
                    except (KeyError, TypeError):
                        break
                else:
                    props[dst] = v

        if isinstance(props.get("pattern"), str):
            # 如果 pattern 属性是字符串类型
            # TODO: 支持其他填充模式
            props["pattern"] = 0 if props["pattern"] == "none" else 1

        for k in ["border", "top", "right", "bottom", "left"]:
            if isinstance(props.get(k), str):
                # 将边框相关属性从字符串形式转换为索引
                try:
                    props[k] = [
                        "none",
                        "thin",
                        "medium",
                        "dashed",
                        "dotted",
                        "thick",
                        "double",
                        "hair",
                        "mediumDashed",
                        "dashDot",
                        "mediumDashDot",
                        "dashDotDot",
                        "mediumDashDotDot",
                        "slantDashDot",
                    ].index(props[k])
                except ValueError:
                    # 如果未找到匹配的字符串，设置默认值
                    props[k] = 2

        if isinstance(props.get("font_script"), str):
            # 将字体脚本属性从字符串形式转换为索引
            props["font_script"] = ["baseline", "superscript", "subscript"].index(
                props["font_script"]
            )

        if isinstance(props.get("underline"), str):
            # 将下划线属性从字符串形式转换为对应的数值
            props["underline"] = {
                "none": 0,
                "single": 1,
                "double": 2,
                "singleAccounting": 33,
                "doubleAccounting": 34,
            }[props["underline"]]

        # GH 30107 - xlsxwriter 使用不同的名称
        if props.get("valign") == "center":
            # 如果垂直对齐方式是 "center"，将其改为 "vcenter"
            props["valign"] = "vcenter"

        return props
# 定义 XlsxWriter 类，继承自 ExcelWriter
class XlsxWriter(ExcelWriter):
    # 定义引擎类型为 "xlsxwriter"
    _engine = "xlsxwriter"
    # 支持的文件扩展名为 .xlsx
    _supported_extensions = (".xlsx",)

    # 初始化方法
    def __init__(
        self,
        path: FilePath | WriteExcelBuffer | ExcelWriter,
        engine: str | None = None,
        date_format: str | None = None,
        datetime_format: str | None = None,
        mode: str = "w",
        storage_options: StorageOptions | None = None,
        if_sheet_exists: ExcelWriterIfSheetExists | None = None,
        engine_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        # 导入 xlsxwriter 模块中的 Workbook 类
        from xlsxwriter import Workbook
        
        # 合并 engine_kwargs 和 kwargs 参数
        engine_kwargs = combine_kwargs(engine_kwargs, kwargs)
        
        # 如果模式为 "a"，则抛出 ValueError 异常
        if mode == "a":
            raise ValueError("Append mode is not supported with xlsxwriter!")
        
        # 调用父类的初始化方法
        super().__init__(
            path,
            engine=engine,
            date_format=date_format,
            datetime_format=datetime_format,
            mode=mode,
            storage_options=storage_options,
            if_sheet_exists=if_sheet_exists,
            engine_kwargs=engine_kwargs,
        )
        
        # 尝试创建 Workbook 实例
        try:
            self._book = Workbook(self._handles.handle, **engine_kwargs)
        except TypeError:
            # 如果创建失败，关闭文件句柄并抛出异常
            self._handles.handle.close()
            raise

    # book 属性，返回 Workbook 实例，用于访问引擎特定功能
    @property
    def book(self):
        """
        Book instance of class xlsxwriter.Workbook.

        This attribute can be used to access engine-specific features.
        """
        return self._book

    # sheets 属性，返回所有工作表的名称字典
    @property
    def sheets(self) -> dict[str, Any]:
        result = self.book.sheetnames
        return result

    # _save 方法，关闭 Workbook 对象，保存工作簿到磁盘
    def _save(self) -> None:
        """
        Save workbook to disk.
        """
        self.book.close()

    # _write_cells 方法，用于写入单元格内容到指定工作表的特定位置
    def _write_cells(
        self,
        cells,
        sheet_name: str | None = None,
        startrow: int = 0,
        startcol: int = 0,
        freeze_panes: tuple[int, int] | None = None,
        ```
    ) -> None:
        # 使用 xlsxwriter 写入表格单元格内容。

        # 根据传入的表格名获取实际使用的工作表名
        sheet_name = self._get_sheet_name(sheet_name)

        # 根据工作表名从工作簿中获取工作表对象，如果不存在则创建新的工作表
        wks = self.book.get_worksheet_by_name(sheet_name)
        if wks is None:
            wks = self.book.add_worksheet(sheet_name)

        # 定义样式字典，初始只包含一个空样式
        style_dict = {"null": None}

        # 如果传入的冻结窗格参数有效，则设置工作表的冻结窗格
        if validate_freeze_panes(freeze_panes):
            wks.freeze_panes(*(freeze_panes))

        # 遍历传入的单元格列表
        for cell in cells:
            # 调用 _value_with_fmt 方法获取单元格的值和格式
            val, fmt = self._value_with_fmt(cell.val)

            # 将单元格的样式转换成可哈希的 JSON 格式作为样式字典的键
            stylekey = json.dumps(cell.style)
            if fmt:
                stylekey += fmt

            # 如果样式键已存在于样式字典中，则直接使用现有样式，否则创建新样式并添加到样式字典中
            if stylekey in style_dict:
                style = style_dict[stylekey]
            else:
                style = self.book.add_format(_XlsxStyler.convert(cell.style, fmt))
                style_dict[stylekey] = style

            # 如果单元格具有合并的起始和结束行列索引，则调用 merge_range 方法合并单元格，否则直接写入单元格内容
            if cell.mergestart is not None and cell.mergeend is not None:
                wks.merge_range(
                    startrow + cell.row,
                    startcol + cell.col,
                    startrow + cell.mergestart,
                    startcol + cell.mergeend,
                    val,
                    style,
                )
            else:
                wks.write(startrow + cell.row, startcol + cell.col, val, style)
```