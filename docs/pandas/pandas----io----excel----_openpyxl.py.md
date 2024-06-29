# `D:\src\scipysrc\pandas\pandas\io\excel\_openpyxl.py`

```
# 从未来导入模块注解功能
from __future__ import annotations

# 导入需要的模块和类型声明
import mmap
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

# 导入第三方库numpy
import numpy as np

# 导入pandas的依赖模块
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc

# 导入pandas共享文档模块
from pandas.core.shared_docs import _shared_docs

# 导入Excel读写相关的基础类和实用函数
from pandas.io.excel._base import (
    BaseExcelReader,
    ExcelWriter,
)
from pandas.io.excel._util import (
    combine_kwargs,
    validate_freeze_panes,
)

# 如果在类型检查模式下，则导入特定类型声明
if TYPE_CHECKING:
    from openpyxl import Workbook
    from openpyxl.descriptors.serialisable import Serialisable
    from openpyxl.styles import Fill

    from pandas._typing import (
        ExcelWriterIfSheetExists,
        FilePath,
        ReadBuffer,
        Scalar,
        StorageOptions,
        WriteExcelBuffer,
    )


# 定义一个Excel写入器类，继承自ExcelWriter
class OpenpyxlWriter(ExcelWriter):
    _engine = "openpyxl"  # 引擎名称为openpyxl
    _supported_extensions = (".xlsx", ".xlsm")  # 支持的文件扩展名

    # 初始化方法，接收多个参数来配置Excel写入器
    def __init__(
        self,
        path: FilePath | WriteExcelBuffer | ExcelWriter,  # 文件路径或写入缓冲区或ExcelWriter对象
        engine: str | None = None,  # 引擎名称，默认为None
        date_format: str | None = None,  # 日期格式，默认为None
        datetime_format: str | None = None,  # 日期时间格式，默认为None
        mode: str = "w",  # 打开文件模式，默认为写模式
        storage_options: StorageOptions | None = None,  # 存储选项，默认为None
        if_sheet_exists: ExcelWriterIfSheetExists | None = None,  # 如果工作表存在时的处理方式，默认为None
        engine_kwargs: dict[str, Any] | None = None,  # 引擎关键字参数，默认为None
        **kwargs,  # 其他关键字参数
    ) -> None:
        # 导入openpyxl中的Workbook类
        from openpyxl.workbook import Workbook

        # 合并引擎关键字参数和其他关键字参数
        engine_kwargs = combine_kwargs(engine_kwargs, kwargs)

        # 调用父类ExcelWriter的初始化方法，传递相应的参数
        super().__init__(
            path,
            mode=mode,
            storage_options=storage_options,
            if_sheet_exists=if_sheet_exists,
            engine_kwargs=engine_kwargs,
        )

        # 如果打开模式包含"r+"，表示要从现有工作簿中加载
        if "r+" in self._mode:  # 从现有工作簿中加载
            from openpyxl import load_workbook

            try:
                # 使用load_workbook函数加载现有的工作簿文件
                self._book = load_workbook(self._handles.handle, **engine_kwargs)
            except TypeError:
                # 如果加载出错，关闭文件句柄并抛出异常
                self._handles.handle.close()
                raise
            # 将文件句柄移到文件开始位置
            self._handles.handle.seek(0)
        else:
            # 否则，创建一个新的工作簿对象，默认使用优化写入为True
            try:
                self._book = Workbook(**engine_kwargs)
            except TypeError:
                # 如果创建出错，关闭文件句柄并抛出异常
                self._handles.handle.close()
                raise

            # 如果工作簿中已经有工作表，移除第一个工作表
            if self.book.worksheets:
                self.book.remove(self.book.worksheets[0])

    # book属性的getter方法，返回工作簿对象
    @property
    def book(self) -> Workbook:
        """
        Book instance of class openpyxl.workbook.Workbook.

        This attribute can be used to access engine-specific features.
        """
        return self._book

    # sheets属性的getter方法，返回一个字典，映射工作表名称到工作表对象
    @property
    def sheets(self) -> dict[str, Any]:
        """Mapping of sheet names to sheet objects."""
        result = {name: self.book[name] for name in self.book.sheetnames}
        return result
    def _save(self) -> None:
        """
        Save workbook to disk.
        """
        # 调用 openpyxl Workbook 对象的 save 方法，将工作簿保存到磁盘上指定的路径
        self.book.save(self._handles.handle)
        # 如果模式包含 "r+" 并且文件句柄不是 mmap.mmap 类型的对象
        if "r+" in self._mode and not isinstance(self._handles.handle, mmap.mmap):
            # 截断文件到已写入的内容长度
            self._handles.handle.truncate()

    @classmethod
    def _convert_to_style_kwargs(cls, style_dict: dict) -> dict[str, Serialisable]:
        """
        Convert a style_dict to a set of kwargs suitable for initializing
        or updating-on-copy an openpyxl v2 style object.

        Parameters
        ----------
        style_dict : dict
            A dict with zero or more of the following keys (or their synonyms).
                'font'
                'fill'
                'border' ('borders')
                'alignment'
                'number_format'
                'protection'

        Returns
        -------
        style_kwargs : dict
            A dict with the same, normalized keys as ``style_dict`` but each
            value has been replaced with a native openpyxl style object of the
            appropriate class.
        """
        # 定义一个映射表，用于将 'borders' 转换为 'border'
        _style_key_map = {"borders": "border"}

        # 初始化一个空字典，用于存储转换后的样式参数
        style_kwargs: dict[str, Serialisable] = {}
        # 遍历 style_dict 中的键值对
        for k, v in style_dict.items():
            # 将键 k 转换为映射表中的值，如果没有映射则保持原样
            k = _style_key_map.get(k, k)
            # 获取类方法 cls._convert_to_{k}，默认为 lambda x: None
            _conv_to_x = getattr(cls, f"_convert_to_{k}", lambda x: None)
            # 对值 v 进行转换
            new_v = _conv_to_x(v)
            # 如果转换后的值存在，则将其添加到 style_kwargs 字典中
            if new_v:
                style_kwargs[k] = new_v

        return style_kwargs

    @classmethod
    def _convert_to_color(cls, color_spec):
        """
        Convert ``color_spec`` to an openpyxl v2 Color object.

        Parameters
        ----------
        color_spec : str, dict
            A 32-bit ARGB hex string, or a dict with zero or more of the
            following keys.
                'rgb'
                'indexed'
                'auto'
                'theme'
                'tint'
                'index'
                'type'

        Returns
        -------
        color : openpyxl.styles.Color
        """
        # 导入 openpyxl.styles 中的 Color 类
        from openpyxl.styles import Color

        # 如果 color_spec 是字符串类型，则直接使用该字符串创建 Color 对象并返回
        if isinstance(color_spec, str):
            return Color(color_spec)
        # 否则，使用 color_spec 字典创建 Color 对象并返回
        else:
            return Color(**color_spec)
    @classmethod
    def _convert_to_stop(cls, stop_seq):
        """
        Convert ``stop_seq`` to a list of openpyxl v2 Color objects,
        suitable for initializing the ``GradientFill`` ``stop`` parameter.

        Parameters
        ----------
        stop_seq : iterable
            An iterable that yields objects suitable for consumption by
            ``_convert_to_color``.

        Returns
        -------
        stop : list of openpyxl.styles.Color
        """
        # 使用 map 函数将 stop_seq 中的每个元素转换为 openpyxl v2 的 Color 对象列表
        return map(cls._convert_to_color, stop_seq)
    @classmethod
    def _convert_to_fill(cls, fill_dict: dict[str, Any]) -> Fill:
        """
        Convert ``fill_dict`` to an openpyxl v2 Fill object.

        Parameters
        ----------
        fill_dict : dict
            A dict with one or more of the following keys (or their synonyms),
                'fill_type' ('patternType', 'patterntype')
                'start_color' ('fgColor', 'fgcolor')
                'end_color' ('bgColor', 'bgcolor')
            or one or more of the following keys (or their synonyms).
                'type' ('fill_type')
                'degree'
                'left'
                'right'
                'top'
                'bottom'
                'stop'

        Returns
        -------
        fill : openpyxl.styles.Fill
        """
        from openpyxl.styles import (
            GradientFill,
            PatternFill,
        )

        # 映射用于填充类型的关键字，将不同的命名映射到标准的关键字
        _pattern_fill_key_map = {
            "patternType": "fill_type",
            "patterntype": "fill_type",
            "fgColor": "start_color",
            "fgcolor": "start_color",
            "bgColor": "end_color",
            "bgcolor": "end_color",
        }

        # 映射用于渐变填充的关键字，将不同的命名映射到标准的关键字
        _gradient_fill_key_map = {"fill_type": "type"}

        # 初始化填充和渐变填充的关键字字典
        pfill_kwargs = {}
        gfill_kwargs = {}

        # 遍历填充字典中的每一个键值对
        for k, v in fill_dict.items():
            # 尝试从模式填充映射中获取标准关键字
            pk = _pattern_fill_key_map.get(k)
            # 尝试从渐变填充映射中获取标准关键字
            gk = _gradient_fill_key_map.get(k)

            # 如果键是颜色相关的，则转换其值为标准颜色对象
            if pk in ["start_color", "end_color"]:
                v = cls._convert_to_color(v)
            # 如果是渐变填充中的停止颜色，则进行特殊的转换处理
            if gk == "stop":
                v = cls._convert_to_stop(v)

            # 如果是模式填充的标准关键字，则添加到模式填充关键字字典中
            if pk:
                pfill_kwargs[pk] = v
            # 如果是渐变填充的标准关键字，则添加到渐变填充关键字字典中
            elif gk:
                gfill_kwargs[gk] = v
            # 否则将键值对添加到两个填充字典中
            else:
                pfill_kwargs[k] = v
                gfill_kwargs[k] = v

        try:
            # 尝试使用模式填充关键字字典创建 PatternFill 对象
            return PatternFill(**pfill_kwargs)
        except TypeError:
            # 如果出现类型错误，则使用渐变填充关键字字典创建 GradientFill 对象
            return GradientFill(**gfill_kwargs)

    @classmethod
    def _convert_to_side(cls, side_spec):
        """
        Convert ``side_spec`` to an openpyxl v2 Side object.

        Parameters
        ----------
        side_spec : str, dict
            A string specifying the border style, or a dict with zero or more
            of the following keys (or their synonyms).
                'style' ('border_style')
                'color'

        Returns
        -------
        side : openpyxl.styles.Side
        """
        from openpyxl.styles import Side

        # 映射用于边框样式的关键字，将不同的命名映射到标准的关键字
        _side_key_map = {"border_style": "style"}

        # 如果 side_spec 是字符串，则直接创建 Side 对象
        if isinstance(side_spec, str):
            return Side(style=side_spec)

        # 初始化边框关键字字典
        side_kwargs = {}

        # 遍历边框规格字典中的每一个键值对
        for k, v in side_spec.items():
            # 尝试从边框关键字映射中获取标准关键字
            k = _side_key_map.get(k, k)

            # 如果是颜色关键字，则将其值转换为标准颜色对象
            if k == "color":
                v = cls._convert_to_color(v)

            # 将标准关键字及其对应的值添加到边框关键字字典中
            side_kwargs[k] = v

        # 使用边框关键字字典创建 Side 对象并返回
        return Side(**side_kwargs)

    @classmethod
    def _convert_to_border(cls, border_dict):
        """
        Convert ``border_dict`` to an openpyxl v2 Border object.

        Parameters
        ----------
        border_dict : dict
            A dict with zero or more of the following keys (or their synonyms).
                'left'
                'right'
                'top'
                'bottom'
                'diagonal'
                'diagonal_direction'
                'vertical'
                'horizontal'
                'diagonalUp' ('diagonalup')
                'diagonalDown' ('diagonaldown')
                'outline'

        Returns
        -------
        border : openpyxl.styles.Border
            An instance of openpyxl's Border style configured based on provided dictionary.
        """
        from openpyxl.styles import Border

        _border_key_map = {"diagonalup": "diagonalUp", "diagonaldown": "diagonalDown"}

        border_kwargs = {}
        # Iterate over each key-value pair in border_dict
        for k, v in border_dict.items():
            # Map certain keys to their corresponding names in openpyxl
            k = _border_key_map.get(k, k)
            # Convert 'color' value to openpyxl's Color object if key is 'color'
            if k == "color":
                v = cls._convert_to_color(v)
            # Convert 'left', 'right', 'top', 'bottom', 'diagonal' values to Side objects
            if k in ["left", "right", "top", "bottom", "diagonal"]:
                v = cls._convert_to_side(v)
            # Store the key-value pair in border_kwargs dictionary
            border_kwargs[k] = v

        # Create and return an openpyxl Border object using the collected keyword arguments
        return Border(**border_kwargs)

    @classmethod
    def _convert_to_alignment(cls, alignment_dict):
        """
        Convert ``alignment_dict`` to an openpyxl v2 Alignment object.

        Parameters
        ----------
        alignment_dict : dict
            A dict with zero or more of the following keys (or their synonyms).
                'horizontal'
                'vertical'
                'text_rotation'
                'wrap_text'
                'shrink_to_fit'
                'indent'

        Returns
        -------
        alignment : openpyxl.styles.Alignment
            An instance of openpyxl's Alignment style configured based on provided dictionary.
        """
        from openpyxl.styles import Alignment

        # Directly create and return an openpyxl Alignment object using alignment_dict
        return Alignment(**alignment_dict)

    @classmethod
    def _convert_to_number_format(cls, number_format_dict):
        """
        Convert ``number_format_dict`` to an openpyxl v2.1.0 number format
        initializer.

        Parameters
        ----------
        number_format_dict : dict
            A dict with zero or more of the following keys.
                'format_code' : str

        Returns
        -------
        number_format : str
            The format code as a string from number_format_dict.
        """
        # Return the 'format_code' value directly from number_format_dict
        return number_format_dict["format_code"]

    @classmethod
    def _convert_to_protection(cls, protection_dict):
        """
        Convert ``protection_dict`` to an openpyxl v2 Protection object.

        Parameters
        ----------
        protection_dict : dict
            A dict with zero or more of the following keys.
                'locked'
                'hidden'

        Returns
        -------
        protection : openpyxl.styles.Protection
            An instance of openpyxl's Protection style configured based on provided dictionary.
        """
        from openpyxl.styles import Protection

        # Create and return an openpyxl Protection object using protection_dict
        return Protection(**protection_dict)

    def _write_cells(
        self,
        cells,
        sheet_name: str | None = None,
        startrow: int = 0,
        startcol: int = 0,
        freeze_panes: tuple[int, int] | None = None,
    # 定义 OpenpyxlReader 类，继承自 BaseExcelReader["Workbook"]
    class OpenpyxlReader(BaseExcelReader["Workbook"]):
        
        # 使用文档生成器注解初始化方法，接受文件路径或缓冲区、存储选项和引擎关键字参数
        @doc(storage_options=_shared_docs["storage_options"])
        def __init__(
            self,
            filepath_or_buffer: FilePath | ReadBuffer[bytes],
            storage_options: StorageOptions | None = None,
            engine_kwargs: dict | None = None,
        ) -> None:
            """
            Reader using openpyxl engine.

            Parameters
            ----------
            filepath_or_buffer : str, path object or Workbook
                Object to be parsed.
            {storage_options}
            engine_kwargs : dict, optional
                Arbitrary keyword arguments passed to excel engine.
            """
            # 导入必要的 openpyxl 依赖
            import_optional_dependency("openpyxl")
            # 调用父类构造函数，传递文件路径或缓冲区，存储选项和引擎关键字参数
            super().__init__(
                filepath_or_buffer,
                storage_options=storage_options,
                engine_kwargs=engine_kwargs,
            )

        # 返回 openpyxl 中的 Workbook 类型
        @property
        def _workbook_class(self) -> type[Workbook]:
            from openpyxl import Workbook

            return Workbook

        # 加载工作簿，返回 Workbook 对象
        def load_workbook(
            self, filepath_or_buffer: FilePath | ReadBuffer[bytes], engine_kwargs
        ) -> Workbook:
            from openpyxl import load_workbook

            # 默认的加载参数设置
            default_kwargs = {"read_only": True, "data_only": True, "keep_links": False}

            # 使用指定的引擎参数加载工作簿
            return load_workbook(
                filepath_or_buffer,
                **(default_kwargs | engine_kwargs),
            )

        # 返回所有工作表的名称列表
        @property
        def sheet_names(self) -> list[str]:
            return [sheet.title for sheet in self.book.worksheets]

        # 根据名称获取工作表对象
        def get_sheet_by_name(self, name: str):
            # 如果名称不合法，抛出异常
            self.raise_if_bad_sheet_by_name(name)
            return self.book[name]

        # 根据索引获取工作表对象
        def get_sheet_by_index(self, index: int):
            # 如果索引不合法，抛出异常
            self.raise_if_bad_sheet_by_index(index)
            return self.book.worksheets[index]

        # 将单元格数据转换为标量值
        def _convert_cell(self, cell) -> Scalar:
            from openpyxl.cell.cell import (
                TYPE_ERROR,
                TYPE_NUMERIC,
            )

            # 如果单元格值为空，返回空字符串（与 xlrd 兼容）
            if cell.value is None:
                return ""
            # 如果单元格数据类型是错误类型，返回 NaN
            elif cell.data_type == TYPE_ERROR:
                return np.nan
            # 如果单元格数据类型是数值类型，尝试转换为整数或浮点数
            elif cell.data_type == TYPE_NUMERIC:
                val = int(cell.value)
                if val == cell.value:
                    return val
                return float(cell.value)

            return cell.value

        # 获取工作表数据
        def get_sheet_data(
            self, sheet, file_rows_needed: int | None = None
        ):
    # 定义函数的参数和返回类型，接收一个自变量并返回一个列表，其中每个元素都是一个列表，每个元素都包含标量值
    ) -> list[list[Scalar]]:
        # 如果工作簿为只读，则重置工作表的维度
        if self.book.read_only:
            sheet.reset_dimensions()

        # 初始化空数据列表
        data: list[list[Scalar]] = []
        # 记录最后一个含有数据的行的索引，默认为-1
        last_row_with_data = -1
        # 遍历工作表的每一行
        for row_number, row in enumerate(sheet.rows):
            # 将每个单元格转换为标量值并存储到converted_row中
            converted_row = [self._convert_cell(cell) for cell in row]
            # 剔除末尾的空元素
            while converted_row and converted_row[-1] == "":
                converted_row.pop()
            # 如果converted_row非空，则更新最后一个含有数据的行的索引
            if converted_row:
                last_row_with_data = row_number
            # 将转换后的行添加到数据列表中
            data.append(converted_row)
            # 如果指定了需要的文件行数，并且数据列表长度达到了指定的文件行数，则提前结束循环
            if file_rows_needed is not None and len(data) >= file_rows_needed:
                break

        # 剔除末尾的空行
        data = data[: last_row_with_data + 1]

        # 如果数据列表不为空
        if len(data) > 0:
            # 扩展每行以匹配最大宽度
            max_width = max(len(data_row) for data_row in data)
            # 如果最小行的长度小于最大宽度，则在每行末尾添加空单元格，使其长度匹配最大宽度
            if min(len(data_row) for data_row in data) < max_width:
                empty_cell: list[Scalar] = [""]
                data = [
                    data_row + (max_width - len(data_row)) * empty_cell
                    for data_row in data
                ]

        # 返回处理后的数据列表
        return data
```