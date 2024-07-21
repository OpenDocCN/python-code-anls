# `.\pytorch\torch\ao\ns\fx\qconfig_multi_mapping.py`

```py
# mypy: allow-untyped-defs
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Union, TYPE_CHECKING

import torch
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig_mapping import _QCONFIG_STYLE_ORDER

if TYPE_CHECKING:
    from torch.ao.quantization.qconfig import QConfigAny

__all__ = ["QConfigMultiMapping"]

# 字典，将配置风格映射到对应的方法名
_QCONFIG_STYLE_TO_METHOD: Dict[str, str] = {
    "global_qconfig": "set_global",
    "object_type_qconfigs": "set_object_type",
    "module_name_regex_qconfigs": "set_module_name_regex",
    "module_name_qconfigs": "set_module_name",
    "module_name_object_type_order_qconfigs": "set_module_name_object_type_order",
}

def _remove_duplicates_and_none(qconfig_list: List[QConfigAny]) -> None:
    # 移除重复项和None元素
    to_remove = []
    for index, cur_qconfig in enumerate(qconfig_list):
        if cur_qconfig is None:
            to_remove.append(index)
            break
        for checked_qconfig in qconfig_list[:index]:
            if torch.ao.quantization.qconfig_equals(cur_qconfig, checked_qconfig):
                to_remove.append(index)
                break
    for index in to_remove[::-1]:
        qconfig_list.pop(index)

class QConfigMultiMapping:
    """
    This class, used with the prepare_n_shadows_model API, stores a list of :class:`torch.ao.quantization.QConfigMapping`s
    so that multiple QConfigs can be specified for each QConfig matching style.

    The user can specify QConfigs using the following methods (in increasing match priority):

        ``set_global`` : sets the global (default) QConfigs

        ``set_object_type`` : sets the QConfigs for a given module type, function, or method name

        ``set_module_name_regex`` : sets the QConfigs for modules matching the given regex string

        ``set_module_name`` : sets the QConfigs for modules matching the given module name

        ``set_module_name_object_type_order`` : sets the QConfigs for modules matching a combination
        of the given module name, object type, and the index at which the module appears

    Note: Usage of set methods is the same as in QConfigMapping except with a passed in list of QConfigs rather than a
    single QConfig.

    Example usage::

        qconfig_mapping = QConfigMultiMapping()
            .set_global([qconfig1, qconfig2])
            .set_object_type(torch.nn.Linear, [qconfig2, qconfig3])
            .set_object_type(torch.nn.ReLU, [qconfig1])
            .set_module_name_regex("foo.*bar.*conv[0-9]+", [qconfig2])
            .set_module_name_regex("foo.*", [qconfig1, qconfig2, qconfig3])
            .set_module_name("module1", [None])
            .set_module_name("module2", [qconfig2])
            .set_module_name_object_type_order("foo.bar", torch.nn.functional.linear, 0, [qconfig3])

    """

    def __init__(self):
        # 初始化时，创建一个包含单个 QConfigMapping 的列表，以避免边界情况
        self.qconfig_mappings_list: List[QConfigMapping] = [QConfigMapping()]
    # 处理当 qconfig_list 的大小与 qconfig_mappings_list 的大小不匹配的情况
    def _handle_list_size_mismatch(
        self, qconfig_list: List[QConfigAny], style: str
    ) -> None:
        # 当 qconfig_list 的大小与 qconfig_mappings_list 的大小不匹配时，处理方法

        # 如果 qconfig_list 比 qconfig_mappings_list 更长
        if len(qconfig_list) > len(self.qconfig_mappings_list):
            # 情况：qconfig_list 中的 qconfig 数量多于 QConfigMappings 的数量

            # 添加新的 QConfigMapping 来保持不变性
            new_qconfig_mapping = QConfigMapping()

            # 搜索其他 QConfigMappings，找出需要作为 `None` 插入到新 QConfigMapping 中的 qconfig style+key
            for qconfig_mapping in self.qconfig_mappings_list:
                # 默认情况下，global_qconfig 的值为 None
                for check_style in _QCONFIG_STYLE_ORDER[1:]:
                    qconfigs_dict = getattr(qconfig_mapping, check_style)
                    target_qconfigs_dict = getattr(new_qconfig_mapping, check_style)
                    for key in qconfigs_dict:
                        target_qconfigs_dict[key] = None
                break  # 只需处理第一个 QConfigMapping

            # 复制这个新的 QConfigMapping，直到 qconfig_list 中的所有条目都能适应 QConfigMappings
            while len(qconfig_list) > len(self.qconfig_mappings_list):
                self.qconfig_mappings_list.append(copy.deepcopy(new_qconfig_mapping))
        else:
            # 情况：qconfig_list 中的 qconfig 数量少于 QConfigMappings 的数量

            # 使用 `None` 填充 qconfig_list 直到长度相同
            while len(qconfig_list) < len(self.qconfig_mappings_list):
                qconfig_list.append(None)

    # 对每个 QConfigMapping 应用插入方法的函数
    def _insert_qconfig_list(
        self,
        style: str,
        args: List[Union[str, int, Callable]],
        qconfig_list: List[QConfigAny],
    ) -> None:
        """
        Set QConfig mappings for a specific configuration style.

        Args:
            qconfig_list (List[QConfigAny]): List of QConfig objects to set.

        Notes:
            - Removes duplicates and None values from qconfig_list for deterministic ordering.
            - Calls self._handle_list_size_mismatch to handle mismatches in list sizes.
            - Uses a method specified by style from _QCONFIG_STYLE_TO_METHOD to set qconfigs in qconfig_mappings_list.
        """
        _remove_duplicates_and_none(qconfig_list)  # Remove duplicates and None values from qconfig_list

        self._handle_list_size_mismatch(qconfig_list, style)  # Handle size mismatch in qconfig_list

        method_name = _QCONFIG_STYLE_TO_METHOD[style]  # Determine method name based on style

        # Iterate through qconfig_mappings_list and qconfig_list to set qconfigs
        for qconfig_mapping, qconfig in zip(self.qconfig_mappings_list, qconfig_list):
            set_method = getattr(qconfig_mapping, method_name)  # Get method from qconfig_mapping
            set_method(*args, qconfig)  # Call set_method to insert qconfig

    def set_global(self, global_qconfig_list: List[QConfigAny]) -> QConfigMultiMapping:
        """
        Set global QConfigs.

        Args:
            global_qconfig_list (List[QConfigAny]): List of global QConfig objects to set.

        Returns:
            QConfigMultiMapping: Returns self after setting global QConfigs.
        """
        self._insert_qconfig_list("global_qconfig", [], global_qconfig_list)  # Insert global_qconfig into qconfig_mappings_list
        return self

    def set_object_type(
        self, object_type: Union[Callable, str], qconfig_list: List[QConfigAny]
    ) -> QConfigMultiMapping:
        """
        Set QConfigs for a specific object type.

        Args:
            object_type (Union[Callable, str]): Object type for which QConfigs are set.
            qconfig_list (List[QConfigAny]): List of QConfig objects to set.

        Returns:
            QConfigMultiMapping: Returns self after setting QConfigs for object type.
        """
        self._insert_qconfig_list("object_type_qconfigs", [object_type], qconfig_list)  # Insert object_type_qconfigs into qconfig_mappings_list
        return self

    def set_module_name_regex(
        self, module_name_regex: str, qconfig_list: List[QConfigAny]
    ) -> QConfigMultiMapping:
        """
        Set QConfigs for a module name regex pattern.

        Args:
            module_name_regex (str): Regular expression pattern for module name.
            qconfig_list (List[QConfigAny]): List of QConfig objects to set.

        Returns:
            QConfigMultiMapping: Returns self after setting QConfigs for module name regex.
        """
        self._insert_qconfig_list(
            "module_name_regex_qconfigs", [module_name_regex], qconfig_list
        )  # Insert module_name_regex_qconfigs into qconfig_mappings_list
        return self

    def set_module_name(
        self, module_name: str, qconfig_list: List[QConfigAny]
    ) -> QConfigMultiMapping:
        """
        Set QConfigs for a specific module name.

        Args:
            module_name (str): Module name for which QConfigs are set.
            qconfig_list (List[QConfigAny]): List of QConfig objects to set.

        Returns:
            QConfigMultiMapping: Returns self after setting QConfigs for module name.
        """
        self._insert_qconfig_list("module_name_qconfigs", [module_name], qconfig_list)  # Insert module_name_qconfigs into qconfig_mappings_list
        return self

    def set_module_name_object_type_order(
        self,
        module_name: str,
        object_type: Callable,
        index: int,
        qconfig_list: List[QConfigAny],
    ) -> QConfigMultiMapping:
        """
        Set QConfigs for a module name and object type with specific order.

        Args:
            module_name (str): Module name for which QConfigs are set.
            object_type (Callable): Object type for which QConfigs are set.
            index (int): Index to specify order of QConfig.
            qconfig_list (List[QConfigAny]): List of QConfig objects to set.

        Returns:
            QConfigMultiMapping: Returns self after setting QConfigs for module name and object type order.
        """
        self._insert_qconfig_list(
            "module_name_object_type_order_qconfigs",
            [module_name, object_type, index],
            qconfig_list,
        )  # Insert module_name_object_type_order_qconfigs into qconfig_mappings_list
        return self

    def __repr__(self):
        """
        Return a string representation of the QConfigMultiMapping object.

        Returns:
            str: String representation of the object, including representations of each qconfig_mapping in qconfig_mappings_list.
        """
        return (
            self.__class__.__name__ +
            " [" +
            "".join(f"\n{qconfig_mapping.__repr__()}," for qconfig_mapping in self.qconfig_mappings_list) +
            "\n]"
        )

    @classmethod
    def from_list_qconfig_mapping(
        cls, qconfig_mapping_list: List[QConfigMapping]
    ) -> 'QConfigMultiMapping':
        """
        Construct a QConfigMultiMapping object from a list of QConfigMapping objects.

        Args:
            qconfig_mapping_list (List[QConfigMapping]): List of QConfigMapping objects.

        Returns:
            QConfigMultiMapping: Returns a new QConfigMultiMapping object initialized with qconfig_mapping_list.
        """
    ) -> QConfigMultiMapping:
        """
        从 QConfigMappings 列表创建一个 QConfigMultiMapping 对象
        """
        # 创建一个空的 QConfigMultiMapping 对象
        new_qconfig_multi_mapping = cls()

        # 深度复制 qconfig_mapping_list 中的内容到新对象的 qconfig_mappings_list 属性中
        new_qconfig_multi_mapping.qconfig_mappings_list = copy.deepcopy(
            qconfig_mapping_list
        )

        # 遍历所有的 qconfig 样式，忽略 global，因为默认为 None
        for style in _QCONFIG_STYLE_ORDER[1:]:

            # 收集当前样式下所有的 key+qconfig 组合，存放在 qconfig_dict_list 中
            qconfig_dict_list: Dict[Any, List[QConfigAny]] = {}
            for qconfig_mapping in qconfig_mapping_list:
                qconfig_dict = getattr(qconfig_mapping, style)
                for key, qconfig in qconfig_dict.items():
                    if key not in qconfig_dict_list:
                        qconfig_dict_list[key] = []
                    qconfig_dict_list[key].append(qconfig)

            # 使用 QConfigMultiMapping 对象的 set 方法重新插入所有收集到的 key+qconfig 组合
            set_method_name = _QCONFIG_STYLE_TO_METHOD[style]
            set_method = getattr(new_qconfig_multi_mapping, set_method_name)
            for key, qconfig_list in qconfig_dict_list.items():
                if isinstance(key, tuple):
                    set_method(*key, qconfig_list)
                else:
                    set_method(key, qconfig_list)

        # 返回新创建的 QConfigMultiMapping 对象
        return new_qconfig_multi_mapping
```