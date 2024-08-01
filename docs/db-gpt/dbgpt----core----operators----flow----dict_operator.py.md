# `.\DB-GPT-src\dbgpt\core\operators\flow\dict_operator.py`

```py
"""Dict tool operators."""

from typing import Dict  # 导入字典类型的类型提示

from dbgpt.core.awel import JoinOperator  # 从 dbgpt.core.awel 模块导入 JoinOperator 类
from dbgpt.core.awel.flow import IOField, OperatorCategory, Parameter, ViewMetadata  # 从 dbgpt.core.awel.flow 模块导入 IOField, OperatorCategory, Parameter, ViewMetadata 类
from dbgpt.util.i18n_utils import _  # 导入国际化翻译函数 _

class MergeStringToDictOperator(JoinOperator[Dict[str, str]]):
    """Merge two strings to a dict."""
    
    metadata = ViewMetadata(
        label=_("Merge String to Dict Operator"),  # 操作符的标签，使用国际化翻译
        name="merge_string_to_dict_operator",  # 操作符的名称
        category=OperatorCategory.COMMON,  # 操作符所属的类别
        description=_(
            "Merge two strings to a dict, the fist string which is the value from first"
            " upstream is the value of the key `first_key`, the second string which is "
            "the value from second upstream is the value of the key `second_key`."
        ),  # 操作符的描述，使用国际化翻译
        parameters=[
            Parameter.build_from(
                _("First Key"),  # 第一个参数的标签，使用国际化翻译
                "first_key",  # 第一个参数的名称
                str,  # 参数类型为字符串
                optional=True,  # 参数是可选的
                default="user_input",  # 参数的默认值为'user_input'
                description=_("The key for the first string, default is `user_input`."),
            ),
            Parameter.build_from(
                _("Second Key"),  # 第二个参数的标签，使用国际化翻译
                "second_key",  # 第二个参数的名称
                str,  # 参数类型为字符串
                optional=True,  # 参数是可选的
                default="context",  # 参数的默认值为'context'
                description=_("The key for the second string, default is `context`."),
            ),
        ],
        inputs=[
            IOField.build_from(
                _("First String"),  # 输入字段的标签，使用国际化翻译
                "first",  # 输入字段的名称
                str,  # 输入字段的类型为字符串
                description=_("The first string from first upstream."),
            ),
            IOField.build_from(
                _("Second String"),  # 输入字段的标签，使用国际化翻译
                "second",  # 输入字段的名称
                str,  # 输入字段的类型为字符串
                description=_("The second string from second upstream."),
            ),
        ],
        outputs=[
            IOField.build_from(
                _("Output"),  # 输出字段的标签，使用国际化翻译
                "output",  # 输出字段的名称
                dict,  # 输出字段的类型为字典
                description=_(
                    "The merged dict. example: "
                    "{'user_input': 'first', 'context': 'second'}."
                ),  # 输出字段的描述
            ),
        ],
    )

    def __init__(
        self, first_key: str = "user_input", second_key: str = "context", **kwargs
    ):
        """Create a MergeStringToDictOperator instance."""
        self._first_key = first_key  # 初始化第一个键
        self._second_key = second_key  # 初始化第二个键
        super().__init__(combine_function=self._merge_to_dict, **kwargs)  # 调用父类的初始化方法，指定合并函数为 self._merge_to_dict

    def _merge_to_dict(self, first: str, second: str) -> Dict[str, str]:
        """Merge two strings into a dictionary."""
        return {self._first_key: first, self._second_key: second}  # 将两个字符串合并成字典的键值对
```