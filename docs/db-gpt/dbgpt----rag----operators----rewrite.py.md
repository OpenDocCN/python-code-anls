# `.\DB-GPT-src\dbgpt\rag\operators\rewrite.py`

```py
"""The rewrite operator."""

from typing import Any, List, Optional

from dbgpt.core import LLMClient  # 导入LLMClient类
from dbgpt.core.awel import MapOperator  # 导入MapOperator类
from dbgpt.core.awel.flow import IOField, OperatorCategory, Parameter, ViewMetadata  # 导入IOField, OperatorCategory, Parameter, ViewMetadata类
from dbgpt.rag.retriever.rewrite import QueryRewrite  # 导入QueryRewrite类
from dbgpt.util.i18n_utils import _  # 导入翻译函数_

class QueryRewriteOperator(MapOperator[dict, Any]):
    """The Rewrite Operator."""

    metadata = ViewMetadata(  # 定义元数据
        label=_("Query Rewrite Operator"),  # 标签，用于显示
        name="query_rewrite_operator",  # 名称，用于标识
        category=OperatorCategory.RAG,  # 操作符类别为RAG
        description=_("Query rewrite operator."),  # 描述，操作符功能说明
        inputs=[  # 输入定义
            IOField.build_from(  # 构建输入字段
                _("Query context"), "query_context", dict, _("query context")  # 输入字段的名称、类型和描述
            )
        ],
        outputs=[  # 输出定义
            IOField.build_from(  # 构建输出字段
                _("Rewritten queries"),  # 输出字段名称
                "queries",  # 输出字段标识
                str,  # 输出字段类型为字符串
                is_list=True,  # 输出为列表形式
                description=_("Rewritten queries"),  # 输出字段描述
            )
        ],
        parameters=[  # 参数定义
            Parameter.build_from(  # 构建参数
                _("LLM Client"),  # 参数名称
                "llm_client",  # 参数标识
                LLMClient,  # 参数类型为LLMClient类
                description=_("The LLM Client."),  # 参数描述
            ),
            Parameter.build_from(  # 构建参数
                label=_("Model name"),  # 参数标签
                name="model_name",  # 参数名称
                type=str,  # 参数类型为字符串
                optional=True,  # 参数可选
                default="gpt-3.5-turbo",  # 参数默认值
                description=_("LLM model name."),  # 参数描述
            ),
            Parameter.build_from(  # 构建参数
                label=_("Prompt language"),  # 参数标签
                name="language",  # 参数名称
                type=str,  # 参数类型为字符串
                optional=True,  # 参数可选
                default="en",  # 参数默认值
                description=_("Prompt language."),  # 参数描述
            ),
            Parameter.build_from(  # 构建参数
                label=_("Number of results"),  # 参数标签
                name="nums",  # 参数名称
                type=int,  # 参数类型为整数
                optional=True,  # 参数可选
                default=5,  # 参数默认值
                description=_("rewrite query number."),  # 参数描述
            ),
        ],
        documentation_url="https://github.com/openai/openai-python",  # 文档URL
    )

    def __init__(  # 初始化方法
        self,
        llm_client: LLMClient,  # LLM客户端参数
        model_name: str = "gpt-3.5-turbo",  # 模型名称参数，默认为"gpt-3.5-turbo"
        language: Optional[str] = "en",  # 语言参数，默认为"en"
        nums: Optional[int] = 1,  # 结果数量参数，默认为1
        **kwargs  # 其他关键字参数
    ):
        """Init the query rewrite operator.

        Args:
            llm_client (Optional[LLMClient]): The LLM client. LLM客户端对象
            model_name (Optional[str]): The model name. 模型名称
            language (Optional[str]): The prompt language. 提示语言
            nums (Optional[int]): The number of the rewrite results. 重写结果数量
        """
        super().__init__(**kwargs)  # 调用父类初始化方法
        self._nums = nums  # 设置结果数量属性
        self._rewrite = QueryRewrite(  # 初始化QueryRewrite对象
            llm_client=llm_client,  # 设置LLM客户端
            model_name=model_name,  # 设置模型名称
            language=language,  # 设置语言
        )
    # 定义异步方法 `map`，接受一个字典类型的查询上下文 `query_context` 并返回一个字符串列表
    async def map(self, query_context: dict) -> List[str]:
        """Rewrite the query."""
        # 从查询上下文中获取查询字符串 `query`
        query = query_context.get("query")
        # 从查询上下文中获取上下文信息 `context`
        context = query_context.get("context")
        # 如果没有查询字符串，则抛出值错误异常
        if not query:
            raise ValueError("query is required")
        # 调用 `_rewrite` 对象的 `rewrite` 方法，传入原始查询 `query`、上下文 `context` 和类的 `_nums` 属性
        return await self._rewrite.rewrite(
            origin_query=query, context=context, nums=self._nums
        )
```