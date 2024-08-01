# `.\DB-GPT-src\dbgpt\rag\operators\summary.py`

```py
"""The summary operator."""

# 导入必要的模块和类
from typing import Any, Optional

from dbgpt.core import LLMClient
from dbgpt.core.awel.flow import IOField, OperatorCategory, Parameter, ViewMetadata
from dbgpt.rag.assembler.summary import SummaryAssembler
from dbgpt.rag.knowledge.base import Knowledge
from dbgpt.rag.operators.assembler import AssemblerOperator
from dbgpt.util.i18n_utils import _

# 定义摘要装配器操作符类，继承自装配器操作符基类
class SummaryAssemblerOperator(AssemblerOperator[Any, Any]):
    """The summary assembler operator."""

    # 定义视图元数据，包括标签、名称、类别、描述、输入输出字段和参数
    metadata = ViewMetadata(
        label=_("Summary Operator"),
        name="summary_assembler_operator",
        category=OperatorCategory.RAG,
        description=_("The summary assembler operator."),
        inputs=[
            IOField.build_from(
                _("Knowledge"), "knowledge", Knowledge, _("Knowledge datasource")
            )
        ],
        outputs=[
            IOField.build_from(
                _("Document summary"),
                "summary",
                str,
                description="document summary",
            )
        ],
        parameters=[
            Parameter.build_from(
                _("LLM Client"),
                "llm_client",
                LLMClient,
                optional=True,
                default=None,
                description=_("The LLM Client."),
            ),
            Parameter.build_from(
                label=_("Model name"),
                name="model_name",
                type=str,
                optional=True,
                default="gpt-3.5-turbo",
                description=_("LLM model name"),
            ),
            Parameter.build_from(
                label=_("prompt language"),
                name="language",
                type=str,
                optional=True,
                default="en",
                description=_("prompt language"),
            ),
            Parameter.build_from(
                label=_("Max iteration with LLM"),
                name="max_iteration_with_llm",
                type=int,
                optional=True,
                default=5,
                description=_("prompt language"),
            ),
            Parameter.build_from(
                label=_("Concurrency limit with LLM"),
                name="concurrency_limit_with_llm",
                type=int,
                optional=True,
                default=3,
                description=_("The concurrency limit with llm"),
            ),
        ],
        documentation_url="https://github.com/openai/openai-python",
    )

    # 定义初始化方法，接收多个可选参数和关键字参数
    def __init__(
        self,
        llm_client: Optional[LLMClient],
        model_name: Optional[str] = "gpt-3.5-turbo",
        language: Optional[str] = "en",
        max_iteration_with_llm: Optional[int] = 5,
        concurrency_limit_with_llm: Optional[int] = 3,
        **kwargs
    ):
        """
        创建汇总组装操作符。

        Args:
              llm_client: (Optional[LLMClient]) LLM 客户端。
              model_name: (Optional[str]) 模型名称。
              language: (Optional[str]) 提示语言。
              max_iteration_with_llm: (Optional[int]) 使用LLM的最大迭代次数。
              concurrency_limit_with_llm: (Optional[int]) 使用LLM的并发限制。
        """
        super().__init__(**kwargs)
        self._llm_client = llm_client
        self._model_name = model_name
        self._language = language
        self._max_iteration_with_llm = max_iteration_with_llm
        self._concurrency_limit_with_llm = concurrency_limit_with_llm

    async def map(self, knowledge: Knowledge) -> str:
        """
        组装摘要。

        使用知识加载摘要组装器，设置所需的参数。

        Args:
            knowledge: (Knowledge) 知识对象。

        Returns:
            str: 生成的摘要字符串。
        """
        assembler = SummaryAssembler.load_from_knowledge(
            knowledge=knowledge,
            llm_client=self._llm_client,
            model_name=self._model_name,
            language=self._language,
            max_iteration_with_llm=self._max_iteration_with_llm,
            concurrency_limit_with_llm=self._concurrency_limit_with_llm,
        )
        return await assembler.generate_summary()

    def assemble(self, knowledge: Knowledge) -> Any:
        """
        组装摘要。

        此方法目前未实现，仅声明返回类型。

        Args:
            knowledge: (Knowledge) 知识对象。

        Returns:
            Any: 任意类型，当前方法未实现。
        """
        pass
```