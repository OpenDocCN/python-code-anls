# `.\DB-GPT-src\dbgpt\rag\retriever\rewrite.py`

```py
"""Query rewrite."""

# 导入所需模块和类
from typing import List, Optional

from dbgpt.core import LLMClient, ModelMessage, ModelMessageRoleType, ModelRequest
from dbgpt.core.awel.flow import Parameter, ResourceCategory, register_resource
from dbgpt.util.i18n_utils import _

# 英文版重写提示模板，用于生成查询
REWRITE_PROMPT_TEMPLATE_EN = """
Based on the given context {context}, Generate {nums} search queries related to:
{original_query}, Provide following comma-separated format: 'queries: <queries>'":
    "original query:{original_query}\n"
    "queries:"
"""

# 中文版重写提示模板，用于生成查询
REWRITE_PROMPT_TEMPLATE_ZH = """请根据上下文{context}, 将原问题优化生成{nums}个相关的搜索查询，
这些查询应与原始查询相似并且是人们可能会提出的可回答的搜索问题。请勿使用任何示例中提到的内容，确保所有
生成的查询均独立于示例，仅基于提供的原始查询。请按照以下逗号分隔的格式提供: 'queries:<queries>'
"original_query:{original_query}\n"
"queries:"
"""

# 注册查询重写资源，包括名称、标识符、类别和描述等信息
@register_resource(
    _("Query Rewrite"),
    "query_rewrite",
    category=ResourceCategory.RAG,
    description=_("Query rewrite."),
    parameters=[
        Parameter.build_from(
            _("Model Name"),
            "model_name",
            str,
            description=_("The LLM model name."),
        ),
        Parameter.build_from(
            _("LLM Client"),
            "llm_client",
            LLMClient,
            description=_("The llm client."),
        ),
        Parameter.build_from(
            _("Language"),
            "language",
            str,
            description=_("The language of the query rewrite prompt."),
            optional=True,
            default="en",
        ),
    ],
)
class QueryRewrite:
    """Query rewrite.

    query reinforce, include query rewrite, query correct
    """

    def __init__(
        self,
        model_name: str,
        llm_client: LLMClient,
        language: Optional[str] = "en",
    ) -> None:
        """Create QueryRewrite with model_name, llm_client, language.

        Args:
            model_name(str): model name
            llm_client(LLMClient, optional): llm client
            language(str, optional): language
        """
        # 初始化实例变量
        self._model_name = model_name  # 设置模型名称
        self._llm_client = llm_client  # 设置LLM客户端
        self._language = language  # 设置语言选项
        # 根据语言选项选择对应的重写提示模板
        self._prompt_template = (
            REWRITE_PROMPT_TEMPLATE_EN
            if language == "en"
            else REWRITE_PROMPT_TEMPLATE_ZH
        )

    async def rewrite(
        self, origin_query: str, context: Optional[str], nums: Optional[int] = 1
    ):
        """Generate rewritten queries based on the original query and context.

        Args:
            origin_query (str): The original query to be rewritten.
            context (str, optional): The context provided for rewriting.
            nums (int, optional): Number of rewritten queries to generate. Default is 1.
        """
        # 进行异步重写查询的操作，返回生成的重写查询结果
        pass  # Placeholder for actual implementation
    def rewrite_queries(
        origin_query: str,
        context: Optional[str],
        nums: Optional[int]
    ) -> List[str]:
        """Rewrite the original query multiple times based on a given context and number.

        Args:
            origin_query: str original query
            context: Optional[str] context
            nums: Optional[int] rewrite nums

        Returns:
            queries: List[str]
        """
        # Import necessary function for running asynchronous tasks
        from dbgpt.util.chat_util import run_async_tasks

        # Construct the prompt using the provided template
        prompt = self._prompt_template.format(
            context=context, original_query=origin_query, nums=nums
        )

        # Create a system message containing the prompt
        messages = [ModelMessage(role=ModelMessageRoleType.SYSTEM, content=prompt)]

        # Create a request object for the language model with the messages
        request = ModelRequest(model=self._model_name, messages=messages)

        # Generate responses asynchronously using the language model client
        tasks = [self._llm_client.generate(request)]
        queries = await run_async_tasks(tasks=tasks, concurrency_limit=1)

        # Extract text content from model responses
        queries = [model_out.text for model_out in queries]

        # Filter out queries that indicate generation errors
        queries = list(
            filter(
                lambda content: "LLMServer Generate Error" not in content,
                queries,
            )
        )

        # Handle case where no valid rewrite queries are generated
        if len(queries) == 0:
            print("llm generate no rewrite queries.")
            return queries

        # Parse the first generated query output and limit to specified number of queries
        new_queries = self._parse_llm_output(output=queries[0])[0:nums]
        print(f"rewrite queries: {new_queries}")

        return new_queries

    def correct(self) -> List[str] | None:
        """Placeholder method for query correction."""
        pass

    def _parse_llm_output(self, output: str) -> List[str]:
        """Parse the output from the language model.

        Args:
            output: str

        Returns:
            results: List[str]
        """
        # Initialize variables
        lowercase = True
        try:
            results = []
            response = output.strip()

            # Handle variations of delimiters in the model output
            if response.startswith("queries:"):
                response = response[len("queries:") :]
            if response.startswith("queries："):
                response = response[len("queries：") :]
            
            # Split the response into individual queries based on different separators
            queries = response.split(",")
            if len(queries) == 1:
                queries = response.split("，")
            if len(queries) == 1:
                queries = response.split("?")
            if len(queries) == 1:
                queries = response.split("？")
            
            # Process each query, removing unnecessary prefixes and applying lowercase if specified
            for k in queries:
                if k.startswith("queries:"):
                    k = k[len("queries:") :]
                if k.startswith("queries："):
                    k = response[len("queries：") :]
                rk = k
                if lowercase:
                    rk = rk.lower()
                s = rk.strip()
                if s == "":
                    continue
                results.append(s)
        except Exception as e:
            # Handle any exceptions that occur during parsing
            print(f"parse query rewrite prompt_response error: {e}")
            return []

        return results
```