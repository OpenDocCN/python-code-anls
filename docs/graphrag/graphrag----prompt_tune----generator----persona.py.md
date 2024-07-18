# `.\graphrag\graphrag\prompt_tune\generator\persona.py`

```py
# 版权声明，版权由2024年Microsoft Corporation所有，根据MIT许可证授权

"""用于微调GraphRAG提示的Persona生成模块。"""

# 导入所需的库和模块
from graphrag.llm.types.llm_types import CompletionLLM
from graphrag.prompt_tune.generator.defaults import DEFAULT_TASK
from graphrag.prompt_tune.prompt import GENERATE_PERSONA_PROMPT

# 异步函数，用于生成LLM persona以供GraphRAG提示使用
async def generate_persona(
    llm: CompletionLLM, domain: str, task: str = DEFAULT_TASK
) -> str:
    """Generate an LLM persona to use for GraphRAG prompts.

    Parameters
    ----------
    - llm (CompletionLLM): The LLM to use for generation
    - domain (str): The domain to generate a persona for
    - task (str): The task to generate a persona for. Default is DEFAULT_TASK
    """
    # 格式化任务名称并拼接领域信息
    formatted_task = task.format(domain=domain)
    # 生成persona提示用于请求LLM生成persona
    persona_prompt = GENERATE_PERSONA_PROMPT.format(sample_task=formatted_task)

    # 调用LLM生成persona
    response = await llm(persona_prompt)

    # 返回生成的persona作为字符串
    return str(response.output)
```