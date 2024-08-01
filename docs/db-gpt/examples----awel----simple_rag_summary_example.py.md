# `.\DB-GPT-src\examples\awel\simple_rag_summary_example.py`

```py
"""AWEL:
This example shows how to use AWEL to build a simple rag summary example.
    pre-requirements:
        1. install openai python sdk
        ```
            pip install openai
        ```py
        2. set openai key and base
        ```
            export OPENAI_API_KEY={your_openai_key}
            export OPENAI_API_BASE={your_openai_base}
        ```py
        or
        ```
            import os
            os.environ["OPENAI_API_KEY"] = {your_openai_key}
            os.environ["OPENAI_API_BASE"] = {your_openai_base}
        ```py
        python examples/awel/simple_rag_summary_example.py
    Example:

    .. code-block:: shell

        curl -X POST http://127.0.0.1:5555/api/v1/awel/trigger/examples/rag/summary \
        -H "Content-Type: application/json" -d '{
            "url": "https://docs.dbgpt.site/docs/awel"
        }'
"""
from typing import Dict

from dbgpt._private.pydantic import BaseModel, Field
from dbgpt.core.awel import DAG, HttpTrigger, MapOperator
from dbgpt.model.proxy import OpenAILLMClient
from dbgpt.rag.knowledge import KnowledgeType
from dbgpt.rag.operators import KnowledgeOperator, SummaryAssemblerOperator


class TriggerReqBody(BaseModel):
    url: str = Field(..., description="url")


class RequestHandleOperator(MapOperator[TriggerReqBody, Dict]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def map(self, input_value: TriggerReqBody) -> Dict:
        params = {
            "url": input_value.url,
        }
        print(f"Receive input value: {input_value}")
        return params


# Create a Directed Acyclic Graph (DAG) with the name "dbgpt_awel_simple_rag_summary_example"
with DAG("dbgpt_awel_simple_rag_summary_example") as dag:
    # Define an HTTP trigger for the DAG with specified endpoint and request body structure
    trigger = HttpTrigger(
        "/examples/rag/summary", methods="POST", request_body=TriggerReqBody
    )
    # Create a task to handle the request and extract necessary information
    request_handle_task = RequestHandleOperator()
    # Extract the URL from the request parameters
    path_operator = MapOperator(lambda request: request["url"])
    # Create a knowledge operator with the type as URL
    knowledge_operator = KnowledgeOperator(knowledge_type=KnowledgeType.URL.name)
    # Create a summary assembler operator with OpenAI language model client and language set to English
    summary_operator = SummaryAssemblerOperator(
        llm_client=OpenAILLMClient(), language="en"
    )
    # Define the workflow by chaining the operators together
    (
        trigger
        >> request_handle_task
        >> path_operator
        >> knowledge_operator
        >> summary_operator
    )

# Check if the script is being run directly
if __name__ == "__main__":
    # Check if the DAG is in development mode
    if dag.leaf_nodes[0].dev_mode:
        # If in development mode, set up a local environment for debugging
        from dbgpt.core.awel import setup_dev_environment

        setup_dev_environment([dag], port=5555)
    else:
        pass
```