# `.\DB-GPT-src\examples\agents\single_agent_dialogue_example.py`

```py
"""Define an asynchronous function 'main' to run the program.

    The 'main' function integrates several components:
    - Imports necessary modules and classes from dbgpt.agent.
    - Initializes an OpenAILLMClient for interaction with a specified GPT model.
    - Sets up an AgentContext object with a conversation ID 'test123'.
    - Creates an AgentMemory instance using HybridMemory with AgentMemoryFragment data.

    The function then proceeds to:
    - Instantiate a CodeAssistantAgent and configure it with context, LLMConfig, and agent_memory.
    - Initializes a UserProxyAgent with context and agent_memory.
    - Initiates a chat session between user_proxy and coder with a specified message.

    Finally, it prints the result of a query to agent_memory, specifically one_chat_completions("test123").

    This script is executed only if it is run directly as the main program.
"""

import asyncio

# Import necessary modules and classes from dbgpt.agent
from dbgpt.agent import (
    AgentContext,
    AgentMemory,
    AgentMemoryFragment,
    HybridMemory,
    LLMConfig,
    UserProxyAgent,
)
# Import CodeAssistantAgent class from dbgpt.agent.expand.code_assistant_agent
from dbgpt.agent.expand.code_assistant_agent import CodeAssistantAgent


async def main():
    # Import OpenAILLMClient from dbgpt.model.proxy
    from dbgpt.model.proxy import OpenAILLMClient

    # Create an instance of OpenAILLMClient configured for 'gpt-3.5-turbo'
    llm_client = OpenAILLMClient(model_alias="gpt-3.5-turbo")
    
    # Create an AgentContext object with a conversation ID 'test123'
    context: AgentContext = AgentContext(conv_id="test123")
    
    # Initialize an AgentMemory instance using HybridMemory with AgentMemoryFragment data
    agent_memory = AgentMemory(HybridMemory[AgentMemoryFragment].from_chroma())

    # Instantiate a CodeAssistantAgent and configure it with context, LLMConfig, and agent_memory
    coder = (
        await CodeAssistantAgent()
        .bind(context)
        .bind(LLMConfig(llm_client=llm_client))
        .bind(agent_memory)
        .build()
    )

    # Initialize a UserProxyAgent and bind it with context and agent_memory
    user_proxy = await UserProxyAgent().bind(context).bind(agent_memory).build()

    # Initiate a chat session between user_proxy and coder with a specified message
    await user_proxy.initiate_chat(
        recipient=coder,
        reviewer=user_proxy,
        message="计算下321 * 123等于多少",  # Compute 321 * 123 using Python code
        # message="download data from https://raw.githubusercontent.com/uwdata/draco/master/data/cars.csv and plot a visualization that tells us about the relationship between weight and horsepower. Save the plot to a file. Print the fields in a dataset before visualizing it.",
    )
    
    # Print the result of a query to agent_memory using one_chat_completions("test123")
    print(await agent_memory.gpts_memory.one_chat_completions("test123"))


if __name__ == "__main__":
    # Run the main function asynchronously
    asyncio.run(main())
```