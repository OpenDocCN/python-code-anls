# AutoGPT源码解析 23

# AutoGPT Forge: Crafting Intelligent Agent Logic

![Header](../../../docs/content/imgs/quickstart/t3_01.png)
**By Craig Swift & [Ryan Brandt](https://github.com/paperMoose)**

Hey there! Ready for part 3 of our AutoGPT Forge tutorial series? If you missed the earlier parts, catch up here:

- [Getting Started](001_getting_started.md)
- [Blueprint of an Agent](002_blueprint_of_an_agent.md)

Now, let's get hands-on! We'll use an LLM to power our agent and complete a task. The challenge? Making the agent write "Washington" to a .txt file. We won't give it step-by-step instructions—just the task. Let's see our agent in action and watch it figure out the steps on its own!


## Get Your Smart Agent Project Ready

Make sure you've set up your project and created an agent as described in our initial guide. If you skipped that part, [click here](#) to get started. Once you're done, come back, and we'll move forward.

In the image below, you'll see my "SmartAgent" and the agent.py file inside the 'forge' folder. That's where we'll be adding our LLM-based logic. If you're unsure about the project structure or agent functions from our last guide, don't worry. We'll cover the basics as we go!

![SmartAgent](../../../docs/content/imgs/quickstart/t3_02.png)

---

## The Task Lifecycle

The lifecycle of a task, from its creation to execution, is outlined in the agent protocol. In simple terms: a task is initiated, its steps are systematically executed, and it concludes once completed.

Want your agent to perform an action? Start by dispatching a create_task request. This crucial step involves specifying the task details, much like how you'd send a prompt to ChatGPT, using the input field. If you're giving this a shot on your own, the UI is your best friend; it effortlessly handles all the API calls on your behalf.

When the agent gets this, it runs the create_task function. The code `super().create_task(task_request)` takes care of protocol steps. It then logs the task's start. For this guide, you don't need to change this function.

```python
async def create_task(self, task_request: TaskRequestBody) -> Task:
    """
    The agent protocol, which is the core of the Forge, works by creating a task and then
    executing steps for that task. This method is called when the agent is asked to create
    a task.

    We are hooking into function to add a custom log message. Though you can do anything you
    want here.
    """
    task = await super().create_task(task_request)
    LOG.info(
        f"📦 Task created: {task.task_id} input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
    )
    return task
```py

After starting a task, the `execute_step` function runs until all steps are done. Here's a basic view of `execute_step`. I've left out the detailed comments for simplicity, but you'll find them in your project.

```python
async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
    # An example that
      step = await self.db.create_step(
          task_id=task_id, input=step_request, is_last=True
      )

      self.workspace.write(task_id=task_id, path="output.txt", data=b"Washington D.C")

      await self.db.create_artifact(
          task_id=task_id,
          step_id=step.step_id,
          file_name="output.txt",
          relative_path="",
          agent_created=True,
      )
      
      step.output = "Washington D.C"

      LOG.info(f"\t✅ Final Step completed: {step.step_id}")

      return step
```py

Here's the breakdown of the 'write file' process in four steps:

1. **Database Step Creation**: The first stage is all about creating a step within the database, an essential aspect of the agent protocol. You'll observe that while setting up this step, we've flagged it with `is_last=True`. This signals to the agent protocol that no more steps are pending. For the purpose of this guide, let's work under the assumption that our agent will only tackle single-step tasks. However, hang tight for future tutorials, where we'll level up and let the agent determine its completion point.

2. **File Writing**: Next, we pen down "Washington D.C." using the workspace.write function.

3. **Artifact Database Update**: After writing, we record the file in the agent's artifact database.

4. **Step Output & Logging**: Finally, we set the step output to match the file content, log the executed step, and use the step object.

With the 'write file' process clear, let's make our agent smarter and more autonomous. Ready to dive in?

---

## Building the Foundations For Our Smart Agent

First, we need to update the `execute_step()` function. Instead of a fixed solution, it should use the given request.

To do this, we'll fetch the task details using the provided `task_id`:

```python
task = await self.db.get_task(task_id)
```py

Next, remember to create a database record and mark it as a single-step task with `is_last=True`:

```python
step = await self.db.create_step(
    task_id=task_id, input=step_request, is_last=True
)
```py

Your updated `execute_step` function will look like this:

```python
async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
    # Get the task details
    task = await self.db.get_task(task_id)

    # Add a new step to the database
    step = await self.db.create_step(
        task_id=task_id, input=step_request, is_last=True
    )
    return step
```py

Now that we've set this up, let's move to the next exciting part: The PromptEngine.

---


**The Art of Prompting**  

![Prompting 101](../../../docs/content/imgs/quickstart/t3_03.png)

Prompting is like shaping messages for powerful language models like ChatGPT. Since these models respond to input details, creating the right prompt can be a challenge. That's where the **PromptEngine** comes in.

The "PromptEngine" helps you store prompts in text files, specifically in Jinja2 templates. This means you can change the prompts without changing the code. It also lets you adjust prompts for different LLMs. Here's how to use it:

First, add the PromptEngine from the SDK:

```python
from .sdk import PromptEngine
```py

In your `execute_step` function, set up the engine for the `gpt-3.5-turbo` LLM:

```python
prompt_engine = PromptEngine("gpt-3.5-turbo")
```py

Loading a prompt is straightforward. For instance, loading the `system-format` prompt, which dictates the response format from the LLM, is as easy as:

```python
system_prompt = prompt_engine.load_prompt("system-format")
```py

For intricate use cases, like the `task-step` prompt which requires parameters, employ the following method:

```python
# Define the task parameters
task_kwargs = {
    "task": task.input,
    "abilities": self.abilities.list_abilities_for_prompt(),
}

# Load the task prompt with those parameters
task_prompt = prompt_engine.load_prompt("task-step", **task_kwargs)
```py



Delving deeper, let's look at the `task-step` prompt template in `prompts/gpt-3.5-turbo/task-step.j2`:

```jinja
{% extends "techniques/expert.j2" %}
{% block expert %}Planner{% endblock %}
{% block prompt %}
Your task is:

{{ task }}

Ensure to respond in the given format. Always make autonomous decisions, devoid of user guidance. Harness the power of your LLM, opting for straightforward tactics sans any legal entanglements.
{% if constraints %}
## Constraints
Operate under these confines:
{% for constraint in constraints %}
- {{ constraint }}
{% endfor %}
{% endif %}
{% if resources %}
## Resources
Utilize these resources:
{% for resource in resources %}
- {{ resource }}
{% endfor %}
{% endif %}
{% if abilities %}
## Abilities
Summon these abilities:
{% for ability in abilities %}
- {{ ability }}
{% endfor %}
{% endif %}

{% if abilities %}
## Abilities
Use these abilities:
{% for ability in abilities %}
- {{ ability }}
{% endfor %}
{% endif %}

{% if best_practices %}
## Best Practices
{% for best_practice in best_practices %}
- {{ best_practice }}
{% endfor %}
{% endif %}
{% endblock %}
```py

This template is modular. It uses the `extends` directive to build on the `expert.j2` template. The different sections like constraints, resources, abilities, and best practices make the prompt dynamic. It guides the LLM in understanding the task and using resources and abilities.

The PromptEngine equips us with a potent tool to converse seamlessly with large language models. By externalizing prompts and using templates, we can ensure that our agent remains agile, adapting to new challenges without a code overhaul. As we march forward, keep this foundation in mind—it's the bedrock of our agent's intelligence.

---

## Engaging with your LLM

To make the most of the LLM, you'll send a series of organized instructions, not just one prompt. Structure your prompts as a list of messages for the LLM. Using the `system_prompt` and `task_prompt` from before, create the `messages` list:

```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": task_prompt}
]
```py

With the prompt set, send it to the LLM. This step involves foundational code, focusing on the `chat_completion_request`. This function gives the LLM your prompt, and then gets the LLM's output. The other code sets up our request and interprets the feedback:

```python
try:
    # Set the parameters for the chat completion
    chat_completion_kwargs = {
        "messages": messages,
        "model": "gpt-3.5-turbo",
    }
    # Get the LLM's response and interpret it
    chat_response = await chat_completion_request(**chat_completion_kwargs)
    answer = json.loads(chat_response["choices"][0]["message"]["content"])

    # Log the answer for reference
    LOG.info(pprint.pformat(answer))

except json.JSONDecodeError as e:
    # Handle JSON decoding errors
    LOG.error(f"Can't decode chat response: {chat_response}")
except Exception as e:
    # Handle other errors
    LOG.error(f"Can't get chat response: {e}")
```py

Extracting clear messages from LLM outputs can be complex. Our method is simple and works with GPT-3.5 and GPT-4. Future guides will show more ways to interpret LLM outputs. The goal? To go beyond JSON, as some LLMs work best with other response types. Stay tuned!

---


## Using and Creating Abilities

Abilities are the gears and levers that enable the agent to interact with tasks at hand. Let's unpack the mechanisms behind these abilities and how you can harness, and even extend, them.

In the SDK, there's a `abilities` folder containing `registry.py`, `finish.py`, and a `file_system` subfolder. You can also add your own abilities here. `registry.py` is the main file for abilities. It contains the `@ability` decorator and the `AbilityRegister` class. This class actively tracks abilities and outlines their function. The base Agent class includes a default ability register available via `self.abilities`. It looks like this:

```python
self.abilities = AbilityRegister(self)
```py

The `AbilityRegister` has two key methods. `list_abilities_for_prompt` prepares abilities for prompts. `run_ability` makes the ability work. An ability is a function with the `@ability` decorator. It must have specific parameters, including the agent and `task_id`.

```python
@ability(
    name="write_file",
    description="Write data to a file",
    parameters=[
        {
            "name": "file_path",
            "description": "Path to the file",
            "type": "string",
            "required": True,
        },
        {
            "name": "data",
            "description": "Data to write to the file",
            "type": "bytes",
            "required": True,
        },
    ],
    output_type="None",
)
async def write_file(agent, task_id: str, file_path: str, data: bytes) -> None:
    pass
```py

The `@ability` decorator defines the ability's details, like its identity (name), functionality (description), and operational parameters.

## Example of a Custom Ability: Webpage Fetcher

```python
import requests

@ability(
  name="fetch_webpage",
  description="Retrieve the content of a webpage",
  parameters=[
      {
          "name": "url",
          "description": "Webpage URL",
          "type": "string",
          "required": True,
      }
  ],
  output_type="string",
)
async def fetch_webpage(agent, task_id: str, url: str) -> str:
  response = requests.get(url)
  return response.text
```py

This ability, `fetch_webpage`, accepts a URL as input and returns the HTML content of the webpage as a string. Custom abilities let you add more features to your agent. They can integrate other tools and libraries to enhance its functions. To make a custom ability, you need to understand the structure and add technical details. With abilities like "fetch_webpage", your agent can handle complex tasks efficiently.

## Running an Ability

Now that you understand abilities and how to create them, let's use them. The last piece is the `execute_step` function. Our goal is to understand the agent's response, find the ability, and use it. 

First, we get the ability details from the agent's answer:

```python
# Extract the ability from the answer
ability = answer["ability"]
```py

With the ability details, we use it. We call the `run_ability` function:

```python
# Run the ability and get the output
# We don't actually use the output in this example
output = await self.abilities.run_ability(
    task_id, ability["name"], **ability["args"]
)
```py

Here, we’re invoking the specified ability. The task_id ensures continuity, ability['name'] pinpoints the exact function, and the arguments (ability["args"]) provide necessary context.

Finally, we make the step's output show the agent's thinking:

```python
# Set the step output to the "speak" part of the answer
step.output = answer["thoughts"]["speak"]

# Return the completed step
return step
```py

And there you have it! Your first Smart Agent, sculpted with precision and purpose, stands ready to take on challenges. The stage is set. It’s showtime!

Here is what your function should look like:

```python
async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
    # Firstly we get the task this step is for so we can access the task input
    task = await self.db.get_task(task_id)

    # Create a new step in the database
    step = await self.db.create_step(
        task_id=task_id, input=step_request, is_last=True
    )

    # Log the message
    LOG.info(f"\t✅ Final Step completed: {step.step_id} input: {step.input[:19]}")

    # Initialize the PromptEngine with the "gpt-3.5-turbo" model
    prompt_engine = PromptEngine("gpt-3.5-turbo")

    # Load the system and task prompts
    system_prompt = prompt_engine.load_prompt("system-format")

    # Initialize the messages list with the system prompt
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    # Define the task parameters
    task_kwargs = {
        "task": task.input,
        "abilities": self.abilities.list_abilities_for_prompt(),
    }

    # Load the task prompt with the defined task parameters
    task_prompt = prompt_engine.load_prompt("task-step", **task_kwargs)

    # Append the task prompt to the messages list
    messages.append({"role": "user", "content": task_prompt})

    try:
        # Define the parameters for the chat completion request
        chat_completion_kwargs = {
            "messages": messages,
            "model": "gpt-3.5-turbo",
        }
        # Make the chat completion request and parse the response
        chat_response = await chat_completion_request(**chat_completion_kwargs)
        answer = json.loads(chat_response["choices"][0]["message"]["content"])

        # Log the answer for debugging purposes
        LOG.info(pprint.pformat(answer))

    except json.JSONDecodeError as e:
        # Handle JSON decoding errors
        LOG.error(f"Unable to decode chat response: {chat_response}")
    except Exception as e:
        # Handle other exceptions
        LOG.error(f"Unable to generate chat response: {e}")

    # Extract the ability from the answer
    ability = answer["ability"]

    # Run the ability and get the output
    # We don't actually use the output in this example
    output = await self.abilities.run_ability(
        task_id, ability["name"], **ability["args"]
    )

    # Set the step output to the "speak" part of the answer
    step.output = answer["thoughts"]["speak"]

    # Return the completed step
    return step
```py

## Interacting with your Agent
> ⚠️ Heads up: The UI and benchmark are still in the oven, so they might be a tad glitchy.

With the heavy lifting of crafting our Smart Agent behind us, it’s high time to see it in action. Kick things off by firing up the agent with this command:
```bash
./run agent start SmartAgent.
```py

Once your digital playground is all set, your terminal should light up with:
```bash


       d8888          888             .d8888b.  8888888b. 88888888888 
      d88888          888            d88P  Y88b 888   Y88b    888     
     d88P888          888            888    888 888    888    888     
    d88P 888 888  888 888888 .d88b.  888        888   d88P    888     
   d88P  888 888  888 888   d88""88b 888  88888 8888888P"     888     
  d88P   888 888  888 888   888  888 888    888 888           888     
 d8888888888 Y88b 888 Y88b. Y88..88P Y88b  d88P 888           888     
d88P     888  "Y88888  "Y888 "Y88P"   "Y8888P88 888           888     
                                                                      
                                                                      
                                                                      
                8888888888                                            
                888                                                   
                888                                                   
                8888888  .d88b.  888d888 .d88b.   .d88b.              
                888     d88""88b 888P"  d88P"88b d8P  Y8b             
                888     888  888 888    888  888 88888888             
                888     Y88..88P 888    Y88b 888 Y8b.                 
                888      "Y88P"  888     "Y88888  "Y8888              
                                             888                      
                                        Y8b d88P                      
                                         "Y88P"                v0.1.0


[2023-09-27 15:39:07,832] [forge.sdk.agent] [INFO]      📝  Agent server starting on http://localhost:8000

```py
1. **Get Started**
   - Click the link to access the AutoGPT Agent UI.

2. **Login**
   - Log in using your Gmail or Github credentials.

3. **Navigate to Benchmarking**
   - Look to the left, and you'll spot a trophy icon. Click it to enter the benchmarking arena.
  
![Benchmarking page of the AutoGPT UI](../../../docs/content/imgs/quickstart/t3_04.png)

4. **Select the 'WriteFile' Test**
   - Choose the 'WriteFile' test from the available options.

5. **Initiate the Test Suite**
   - Hit 'Initiate test suite' to start the benchmarking process.

6. **Monitor in Real-Time**
   - Keep your eyes on the right panel as it displays real-time output.

7. **Check the Console**
   - For additional information, you can also monitor your console for progress updates and messages.
```bash
📝  📦 Task created: 70518b75-0104-49b0-923e-f607719d042b input: Write the word 'Washington' to a .txt fi...
📝      ✅ Final Step completed: a736c45f-65a5-4c44-a697-f1d6dcd94d5c input: y
```
If you see this, you've done it!

8. **Troubleshooting**
   - If you encounter any issues or see cryptic error messages, don't worry. Just hit the retry button. Remember, LLMs are powerful but may occasionally need some guidance.

## Wrap Up
- Stay tuned for our next tutorial, where we'll enhance the agent's capabilities by adding memory!

## Keep Exploring
- Keep experimenting and pushing the boundaries of AI. Happy coding! 🚀

## Wrap Up
In our next tutorial, we’ll further refine this process, enhancing the agent’s capabilities, through the addition of memory!

Until then, keep experimenting and pushing the boundaries of AI. Happy coding! 🚀


# Memory Integration: Enabling Your Agent to Remember and Learn

## Introduction
- Importance of Memory Integration in AI Agents
- Overview of Memory Mechanisms in AutoGPT

## Section 1: Understanding Memory Integration
- Concept of Memory in AI Agents
- Types of Memory: Short-term vs. Long-term

## Section 2: Implementing Memory in Your Agent
- Setting up Memory Structures in the Forge Environment
- Utilizing Agent Protocol for Memory Integration

## Section 3: Developing Learning Mechanisms
- Creating Learning Algorithms for Your Agent
- Implementing Learning Mechanisms using Task and Artifact Schemas

## Section 4: Testing and Optimizing Memory Integration
- Employing AGBenchmark for Memory Testing
- Optimizing Memory for Enhanced Performance and Efficiency

## Section 5: Best Practices in Memory Integration
- Tips and Strategies for Effective Memory Integration
- Avoiding Common Pitfalls in Memory Development

## Conclusion
- Recap of the Tutorial
- Future Directions in Memory Integration

## Additional Resources

From **The Rise and Potential of Large Language Model Based Agents: A Survey** *Zhiheng Xi (Fudan University) et al. arXiv.* [[paper](https://arxiv.org/abs/2305.14497)] [[code](https://github.com/woooodyy/llm-agent-paper-list)]

##### Memory capability

###### Raising the length limit of Transformers

- [2023/05] **Randomized Positional Encodings Boost Length Generalization of Transformers.** *Anian Ruoss (DeepMind) et al. arXiv.* [[paper](https://arxiv.org/abs/2305.16843)] [[code](https://github.com/google-deepmind/randomized_positional_encodings)]
- [2023-03] **CoLT5: Faster Long-Range Transformers with Conditional Computation.** *Joshua Ainslie (Google Research) et al. arXiv.* [[paper](https://arxiv.org/abs/2303.09752)]
- [2022/03] **Efficient Classification of Long Documents Using Transformers.** *Hyunji Hayley Park (Illinois University) et al. arXiv.* [[paper](https://arxiv.org/abs/2203.11258)] [[code](https://github.com/amazon-science/efficient-longdoc-classification)]
- [2021/12] **LongT5: Efficient Text-To-Text Transformer for Long Sequences.** *Mandy Guo (Google Research) et al. arXiv.* [[paper](https://arxiv.org/abs/2112.07916)] [[code](https://github.com/google-research/longt5)]
- [2019/10] **BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.** *Michael Lewis(Facebook AI) et al. arXiv.* [[paper](https://arxiv.org/abs/1910.13461)] [[code](https://github.com/huggingface/transformers/tree/main/src/transformers/models/bart)]

###### Summarizing memory

- [2023/08] **ExpeL: LLM Agents Are Experiential Learners.** *Andrew Zhao (Tsinghua University) et al. arXiv.* [[paper](https://arxiv.org/abs/2308.10144)] [[code]([https://github.com/thunlp/ChatEval](https://github.com/Andrewzh112/ExpeL))]
- [2023/08] **ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate.** *Chi-Min Chan (Tsinghua University) et al. arXiv.* [[paper](https://arxiv.org/abs/2308.07201)] [[code](https://github.com/thunlp/ChatEval)]
- [2023/05] **MemoryBank: Enhancing Large Language Models with Long-Term Memory.** *Wanjun Zhong (Harbin Institute of Technology) et al. arXiv.* [[paper](https://arxiv.org/abs/2305.10250)] [[code](https://github.com/zhongwanjun/memorybank-siliconfriend)]
- [2023/04] **Generative Agents: Interactive Simulacra of Human Behavior.** *Joon Sung Park (Stanford University) et al. arXiv.* [[paper](https://arxiv.org/abs/2304.03442)] [[code](https://github.com/joonspk-research/generative_agents)]
- [2023/04] **Unleashing Infinite-Length Input Capacity for Large-scale Language Models with Self-Controlled Memory System.** *Xinnian Liang(Beihang University) et al. arXiv.* [[paper](https://arxiv.org/abs/2304.13343)] [[code](https://github.com/wbbeyourself/scm4llms)]
- [2023/03] **Reflexion: Language Agents with Verbal Reinforcement Learning.** *Noah Shinn (Northeastern University) et al. arXiv.* [[paper](https://arxiv.org/abs/2303.11366)] [[code](https://github.com/noahshinn024/reflexion)]
- [2023/05] **RecurrentGPT: Interactive Generation of (Arbitrarily) Long Text.** Wangchunshu Zhou (AIWaves) et al. arXiv.* [[paper](https://arxiv.org/pdf/2305.13304.pdf)] [[code](https://github.com/aiwaves-cn/RecurrentGPT)]  


###### Compressing memories with vectors or data structures

- [2023/07] **Communicative Agents for Software Development.** *Chen Qian (Tsinghua University) et al. arXiv.* [[paper](https://arxiv.org/abs/2307.07924)] [[code](https://github.com/openbmb/chatdev)]
- [2023/06] **ChatDB: Augmenting LLMs with Databases as Their Symbolic Memory.** *Chenxu Hu(Tsinghua University) et al. arXiv.* [[paper](https://arxiv.org/abs/2306.03901)] [[code](https://github.com/huchenxucs/ChatDB)]
- [2023/05] **Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory.** *Xizhou Zhu (Tsinghua University) et al. arXiv.* [[paper](https://arxiv.org/abs/2305.17144)] [[code](https://github.com/OpenGVLab/GITM)]
- [2023/05] **RET-LLM: Towards a General Read-Write Memory for Large Language Models.** *Ali Modarressi (LMU Munich) et al. arXiv.* [[paper](https://arxiv.org/abs/2305.14322)] [[code](https://github.com/tloen/alpaca-lora)]
- [2023/05] **RecurrentGPT: Interactive Generation of (Arbitrarily) Long Text.** Wangchunshu Zhou (AIWaves) et al. arXiv.* [[paper](https://arxiv.org/pdf/2305.13304.pdf)] [[code](https://github.com/aiwaves-cn/RecurrentGPT)]

##### Memory retrieval

- [2023/08] **Memory Sandbox: Transparent and Interactive Memory Management for Conversational Agents.** *Ziheng Huang(University of California—San Diego) et al. arXiv.* [[paper](https://arxiv.org/abs/2308.01542)]
- [2023/08] **AgentSims: An Open-Source Sandbox for Large Language Model Evaluation.** *Jiaju Lin (PTA Studio) et al. arXiv.* [[paper](https://arxiv.org/abs/2308.04026)] [[project page](https://www.agentsims.com/)] [[code](https://github.com/py499372727/AgentSims/)] 
- [2023/06] **ChatDB: Augmenting LLMs with Databases as Their Symbolic Memory.** *Chenxu Hu(Tsinghua University) et al. arXiv.* [[paper](https://arxiv.org/abs/2306.03901)] [[code](https://github.com/huchenxucs/ChatDB)]
- [2023/05] **MemoryBank: Enhancing Large Language Models with Long-Term Memory.** *Wanjun Zhong (Harbin Institute of Technology) et al. arXiv.* [[paper](https://arxiv.org/abs/2305.10250)] [[code](https://github.com/zhongwanjun/memorybank-siliconfriend)]
- [2023/04] **Generative Agents: Interactive Simulacra of Human Behavior.** *Joon Sung Park (Stanford) et al. arXiv.* [[paper](https://arxiv.org/abs/2304.03442)] [[code](https://github.com/joonspk-research/generative_agents)]
- [2023/05] **RecurrentGPT: Interactive Generation of (Arbitrarily) Long Text.** Wangchunshu Zhou (AIWaves) et al. arXiv.* [[paper](https://arxiv.org/pdf/2305.13304.pdf)] [[code](https://github.com/aiwaves-cn/RecurrentGPT)]

## Appendix
- Examples of Memory Integration Implementations
- Glossary of Memory-Related Terms


# Auto-GPT Benchmarks

Built for the purpose of benchmarking the performance of agents regardless of how they work.

Objectively know how well your agent is performing in categories like code, retrieval, memory, and safety.

Save time and money while doing it through smart dependencies. The best part? It's all automated.

## Scores:

<img width="733" alt="Screenshot 2023-07-25 at 10 35 01 AM" src="https://github.com/Significant-Gravitas/Auto-GPT-Benchmarks/assets/9652976/98963e0b-18b9-4b17-9a6a-4d3e4418af70">

## Ranking overall:

- 1- [Beebot](https://github.com/AutoPackAI/beebot)
- 2- [mini-agi](https://github.com/muellerberndt/mini-agi)
- 3- [Auto-GPT](https://github.com/Significant-Gravitas/AutoGPT)

## Detailed results:

<img width="733" alt="Screenshot 2023-07-25 at 10 42 15 AM" src="https://github.com/Significant-Gravitas/Auto-GPT-Benchmarks/assets/9652976/39be464c-c842-4437-b28a-07d878542a83">

[Click here to see the results and the raw data!](https://docs.google.com/spreadsheets/d/1WXm16P2AHNbKpkOI0LYBpcsGG0O7D8HYTG5Uj0PaJjA/edit#gid=203558751)!

More agents coming soon !


# `benchmark/server.py`

这段代码使用了多种Python库，包括fastapi、json、logging和shutil，作用是为了解决特定的任务或问题。下面是每部分代码的作用：

1. `import io`：用于导入io库，以便我们使用其中的流对象（如文件流或字符流等）。
2. `import json`：用于导入json库，以便我们能够方便地解析和生成JSON格式的数据。
3. `import logging`：用于导入logging库，以便我们能够方便地使用日志记录功能。
4. `import shutil`：用于导入shutil库，以便我们能够方便地处理文件和目录操作。
5. `from pathlib import Path`：用于从pathlib库中导入Path对象，以便我们能够使用它们来表示文件或目录的路径。
6. `from random import randint`：用于从random库中导入randint函数，以便我们能够生成一个随机的整数。
7. `from typing import Annotated, Any, Dict, List`：用于从typing库中导入Annotated、Any、Dict和List类型，以便我们能够定义这些类型。
8. `from fastapi import FastAPI, File, Form, HTTPException, UploadFile`：用于导入fastapi库，以便我们能够使用它们来创建API和处理文件上传。
9. `from fastapi.responses import StreamingResponse`：用于导入fastapi.responses库，以便我们能够使用它们来返回不同类型的响应（如JSON、图片等）。
10. `from pydantic import BaseModel`：用于导入pydantic库，以便我们能够使用它们来定义API和数据模型。


```py
import io
import json
import logging
import shutil
from pathlib import Path
from random import randint
from typing import Annotated, Any, Dict, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

```

这段代码定义了一个 FastAPI 应用和一组 Artifacts。

FastAPI 是一个用于构建 API 的现代 Python web 框架。这里我们创建了一个 FastAPI 应用，表示我们的 API 将如何处理客户端请求。

Artifacts 是一个列表，其中每个 Artifact 都是一个字典，包含一个或多个二进制数据、一个或多个键值对（即相对路径）和一个名为 ArtifactID 的随机生成的字符串。

`Task` 是一个类，它是 FastAPI 的模型，定义了上传文件的输入参数。

`@app.post("/agent/tasks/{task_id}/artifacts")` 是 FastAPI 的路由，当客户端发出一个 /agent/tasks/{task_id}/artifacts 请求时，这个路由将被调用。这个路由将接收一个任务 ID 和一个文件，然后将文件上传到指定的路径。

`file` 是文件参数，它通过 `Annotated[UploadFile, File()]` 类型进行约束，只能上传文件或者从 `File` 类中读取文件。

`relative_path` 是可选的，它指定了文件在 API 请求中的路径。如果 `relative_path` 被提供，则它将被用作文件的相对路径。

`artifacts` 列表用于存储 Artifacts，它被初始化为一个空列表。

`upload_file` 是 `Task` 类的一个方法，用于上传文件到指定的路径。这个方法将读取文件内容，并将其存储为 `artifact_data` 字典，然后将其添加到 `artifacts` 列表中。

最后，`logger.info` 将用于记录上传文件的日志信息，`return` 表示路由的响应，其中 `artifact_id` 是随机生成的字符串，`file_name` 是文件名，`relative_path` 是文件的相对路径（如果提供了的话）。


```py
app = FastAPI()
artifacts: List[Dict[str, Any]] = []


class Task(BaseModel):
    input: str


@app.post("/agent/tasks/{task_id}/artifacts")
async def upload_file(
    task_id: str, file: Annotated[UploadFile, File()], relative_path: str = Form("")
) -> Dict[str, Any]:
    logger.info(
        "Uploading file for task_id: %s with relative path: %s", task_id, relative_path
    )
    absolute_directory_path = Path(__file__).parent.absolute()
    save_path = (
        absolute_directory_path
        / "agent/gpt-engineer"
        / "projects/my-new-project/workspace"
    )

    random_string = str(randint(0, 100000))
    while random_string in artifacts:
        random_string = str(randint(0, 100000))

    artifact_data = await file.read()
    artifacts.append(
        {
            "binary": artifact_data,
            "relative_path": relative_path,
            "file_name": file.filename,
            "artifact_id": random_string,
        }
    )

    print(artifacts)
    return {
        "artifact_id": random_string,
        "file_name": "file_name",
        "relative_path": "relative_path",
    }


```

这段代码是一个 Python 编写的 Flask 应用程序中的两个路由，分别是 `/agent/tasks/{task_id}/artifacts` 和 `/agent/tasks/{task_id}/artifacts/{artifact_id}`。它们的作用是获取指定任务下的艺术品列表和指定艺术品的信息。

具体来说，第一个路由 `/agent/tasks/{task_id}/artifacts` 将会获取包含指定任务的所有艺术品列表。这个路由中的 `{task_id}` 参数将会被替换为任务 ID，因此如果你在一个代理程序中运行了 `app.get("/agent/tasks/123456/artifacts")`，它将会获取到代理程序中所有任务下的艺术品列表。这个路由中的 `artifacts` 参数将会被解析成一个 JSON 响应，其中包含指定任务下的所有艺术品对象。由于 `artifacts` 是一个 JSON 数据结构，因此 `get_files()` 函数将作为一个 `List[Dict[str, Any]]` 类型的函数，它将会返回一个包含所有文件名的列表。

第二个路由 `/agent/tasks/{task_id}/artifacts/{artifact_id}` 将会获取指定艺术品的信息。这个路由中的 `{task_id}` 参数将会被替换为任务 ID，因此如果你在一个代理程序中运行了 `app.get("/agent/tasks/123456/artifacts/678901")`，它将会获取到代理程序中指定任务下的指定艺术品的信息。这个路由中的 `{artifact_id}` 参数将会被替换为艺术品 ID，因此如果你在一个代理程序中运行了 `app.get("/agent/tasks/123456/artifacts/678901")`，它将会获取到指定任务下的指定艺术品的信息。

在 `get_file()` 函数中，我们通过 `artifacts` 数据结构查找到了指定艺术品的信息，并返回了该艺术品的信息 as a file-like object。我们通过 `for` 循环来查找指定艺术品，如果查找到了该艺术品，就返回了该文件的信息，否则返回了一个 HTTP 404 错误并返回了一个错误消息。


```py
@app.get("/agent/tasks/{task_id}/artifacts")
async def get_files() -> List[Dict[str, Any]]:
    logger.info("Fetching list of files for task")
    return artifacts


@app.get("/agent/tasks/{task_id}/artifacts/{artifact_id}")
async def get_file(artifact_id: str):
    for artifact in artifacts:
        if artifact["artifact_id"] == artifact_id:
            break
    else:
        logger.error("Attempt to access nonexistent artifact with ID: %s", artifact_id)
        raise HTTPException(status_code=404, detail="Artifact not found")

    logger.info("Fetching artifact with ID: %s", artifact_id)
    # find aritifact where artifact_id = artifact_id

    for artifact in artifacts:
        if artifact["artifact_id"] == artifact_id:
            return StreamingResponse(
                io.BytesIO(artifact["binary"]),
                media_type="application/octet-stream",
                headers={"Content-Disposition": f"attachment; filename=test.txt"},
            )
    # return 404
    return HTTPException(status_code=404, detail="Artifact not found")


```

这段代码是一个异步函数，使用Python的作弊接口asyncio来编写。

它的作用是当有一个请求到达时，以post的方式发送一个任务编号（task_id），并获取该任务的所有步骤。

具体来说，它接收一个参数，创建一个新的步骤，并将其存储在本地变量中。然后，它将步骤添加到任务元数据中，以便在任务成功完成后向客户端发送。

步骤元数据包含输入、附加输入、任务编号、步骤编号、名称、状态、输出、附加输出、艺术品和是否是最后一步等信息。在这里，我们只是简单地创建一个随机步骤，并将它添加到任务元数据中。

最后，它还返回一个表示已创建步骤的布尔值，以便在稍后的检查中进行使用。


```py
@app.post("/agent/tasks/{task_id}/steps")
async def create_steps(task_id: str):
    logger.info("Creating step for task_id: %s", task_id)
    return {
        "input": "random",
        "additional_input": {},
        "task_id": task_id,
        "step_id": "random_step",
        "name": "random",
        "status": "created",
        "output": "random",
        "additional_output": {},
        "artifacts": [],
        "is_last": True,
    }


```

这段代码是一个异步函数，用于在狗焕应用中创建新任务。具体来说，它通过向 `/agent/tasks` 路径发送一个 POST 请求来创建一个新的任务，并将任务的信息存储在 `artifacts` 集合中。

当这段代码在没有其他代码直接调用时，它会在创建新任务时执行。在这种情况下，它将清除 `artifacts` 集合并返回一个包含新任务信息的字典，其中包含一个 `input` 字段，用于指定任务接收者的输入，以及一个 `additional_input` 字段，用于存储任务的其他附加信息。此外，它还将 `task_id` 字段设置为 `static_task_id`，并将 `artifacts` 列表初始化为一个空列表。

如果这段代码在被称为主程序的程序中被调用，它将使用 `uvicorn` 函数服务器运行该应用程序。在这种情况下，它将监听所有经过 `/agent/tasks` 路径的请求，并执行相应的创建任务操作。


```py
@app.post("/agent/tasks")
async def create_tasks(task: Task):
    artifacts.clear()
    return {
        "input": "random",
        "additional_input": {},
        "task_id": "static_task_id",
        "artifacts": [],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

```

# `benchmark/agbenchmark/agent_api_interface.py`

这段代码的作用是实现一个命令行工具，用于运行AG Benchmark agent的命令行工具。具体实现包括以下几个步骤：

1. 导入需要用到的库：import json, logging, os, pathlib, time, typing
2. 导入AG Benchmark agent的类：from agbenchmark.__main__ import TEMP_FOLDER_ABS_PATH, UPDATES_JSON_PATH
3. 导入文件操作类：import pathlib
4. 导入用于获取命令行参数的库：import sys
5. 初始化AG Benchmark agent：from agbenchmark.agent_interface import get_list_of_file_paths
6. 创建一个Agbenchmark类：from agbenchmark.agent_protocol_client import (
   AgentApi,
   ApiClient,
   Configuration,
   TaskRequestBody,
)
7. 定义获取文件列表的方法：class Agbenchmark:
   def __init__(self, folder_path: str = TEMP_FOLDER_ABS_PATH):
       self.folder_path = folder_path
       if not os.path.exists(folder_path):
           self.folder_path = pathlib.Path(os.path.join(sys.path[0], folder_path))
       self.api_client = AgentApi()

   def get_file_paths(self) -> Optional[List[str]]:
       return self.api_client.list_file_paths(self.folder_path)

   def run_agent(self, agent_options: Dict[str, Any]) -> Optional[Dict[str, Any]]:
       return self.api_client.run_agent(agent_options, self.folder_path)

   def run_suite(self, suite_options: Dict[str, Any]) -> Optional[Dict[str, Any]]:
       return self.run_agent(suite_options)

   def run(self) -> Optional[Dict[str, Any]]:
       folder_path = self.folder_path
       options = {"files": [f"{folder_path}/**/*"]}
       return self.run_suite(options)
```py

这段代码实现了一个Agbenchmark类，用于运行AG Benchmark agent的命令行工具。该类包含以下方法：

* `__init__`：初始化AG Benchmark agent，并设置一个文件操作类`Agbenchmark`。
* `get_file_paths`：方法用于获取指定文件夹下的所有文件路径，并在需要时创建文件夹。
* `run_agent`：方法用于与AG Benchmark agent通信，并获取或设置一些参数。
* `run_suite`：方法用于运行一个测试组合，包括运行`run_agent`方法。
* `run`：方法用于运行所有文件，包括运行`run_suite`方法。

通过调用这些方法，可以实现AG Benchmark agent命令行工具的所有功能。


```
import json
import logging
import os
import pathlib
import time
from typing import Any, Dict, Optional

from agbenchmark.__main__ import TEMP_FOLDER_ABS_PATH, UPDATES_JSON_PATH
from agbenchmark.agent_interface import get_list_of_file_paths
from agbenchmark.agent_protocol_client import (
    AgentApi,
    ApiClient,
    Configuration,
    TaskRequestBody,
)
```py

This is a Python function that runs an API agent to perform a task and returns its results. The function takes in several parameters, including a `ChallengeData` object, which contains the task data, and a dictionary of configuration settings. The function uses the `AgentApi` class to interact with the API, and the `TaskRequestBody` class to construct the request for the task. The function also makes use of the `append_updates_file` and `upload_artifacts` functions to handle the uploading of artifacts and the artifact location. The function runs for a specified timeout, and if the timeout is reached it will raise a `TimeoutError`.


```
from agbenchmark.agent_protocol_client.models.step import Step
from agbenchmark.utils.data_types import ChallengeData

LOG = logging.getLogger(__name__)


async def run_api_agent(
    task: ChallengeData, config: Dict[str, Any], artifacts_location: str, timeout: int
) -> None:
    host_value = None

    configuration = Configuration(host=config["AgentBenchmarkConfig"].host + "/ap/v1")
    async with ApiClient(configuration) as api_client:
        api_instance = AgentApi(api_client)
        task_request_body = TaskRequestBody(input=task.task)

        start_time = time.time()
        response = await api_instance.create_agent_task(
            task_request_body=task_request_body
        )
        task_id = response.task_id

        await upload_artifacts(
            api_instance, artifacts_location, task_id, "artifacts_in"
        )

        i = 1
        steps_remaining = True
        while steps_remaining:
            # Read the existing JSON data from the file

            step = await api_instance.execute_agent_task_step(task_id=task_id)
            await append_updates_file(step)

            print(f"[{task.name}] - step {step.name} ({i}. request)")
            i += 1

            if time.time() - start_time > timeout:
                raise TimeoutError("Time limit exceeded")
            if not step or step.is_last:
                steps_remaining = False
        # if we're calling a mock agent, we "cheat" and give the correct artifacts to pass the tests
        if os.getenv("IS_MOCK"):
            await upload_artifacts(
                api_instance, artifacts_location, task_id, "artifacts_out"
            )

        await copy_agent_artifacts_into_temp_folder(api_instance, task_id)


```py

这段代码是一个 Python 函数，名为 `copy_agent_artifacts_into_temp_folder`，它实现了将指定任务（`task_id`）的代理任务（`api_instance`）中生成的 artifacts 复制到临时文件夹中。

具体来说，这段代码执行以下操作：

1. 从 `api_instance` 中下载指定任务的所有机件（或 artifacts）的副本。
2. 遍历下载的每个机件（或 artifacts）。
3. 如果机件（或 artifacts）有相对路径，则将其复制到临时文件夹中。
4. 创建临时文件夹（如果它们不存在的话）。
5. 写入文件内容。

值得注意的是，这段代码使用了 `pathlib` 包来管理文件和目录操作，并使用了 `LOG.info` 函数来输出日志信息。另外，由于这段代码使用了 `await` 关键字，因此需要确保 `api_instance` 引用了一个可以调用 `download_agent_task_artifact` 函数的 Api 客户端。


```
async def copy_agent_artifacts_into_temp_folder(api_instance, task_id):
    artifacts = await api_instance.list_agent_task_artifacts(task_id=task_id)
    for artifact in artifacts.artifacts:
        # current absolute path of the directory of the file
        directory_location = pathlib.Path(TEMP_FOLDER_ABS_PATH)
        if artifact.relative_path:
            path = (
                artifact.relative_path
                if not artifact.relative_path.startswith("/")
                else artifact.relative_path[1:]
            )
            directory_location = pathlib.Path(
                os.path.dirname(directory_location / path)
            )
            LOG.info(f"Creating directory {directory_location}")

        directory_location.mkdir(parents=True, exist_ok=True)

        file_path = directory_location / artifact.file_name
        LOG.info(f"Writing file {file_path}")
        with open(file_path, "wb") as f:
            content = await api_instance.download_agent_task_artifact(
                task_id=task_id, artifact_id=artifact.artifact_id
            )

            f.write(content)


```py

这段代码定义了一个名为 `append_updates_file` 的异步函数，它接受一个名为 `step` 的参数。函数的作用是将 `create_update_json` 函数生成的新的更新信息添加到已有的数据列表中，然后将更新后的数据列表写回到文件中。

具体来说，代码首先打开名为 `UPDATES_JSON_PATH` 的文件，并将其中的数据读取到内存中，定义了一个变量 `existing_data`。然后，定义了一个名为 `create_update_json` 的函数，它接受一个参数 `step`，并使用这个参数创建一个新的更新信息对象。

接着，代码使用列表推导式将新的更新信息对象添加到 `existing_data` 列表中，并将更新后的列表写回到文件中，最后在函数内部关闭文件。


```
async def append_updates_file(step: Step):
    with open(UPDATES_JSON_PATH, "r") as file:
        existing_data = json.load(file)
    # Append the new update to the existing array
    new_update = create_update_json(step)

    existing_data.append(new_update)
    # Write the updated array back to the file
    with open(UPDATES_JSON_PATH, "w") as file:
        file.write(json.dumps(existing_data, indent=2))


async def upload_artifacts(
    api_instance: ApiClient, artifacts_location: str, task_id: str, type: str
) -> None:
    for file_path in get_list_of_file_paths(artifacts_location, type):
        relative_path: Optional[str] = "/".join(
            file_path.split(f"{type}/", 1)[-1].split("/")[:-1]
        )
        if not relative_path:
            relative_path = None

        await api_instance.upload_agent_task_artifacts(
            task_id=task_id, file=file_path, relative_path=relative_path
        )


```py

这段代码定义了一个名为 `create_update_json` 的函数，它接受一个名为 `step` 的参数，代表一个 `Step` 对象。

函数内部先获取当前时间戳并将其存储在整数变量 `now` 中，然后创建一个字典对象 `content`，该对象包含两个键，分别为 `"content"` 和 `"timestamp"`。

键 `"content"` 的值为 `step` 对象以其 `to_dict()` 方法返回的 JSON 对象的 Python 代码表示，而键 `"timestamp"` 的值为当前时间戳。

最后，函数返回 `content` 对象。

该函数的作用是创建一个 JSON 对象，其中包含一个键 `"content"` 的值为 `step` 对象的 Python 代码表示，以及一个键 `"timestamp"` 的值为当前时间戳。


```
def create_update_json(step: Step):
    now = int(time.time())
    content = {"content": step.to_dict(), "timestamp": now}

    return content

```py

# `benchmark/agbenchmark/agent_interface.py`

这段代码的作用是：

1. 导入 `os`、`shutil` 和 `sys` 模块。
2. 从 `sys.environment` 中读取 `HELICONE_GRAPHQL_LOGS` 环境变量，如果当前环境中存在该变量，则将其值存储到一个名为 `helicone_graphql_logs` 的常量中，否则创建一个新的环境变量并将其值设置为 `False`。
3. 从 `dotenv` 包中使用 `load_dotenv` 函数加载是否存在名为 `helicone_graphql_logs` 的环境变量。
4. 从 `agbenchmark.execute_sub_process` 类中使用 `execute_subprocess` 函数执行名为 `graphql` 的命令，并将 `helicone_graphql_logs` 和 `graphql` 作为参数传递给该函数。
5. 对 `graphql` 命令的输出进行处理，使用列表推导式获取所有输出行，并将它们存储到一个名为 `outputs` 的列表中。
6. 将 `graphql` 命令的输出作为 `helicone_graphql_logs` 环境变量的更新值，如果 `helicone_graphql_logs` 的值为 `True`，则执行 `graphql` 命令并将 `helicone_graphql_logs` 作为参数传递给该命令，否则将 `False` 作为 `helicone_graphql_logs` 的值。


```
import os
import shutil
import sys
from typing import List

from dotenv import load_dotenv

from agbenchmark.execute_sub_process import execute_subprocess

load_dotenv()

helicone_graphql_logs = os.getenv("HELICONE_GRAPHQL_LOGS")
HELICONE_GRAPHQL_LOGS = (
    helicone_graphql_logs.lower() == "true" if helicone_graphql_logs else False
)


```py



这段代码是一个 Python 函数，名为 `run_agent`，它接受两个参数 `task` 和 `timeout`。这个函数的作用是在给定超时时间和任务的情况下运行 `agbenchmark_config.benchmarks` 脚本，并将任务路径作为参数传递给它。

具体来说，函数会在执行 `agbenchmark_config.benchmarks` 脚本之前输出一条消息，说明正在运行这个任务，然后使用 `sys.executable` 执行 `agbenchmark_config.benchmarks` 脚本，并将 `task` 和 `timeout` 作为参数传递给它。最后，函数通过调用 `execute_subprocess` 函数来运行 `agbenchmark_config.benchmarks` 脚本，这个函数接受一个命令行列表，其中包含 `sys.executable` 和 `-m` 参数，以及 `timeout` 参数。

另外，函数 `get_list_of_file_paths` 接收两个参数 `challenge_dir_path` 和 `artifact_folder_name`，返回列表，其中包含 `artbenchmark_agent_interface.py` 文件中的所有文件名。


```
def run_agent(task: str, timeout: int) -> None:
    print(f"Running agbenchmark/benchmarks.py with timeout {timeout}")

    command = [sys.executable, "-m", "agbenchmark_config.benchmarks", str(task)]

    execute_subprocess(command, timeout)


def get_list_of_file_paths(
    challenge_dir_path: str, artifact_folder_name: str
) -> List[str]:
    # this file is at agbenchmark\agent_interface.py
    source_dir = os.path.join(
        challenge_dir_path,
        artifact_folder_name,
    )
    if not os.path.exists(source_dir):
        return []
    return [os.path.join(source_dir, file_name) for file_name in os.listdir(source_dir)]


```py

这段代码定义了一个名为 `copy_artifacts_into_temp_folder` 的函数，它接受一个 `workspace` 参数，一个 `artifact_folder_name` 参数和一个 `challenge_dir_path` 参数。这个函数的作用是将 `artifact_folder_name` 中的文件复制到指定的临时文件夹中，如果目标文件夹不存在，则会创建它。

具体来说，函数首先通过调用 `get_list_of_file_paths` 函数获取 `artifact_folder_name` 中所有文件的文件路径。接着，函数遍历每个文件路径，判断文件是否存在于系统文件系统中（通常可以通过在命令行中输入 `ls` 命令来验证），如果是，则使用 `shutil.copy` 函数将文件复制到 `workspace` 参数指定的目标文件夹中。如果目标文件夹不存在，则会创建它。


```
def copy_artifacts_into_temp_folder(
    workspace: str | dict[str, str], artifact_folder_name: str, challenge_dir_path: str
) -> None:
    file_paths = get_list_of_file_paths(challenge_dir_path, artifact_folder_name)
    for file_path in file_paths:
        if os.path.isfile(file_path):
            shutil.copy(file_path, workspace)

```py

# `benchmark/agbenchmark/app.py`

这段代码的作用是实现一个基准测试的流程，包括以下几个步骤：

1. 导入需要的库：datetime、uuid、collections、deque、pathlib、httpx。

2. 从collections.defaultdict创建一个任务实体，使用deque存储任务队列。

3. 使用pathlib的Path类创建一个基准测试报告的路径。

4. 从httpx库创建一个HTTP客户端，用于和远程服务器通信。

5. 使用配置文件（可能是环境变量）设置Agbenchmark agent的API配置。

6. 实现一个报告类，用于记录每个任务的基准测试结果，包括任务的ID、启动时间、结束时间和结果（成功或失败）。

7. 实现一个基准测试类，用于执行一个测试任务，包括设置任务的ID、权重、堆栈和基准测试报告。

8. 在测试报告中统计每个任务的基准测试结果，并输出结果。


```
import datetime
import uuid
from collections import defaultdict, deque
from pathlib import Path

import httpx

from agbenchmark.agent_protocol_client import (
    AgentApi,
    ApiClient,
    ApiException,
    Configuration,
)
from agbenchmark.reports.processing.report_types_v2 import BenchmarkRun
from agbenchmark.schema import TaskEvalRequestBody
```py

这段代码使用了多个模块和函数，需要根据具体模块和函数来解释它们的作用。这里主要 fastapi 和 agbenchmark，可以先简要介绍这两个模块的功能和作用。

1. fastapi：FastAPI 是一个用于构建现代 Web 应用程序的 FastAPI 库，提供了 RESTful API 的快速开发和运行。

2. agbenchmark：Agbenchmark 是 AgileBenchmark 的缩写，是一个基于 FastAPI 库的 Agile 测试和运行工具，提供了自动化测试、部署和管理等功能。

现在来看这段代码的具体作用：

1. 从 agbenchmark.utils.utils 模块中引入了 write_pretty_json 函数，该函数的作用是写入一个漂亮的 JSON 数据。

2. 从 Configuration 类中继承了 Configuration 接口，这个接口可能定义了一些配置参数，但并不包含具体的数据写入操作。

3. 从 Configuration 类中继承了 Configuration 接口，这个接口可能定义了一些配置参数，但并不包含具体的数据写入操作。

4. 创建了一个 ApiRouter，这个路由器使用 FastAPI 库作为 API 的基础，为 v1 路径下的用户提供了一个简单的 HTTP 请求处理。

5. 在 ApiRouter 中定义了两个方法：get 和 post，分别对应着获取和发送请求的功能。

6. 在 post 方法的回调函数中，通过调用 agbenchmark.utils.contrib.psutil 库与当前系统的 psutil 工具类来获取系统的 CPU 使用情况，并将其作为请求的信息返回。

这段代码主要作用是为了一个基于 FastAPI 和 AgileBenchmark 的测试和运行工具，提供了一个简单的 API 路由处理，并在路由处理中加入了一些额外的 functionality，比如将系统 CPU 使用情况作为请求信息返回。


```
from agbenchmark.utils.utils import write_pretty_json

configuration = Configuration(host="http://localhost:8000" + "/ap/v1")

import json
import os
import sys
from typing import Any, Optional

import psutil
from fastapi import APIRouter, FastAPI
from fastapi import (
    HTTPException as FastAPIHTTPException,  # Import HTTPException from FastAPI
)
from fastapi import Request, Response
```py

这段代码使用了三个模块：fastapi、agbenchmark和pydantic。具体来说，这段代码：

1. 从fastapi模块中导入了一个名为CORSMiddleware的类，该类实现了跨域资源共享（CORS）功能。
2. 从agbenchmark模块中导入了execute_subprocess和Task、TaskRequestBody两个类，这些类用于执行和管理测试用例。
3. 从./__init__.py__文件中导入了一个名为FastAPI的类，该类是FastAPI项目的基类。
4. 从fastapi模块中导入了os模块中的find_absolute_benchmark_path函数，该函数用于查找基准测试所在的目录。
5. 在设置完基准测试目录后，将当前工作目录（也就是当前目录）切换到基准测试目录。
6. 创建了一个名为router的API路由器，用于定义API的路由。
7. 从fastapi模块中导入了BaseModel和Extra两个类，这些类用于定义API响应的数据结构和传递给外部API的参数。


```
from fastapi.middleware.cors import CORSMiddleware

from agbenchmark.execute_sub_process import execute_subprocess
from agbenchmark.schema import Task, TaskRequestBody

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import FastAPI
from pydantic import BaseModel, Extra

router = APIRouter()
import glob

# Change the current working directory to the benchmark path
# home_path = find_absolute_benchmark_path()
# os.chdir(home_path)

```py

这段代码的作用是执行一个名为"general_command"的命令，其中包括运行"poetry"命令、"run"命令、"agbenchmark"命令、"start"命令以及一个名为"--backend"的参数。同时，它还定义了一个名为"challenges_path"的路径，用于存储一个名为"challenges"的文件夹，这个文件夹中可能包含一些以"data.json"为扩展名的JSON文件。

具体来说，这段代码下面的几个步骤：

1. 导入psutil模块，用于获取与os.path.dirname(__file__)相对路径的当前工作目录。
2. 创建一个名为"challenges_path"的文件夹，如果这个文件夹不存在的话。
3. 循环遍历"challenges_path"文件夹中的所有文件，包括以"data.json"为扩展名的文件。
4. 将遍历到的所有JSON文件的内容存储到"CHALLENGES"字典中，其中键为文件的绝对路径，值为{"line_count": 0, "test_cases": 0}。
5. 遍历"CHALLENGES"字典中的每个键值对，其中键为文件的绝对路径，值为{"line_count": 0, "test_cases": 0}。
6. 将每个文件的"line_count"值和"test_cases"值作为函数参数传入，实现自动测试。


```
general_command = ["poetry", "run", "agbenchmark", "start", "--backend"]

import psutil

challenges_path = os.path.join(os.path.dirname(__file__), "challenges")

json_files = deque(
    glob.glob(
        f"{challenges_path}/**/data.json",
        recursive=True,
    )
)

CHALLENGES = {}
task_informations = defaultdict(dict)

```py

这段代码使用了Python的uuid库和json库，实现了生成唯一ID并写入JSON文件的功能。

while True:
   # 取出json文件名并将其打印
   json_file = json_files.popleft()
   print(f"Processing JSON file: {json_file}")

   # 使用with open打开文件读取数据
   with open(json_file, "r") as file:
       # 使用json.load读取文件中的JSON数据并存储到data变量中
       data = json.load(file)

       # 如果eval_id不在数据中，则将其添加到data中
       if "eval_id" not in data:
           data["eval_id"] = str(uuid.uuid4())

   # 使用write_pretty_json将数据写入到文件中
   write_pretty_json(data, json_file)
   # 存储到Challenges变量中
   CHALLENGES[data["eval_id"]] = data
   CHALLENGES[data["eval_id"]]["path"] = json_file

   # 检查文件是否存在，如果不存在则继续循环
   if not os.path.isfile(json_file):
       continue


```
while json_files:
    json_file = json_files.popleft()

    with open(json_file, "r") as file:
        data = json.load(file)

        if "eval_id" not in data:
            data["eval_id"] = str(uuid.uuid4())
        # this will sort all the keys of the JSON systematically so that the order is always the same
        write_pretty_json(data, json_file)
        # ok
        CHALLENGES[data["eval_id"]] = data
        CHALLENGES[data["eval_id"]]["path"] = json_file


```py

这段代码的作用是找出没有使用uvicorn的基准测试进程。它使用了两个psutil库函数，第一个psutil.process_iter函数用于获取所有进程的pid，第二个psutil.process_iter函数用于获取每个进程的信息字典。

在这段代码中，我们遍历了所有进程，并使用psutil库的过滤器来获取每个进程的信息字典。我们使用and()运算符来获取符合条件的进程，即同时满足“agbenchmark”和“uvicorn”的条件。如果一个进程的信息中包含“agbenchmark”并且不包含“uvicorn”，那么我们将其添加到pids列表中。

最后，我们返回pids列表，其中包含所有符合条件的进程的pid。


```
def find_agbenchmark_without_uvicorn():
    pids = []
    for process in psutil.process_iter(
        attrs=[
            "pid",
            "cmdline",
            "name",
            "username",
            "status",
            "cpu_percent",
            "memory_info",
            "create_time",
            "cwd",
            "connections",
        ]
    ):
        try:
            # Convert the process.info dictionary values to strings and concatenate them
            full_info = " ".join([str(v) for k, v in process.info.items()])

            if "agbenchmark" in full_info and "uvicorn" not in full_info:
                pids.append(process.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return pids


```py

这段代码定义了一个名为CreateReportRequest的类，它继承自Python中的BaseModel类。这个类的目的是定义一个配置类，用于在测试运行时创建报告。

具体来说，这个类包含以下成员变量：

- test: 一个字符串，表示要运行的测试的ID。
- test_run_id: 一个字符串，表示运行测试的报告ID。
- category: 一个可选的字符串，表示一个或多个测试类别。
- mock: 一个可选的布尔值，表示是否在测试中使用一个虚拟的、不会输出任何结果的运行。

这个类的配置类还包含一个 Extra.forbid 属性，用于禁止任何额外的配置字段。

在 upates_list 变量中，有两个空列表，似乎没有进行使用。另外，从代码中无法确定 test 和 test_run_id 变量是在运行时还是用于配置时进行初始化的。


```
class CreateReportRequest(BaseModel):
    test: str = None
    test_run_id: str = None
    # category: Optional[str] = []
    mock: Optional[bool] = False

    class Config:
        extra = Extra.forbid  # this will forbid any extra fields


updates_list = []

updates_list = []

import json

```py

这段代码使用了Python的FastAPI框架来实现Web应用程序的开发，主要作用是定义了一个名为origins的列表，包含了多个Web服务器地址，这些地址都是用来启动FastAPI应用程序的中间件(即FastAPI中间件)的接口地址。

具体来说，origins列表中的每个元素都是一个格式为"<IP地址：端口号>"的元组，其中IP地址表示Web服务器的位置，端口号表示该服务器所监听的端口。这些元组组组成了FastAPI应用程序的中间件，用于将请求转发到相应的Web服务器上进行处理，并返回处理结果给FastAPI应用程序。

在 FastAPI应用程序中，添加CORSMiddleware是一个自动化的过程，可以自动将所有允许的域名添加到origins列表中。而在这个具体的实现中，使用了allow_credentials=True和allow_methods=["*"]两个参数，表示允许应用程序发送包括用户名和密码等身份验证信息以及所有HTTP方法(包括GET、POST等)的请求。


```
origins = [
    "http://localhost:8000",
    "http://localhost:8080",
    "http://127.0.0.1:5000",
    "http://localhost:5000",
]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


```py

This code looks like it is a Flask router for a report generation tool. It has a route for running a single test against an benchmark tool called Agbenchmark, and another route for generating a report of the test results.

The `run_single_test` function takes a `CreateReportRequest` object as its argument, which must contain the configuration options for running the test (e.g. the path to the benchmark tool, the test to run, and optionally the category to test). This function starts the benchmark tool in the background and returns when the test finishes.

The `reports` folder is used to store the test results, and is automatically created if it does not already exist. The last folder in the folder is assumed to be the output folder for the test report, which is read from here.

It appears that the code also includes some additional functionality, such as finding and listing all folders in the current working directory, and sorting the folders based on their names. However, these features are not part of the route for generating reports, so I assume they are not used in this case.


```
def stream_output(pipe):
    for line in pipe:
        print(line, end="")


@router.post("/reports")
def run_single_test(body: CreateReportRequest) -> Any:
    pids = find_agbenchmark_without_uvicorn()
    print(f"pids already running with agbenchmark: {pids}")
    print(body.dict())
    # it's a hack because other parts of the code are using sys.argv
    print(os.getcwd())
    command_options = ["agbenchmark"]
    # if body.category:
    #     sys.argv.append(f"--category={body.category}")
    command_options.append(f"--test={body.test}")
    if body.mock:
        command_options.append("--mock")

    execute_subprocess(command_options, 200)
    import json
    from pathlib import Path

    print("finished running")
    # List all folders in the current working directory
    path_reports = Path.cwd() / "agbenchmark_config" / "reports"
    folders = [folder for folder in path_reports.iterdir() if folder.is_dir()]

    # Sort the folders based on their names
    sorted_folders = sorted(folders, key=lambda x: x.name)

    # Get the last folder
    last_folder = sorted_folders[-1] if sorted_folders else None

    # Read report.json from this folder
    if last_folder:
        report_path = last_folder / "report.json"
        print(report_path)
        if report_path.exists():
            with report_path.open() as file:
                data = json.load(file)
            print(data)
        else:
            print(f"'report.json' does not exist in '{last_folder}'")
    else:
        print("No folders found.")

    return Response(
        content=json.dumps(data),
        status_code=200,
        media_type="application/json",
    )


```py

This is a Flask endpoint that serves as the main entry point for a Flask application. It reads data from a "update.json" file (the path to which is provided via the UPDATES\_JSON\_PATH environment variable), converts it to JSON, and returns it to the client. The data in the JSON file is filtered based on the "timestamp" field, which is lower than a specified query parameter (last\_update\_time). If the query parameter is not provided, the endpoint returns an error message.


```
import json
from typing import Any

from fastapi import FastAPI, Request, Response


@router.get("/updates")
def get_updates(request: Request) -> Any:
    from agbenchmark.__main__ import UPDATES_JSON_PATH

    try:
        # Read data from the "update.json" file (provide the correct file path)
        with open(UPDATES_JSON_PATH, "r") as file:
            data = json.load(file)

        # Get the last_update_time from the query parameter
        query_param = request.query_params.get("last_update_time")

        if query_param is None:
            # Handle the case when last_update_time is not provided
            print("ERROR: last_update_time parameter is missing")
            return Response(
                content=json.dumps({"error": "last_update_time parameter is missing"}),
                status_code=400,
                media_type="application/json",
                headers={"Content-Type": "application/json"},
            )

        # Convert query_param to a Unix timestamp (assuming it's in seconds as a string)
        query_timestamp = int(query_param)

        # Filter the data based on the timestamp (keep timestamps before query_timestamp)
        filtered_data = [item for item in data if item["timestamp"] > query_timestamp]

        # Extract only the "content" field from each item
        filtered_data = [item["content"] for item in filtered_data]

        # Convert the filtered data to JSON
        filtered_json = json.dumps(filtered_data, indent=2)

        print("INFO: Returning filtered data to the client")
        return Response(
            content=filtered_json,
            status_code=200,
            media_type="application/json",
            headers={"Content-Type": "application/json"},
        )
    except FileNotFoundError:
        print("ERROR: File not found: updates.json")
        return Response(
            content=json.dumps({"error": "File not found"}),
            status_code=404,
            media_type="application/json",
            headers={"Content-Type": "application/json"},
        )


```py

This is an Agbenchmark agent that is configured to upload the results of a benchmark test to a specified endpoint. The agent has a task id of "50da533e-3904-4401-8a07-c49adf88b5eb" and an input of "Write the word 'Washington' to a .txt file" and an additional input of "python/code". The task has been created and the agent is waiting for the benchmark start time to be recorded. If the benchmark start time has not been recorded within the specified timeout, the agent will upload the benchmark results to the specified endpoint. The endpoint that the agent is using is not specified in this code.


```
@router.post("/agent/tasks", tags=["agent"], response_model=Task)
async def create_agent_task(task_eval_request: TaskEvalRequestBody) -> Task:
    """
    Creates a new task using the provided TaskRequestBody and returns a Task.

    Args:
        request (Request): FastAPI request object.
        task (TaskRequestBody): The task request containing input and additional input data.

    Returns:
        Task: A new task with task_id, input, additional_input, and empty lists for artifacts and steps.

    Example:
        Request (TaskRequestBody defined in schema.py):
            {
                "input": "Write the words you receive to the file 'output.txt'.",
                "additional_input": "python/code"
            }

        Response (Task defined in schema.py):
            {
                "task_id": "50da533e-3904-4401-8a07-c49adf88b5eb",
                "input": "Write the word 'Washington' to a .txt file",
                "additional_input": "python/code",
                "artifacts": [],
            }
    """
    from agbenchmark.agent_api_interface import upload_artifacts

    try:
        async with ApiClient(configuration) as api_client:
            api_instance = AgentApi(api_client)
            task_input = CHALLENGES[task_eval_request.eval_id]["task"]

            task_request_body = TaskRequestBody(input=task_input)
            task_response = await api_instance.create_agent_task(
                task_request_body=task_request_body
            )
            task_informations[task_response.task_id][
                "benchmark_start_time"
            ] = datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S+00:00"
            )
            task_informations[task_response.task_id][
                "eval_id"
            ] = task_eval_request.eval_id
            await upload_artifacts(
                api_instance,
                str(Path(CHALLENGES[task_eval_request.eval_id]["path"]).parent),
                task_response.task_id,
                "artifacts_in",
            )
            return Response(
                content=task_response.json(),
                status_code=200,
                media_type="application/json",
            )
    except ApiException as e:
        print(f"Error whilst trying to create a task: {task_eval_request}")
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )


```py

这段代码使用了Python中的异步编程库`aiohttp`和`httpx`。它是一个代理URL，可以在任务步度的URL上执行异步操作。

具体来说，这段代码定义了一个名为`proxy`的函数，该函数接收一个`Request`对象和一个任务ID。函数内部使用`httpx.Timeout`创建一个HTTP请求超时时间，并使用`httpx.AsyncClient`发送POST请求。它将请求的新URL设置为`http://localhost:8000/ap/v1/agent/tasks/{task_id}/steps`，并将请求的数据作为参数传递给`client.post()`方法。函数使用`async with`语句来确保在函数内部资源和URL都处于活动状态，以便在函数外部继续执行后续操作。

函数返回一个`Response`对象，其中包含来自原始请求的响应内容以及响应状态码。


```
@router.post("/agent/tasks/{task_id}/steps")
async def proxy(request: Request, task_id: str):
    timeout = httpx.Timeout(300.0, read=300.0)  # 5 minutes
    async with httpx.AsyncClient(timeout=timeout) as client:
        # Construct the new URL
        new_url = f"http://localhost:8000/ap/v1/agent/tasks/{task_id}/steps"

        # Forward the request
        response = await client.post(
            new_url,
            data=await request.body(),
            headers=dict(request.headers),
        )

        # Return the response from the forwarded request
        return Response(content=response.content, status_code=response.status_code)


```py

It seems like you are providing a Python function that takes a JSON file path as input and returns information about a benchmark.

The function first reads the JSON file and parses it into an object using the `json.loads()` method. Then, it calls the `BenchmarkRun.parse_obj()` method from the `BenchmarkRun` class to parse the object into a `BenchmarkRun` object.

If there are any errors, such as an internal server error, the function returns a response with a 500 status code and an error message.

It's worth noting that the `json.dumps()` method is used to convert the `BenchmarkRun` object to a JSON string that can be returned by the API.


```
@router.post("/agent/tasks/{task_id}/evaluations")
async def create_evaluation(task_id: str) -> deque:
    from agbenchmark.__main__ import TEMP_FOLDER_ABS_PATH
    from agbenchmark.agent_api_interface import copy_agent_artifacts_into_temp_folder
    from agbenchmark.agent_interface import copy_artifacts_into_temp_folder
    from agbenchmark.generate_test import create_challenge

    try:
        async with ApiClient(configuration) as api_client:
            api_instance = AgentApi(api_client)
            await copy_agent_artifacts_into_temp_folder(api_instance, task_id)
        # add custom python
        data = CHALLENGES[task_informations[task_id]["eval_id"]]

        artifact_path = str(Path(data["path"]).parent)
        copy_artifacts_into_temp_folder(
            TEMP_FOLDER_ABS_PATH, "custom_python", artifact_path
        )
        json_file = CHALLENGES[task_informations[task_id]["eval_id"]]["path"]
        json_files = deque()

        _, challenge_class = create_challenge(data, json_file, json_files)
        challenge_instance = challenge_class()
        scores = challenge_instance.get_scores(config={})
        test_name = "Test" + data["name"]
        is_score_100 = 1 in scores["values"]

        info_details = {
            "repository_info": {
                "repo_url": None,
                "team_name": None,
                "benchmark_git_commit_sha": None,
                "agent_git_commit_sha": None,
            },
            "run_details": {
                "run_id": None,
                "command": "agbenchmark" + " --test=" + test_name,
                "completion_time": None,
                "benchmark_start_time": task_informations[task_id][
                    "benchmark_start_time"
                ],
                "test_name": data["name"],
            },
            "task_info": {
                "data_path": data["path"].split("benchmark/", 1)[-1],
                "is_regression": None,
                "category": data["category"],
                "task": data["task"],
                "answer": data["ground"]["answer"],
                "description": data["info"]["description"],
            },
            "metrics": {
                "difficulty": None,
                "success": is_score_100,
                "attempted": True,
                "success_percentage": None,
                "cost": None,
                "run_time": None,
            },
            "reached_cutoff": None,
            "config": {},
        }

        BenchmarkRun.parse_obj(info_details)

        print(json.dumps(info_details, indent=4))
        return Response(
            content=json.dumps(info_details),
            status_code=200,
            media_type="application/json",
        )
    except ApiException as e:
        print(f"Error whilst trying to evaluate the task: {task_id}")
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )
    # path = Path(json_file).resolve()


```py

这段代码是使用 Flask-Router 库中的 include_router 方法，用于将一个已经定义好的路由（router）添加到当前应用程序（app）的路由树中。

具体来说，这段代码将会在 app 的路由树中添加一个名为 "/ap/v1" 的前缀，这个前缀将会被添加到所有匹配这个前缀的路由项上。这样，当用户访问 URL 中的 "/ap/v1" 时，就会匹配到 app 中定义好的路由，然后执行该路由对应的处理逻辑。

假设 app 中有一个名为 "my_route" 的路由项，该路由项处理器为 "handle_my_route"，那么当用户访问 URL "/ap/v1/my_route" 时，该路由项中的 "handle_my_route" 将会被 executed，执行完畢后，最终返回结果 "hello"给客户端。


```
app.include_router(router, prefix="/ap/v1")

```