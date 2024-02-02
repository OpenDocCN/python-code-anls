# AutoGPT源码解析 9

# `autogpts/autogpt/autogpt/core/planning/templates.py`

这段代码是一个 Python 代码段，定义了一个名为 `RulesOfThumbs` 的规则。这个规则指出模板不要在字符串的结尾添加新的行。然后，定义了一个名为 `USER_OBJECTIVE` 的变量，它的值为一个关于一个名为 `AutoGPT` 的 Python 项目的维基百科风格的的文章。最后，没有进行任何操作，直接输出了这个规则。


```py
# Rules of thumb:
# - Templates don't add new lines at the end of the string.  This is the
#   responsibility of the or a consuming template.

####################
# Planner defaults #
####################


USER_OBJECTIVE = (
    "Write a wikipedia style article about the project: "
    "https://github.com/significant-gravitas/AutoGPT"
)


```

这是一段 Python 代码，它定义了一个名为 `ABILITIES` 的列表，包含了多个函数，每个函数都对应着不同的功能。下面是每个函数的作用：

1. `analyze_code: Analyze Code, args: "code": "<full_code_string>"` - 这个函数用于分析给定的代码，并返回分析结果。
2. `execute_python_file: Execute Python File, args: "filename": "<filename>"` - 这个函数用于执行给定的 Python 文件。
3. `append_to_file: Append to file, args: "filename": "<filename>", "text": "<text>"` - 这个函数用于将给定文本追加到指定的文件中。
4. `list_files: List Files in Directory, args: "directory": "<directory>"` - 这个函数用于列出指定目录中的所有文件。
5. `read_file: Read a file, args: "filename": "<filename>"` - 这个函数用于从指定文件中读取内容。
6. `write_to_file: Write to file, args: "filename": "<filename>", "text": "<text>"` - 这个函数用于将指定内容写入到指定文件中。
7. `google: Google Search, args: "query": "<query>"` - 这个函数用于在 Google 上进行搜索。
8. `improve_code: Get Improved Code, args: "suggestions": "<list_of_suggestions>", "code": "<full_code_string>"` - 这个函数用于获取改进的代码，并返回建议。
9. `browse_website: Browse Website, args: "url": "<url>", "question": "<what_you_want_to_find_on_website>"` - 这个函数用于浏览指定网站，并返回相关信息。
10. `write_tests: Write Tests, args: "code": "<full_code_string>", "focus": "<list_of_focus_areas>"` - 这个函数用于编写测试，并返回需要关注的内容列表。
11. `get_hyperlinks: Get hyperlinks, args: "url": "<url>"` - 这个函数用于获取链接。
12. `get_text_summary: Get text summary, args: "url": "<url>", "question": "<question>"` - 这个函数用于获取文本摘要。
13. `task_complete: Task Complete (Shutdown)` - 这个函数用于标记任务完成（关闭）。


```py
ABILITIES = (
    'analyze_code: Analyze Code, args: "code": "<full_code_string>"',
    'execute_python_file: Execute Python File, args: "filename": "<filename>"',
    'append_to_file: Append to file, args: "filename": "<filename>", "text": "<text>"',
    'list_files: List Files in Directory, args: "directory": "<directory>"',
    'read_file: Read a file, args: "filename": "<filename>"',
    'write_to_file: Write to file, args: "filename": "<filename>", "text": "<text>"',
    'google: Google Search, args: "query": "<query>"',
    'improve_code: Get Improved Code, args: "suggestions": "<list_of_suggestions>", "code": "<full_code_string>"',
    'browse_website: Browse Website, args: "url": "<url>", "question": "<what_you_want_to_find_on_website>"',
    'write_tests: Write Tests, args: "code": "<full_code_string>", "focus": "<list_of_focus_areas>"',
    'get_hyperlinks: Get hyperlinks, args: "url": "<url>"',
    'get_text_summary: Get text summary, args: "url": "<url>", "question": "<question>"',
    'task_complete: Task Complete (Shutdown), args: "reason": "<reason>"',
)


```

这段代码是一个 Plan Prompt，它用于限制用户在短期记忆中的词汇量，以帮助用户更好地控制自己的记忆。它还建议用户通过思考与当前事件相似的事件来帮助自己记忆，并提示用户不要依赖用户协助。此外，它还建议用户使用指定的命令来获取短期的信息。


```py
# Plan Prompt
# -----------


PLAN_PROMPT_CONSTRAINTS = (
    "~4000 word limit for short term memory. Your short term memory is short, so "
    "immediately save important information to files.",
    "If you are unsure how you previously did something or want to recall past "
    "events, thinking about similar events will help you remember.",
    "No user assistance",
    "Exclusively use the commands listed below e.g. command_name",
)

PLAN_PROMPT_RESOURCES = (
    "Internet access for searches and information gathering.",
    "Long-term memory management.",
    "File output.",
)

```

这段代码定义了一个名为 `PLAN_PROMPT_PERFORMANCE_EVALUATIONS` 的列表，其中包含了几个短语，这些短语是为了鼓励开发人员要回顾和评估他们的行动，以确保他们能够以最佳能力执行任务。

接着，定义了一个名为 `PLAN_PROMPT_RESPONSE_DICT` 的字典，其中包含了一些关于如何评估自己的计划和策略。这些短语包括对自己的行为的 constructive self-criticism(建设性的自我批评)、反思过去的决策和策略，以及计划如何最好地完成任务等。

最后，将上述列表和字典都存储了一个字符串变量中，这个字符串变量可以在需要时被打印出来，用于提醒开发人员要保持良好的编程表现。


```py
PLAN_PROMPT_PERFORMANCE_EVALUATIONS = (
    "Continuously review and analyze your actions to ensure you are performing to"
    " the best of your abilities.",
    "Constructively self-criticize your big-picture behavior constantly.",
    "Reflect on past decisions and strategies to refine your approach.",
    "Every command has a cost, so be smart and efficient. Aim to complete tasks in"
    " the least number of steps.",
    "Write all code to a file",
)


PLAN_PROMPT_RESPONSE_DICT = {
    "thoughts": {
        "text": "thought",
        "reasoning": "reasoning",
        "plan": "- short bulleted\n- list that conveys\n- long-term plan",
        "criticism": "constructive self-criticism",
        "speak": "thoughts summary to say to user",
    },
    "command": {"name": "command name", "args": {"arg name": "value"}},
}

```

这段代码是一个字符串，包含了多个段落式的文本内容。主要用途是指导如何以正确的格式在给定的 JSON 数据上进行交互，并提供了一些相关的信息。

具体来说，PLAN_PROMPT_RESPONSE_FORMAT 是一个提示，告诉用户应该如何回复问题，应该在 JSON 格式中进行。PLAN_TRIGGERING_PROMPT 是一个提示，告诉用户需要决定下一个命令如何使用。PLAN_PROMPT_MAIN 是主要的文本，描述了在交互过程中需要遵循的一些指导方针。其中包括目标、信息、约束、命令、资源以及性能评估等。


```py
PLAN_PROMPT_RESPONSE_FORMAT = (
    "You should only respond in JSON format as described below\n"
    "Response Format:\n"
    "{response_json_structure}\n"
    "Ensure the response can be parsed by Python json.loads"
)

PLAN_TRIGGERING_PROMPT = (
    "Determine which next command to use, and respond using the format specified above:"
)

PLAN_PROMPT_MAIN = (
    "{header}\n\n"
    "GOALS:\n\n{goals}\n\n"
    "Info:\n{info}\n\n"
    "Constraints:\n{constraints}\n\n"
    "Commands:\n{commands}\n\n"
    "Resources:\n{resources}\n\n"
    "Performance Evaluations:\n{performance_evaluations}\n\n"
    "You should only respond in JSON format as described below\n"
    "Response Format:\n{response_json_structure}\n"
    "Ensure the response can be parsed by Python json.loads"
)


```

这段代码是一个参数化模板的例子，它允许用户指定模板的参数。具体来说，它是一个模板，可以用来生成指定类或函数的示例输出。

在这里，`template`是一个参数化模板，它可以接受一个或多个参数。当用户使用这个模板时，他或她需要提供一个模板名和一个或多个参数列表，模板名用于指定要生成的类或函数的名称，而参数列表则用于指定要插入到模板中的参数。

例如，如果用户使用以下代码：
```py
#include <iostream>
using namespace std;

int main() {
 int template_name = 1; // template name
 int num_arguments = 3; // number of arguments
 cout << "Enter the template and arguments: " << template_name << " " << num_arguments << endl;
 int template_arg;
 int arguments[3]; // arguments
 cout << "Enter the values for the arguments: " << endl;
 for (int i = 0; i < num_arguments; i++) {
   cin >> arguments[i];
 }
 cout << "The output of the template is: " << template_name << endl;
 return 0;
}
```
他或她可以提供一个模板名（例如`int main()`）和一个或多个参数列表（例如`int template_name=1  and int num_arguments=3`）。

当用户运行这个程序时，它将输出：
```py
Enter the template and arguments: int main() and int num_arguments=3
The output of the template is: int main()
```
这段代码的作用就是允许用户生成一个模板，根据用户的指定，生成指定类或函数的示例输出。


```py
###########################
# Parameterized templates #
###########################

```

# `autogpts/autogpt/autogpt/core/planning/__init__.py`

这段代码是一个自然语言处理任务，表示为Agent的活动组织计划系统。它使用一个AutogPT模型（从autogpt.core.planning.schema模块中导入）来生成任务，并使用SimplePlanner类来管理任务计划。


```py
"""The planning system organizes the Agent's activities."""
from autogpt.core.planning.schema import Task, TaskStatus, TaskType
from autogpt.core.planning.simple import PlannerSettings, SimplePlanner

```

# `autogpts/autogpt/autogpt/core/planning/prompt_strategies/initial_plan.py`

这段代码是一个自定义的Python库，用于实现类自动对话（Chatbot）应用程序。它主要用于以下几个方面：

1. 引入logging库，用于在运行时记录相关操作的日志信息。
2. 从autogpt库中导入SystemConfiguration、UserConfigurable、Task、TaskType、PromptStrategy、ChatPrompt、LanguageModelClassification和json_loads库。
3. 自定义AssistantChatMessageDict类，实现了一个类比ChatMessage更加丰富的Chat对话实体。
4. 自定义ChatMessage类，实现了ChatMessage实体。
5. 自定义CompletionModelFunction类，实现了一个自定义的对话补全功能。
6. 自定义PromptStrategy类，实现了自定义的Prompt策略。
7. 自定义LanguageModelClassification类，实现了自定义的语言模型分类。
8. 自定义的PromptStrategy类，实现了自定义的Prompt策略。
9. 自定义的utils.json_loads函数，实现了对JSON格式的数据解析。
10. 自定义的to_numbered_list函数，实现了将文本列表转换为数字列表的功能。


```py
import logging

from autogpt.core.configuration import SystemConfiguration, UserConfigurable
from autogpt.core.planning.schema import Task, TaskType
from autogpt.core.prompting import PromptStrategy
from autogpt.core.prompting.schema import ChatPrompt, LanguageModelClassification
from autogpt.core.prompting.utils import json_loads, to_numbered_list
from autogpt.core.resource.model_providers import (
    AssistantChatMessageDict,
    ChatMessage,
    CompletionModelFunction,
)
from autogpt.core.utils.json_schema import JSONSchema

logger = logging.getLogger(__name__)


```

This is a class definition for an `AssistantChatPrompt` which inherits from the `ChatPrompt` class and adds support for an `Agent` and an `OSInfo` parameter.

The `AssistantChatPrompt` class has the following methods:

* `__init__(self, agent_name: str, agent_role: str, os_info: str, api_budget: float, current_time: str, **kwargs)`: Initializes the object with the required parameters.
* `parse_response_content(response_content: AssistantChatMessageDict) -> dict`: Parses the actual text response from the objective model and returns the parsed response.
* `system(template_name: str) -> str`: Format and returns the system response based on the given template.
* `user(template_name: str) -> str`: Format and returns the user response based on the given template.
* `_create_plan_function(agents: List[str], capabilities: List[str], system_info: List[str]) -> None`: A function for creating a plan based on the given capabilities and system information.

This class can be used in an `Assistant` class as follows:
```py
assistant = Assistant(
   __name__ = "assistant_name",
   agent_name = "assistant_agent_name",
   agent_role = "assistant_agent_role",
   os_info = "assistant_os_info",
   api_budget = "assistant_api_budget",
   current_time = "assistant_current_time",
   # other parameters
)

response_content = assistant.parse_response_content(assistant.get_response())
```


```py
class InitialPlanConfiguration(SystemConfiguration):
    model_classification: LanguageModelClassification = UserConfigurable()
    system_prompt_template: str = UserConfigurable()
    system_info: list[str] = UserConfigurable()
    user_prompt_template: str = UserConfigurable()
    create_plan_function: dict = UserConfigurable()


class InitialPlan(PromptStrategy):
    DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
        "You are an expert project planner. You're responsibility is to create work plans for autonomous agents. "
        "You will be given a name, a role, set of goals for the agent to accomplish. Your job is to "
        "break down those goals into a set of tasks that the agent can accomplish to achieve those goals. "
        "Agents are resourceful, but require clear instructions. Each task you create should have clearly defined "
        "`ready_criteria` that the agent can check to see if the task is ready to be started. Each task should "
        "also have clearly defined `acceptance_criteria` that the agent can check to evaluate if the task is complete. "
        "You should create as many tasks as you think is necessary to accomplish the goals.\n\n"
        "System Info:\n{system_info}"
    )

    DEFAULT_SYSTEM_INFO = [
        "The OS you are running on is: {os_info}",
        "It takes money to let you run. Your API budget is ${api_budget:.3f}",
        "The current time and date is {current_time}",
    ]

    DEFAULT_USER_PROMPT_TEMPLATE = (
        "You are {agent_name}, {agent_role}\n" "Your goals are:\n" "{agent_goals}"
    )

    DEFAULT_CREATE_PLAN_FUNCTION = CompletionModelFunction(
        name="create_initial_agent_plan",
        description="Creates a set of tasks that forms the initial plan for an autonomous agent.",
        parameters={
            "task_list": JSONSchema(
                type=JSONSchema.Type.ARRAY,
                items=JSONSchema(
                    type=JSONSchema.Type.OBJECT,
                    properties={
                        "objective": JSONSchema(
                            type=JSONSchema.Type.STRING,
                            description="An imperative verb phrase that succinctly describes the task.",
                        ),
                        "type": JSONSchema(
                            type=JSONSchema.Type.STRING,
                            description="A categorization for the task.",
                            enum=[t.value for t in TaskType],
                        ),
                        "acceptance_criteria": JSONSchema(
                            type=JSONSchema.Type.ARRAY,
                            items=JSONSchema(
                                type=JSONSchema.Type.STRING,
                                description="A list of measurable and testable criteria that must be met for the task to be considered complete.",
                            ),
                        ),
                        "priority": JSONSchema(
                            type=JSONSchema.Type.INTEGER,
                            description="A number between 1 and 10 indicating the priority of the task relative to other generated tasks.",
                            minimum=1,
                            maximum=10,
                        ),
                        "ready_criteria": JSONSchema(
                            type=JSONSchema.Type.ARRAY,
                            items=JSONSchema(
                                type=JSONSchema.Type.STRING,
                                description="A list of measurable and testable criteria that must be met before the task can be started.",
                            ),
                        ),
                    },
                ),
            ),
        },
    )

    default_configuration: InitialPlanConfiguration = InitialPlanConfiguration(
        model_classification=LanguageModelClassification.SMART_MODEL,
        system_prompt_template=DEFAULT_SYSTEM_PROMPT_TEMPLATE,
        system_info=DEFAULT_SYSTEM_INFO,
        user_prompt_template=DEFAULT_USER_PROMPT_TEMPLATE,
        create_plan_function=DEFAULT_CREATE_PLAN_FUNCTION.schema,
    )

    def __init__(
        self,
        model_classification: LanguageModelClassification,
        system_prompt_template: str,
        system_info: list[str],
        user_prompt_template: str,
        create_plan_function: dict,
    ):
        self._model_classification = model_classification
        self._system_prompt_template = system_prompt_template
        self._system_info = system_info
        self._user_prompt_template = user_prompt_template
        self._create_plan_function = CompletionModelFunction.parse(create_plan_function)

    @property
    def model_classification(self) -> LanguageModelClassification:
        return self._model_classification

    def build_prompt(
        self,
        agent_name: str,
        agent_role: str,
        agent_goals: list[str],
        abilities: list[str],
        os_info: str,
        api_budget: float,
        current_time: str,
        **kwargs,
    ) -> ChatPrompt:
        template_kwargs = {
            "agent_name": agent_name,
            "agent_role": agent_role,
            "os_info": os_info,
            "api_budget": api_budget,
            "current_time": current_time,
            **kwargs,
        }
        template_kwargs["agent_goals"] = to_numbered_list(
            agent_goals, **template_kwargs
        )
        template_kwargs["abilities"] = to_numbered_list(abilities, **template_kwargs)
        template_kwargs["system_info"] = to_numbered_list(
            self._system_info, **template_kwargs
        )

        system_prompt = ChatMessage.system(
            self._system_prompt_template.format(**template_kwargs),
        )
        user_prompt = ChatMessage.user(
            self._user_prompt_template.format(**template_kwargs),
        )

        return ChatPrompt(
            messages=[system_prompt, user_prompt],
            functions=[self._create_plan_function],
            # TODO:
            tokens_used=0,
        )

    def parse_response_content(
        self,
        response_content: AssistantChatMessageDict,
    ) -> dict:
        """Parse the actual text response from the objective model.

        Args:
            response_content: The raw response content from the objective model.

        Returns:
            The parsed response.
        """
        try:
            parsed_response = json_loads(response_content["function_call"]["arguments"])
            parsed_response["task_list"] = [
                Task.parse_obj(task) for task in parsed_response["task_list"]
            ]
        except KeyError:
            logger.debug(f"Failed to parse this response content: {response_content}")
            raise
        return parsed_response

```

# `autogpts/autogpt/autogpt/core/planning/prompt_strategies/name_and_goals.py`

这段代码是一个自动化工具有关的 Python 代码，它使用了 Python 的 logging 库来输出信息。它从 autogpt 包中引入了多个类，包括 SystemConfiguration、UserConfigurable、PromptStrategy、ChatPrompt、LanguageModelClassification、json_loads 函数等，这些类和函数都是用于实现和处理自然语言处理任务的工具和组件。

具体来说，这段代码的作用是：

1. 导入 logging 库，以便在需要时输出信息。
2. 从 autogpt 包中引入了 SystemConfiguration、UserConfigurable、PromptStrategy、ChatPrompt、LanguageModelClassification、json_loads 函数等类，这些类和函数都是用于实现和处理自然语言处理任务的工具和组件。
3. 在代码内部，通过多层调用这些类和函数，实现了自然语言处理任务的配置、提示、对话模型的创建和解析等功能。
4. 输出了自动化工具有关的信息，以便开发者更好的了解代码的实现。


```py
import logging

from autogpt.core.configuration import SystemConfiguration, UserConfigurable
from autogpt.core.prompting import PromptStrategy
from autogpt.core.prompting.schema import ChatPrompt, LanguageModelClassification
from autogpt.core.prompting.utils import json_loads
from autogpt.core.resource.model_providers import (
    AssistantChatMessageDict,
    ChatMessage,
    CompletionModelFunction,
)
from autogpt.core.utils.json_schema import JSONSchema

logger = logging.getLogger(__name__)


```


This is a class that wraps an AI language model and defines methods for generating responses
to user prompts. The class takes in several arguments:

- `model_classification`: A `LanguageModelClassification` object that defines the model architecture.
- `system_prompt`: The prompt message for the system to display.
- `user_prompt_template`: The template for prompting the user to provide their objectives.
- `create_agent_function`: A function that defines the agent to use for completing the task.

The class has an `__init__` method that sets the object's attributes and a `build_prompt` method
that generates a prompt for the user to complete their objective.

The `build_prompt` method takes in a user objective,


```py
class NameAndGoalsConfiguration(SystemConfiguration):
    model_classification: LanguageModelClassification = UserConfigurable()
    system_prompt: str = UserConfigurable()
    user_prompt_template: str = UserConfigurable()
    create_agent_function: dict = UserConfigurable()


class NameAndGoals(PromptStrategy):
    DEFAULT_SYSTEM_PROMPT = (
        "Your job is to respond to a user-defined task, given in triple quotes, by "
        "invoking the `create_agent` function to generate an autonomous agent to "
        "complete the task. "
        "You should supply a role-based name for the agent, "
        "an informative description for what the agent does, and "
        "1 to 5 goals that are optimally aligned with the successful completion of "
        "its assigned task.\n"
        "\n"
        "Example Input:\n"
        '"""Help me with marketing my business"""\n\n'
        "Example Function Call:\n"
        "create_agent(name='CMOGPT', "
        "description='A professional digital marketer AI that assists Solopreneurs in "
        "growing their businesses by providing world-class expertise in solving "
        "marketing problems for SaaS, content products, agencies, and more.', "
        "goals=['Engage in effective problem-solving, prioritization, planning, and "
        "supporting execution to address your marketing needs as your virtual Chief "
        "Marketing Officer.', 'Provide specific, actionable, and concise advice to "
        "help you make informed decisions without the use of platitudes or overly "
        "wordy explanations.', 'Identify and prioritize quick wins and cost-effective "
        "campaigns that maximize results with minimal time and budget investment.', "
        "'Proactively take the lead in guiding you and offering suggestions when faced "
        "with unclear information or uncertainty to ensure your marketing strategy "
        "remains on track.'])"
    )

    DEFAULT_USER_PROMPT_TEMPLATE = '"""{user_objective}"""'

    DEFAULT_CREATE_AGENT_FUNCTION = CompletionModelFunction(
        name="create_agent",
        description="Create a new autonomous AI agent to complete a given task.",
        parameters={
            "agent_name": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="A short role-based name for an autonomous agent.",
            ),
            "agent_role": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="An informative one sentence description of what the AI agent does",
            ),
            "agent_goals": JSONSchema(
                type=JSONSchema.Type.ARRAY,
                minItems=1,
                maxItems=5,
                items=JSONSchema(
                    type=JSONSchema.Type.STRING,
                ),
                description=(
                    "One to five highly effective goals that are optimally aligned with the completion of a "
                    "specific task. The number and complexity of the goals should correspond to the "
                    "complexity of the agent's primary objective."
                ),
            ),
        },
    )

    default_configuration: NameAndGoalsConfiguration = NameAndGoalsConfiguration(
        model_classification=LanguageModelClassification.SMART_MODEL,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        user_prompt_template=DEFAULT_USER_PROMPT_TEMPLATE,
        create_agent_function=DEFAULT_CREATE_AGENT_FUNCTION.schema,
    )

    def __init__(
        self,
        model_classification: LanguageModelClassification,
        system_prompt: str,
        user_prompt_template: str,
        create_agent_function: dict,
    ):
        self._model_classification = model_classification
        self._system_prompt_message = system_prompt
        self._user_prompt_template = user_prompt_template
        self._create_agent_function = CompletionModelFunction.parse(
            create_agent_function
        )

    @property
    def model_classification(self) -> LanguageModelClassification:
        return self._model_classification

    def build_prompt(self, user_objective: str = "", **kwargs) -> ChatPrompt:
        system_message = ChatMessage.system(self._system_prompt_message)
        user_message = ChatMessage.user(
            self._user_prompt_template.format(
                user_objective=user_objective,
            )
        )
        prompt = ChatPrompt(
            messages=[system_message, user_message],
            functions=[self._create_agent_function],
            # TODO
            tokens_used=0,
        )
        return prompt

    def parse_response_content(
        self,
        response_content: AssistantChatMessageDict,
    ) -> dict:
        """Parse the actual text response from the objective model.

        Args:
            response_content: The raw response content from the objective model.

        Returns:
            The parsed response.

        """
        try:
            parsed_response = json_loads(response_content["function_call"]["arguments"])
        except KeyError:
            logger.debug(f"Failed to parse this response content: {response_content}")
            raise
        return parsed_response

```

# `autogpts/autogpt/autogpt/core/planning/prompt_strategies/next_ability.py`

这段代码是一个自定义的Python库，它实现了AutogPT的一些核心功能。具体来说，它实现了以下功能：

1. 引入logging库，用于在日志中记录信息。
2. 从autogpt库中导入SystemConfiguration、UserConfigurable和Task。
3. 从autogpt库中导入PromptStrategy和ChatPrompt。
4. 从autogpt库中导入JSONSchema。
5. 定义了一个logger变量，用于在日志中记录信息。
6. 实现了自定义AssistantChatMessageDict和ChatMessage类。
7. 实现了CompletionModelFunction。
8. 实现了从json_loads函数中解析json文件，并将其转换为数字列表。
9. 实现了从ChatPrompt中获取输入，并在PromptStrategy中对其进行处理。
10. 实现了将任务添加到completion_model_function中，以获取用户的输入并返回结果。

总之，这段代码定义了一个自定义的库，用于实现AutogPT的核心功能，包括对输入的任务进行解析和处理，以获取用户的输入并返回结果。


```py
import logging

from autogpt.core.configuration import SystemConfiguration, UserConfigurable
from autogpt.core.planning.schema import Task
from autogpt.core.prompting import PromptStrategy
from autogpt.core.prompting.schema import ChatPrompt, LanguageModelClassification
from autogpt.core.prompting.utils import json_loads, to_numbered_list
from autogpt.core.resource.model_providers import (
    AssistantChatMessageDict,
    ChatMessage,
    CompletionModelFunction,
)
from autogpt.core.utils.json_schema import JSONSchema

logger = logging.getLogger(__name__)


```

This appears to be a class that defines an `AssistantChatMessage` object that inherits from a `ChatMessage` object. It appears to have a `parse_response_content` method that parses the actual text response from the objective model.

The `parse_response_content` method takes an `AssistantChatMessageDict` object as its argument and returns a parsed response. It does this by calling the `ChatMessage` class's `parse_response` method, which takes a JSON response object as its argument.

The `parse_response` method takes an `AssistantChatMessageDict` object as its argument and returns a parsed response. It does this by calling the `ChatMessage` class's `parse_response` method, which takes a JSON response object as its argument.

The `ChatMessage` class appears to have a number of methods for managing the response content, including `system`, `user`, and `ability_specs`. These methods are not defined in the provided code and would need to be implemented in order for the class to work correctly.


```py
class NextAbilityConfiguration(SystemConfiguration):
    model_classification: LanguageModelClassification = UserConfigurable()
    system_prompt_template: str = UserConfigurable()
    system_info: list[str] = UserConfigurable()
    user_prompt_template: str = UserConfigurable()
    additional_ability_arguments: dict = UserConfigurable()


class NextAbility(PromptStrategy):
    DEFAULT_SYSTEM_PROMPT_TEMPLATE = "System Info:\n{system_info}"

    DEFAULT_SYSTEM_INFO = [
        "The OS you are running on is: {os_info}",
        "It takes money to let you run. Your API budget is ${api_budget:.3f}",
        "The current time and date is {current_time}",
    ]

    DEFAULT_USER_PROMPT_TEMPLATE = (
        "Your current task is is {task_objective}.\n"
        "You have taken {cycle_count} actions on this task already. "
        "Here is the actions you have taken and their results:\n"
        "{action_history}\n\n"
        "Here is additional information that may be useful to you:\n"
        "{additional_info}\n\n"
        "Additionally, you should consider the following:\n"
        "{user_input}\n\n"
        "Your task of {task_objective} is complete when the following acceptance criteria have been met:\n"
        "{acceptance_criteria}\n\n"
        "Please choose one of the provided functions to accomplish this task. "
        "Some tasks may require multiple functions to accomplish. If that is the case, choose the function that "
        "you think is most appropriate for the current situation given your progress so far."
    )

    DEFAULT_ADDITIONAL_ABILITY_ARGUMENTS = {
        "motivation": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Your justification for choosing choosing this function instead of a different one.",
        ),
        "self_criticism": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Thoughtful self-criticism that explains why this function may not be the best choice.",
        ),
        "reasoning": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Your reasoning for choosing this function taking into account the `motivation` and weighing the `self_criticism`.",
        ),
    }

    default_configuration: NextAbilityConfiguration = NextAbilityConfiguration(
        model_classification=LanguageModelClassification.SMART_MODEL,
        system_prompt_template=DEFAULT_SYSTEM_PROMPT_TEMPLATE,
        system_info=DEFAULT_SYSTEM_INFO,
        user_prompt_template=DEFAULT_USER_PROMPT_TEMPLATE,
        additional_ability_arguments={
            k: v.to_dict() for k, v in DEFAULT_ADDITIONAL_ABILITY_ARGUMENTS.items()
        },
    )

    def __init__(
        self,
        model_classification: LanguageModelClassification,
        system_prompt_template: str,
        system_info: list[str],
        user_prompt_template: str,
        additional_ability_arguments: dict,
    ):
        self._model_classification = model_classification
        self._system_prompt_template = system_prompt_template
        self._system_info = system_info
        self._user_prompt_template = user_prompt_template
        self._additional_ability_arguments = JSONSchema.parse_properties(
            additional_ability_arguments
        )
        for p in self._additional_ability_arguments.values():
            p.required = True

    @property
    def model_classification(self) -> LanguageModelClassification:
        return self._model_classification

    def build_prompt(
        self,
        task: Task,
        ability_specs: list[CompletionModelFunction],
        os_info: str,
        api_budget: float,
        current_time: str,
        **kwargs,
    ) -> ChatPrompt:
        template_kwargs = {
            "os_info": os_info,
            "api_budget": api_budget,
            "current_time": current_time,
            **kwargs,
        }

        for ability in ability_specs:
            ability.parameters.update(self._additional_ability_arguments)

        template_kwargs["task_objective"] = task.objective
        template_kwargs["cycle_count"] = task.context.cycle_count
        template_kwargs["action_history"] = to_numbered_list(
            [action.summary() for action in task.context.prior_actions],
            no_items_response="You have not taken any actions yet.",
            **template_kwargs,
        )
        template_kwargs["additional_info"] = to_numbered_list(
            [memory.summary() for memory in task.context.memories]
            + [info for info in task.context.supplementary_info],
            no_items_response="There is no additional information available at this time.",
            **template_kwargs,
        )
        template_kwargs["user_input"] = to_numbered_list(
            [user_input for user_input in task.context.user_input],
            no_items_response="There are no additional considerations at this time.",
            **template_kwargs,
        )
        template_kwargs["acceptance_criteria"] = to_numbered_list(
            [acceptance_criteria for acceptance_criteria in task.acceptance_criteria],
            **template_kwargs,
        )

        template_kwargs["system_info"] = to_numbered_list(
            self._system_info,
            **template_kwargs,
        )

        system_prompt = ChatMessage.system(
            self._system_prompt_template.format(**template_kwargs)
        )
        user_prompt = ChatMessage.user(
            self._user_prompt_template.format(**template_kwargs)
        )

        return ChatPrompt(
            messages=[system_prompt, user_prompt],
            functions=ability_specs,
            # TODO:
            tokens_used=0,
        )

    def parse_response_content(
        self,
        response_content: AssistantChatMessageDict,
    ) -> dict:
        """Parse the actual text response from the objective model.

        Args:
            response_content: The raw response content from the objective model.

        Returns:
            The parsed response.

        """
        try:
            function_name = response_content["function_call"]["name"]
            function_arguments = json_loads(
                response_content["function_call"]["arguments"]
            )
            parsed_response = {
                "motivation": function_arguments.pop("motivation"),
                "self_criticism": function_arguments.pop("self_criticism"),
                "reasoning": function_arguments.pop("reasoning"),
                "next_ability": function_name,
                "ability_arguments": function_arguments,
            }
        except KeyError:
            logger.debug(f"Failed to parse this response content: {response_content}")
            raise
        return parsed_response

```

# `autogpts/autogpt/autogpt/core/planning/prompt_strategies/__init__.py`

这段代码定义了三个类，分别是InitialPlan、InitialPlanConfiguration和NameAndGoals。

InitialPlan是一个表示计划或项目的类，可以包含一些可执行任务、时间表和资源限制等。

InitialPlanConfiguration是InitialPlan的子类，提供更具体的配置信息。

NameAndGoals是一个表示项目和任务的类，包含了项目的名称和目标。

NameAndGoalsConfiguration是NameAndGoals的子类，提供了用于定义项目名称和目标的配置信息。

NextAbility是一个表示能力的类，可以比较两个计划或项目的完成情况，返回一个未来的能力。

NextAbilityConfiguration是NextAbility的子类，提供了定义能力的配置信息，包括能力名称、描述、触发条件和结果等。


```py
from .initial_plan import InitialPlan, InitialPlanConfiguration
from .name_and_goals import NameAndGoals, NameAndGoalsConfiguration
from .next_ability import NextAbility, NextAbilityConfiguration

```

# `autogpts/autogpt/autogpt/core/plugin/base.py`

这段代码使用了多种 Python 类型，包括 ABc、enum 和 typing，主要作用是定义了一个 Pydantic BaseModel，用于在训练和运行时保证模型的正确性。通过导入 autogpt 包中的 Configuration、Ability 和 Memory，以及从 autogpt 的类型定义中定义了不同的类型，如 ChatModelProvider 和 embeddingModelProvider，从而实现了模型提供者。同时，通过从 Configuration 中声明了不同的类型，如 Ability、AbilityRegistry 和 Memory，实现了这些类型的能力、注册中心和内存。从而，这段代码的主要目的是为了定义一个通用的 Model 类，用于支持不同的模型类型，从而方便在训练和运行时使用。


```py
import abc
import enum
from typing import TYPE_CHECKING, Type

from pydantic import BaseModel

from autogpt.core.configuration import SystemConfiguration, UserConfigurable

if TYPE_CHECKING:
    from autogpt.core.ability import Ability, AbilityRegistry
    from autogpt.core.memory import Memory
    from autogpt.core.resource.model_providers import (
        ChatModelProvider,
        EmbeddingModelProvider,
    )

    # Expand to other types as needed
    PluginType = (
        Type[Ability]  # Swappable now
        | Type[AbilityRegistry]  # Swappable maybe never
        | Type[ChatModelProvider]  # Swappable soon
        | Type[EmbeddingModelProvider]  # Swappable soon
        | Type[Memory]  # Swappable now
        #    | Type[Planner]  # Swappable soon
    )


```

这段代码定义了一个名为 "PluginStorageFormat" 的类，它属于一个名为 "enum.Enum" 的枚举类型。这个类有两个参数，一个是字符串类型（str），另一个是枚举类型（enum.Enum）。

这个类的目的是定义了 supported(许可的) 插件存储格式，告诉用户他们可以选择把插件储存在这里 supported(许可的) 位置之一。

具体来说，这个类的实例可以有以下两种形式：

1. PluginStorageFormat 类的实例，包括 PluginStorageFormat 类本身和 PluginStorageFormat\_Installer 类（需要从 package.py 导入）。

2. PluginStorageFormat 类的实例，不包括 PluginStorageFormat\_Installer 类。这种形式的作用是为用户提供一个简单的 "立即可用" 的接口，而 PluginStorageFormat\_Installer 类将在稍后提供实际的安装功能。

为了给用户提供一个可以灵活设置的接口，PluginStorageFormat 类包含了一个 "INSTALLED\_PACKAGE"（已安装的包）和一个 "WORKSPACE"（工作区）成员变量，这些成员变量分别表示插件在哪个地方存放，用户可以随时更改。

此外，目前（2023年2月19日）需要使用 "open\_api\_url"（OpenAPI URL）进行注册和获取信息，而 "autogpt\_plugin\_service"（自动注册和自动服务）是一个长远解决方案，目前还在设计中。另外，也需要 "pypi"（pip installer）进行 package 的安装，以及 "git"（git repository manager）和 "pypi"（自动注册和自动服务）进行自动化的安装。


```py
class PluginStorageFormat(str, enum.Enum):
    """Supported plugin storage formats.

    Plugins can be stored at one of these supported locations.

    """

    INSTALLED_PACKAGE = "installed_package"  # Required now, loads system defaults
    WORKSPACE = "workspace"  # Required now
    # OPENAPI_URL = "open_api_url"           # Soon (requires some tooling we don't have yet).
    # OTHER_FILE_PATH = "other_file_path"    # Maybe later (maybe now)
    # GIT = "git"                            # Maybe later (or soon)
    # PYPI = "pypi"                          # Maybe later
    # AUTOGPT_PLUGIN_SERVICE = "autogpt_plugin_service"  # Long term solution, requires design
    # AUTO = "auto"                          # Feature for later maybe, automatically find plugin.


```

这段代码定义了两个 Python 包的安装位置，以及一个 Git 包的安装位置。其中，第一个包是一个名为 "example" 的包，它的安装位置是相对于存储库格式和存储路由。第二个包是一个名为 "workspace" 的包，它的安装位置是相对于存储库格式和存储路由。这两个包的安装位置都包含了 "relative/path/to/plugin...." 这样的路径，这表明它们会在程序的相对路径处寻找所需的插件。而第三个包是一个名为 "git" 的包，它的安装位置是相对于存储库格式。


```py
# Installed package example
# PluginLocation(
#     storage_format='installed_package',
#     storage_route='autogpt_plugins.twitter.SendTwitterMessage'
# )
# Workspace example
# PluginLocation(
#     storage_format='workspace',
#     storage_route='relative/path/to/plugin.pkl'
#     OR
#     storage_route='relative/path/to/plugin.py'
# )
# Git
# PluginLocation(
#     storage_format='git',
```

这段代码定义了两个 `PluginLocation` 类，用于定义存储库中的软件包位置。

第一个 `PluginLocation` 类定义了存储库中软件包的存储格式为 PyPI，并且存储库中的软件包名称为 `"autogpt_wolframalpha"`。

第二个 `PluginLocation` 类定义了存储库中软件包的存储格式为 InstalledPackage，并且存储库中的软件包名称为 `"autogpt_plugins.twitter.SendTwitterMessage"`。

这两个 `PluginLocation` 类用于定义存储库中软件包的位置和格式，以便程序在需要时可以从存储库中下载和安装这些软件包。


```py
#     Exact format TBD.
#     storage_route='https://github.com/gravelBridge/AutoGPT-WolframAlpha/blob/main/autogpt-wolframalpha/wolfram_alpha.py'
# )
# PyPI
# PluginLocation(
#     storage_format='pypi',
#     storage_route='package_name'
# )


# PluginLocation(
#     storage_format='installed_package',
#     storage_route='autogpt_plugins.twitter.SendTwitterMessage'
# )


```

这段代码定义了一个名为 PluginLocation 的类，该类继承自 SystemConfiguration 类。

PluginLocation 类表示一个用于加载插件的路径，它由两个属性组成：存储格式和存储位置。这些属性都是用户可配置的。

PluginStorageRoute 属性指定从何处加载插件。它是一个字符串，以指定加载插件的路径（例如，一个导入路径或文件路径）。

PluginLocation 类是 PluginService 的成员，负责加载插件。当加载插件时，PluginService 会根据 PluginStorageRoute 属性的值加载插件。


```py
# A plugin storage route.
#
# This is a string that specifies where to load a plugin from
# (e.g. an import path or file path).
PluginStorageRoute = str


class PluginLocation(SystemConfiguration):
    """A plugin location.

    This is a combination of a plugin storage format and a plugin storage route.
    It is used by the PluginService to load plugins.

    """

    storage_format: PluginStorageFormat = UserConfigurable()
    storage_route: PluginStorageRoute = UserConfigurable()


```



这段代码定义了一个名为 PluginMetadata 的类，它继承自 BaseModel 类。PluginMetadata 类包含一个名字、描述和一个位置，这些信息都是关于插件的基本信息。

接下来，定义了一个名为 PluginService 的类，它继承自 abstract 的 ABC 类。PluginService 类是用于加载插件的服务，应该是一个空类，因为它包含一个抽象方法 `get_plugin`。这个方法从不同的存储格式中加载插件，例如从文件路径、从安装包、从工作空间等。

具体来说，PluginService 的 `get_plugin` 方法可以通过文件路径、从安装包、从工作空间等不同方式加载插件，并在加载成功后返回插件的类型。而 `load_from_file_path` 和 `load_from_import_path` 方法则是用于在文件路径和安装包路径中查找插件并加载它们。

另外，`resolve_name_to_path` 和 `load_from_workspace` 方法用于将插件名称映射到插件路径，`load_from_installed_package` 方法用于从安装包中加载插件。


```py
class PluginMetadata(BaseModel):
    """Metadata about a plugin."""

    name: str
    description: str
    location: PluginLocation


class PluginService(abc.ABC):
    """Base class for plugin service.

    The plugin service should be stateless. This defines the interface for
    loading plugins from various storage formats.

    """

    @staticmethod
    @abc.abstractmethod
    def get_plugin(plugin_location: PluginLocation) -> "PluginType":
        """Get a plugin from a plugin location."""
        ...

    ####################################
    # Low-level storage format loaders #
    ####################################
    @staticmethod
    @abc.abstractmethod
    def load_from_file_path(plugin_route: PluginStorageRoute) -> "PluginType":
        """Load a plugin from a file path."""

        ...

    @staticmethod
    @abc.abstractmethod
    def load_from_import_path(plugin_route: PluginStorageRoute) -> "PluginType":
        """Load a plugin from an import path."""
        ...

    @staticmethod
    @abc.abstractmethod
    def resolve_name_to_path(
        plugin_route: PluginStorageRoute, path_type: str
    ) -> PluginStorageRoute:
        """Resolve a plugin name to a plugin path."""
        ...

    #####################################
    # High-level storage format loaders #
    #####################################

    @staticmethod
    @abc.abstractmethod
    def load_from_workspace(plugin_route: PluginStorageRoute) -> "PluginType":
        """Load a plugin from the workspace."""
        ...

    @staticmethod
    @abc.abstractmethod
    def load_from_installed_package(plugin_route: PluginStorageRoute) -> "PluginType":
        """Load a plugin from an installed package."""
        ...

```

# `autogpts/autogpt/autogpt/core/plugin/simple.py`

It looks like the `SimplePluginService` class is responsible for loading plugins from various storage routes, including file paths and installed packages. The `load_from_file_path` method is specific to loading from file paths, while the `load_from_import_path` method is specific to loading from import paths. The `resolve_name_to_path` method is meant to be a higher-level API to map plugin names to paths, but it is not yet implemented. It is also noted that the `SimplePluginService` class is not implemented as a full plugin, but rather as a utility class for loading plugins.


```py
from importlib import import_module
from typing import TYPE_CHECKING

from autogpt.core.plugin.base import (
    PluginLocation,
    PluginService,
    PluginStorageFormat,
    PluginStorageRoute,
)

if TYPE_CHECKING:
    from autogpt.core.plugin.base import PluginType


class SimplePluginService(PluginService):
    @staticmethod
    def get_plugin(plugin_location: dict | PluginLocation) -> "PluginType":
        """Get a plugin from a plugin location."""
        if isinstance(plugin_location, dict):
            plugin_location = PluginLocation.parse_obj(plugin_location)
        if plugin_location.storage_format == PluginStorageFormat.WORKSPACE:
            return SimplePluginService.load_from_workspace(
                plugin_location.storage_route
            )
        elif plugin_location.storage_format == PluginStorageFormat.INSTALLED_PACKAGE:
            return SimplePluginService.load_from_installed_package(
                plugin_location.storage_route
            )
        else:
            raise NotImplementedError(
                f"Plugin storage format {plugin_location.storage_format} is not implemented."
            )

    ####################################
    # Low-level storage format loaders #
    ####################################
    @staticmethod
    def load_from_file_path(plugin_route: PluginStorageRoute) -> "PluginType":
        """Load a plugin from a file path."""
        # TODO: Define an on disk storage format and implement this.
        #   Can pull from existing zip file loading implementation
        raise NotImplemented("Loading from file path is not implemented.")

    @staticmethod
    def load_from_import_path(plugin_route: PluginStorageRoute) -> "PluginType":
        """Load a plugin from an import path."""
        module_path, _, class_name = plugin_route.rpartition(".")
        return getattr(import_module(module_path), class_name)

    @staticmethod
    def resolve_name_to_path(
        plugin_route: PluginStorageRoute, path_type: str
    ) -> PluginStorageRoute:
        """Resolve a plugin name to a plugin path."""
        # TODO: Implement a discovery system for finding plugins by name from known
        #   storage locations. E.g. if we know that path_type is a file path, we can
        #   search the workspace for it. If it's an import path, we can check the core
        #   system and the auto_gpt_plugins package.
        raise NotImplemented("Resolving plugin name to path is not implemented.")

    #####################################
    # High-level storage format loaders #
    #####################################

    @staticmethod
    def load_from_workspace(plugin_route: PluginStorageRoute) -> "PluginType":
        """Load a plugin from the workspace."""
        plugin = SimplePluginService.load_from_file_path(plugin_route)
        return plugin

    @staticmethod
    def load_from_installed_package(plugin_route: PluginStorageRoute) -> "PluginType":
        plugin = SimplePluginService.load_from_import_path(plugin_route)
        return plugin

```

# `autogpts/autogpt/autogpt/core/plugin/__init__.py`

这段代码是一个AutogPT插件系统的代码，表明了它可以用来扩展代理的功能。它包含两个主要部分：

1. "The plugin system allows the Agent to be extended with new functionality."(插件系统允许代理扩展新的功能。)
这个短语是一个XML声明，表示这个插件系统支持代理扩展。

2. "from autogpt.core.plugin.base import PluginService。"(从autogpt.core.plugin.base import PluginService。)
这个代码块表示从autogpt.core.plugin.base模块中导入了一个名为PluginService的类。

PluginService是一个类，它可以从其中使用代理扩展的功能。这个代码块导入这个类，以便在需要的时候可以创建一个PluginService实例并使用它来扩展代理的功能。


```py
"""The plugin system allows the Agent to be extended with new functionality."""
from autogpt.core.plugin.base import PluginService

```

# `autogpts/autogpt/autogpt/core/prompting/base.py`

这段代码定义了一个名为`PromptStrategy`的类，它实现了`abc.ABC`接口，继承自`SystemConfiguration`和`AssistantChatMessageDict`类。这个类定义了一些抽象方法，用于配置、构建 prompt 和解析 response content。

具体来说，`PromptStrategy`的`default_configuration`方法继承自`SystemConfiguration`，用于设置默认的配置；`model_classification`方法继承自`LanguageModelClassification`，用于设置自然语言处理模型类型；`build_prompt`方法实现了`abc.abstractmethod`，用于构建 prompt；`parse_response_content`方法实现了`abc.abstractmethod`，用于解析 response content。

由于这些方法都是抽象方法，因此它们的实现将取决于具体的子类。在这个例子中，`PromptStrategy`的实现将取决于具体的子类，这些子类需要实现`build_prompt`和`parse_response_content`方法，以及继承自`abc.ABC`的父类。


```py
import abc

from autogpt.core.configuration import SystemConfiguration
from autogpt.core.resource.model_providers import AssistantChatMessageDict

from .schema import ChatPrompt, LanguageModelClassification


class PromptStrategy(abc.ABC):
    default_configuration: SystemConfiguration

    @property
    @abc.abstractmethod
    def model_classification(self) -> LanguageModelClassification:
        ...

    @abc.abstractmethod
    def build_prompt(self, *_, **kwargs) -> ChatPrompt:
        ...

    @abc.abstractmethod
    def parse_response_content(self, response_content: AssistantChatMessageDict):
        ...

```

# `autogpts/autogpt/autogpt/core/prompting/schema.py`

这段代码使用了Python的枚举类型（enum）为LanguageModelClassification创建了一个新的类。这个枚举类型定义了两种不同的模型类型：FAST_MODEL和SMART_MODEL。每种模型类型对应一个字符串值，分别表示“快速模型”和“智能模型”。

这个类还从pydantic库中继承了BaseModel和Field类，这些类可以用于定义模型和它的字段。

然后，这个类还从autogpt库的模型提供商中获取了一些模型，包括ChatMessage和ChatMessageDict类，以及CompletionModelFunction类型。

最后，这段代码还定义了一个LanguageModelClassification类，这个类继承自LanguageModelClassification枚举类型，它提供了两种不同的模型类型选择。


```py
import enum

from pydantic import BaseModel, Field

from autogpt.core.resource.model_providers.schema import (
    ChatMessage,
    ChatMessageDict,
    CompletionModelFunction,
)


class LanguageModelClassification(str, enum.Enum):
    """The LanguageModelClassification is a functional description of the model.

    This is used to determine what kind of model to use for a given prompt.
    Sometimes we prefer a faster or cheaper model to accomplish a task when
    possible.
    """

    FAST_MODEL = "fast_model"
    SMART_MODEL = "smart_model"


```

这段代码定义了一个名为 ChatPrompt 的类，继承自 BaseModel 类。该类包含两个属性：messages，是一个列表，用于存储用户发送的消息；functions，也是一个列表，用于存储自定义函数，可以是列表、函数或类的实例。

该类的构造函数是空函数，使用了 default_factory 参数，该参数用于指定 initialize 的默认值，如果没有提供默认值，该函数将创建一个空列表。

该类的 raw 方法返回了一个包含 ChatMessageDict 对象的列表，其中每个 ChatMessageDict 对象包含了消息内容、发送者和消息类型等信息。

该类的 __str__ 方法返回了一个字符串，该字符串打印出了 ChatPrompt 对象中所有消息的内容，格式为：{发送者： 消息内容}。

该类的实例可以通过调用 raw 方法获取到消息列表，然后使用 for 循环遍历每个消息，并使用 dict 获取消息内容字典。


```py
class ChatPrompt(BaseModel):
    messages: list[ChatMessage]
    functions: list[CompletionModelFunction] = Field(default_factory=list)

    def raw(self) -> list[ChatMessageDict]:
        return [m.dict() for m in self.messages]

    def __str__(self):
        return "\n\n".join(
            f"{m.role.value.upper()}: {m.content}" for m in self.messages
        )

```

# `autogpts/autogpt/autogpt/core/prompting/utils.py`

这段代码定义了一个名为 `to_numbered_list` 的函数，用于将一个列表中的元素格式化为带编号的列表。以下是该函数的实现步骤：

1. 导入 `ast` 和 `json` 模块。
2. 定义一个名为 `to_numbered_list` 的函数，它接受一个列表 `items` 和一个字符串 `no_items_response`(默认值为空字符串)，以及一个模板参数组 `template_args`。
3. 在函数体内，首先检查 `items` 是否为空，如果是，则输出一个空字符串。否则，我们将 `items` 中的每个元素循环遍历，并将它们格式化为 `{i+1}. {item.format(**template_args)}`，其中 `i` 是当前元素的编号，`item` 是元素本身，`template_args` 是模板参数组。
4. 将格式化后的元素字符串连接起来，并使用 `join` 方法将它们连接成一个字符串。最终的结果是，将编号和格式化后的元素列表字符串输出，或者如果 `no_items_response` 是字符串，则直接输出该字符串。


```py
import ast
import json


def to_numbered_list(
    items: list[str], no_items_response: str = "", **template_args
) -> str:
    if items:
        return "\n".join(
            f"{i+1}. {item.format(**template_args)}" for i, item in enumerate(items)
        )
    else:
        return no_items_response


```

这段代码定义了一个名为 `json_loads` 的函数，接受一个字符串参数 `json_str`，并尝试解析 JSON 字符串。

函数内部包含一个 hack 函数，因为在 Python 2.x 中，`ast.literal_eval()` 函数有时会抛出 `JSONDecodeError` 异常。为避免这个异常，函数尝试使用 `ast.parse()` 函数来解析 JSON 字符串，如果这个函数在解析 JSON 时出错，则使用 `ast.literal_eval()` 来解析 JSON。

如果 `ast.parse()` 函数成功解析 JSON，则返回解析后的数据，否则会执行下面的代码：

```py
try:
   print(f"json decode error {e}. trying literal eval")
   return ast.literal_eval(json_str)
except Exception:
   breakpoint()
```

这段代码会在解析 JSON 时捕获 `JSONDecodeError` 异常，如果发生这个异常，则会执行 `ast.literal_eval()` 函数，并尝试使用 `ast.parse()` 函数来解析 JSON。如果 `ast.parse()` 函数在解析 JSON 时出错，则执行下面的代码，将函数暂停（即挂起程序的执行），以便进一步排查问题。


```py
def json_loads(json_str: str):
    # TODO: this is a hack function for now. Trying to see what errors show up in testing.
    #   Can hopefully just replace with a call to ast.literal_eval (the function api still
    #   sometimes returns json strings with minor issues like trailing commas).
    try:
        json_str = json_str[json_str.index("{") : json_str.rindex("}") + 1]
        return ast.literal_eval(json_str)
    except json.decoder.JSONDecodeError as e:
        try:
            print(f"json decode error {e}. trying literal eval")
            return ast.literal_eval(json_str)
        except Exception:
            breakpoint()

```

# `autogpts/autogpt/autogpt/core/prompting/__init__.py`

这段代码定义了一个命名范围(__all__)，其中包含了一些模块名，如"LanguageModelClassification"、"ChatPrompt"和"PromptStrategy"。这些模块的作用是导入相关的类和函数，以便在程序中使用。

具体来说，从."base"模块中，引入了"PromptStrategy"类，从".schema"模块中，引入了"ChatPrompt"类，"LanguageModelClassification"类和"。这些模块和类可以被用于程序中的不同部分，例如在 ChatPrompt 和 LanguageModelClassification 中定义的函数和类，以及定义整个程序时需要使用的类和函数。


```py
from .base import PromptStrategy
from .schema import ChatPrompt, LanguageModelClassification

__all__ = [
    "LanguageModelClassification",
    "ChatPrompt",
    "PromptStrategy",
]

```

# `autogpts/autogpt/autogpt/core/resource/schema.py`

这段代码使用了Python的面向对象编程技术，主要用途是创建一个自定义的ABC类，用于定义资源的类型。

具体来说，该代码实现了一个名为ResourceType的枚举类型，它有两个枚举值：MODEL和MEMORY。这些枚举值可以用来定义一个ResourceType类，这个类可以接受一个字符串类型的参数，表示资源的类型。

在代码的下一行中，该类使用了pydantic库的AbstractBase class，这可能是用于将该类的定义与其他定义一致，以便与其他使用pydantic的代码进行交互。

在接下来的几行中，代码定义了一个名为MyResource的类，该类使用了pydantic的SecureBytes、SecureField和SecureStr修饰符。这些修饰符可以用于定义加密的数据，以便在传输过程中得到保护。

最后，代码定义了一个名为SystemConfiguration的类，该类使用了SystemSettings和UserConfigurable，这两个类似乎用于配置系统的设置。


```py
import abc
import enum

from pydantic import SecretBytes, SecretField, SecretStr

from autogpt.core.configuration import (
    SystemConfiguration,
    SystemSettings,
    UserConfigurable,
)


class ResourceType(str, enum.Enum):
    """An enumeration of resource types."""

    MODEL = "model"
    MEMORY = "memory"


```



这段代码定义了一个名为 "ProviderUsage" 的类，继承自 "abc.ABC" 类。这个类的抽象方法 "update_usage" 中包含了一个抽象方法内部的方法 "update_usage"，它接受任意数量的参数 "args" 和任意数量的参数 "kwargs"，返回值为 "None"。

在这个类的 "ProviderBudget" 类中，定义了一个名为 "update_usage_and_cost" 的抽象方法。这个方法中包含了一个 "update_usage" 方法和一个 "update_cost" 方法，它们都在私有状态下实现。

在 "ProviderUsage" 和 "ProviderBudget" 的类的后面，都使用了 "@abc.abstractmethod" 注释来定义它们的方法。这意味着它们所定义的方法是抽象方法，它们只能在子类中实现。

具体来说，"ProviderUsage" 类中的 "update_usage" 方法用于更新资源的使用情况，"ProviderBudget" 类中的 "update_usage_and_cost" 方法用于更新资源的使用情况和预算。这两个方法的具体实现都在各自的子类中完成。


```py
class ProviderUsage(SystemConfiguration, abc.ABC):
    @abc.abstractmethod
    def update_usage(self, *args, **kwargs) -> None:
        """Update the usage of the resource."""
        ...


class ProviderBudget(SystemConfiguration):
    total_budget: float = UserConfigurable()
    total_cost: float
    remaining_budget: float
    usage: ProviderUsage

    @abc.abstractmethod
    def update_usage_and_cost(self, *args, **kwargs) -> None:
        """Update the usage and cost of the resource."""
        ...


```

这段代码定义了一个名为ProviderCredentials的类，该类包含用于证书野程序的凭据。接下来，定义了一个名为Config的类，该类包含两个字典，其中第一个键是JSON编码器，包含一个函数，用于从给定的JSON字符串中获取秘密值。

然后，定义了一个名为ProviderSettings的类，该类包含一个名为ResourceType的属性，用于指定资源类型。此外，还包含一个名为ProviderCredentials的属性，用于证书凭据，以及一个名为Budget的属性，用于预算。

这段代码的主要目的是定义一个可以配置证书凭据和预算的类，用于证书野程序。通过这个类，用户可以指定从秘密存储库中获取证书凭据的方式，可以是JSON字符串、字节序列还是定义的函数。此外，还可以指定预算，用于控制证书野程序的花费。


```py
class ProviderCredentials(SystemConfiguration):
    """Struct for credentials."""

    class Config:
        json_encoders = {
            SecretStr: lambda v: v.get_secret_value() if v else None,
            SecretBytes: lambda v: v.get_secret_value() if v else None,
            SecretField: lambda v: v.get_secret_value() if v else None,
        }


class ProviderSettings(SystemSettings):
    resource_type: ResourceType
    credentials: ProviderCredentials | None = None
    budget: ProviderBudget | None = None


```

这段代码定义了一个名为"Embedding"的列表类型，该列表类型包含float类型的元素。这个列表类型的变量可以被用来作为模型提供商的输入特征，也可以被用来作为内存提供商的数据结构。通常情况下，模型提供商使用这种列表类型来存储模型中的参数，而内存提供商则可以使用这种列表类型来存储各种数据结构。


```py
# Used both by model providers and memory providers
Embedding = list[float]

```

# `autogpts/autogpt/autogpt/core/resource/__init__.py`

这段代码定义了五个来自 autogpt.core.resource.schema 模型的类：ProviderBudget、ProviderCredentials、ProviderSettings 和 ProviderUsage，以及一个名为 ResourceType 的枚举类型。

具体来说，这段代码描述了一个基于自动生成文本数据的预训练 GPT 模型的资源配置，其中包括以下内容：

- ProviderBudget 类表示预算信息，用于描述预训练模型的成本限制。
- ProviderCredentials 类表示访问模型的凭据，包括用户名和密码。
- ProviderSettings 类表示设置信息，包括模型架构、轮询策略、学习率等。
- ProviderUsage 类表示已使用的预训练模型使用情况。
- ResourceType 类表示资源类型，包括付费和免费两种类型。

这些类和枚举类型可以用于创建一个预训练 GPT 模型资源实例，根据用户提供的参数进行配置，并返回一个可以使用的对象。


```py
from autogpt.core.resource.schema import (
    ProviderBudget,
    ProviderCredentials,
    ProviderSettings,
    ProviderUsage,
    ResourceType,
)

```

# `autogpts/autogpt/autogpt/core/resource/model_providers/openai.py`

这段代码使用了多种Python模块，包括enum、functools、logging、math和time，以及从typing模块中定义的一些类型变量。它们的具体作用如下：

1. 导入`enum`模块，用于定义状态机中的状态枚举类型。
2. 导入`functools`模块，用于定义闭包函数。
3. 导入`logging`模块，用于定义日志记录的类。
4. 导入`math`模块，用于定义数学相关的类型。
5. 导入`time`模块，用于定义时间相关的类型。
6. 从`typing`模块中导入`Callable`类型变量，用于定义可调用函数。
7. 从`typing`模块中导入`Optional`类型变量，用于定义非空可选项。
8. 从`typing`模块中导入`ParamSpec`类型变量，用于定义参数规范。
9. 从`openai`模块中导入`APIError`和`RateLimitError`类型。
10. 从`tiktoken`模块中导入`Image`类型。
11. 定义一个名为`MyEnum`的类，该类继承自`enum`中的`State`类。
12. 在`MyEnum`类中定义了一个名为`MyEnum`的元组类型变量`state_normal`，该元组包含一个字符串类型的成员变量`power`和一个数字类型的成员变量`current_energy`。
13. 在`__init__`方法中，将`state_normal`元组成员赋值，使其成为`MyEnum`类的初始状态。
14. 在`draw_line`方法中，使用`tiktoken`模块中的`Image`类型，创建一个包含`MyEnum`类中所有状态的2D图像，并将其显示出来。
15. 在主程序中，首先创建一个`SystemConfiguration`实例，然后创建一个`UserConfigurable`实例，最后创建一个`Configurable`实例。
16. 将`APIError`和`RateLimitError`实例的`__init__`方法中定义的`api_error`和`rate_limit_error`函数分别绑定到`MyEnum`类的`__init__`和`draw_line`方法中。
17. 在主程序中，创建一个`State`类的实例，该实例初始化时调用`api_error`函数，以获取一个API错误，并尝试加载一个缓存图像以绘制图像。
18. 调用`draw_line`方法绘制图像，并使用`openai`模块中的`APIError`实例，作为绘制操作的失败时转换为成功的情况。


```py
import enum
import functools
import logging
import math
import time
from typing import Callable, Optional, ParamSpec, TypeVar

import openai
import tiktoken
from openai.error import APIError, RateLimitError

from autogpt.core.configuration import (
    Configurable,
    SystemConfiguration,
    UserConfigurable,
)
```

这段代码是一个自定义函数，它从autogpt库中定义了一系列聊天模型相关的类和函数。具体来说，它包括AssistantChatMessageDict类用于定义助手聊天消息的格式，AssistantFunctionCallDict类用于定义助手功能调用的消息格式，ChatMessage类用于定义聊天消息的格式，ChatModelInfo类用于定义聊天模型的信息，ChatModelProvider类用于定义聊天模型的提供者，ChatModelResponse类用于定义聊天模型的响应，CompletionModelFunction类用于定义对话完成的模型函数，Embedding类用于定义实体嵌入的格式，EmbeddingModelInfo类用于定义实体嵌入的详细信息，EmbeddingModelProvider类用于定义实体嵌入的提供者，EmbeddingModelResponse类用于定义实体嵌入的响应，ModelProviderBudget类用于定义模型提供商的预算，ModelProviderCredentials类用于定义模型提供商的凭证，ModelProviderName类用于定义模型提供商的名称，ModelProviderService类用于定义模型提供商的服務，ModelProviderSettings类用于定义模型提供商的設置，ModelProviderUsage类用于定义模型使用情况，ModelTokenizer类用于定义模型 tokenizer。


```py
from autogpt.core.resource.model_providers.schema import (
    AssistantChatMessageDict,
    AssistantFunctionCallDict,
    ChatMessage,
    ChatModelInfo,
    ChatModelProvider,
    ChatModelResponse,
    CompletionModelFunction,
    Embedding,
    EmbeddingModelInfo,
    EmbeddingModelProvider,
    EmbeddingModelResponse,
    ModelProviderBudget,
    ModelProviderCredentials,
    ModelProviderName,
    ModelProviderService,
    ModelProviderSettings,
    ModelProviderUsage,
    ModelTokenizer,
)
```

这段代码定义了一个名为`OpenAIModelName`的枚举类型，用于指定OpenAI模型名称。这个枚举类型包含了四个不同的模型名称：`GPT3_v1`、`GPT3_v2`、`GPT3_v2_16k`和`GPT3_ROLLING`。这些名称后面都跟了一个数字，表示模型的版本号。

这个枚举类型还定义了一个`<meta>`类，其中的`__get__`方法返回一个`JSON Schema`对象。这个`JSON Schema`定义了OpenAI模型的JSON数据结构，其中包括模型的`name`字段，它的类型为`string`，长度为`40`字节。

最后，这段代码还定义了一个名为`OpenAIEmbeddingParser`的函数，它接受两个参数：`Embedding`类型的数据和另一个参数，也是一个`Callable`类型，它的返回类型为`Embedding`。这个函数的作用是将输入的`Embedding`数据和另一个`Callable`函数的返回结果合并，并将合并后的结果返回。


```py
from autogpt.core.utils.json_schema import JSONSchema

_T = TypeVar("_T")
_P = ParamSpec("_P")

OpenAIEmbeddingParser = Callable[[Embedding], Embedding]
OpenAIChatParser = Callable[[str], dict]


class OpenAIModelName(str, enum.Enum):
    ADA = "text-embedding-ada-002"

    GPT3_v1 = "gpt-3.5-turbo-0301"
    GPT3_v2 = "gpt-3.5-turbo-0613"
    GPT3_v2_16k = "gpt-3.5-turbo-16k-0613"
    GPT3_ROLLING = "gpt-3.5-turbo"
    GPT3_ROLLING_16k = "gpt-3.5-turbo-16k"
    GPT3 = GPT3_ROLLING
    GPT3_16k = GPT3_ROLLING_16k

    GPT4_v1 = "gpt-4-0314"
    GPT4_v1_32k = "gpt-4-32k-0314"
    GPT4_v2 = "gpt-4-0613"
    GPT4_v2_32k = "gpt-4-32k-0613"
    GPT4_ROLLING = "gpt-4"
    GPT4_ROLLING_32k = "gpt-4-32k"
    GPT4 = GPT4_ROLLING
    GPT4_32k = GPT4_ROLLING_32k


```

It looks like OpenAI has defined several chat models in the OpenAIModelName.GPT* series. These models are designed to be large language models that can be used for natural language processing tasks, such as chatbots.

The `ChatModelInfo` class seems to define the properties of each model, such as the name, service provider, and token costs. The models also have a `prompt_token_cost` and `completion_token_cost` property, which appear to be the costs of generating a prompt or a completion token, respectively.

It's worth noting that these models are just examples and have not been tested or evaluated in any way. It's up to the user to determine how to use these models for their own chatbot applications.


```py
OPEN_AI_EMBEDDING_MODELS = {
    OpenAIModelName.ADA: EmbeddingModelInfo(
        name=OpenAIModelName.ADA,
        service=ModelProviderService.EMBEDDING,
        provider_name=ModelProviderName.OPENAI,
        prompt_token_cost=0.0001 / 1000,
        max_tokens=8191,
        embedding_dimensions=1536,
    ),
}


OPEN_AI_CHAT_MODELS = {
    info.name: info
    for info in [
        ChatModelInfo(
            name=OpenAIModelName.GPT3,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.0015 / 1000,
            completion_token_cost=0.002 / 1000,
            max_tokens=4096,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT3_16k,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.003 / 1000,
            completion_token_cost=0.004 / 1000,
            max_tokens=16384,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT4,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.03 / 1000,
            completion_token_cost=0.06 / 1000,
            max_tokens=8191,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT4_32k,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.06 / 1000,
            completion_token_cost=0.12 / 1000,
            max_tokens=32768,
            has_function_call_api=True,
        ),
    ]
}
```

这段代码的作用是复制具有相同规格的模型名称（即预训练模型），并将它们映射到对应的模型名称。该代码将 OpenAIModelName.GPT3 和 OpenAIModelName.GPT3_v2 映射为 OpenAIModelName.GPT3 和 OpenAIModelName.GPT3_v2，将 OpenAIModelName.GPT4 和 OpenAIModelName.GPT4_v1 映射为 OpenAIModelName.GPT4 和 OpenAIModelName.GPT4_v1，将 ChatModelName.GPT4 和 ChatModelName.GPT4_v1_32k 映射为 ChatModelName.GPT4 和 ChatModelName.GPT4_v1_32k。对于每个模型，代码创建一个 ChatModelInfo 对象，其中包含 ChatModel 类的属性，例如 name、endswith 等。然后，将模型名称和对应的预训练模型名称存储在 ChatModelInfo 对象的字典中，并将 ChatModel 类的函数调用 API 设置为 False。


```py
# Copy entries for models with equivalent specs
chat_model_mapping = {
    OpenAIModelName.GPT3: [OpenAIModelName.GPT3_v1, OpenAIModelName.GPT3_v2],
    OpenAIModelName.GPT3_16k: [OpenAIModelName.GPT3_v2_16k],
    OpenAIModelName.GPT4: [OpenAIModelName.GPT4_v1, OpenAIModelName.GPT4_v2],
    OpenAIModelName.GPT4_32k: [
        OpenAIModelName.GPT4_v1_32k,
        OpenAIModelName.GPT4_v2_32k,
    ],
}
for base, copies in chat_model_mapping.items():
    for copy in copies:
        copy_info = ChatModelInfo(**OPEN_AI_CHAT_MODELS[base].__dict__)
        copy_info.name = copy
        OPEN_AI_CHAT_MODELS[copy] = copy_info
        if copy.endswith(("-0301", "-0314")):
            copy_info.has_function_call_api = False


```

这段代码定义了一个名为 `OpenAIConfiguration` 的类，它是 `SystemConfiguration` 的子类。这个类的两个重载的 `**` 符号成员 `OPEN_AI_CHAT_MODELS` 和 `OPEN_AI_EMBEDDING_MODELS` 都使用了 `**` 符号成员，这意味着它们在以后继承自 `SystemConfiguration` 类时，将继承 `SystemConfiguration` 类中的所有成员。

接着，OpenAIConfiguration 类有一个名为 `retries_per_request` 的成员变量，它是 `UserConfigurable` 方法，这个方法的参数是一个整数，表示在每次请求中尝试运行 OpenAIChatModelProviderBudget 的尝试次数。

然后，OpenAIModelProviderBudget 类继承自 `ModelProviderBudget` 类。这个类的两个重载的 `**` 符号成员 `graceful_shutdown_threshold` 和 `warning_threshold` 都使用了 `UserConfigurable` 方法，这个方法的参数是一个浮点数，表示在模型提供者被停止工作之前，警告阈值。


```py
OPEN_AI_MODELS = {
    **OPEN_AI_CHAT_MODELS,
    **OPEN_AI_EMBEDDING_MODELS,
}


class OpenAIConfiguration(SystemConfiguration):
    retries_per_request: int = UserConfigurable()


class OpenAIModelProviderBudget(ModelProviderBudget):
    graceful_shutdown_threshold: float = UserConfigurable()
    warning_threshold: float = UserConfigurable()


```

This is a class called `OpenAIProvider()` which is part of the OpenAI language model API.

This class has several methods for creating an instance of the `OpenAIModel` class which can be used to interact with the OpenAIModel API.

The main method for creating an instance of the `OpenAIModel` class is `__init__()` which takes several arguments including the name of the model, an optional `functions` list, and any other keyword arguments provided by the user. This method initializes the model with the given information and returns an instance of the `OpenAIModel` class.

The `create_model_response()` method is used to create a response object for an embedding request. This method takes a keyword argument `response_args` which includes the user's preferences for the embedding, such as the size of the embeddings, the format of the embeddings, and any other settings. This method returns an instance of the `EmbeddingModelResponse` class which can be used to interact with the OpenAIModel API.

The `_budget.update_usage_and_cost()` method is used to update the budget for the usage of the OpenAIModel API. This method takes no arguments and returns no value.

The `_get_completion_kwargs()` method is used to retrieve the keywords for completing an embedding request using the OpenAIModel API. This method takes the name of the model and an optional `functions` list as keyword arguments.

The `_get_embedding_kwargs()` method is used to retrieve the keywords for an embedding request using the OpenAIModel API. This method takes the name of the model as a keyword argument and any other keyword arguments provided by the user.

The `__repr__()` method is a special method for representing the `OpenAIModel` class and returns the string representation of the class.


```py
class OpenAISettings(ModelProviderSettings):
    configuration: OpenAIConfiguration
    credentials: ModelProviderCredentials
    budget: OpenAIModelProviderBudget


class OpenAIProvider(
    Configurable[OpenAISettings], ChatModelProvider, EmbeddingModelProvider
):
    default_settings = OpenAISettings(
        name="openai_provider",
        description="Provides access to OpenAI's API.",
        configuration=OpenAIConfiguration(
            retries_per_request=10,
        ),
        credentials=ModelProviderCredentials(),
        budget=OpenAIModelProviderBudget(
            total_budget=math.inf,
            total_cost=0.0,
            remaining_budget=math.inf,
            usage=ModelProviderUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
            graceful_shutdown_threshold=0.005,
            warning_threshold=0.01,
        ),
    )

    def __init__(
        self,
        settings: OpenAISettings,
        logger: logging.Logger,
    ):
        self._configuration = settings.configuration
        self._credentials = settings.credentials
        self._budget = settings.budget

        self._logger = logger

        retry_handler = _OpenAIRetryHandler(
            logger=self._logger,
            num_retries=self._configuration.retries_per_request,
        )

        self._create_chat_completion = retry_handler(_create_chat_completion)
        self._create_embedding = retry_handler(_create_embedding)

    def get_token_limit(self, model_name: str) -> int:
        """Get the token limit for a given model."""
        return OPEN_AI_MODELS[model_name].max_tokens

    def get_remaining_budget(self) -> float:
        """Get the remaining budget."""
        return self._budget.remaining_budget

    @classmethod
    def get_tokenizer(cls, model_name: OpenAIModelName) -> ModelTokenizer:
        return tiktoken.encoding_for_model(model_name)

    @classmethod
    def count_tokens(cls, text: str, model_name: OpenAIModelName) -> int:
        encoding = cls.get_tokenizer(model_name)
        return len(encoding.encode(text))

    @classmethod
    def count_message_tokens(
        cls,
        messages: ChatMessage | list[ChatMessage],
        model_name: OpenAIModelName,
    ) -> int:
        if isinstance(messages, ChatMessage):
            messages = [messages]

        if model_name.startswith("gpt-3.5-turbo"):
            tokens_per_message = (
                4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            )
            tokens_per_name = -1  # if there's a name, the role is omitted
            encoding_model = "gpt-3.5-turbo"
        elif model_name.startswith("gpt-4"):
            tokens_per_message = 3
            tokens_per_name = 1
            encoding_model = "gpt-4"
        else:
            raise NotImplementedError(
                f"count_message_tokens() is not implemented for model {model_name}.\n"
                " See https://github.com/openai/openai-python/blob/main/chatml.md for"
                " information on how messages are converted to tokens."
            )
        try:
            encoding = tiktoken.encoding_for_model(encoding_model)
        except KeyError:
            cls._logger.warn(
                f"Model {model_name} not found. Defaulting to cl100k_base encoding."
            )
            encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.dict().items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    async def create_chat_completion(
        self,
        model_prompt: list[ChatMessage],
        model_name: OpenAIModelName,
        completion_parser: Callable[[AssistantChatMessageDict], _T] = lambda _: None,
        functions: Optional[list[CompletionModelFunction]] = None,
        **kwargs,
    ) -> ChatModelResponse[_T]:
        """Create a completion using the OpenAI API."""

        completion_kwargs = self._get_completion_kwargs(model_name, functions, **kwargs)
        functions_compat_mode = functions and "functions" not in completion_kwargs
        if "messages" in completion_kwargs:
            model_prompt += completion_kwargs["messages"]
            del completion_kwargs["messages"]

        response = await self._create_chat_completion(
            messages=model_prompt,
            **completion_kwargs,
        )
        response_args = {
            "model_info": OPEN_AI_CHAT_MODELS[model_name],
            "prompt_tokens_used": response.usage.prompt_tokens,
            "completion_tokens_used": response.usage.completion_tokens,
        }

        response_message = response.choices[0].message.to_dict_recursive()
        if functions_compat_mode:
            response_message["function_call"] = _functions_compat_extract_call(
                response_message["content"]
            )
        response = ChatModelResponse(
            response=response_message,
            parsed_result=completion_parser(response_message),
            **response_args,
        )
        self._budget.update_usage_and_cost(response)
        return response

    async def create_embedding(
        self,
        text: str,
        model_name: OpenAIModelName,
        embedding_parser: Callable[[Embedding], Embedding],
        **kwargs,
    ) -> EmbeddingModelResponse:
        """Create an embedding using the OpenAI API."""
        embedding_kwargs = self._get_embedding_kwargs(model_name, **kwargs)
        response = await self._create_embedding(text=text, **embedding_kwargs)

        response_args = {
            "model_info": OPEN_AI_EMBEDDING_MODELS[model_name],
            "prompt_tokens_used": response.usage.prompt_tokens,
            "completion_tokens_used": response.usage.completion_tokens,
        }
        response = EmbeddingModelResponse(
            **response_args,
            embedding=embedding_parser(response.embeddings[0]),
        )
        self._budget.update_usage_and_cost(response)
        return response

    def _get_completion_kwargs(
        self,
        model_name: OpenAIModelName,
        functions: Optional[list[CompletionModelFunction]] = None,
        **kwargs,
    ) -> dict:
        """Get kwargs for completion API call.

        Args:
            model: The model to use.
            kwargs: Keyword arguments to override the default values.

        Returns:
            The kwargs for the chat API call.

        """
        completion_kwargs = {
            "model": model_name,
            **kwargs,
            **self._credentials.unmasked(),
        }

        if functions:
            if OPEN_AI_CHAT_MODELS[model_name].has_function_call_api:
                completion_kwargs["functions"] = [f.schema for f in functions]
                if len(functions) == 1:
                    # force the model to call the only specified function
                    completion_kwargs["function_call"] = {"name": functions[0].name}
            else:
                # Provide compatibility with older models
                _functions_compat_fix_kwargs(functions, completion_kwargs)

        return completion_kwargs

    def _get_embedding_kwargs(
        self,
        model_name: OpenAIModelName,
        **kwargs,
    ) -> dict:
        """Get kwargs for embedding API call.

        Args:
            model: The model to use.
            kwargs: Keyword arguments to override the default values.

        Returns:
            The kwargs for the embedding API call.

        """
        embedding_kwargs = {
            "model": model_name,
            **kwargs,
            **self._credentials.unmasked(),
        }

        return embedding_kwargs

    def __repr__(self):
        return "OpenAIProvider()"


```

这段代码定义了一个名为 `_create_embedding` 的函数，它接受一个字符串参数 `text` 和一个或多个参数 `model` 和一个 keyword参数 `**kwargs`。这个函数的作用是使用指定的语言模型将给定的 `text` 转换为相应的 `model` 语言模型的编码。

具体来说，这个函数首先通过调用 `openai.Embedding.acreate` 函数来获取指定 `model` 模型的编码器。然后，它将输入的 `text` 参数输入编码器中，并将获取的编码器返回。最后，它返回编码器的最终结果，这个结果就是将 `text` 嵌入到指定的 `model` 中的编码。

由于这个函数使用了 `openai.Embedding.acreate` 函数，它需要一个可用的 `model` 参数。如果你没有一个特定的模型，你就可以将 `model` 参数留空，这样函数将返回一个空字符串。如果你提供了一个模型，但是该模型不在 `openai.Embedding.acreate` 函数的支持范围内，那么函数也会失败。


```py
async def _create_embedding(text: str, *_, **kwargs) -> openai.Embedding:
    """Embed text using the OpenAI API.

    Args:
        text str: The text to embed.
        model str: The name of the model to use.

    Returns:
        str: The embedding.
    """
    return await openai.Embedding.acreate(
        input=[text],
        **kwargs,
    )


```

这段代码定义了一个名为 `_create_chat_completion` 的异步函数，它接受一个名为 `messages` 的列表参数。该函数返回一个名为 `openai.Completion` 的类实例。

函数的作用是通过调用 OpenAI API 中的 `acreate` 方法，创建一个聊天完成。`**kwargs` 参数是一个可选的参数，用于传递任何其他参数。

具体来说，函数会将传入的 `messages` 列表中的每个消息转换为字典，并使用 `openai.ChatCompletion.acreate` 方法将消息作为参数传递给 API。这个 API 会根据消息内容生成一个自然语言处理任务，并在完成任务后返回一个完成结果，这个结果就是 `openai.Completion` 类实例。


```py
async def _create_chat_completion(
    messages: list[ChatMessage], *_, **kwargs
) -> openai.Completion:
    """Create a chat completion using the OpenAI API.

    Args:
        messages: The prompt to use.

    Returns:
        The completion.
    """
    raw_messages = [
        message.dict(include={"role", "content", "function_call", "name"})
        for message in messages
    ]
    return await openai.ChatCompletion.acreate(
        messages=raw_messages,
        **kwargs,
    )


```

This is a class that wraps an asynchronous function (a `Callable[_P, _T]`) and implements the basic retry logic for an API key. The class takes an logger, the number of retries, the backoff base, and a flag to raise a warning if the API client fails.

The class has a `__init__` method that initializes the logger, the number of retries, the backoff base, and the flag to raise a warning. The `__call__` method is the entry point for the retry logic, which is wrapped in a function that takes the function to be called, the arguments and keyword arguments for the function, and returns the result of the function call.

The retry logic is implemented in the `_backoff` method, which calculates the backoff factor based on the number of attempts and the backoff base, and logs a message if the API client fails. The `_log_rate_limit_error` method is used to log a message if the API client fails, and the `_api_key_error_msg` method is used as a fallback if the `_log_rate_limit_error` method fails.

The `_warn_user` flag is used to raise a warning if the API client fails and the number of attempts is equal to the number of retries.


```py
class _OpenAIRetryHandler:
    """Retry Handler for OpenAI API call.

    Args:
        num_retries int: Number of retries. Defaults to 10.
        backoff_base float: Base for exponential backoff. Defaults to 2.
        warn_user bool: Whether to warn the user. Defaults to True.
    """

    _retry_limit_msg = "Error: Reached rate limit, passing..."
    _api_key_error_msg = (
        "Please double check that you have setup a PAID OpenAI API Account. You can "
        "read more here: https://docs.agpt.co/setup/#getting-an-api-key"
    )
    _backoff_msg = "Error: API Bad gateway. Waiting {backoff} seconds..."

    def __init__(
        self,
        logger: logging.Logger,
        num_retries: int = 10,
        backoff_base: float = 2.0,
        warn_user: bool = True,
    ):
        self._logger = logger
        self._num_retries = num_retries
        self._backoff_base = backoff_base
        self._warn_user = warn_user

    def _log_rate_limit_error(self) -> None:
        self._logger.debug(self._retry_limit_msg)
        if self._warn_user:
            self._logger.warning(self._api_key_error_msg)
            self._warn_user = False

    def _backoff(self, attempt: int) -> None:
        backoff = self._backoff_base ** (attempt + 2)
        self._logger.debug(self._backoff_msg.format(backoff=backoff))
        time.sleep(backoff)

    def __call__(self, func: Callable[_P, _T]) -> Callable[_P, _T]:
        @functools.wraps(func)
        async def _wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            num_attempts = self._num_retries + 1  # +1 for the first attempt
            for attempt in range(1, num_attempts + 1):
                try:
                    return await func(*args, **kwargs)

                except RateLimitError:
                    if attempt == num_attempts:
                        raise
                    self._log_rate_limit_error()

                except APIError as e:
                    if (e.http_status != 502) or (attempt == num_attempts):
                        raise

                self._backoff(attempt)

        return _wrapped


```

这段代码定义了一个名为 `format_function_specs_as_typescript_ns` 的函数，它的输入参数是一个包含函数模型定义的列表 `functions`。函数的作用是返回一个函数签名块，格式符合 OpenAI 内部使用的格式。

函数的具体实现是通过遍历输入列表中的函数模型定义，并将它们格式化为 OpenAI 内部使用的字符串格式。这个字符串格式包括函数名称、参数列表和函数返回值类型等信息。

函数的实现返回了一个字符串，该字符串是一个函数签名块，使用了 `namespace` 语法。这种语法允许在函数签名中使用自定义的命名空间。在示例中，函数签名是类似于 OpenAI 内部使用的格式，其中包括函数名称、参数列表和返回值类型等信息。


```py
def format_function_specs_as_typescript_ns(
    functions: list[CompletionModelFunction],
) -> str:
    """Returns a function signature block in the format used by OpenAI internally:
    https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/18

    For use with `count_tokens` to determine token usage of provided functions.

    Example:
    ```ts
    namespace functions {

    // Get the current weather in a given location
    type get_current_weather = (_: {
    // The city and state, e.g. San Francisco, CA
    location: string,
    unit?: "celsius" | "fahrenheit",
    }) => any;

    } // namespace functions
    ```py
    """

    return (
        "namespace functions {\n\n"
        + "\n\n".join(format_openai_function_for_prompt(f) for f in functions)
        + "\n\n} // namespace functions"
    )


```



这段代码是一个 Python 函数 `format_openai_function_for_prompt`，它接受一个 `CompletionModelFunction` 类型的参数，并返回一个类似于 OpenAI 函数格式的方式来描述这个函数的文本。

具体来说，这个函数的实现过程可以分为以下几步：

1. 定义了一个名为 `param_signature` 的函数，它接收一个 `str` 类型的参数 `name` 和一个 `JSONSchema` 类型的参数 `spec`，并返回一个字符串，其中包含了函数参数的名称、类型、描述等信息。

2. 定义了 `format_openai_function_for_prompt` 函数，它接收一个 `CompletionModelFunction` 类型的参数 `func`，并返回一个字符串，其中包含了函数定义的头部信息、参数列表和返回值类型等信息。

3. 在 `format_openai_function_for_prompt` 函数中，遍历 `func.parameters` 列表，为每个参数定义一个 `param_signature` 函数，并将其结果与之前的字符串结果合并，最终得到了一个完整的函数定义。

4. 将 `func.description`、`func.name`、`func.parameters` 和 `func.typescript_type` 等信息加入了字符串中，最终生成了一个类似于 OpenAI 函数格式的方式来描述这个函数的文本。


```py
def format_openai_function_for_prompt(func: CompletionModelFunction) -> str:
    """Returns the function formatted similarly to the way OpenAI does it internally:
    https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/18

    Example:
    ```ts
    // Get the current weather in a given location
    type get_current_weather = (_: {
    // The city and state, e.g. San Francisco, CA
    location: string,
    unit?: "celsius" | "fahrenheit",
    }) => any;
    ```py
    """

    def param_signature(name: str, spec: JSONSchema) -> str:
        return (
            f"// {spec.description}\n" if spec.description else ""
        ) + f"{name}{'' if spec.required else '?'}: {spec.typescript_type},"

    return "\n".join(
        [
            f"// {func.description}",
            f"type {func.name} = (_ :{{",
            *[param_signature(name, p) for name, p in func.parameters.items()],
            "}) => any;",
        ]
    )


```



This function appears to be a script for processing function calls in an OpenAI chatbot. The function takes a list of function definitions and a completion keyword (also a dictionary). It returns the number of tokens taken up by the set of function definitions.

The function first formats the function definitions into a TS default object. It then defines a callable that takes a string with a `#` prefix and a completion keyword, and returns the number of tokens taken up by the function call.

The function uses a helper function `count_tokens` to keep track of the number of tokens taken up by each function. This function takes a string and an integer, and returns the number of tokens.

The main function then iterates over the function definitions and calls the `count_tokens` function for each definition, updating the completion keyword with the function name and arguments.

The function also includes some code for handling function calls in a chatbot. When a user specifies a function call in a chatbot, the function will respond with an instruction to specify the function block using a valid JSON object. The function will then be executed with the specified arguments.


```py
def count_openai_functions_tokens(
    functions: list[CompletionModelFunction], count_tokens: Callable[[str], int]
) -> int:
    """Returns the number of tokens taken up by a set of function definitions

    Reference: https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/18
    """
    return count_tokens(
        f"# Tools\n\n## functions\n\n{format_function_specs_as_typescript_ns(functions)}"
    )


def _functions_compat_fix_kwargs(
    functions: list[CompletionModelFunction],
    completion_kwargs: dict,
):
    function_definitions = format_function_specs_as_typescript_ns(functions)
    function_call_schema = JSONSchema(
        type=JSONSchema.Type.OBJECT,
        properties={
            "name": JSONSchema(
                description="The name of the function to call",
                enum=[f.name for f in functions],
                required=True,
            ),
            "arguments": JSONSchema(
                description="The arguments for the function call",
                type=JSONSchema.Type.OBJECT,
                required=True,
            ),
        },
    )
    completion_kwargs["messages"] = [
        ChatMessage.system(
            "# function_call instructions\n\n"
            "Specify a '```function_call' block in your response,"
            " enclosing a function call in the form of a valid JSON object"
            " that adheres to the following schema:\n\n"
            f"{function_call_schema.to_dict()}\n\n"
            "Put the function_call block at the end of your response"
            " and include its fences if it is not the only content.\n\n"
            "## functions\n\n"
            "For the function call itself, use one of the following"
            f" functions:\n\n{function_definitions}"
        ),
    ]


```py

这段代码的作用是提取并返回Assistant中的一个函数调用，根据给定的响应字符串。函数提取的过程如下：

1. 如果给定的响应字符串包含一个双引号(`)，则尝试解析该字符串中包含的JSON对象。
2. 如果给定的响应字符串中包含一个函数定义，则使用正则表达式将函数定义匹配出来，并将其转换为JSON对象。
3. 对于匹配到的函数定义，提取出其arguments参数，并将其类型更改为字符串类型。
4. 返回提取出的函数调用和arguments参数。

该函数可能会被用于以下场景：当需要从给定 response 中提取出一个 Assistant 函数的调用时，该函数将调用函数的唯一标识符(例如函数 ID)或函数定义，并返回该函数的上下文信息。


```
def _functions_compat_extract_call(response: str) -> AssistantFunctionCallDict:
    import json
    import re

    logging.debug(f"Trying to extract function call from response:\n{response}")

    if response[0] == "{":
        function_call = json.loads(response)
    else:
        block = re.search(r"```py(?:function_call)?\n(.*)\n```\s*$", response, re.DOTALL)
        if not block:
            raise ValueError("Could not find function call block in response")
        function_call = json.loads(block.group(1))

    function_call["arguments"] = str(function_call["arguments"])  # HACK
    return function_call

```