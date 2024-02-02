# AutoGPT源码解析 32

# `benchmark/agbenchmark/reports/processing/report_types_v2.py`

这段代码定义了一个名为 "BaseModelBenchmark" 的类，该类继承自 "BaseModel" 类。通过 `from typing import Dict, List` 引入了 "typing" 和 "pydantic" 两个库。

```py
class BaseModelBenchmark(BaseModel):
   class Config:
       extra = "forbid"
```

在这里定义了一个名为 "Config" 的类，其中 `extra = "forbid"` 表示禁用任何额外配置，以使该类更简洁。

```py
class TaskInfo(BaseModelBenchmark):
   data_path: str
   is_regression: bool | None
   answer: str
   description: str
   category: List[str]
   task: str
```

在这里定义了一个名为 "TaskInfo" 的类，该类继承自 "BaseModelBenchmark" 类。

```py
from datetime import datetime
from pydantic import BaseModel, constr
```

在这里引入了 "datetime" 和 "pydantic" 两个库，分别用于日期时间处理和 Pydantic 类型定义。

```py
from typing import Dict, List
```

在这里再次引入了 "typing" 库，用于提供类型声明。

```py
class Benchmark(BaseModel):
   datetime_format: str = constr(datetime.date.format, "%d-%b-%Y %H:%M:%S %20:00:00")
```

在这里定义了一个名为 "Benchmark" 的类，该类继承自 "BaseModel" 类。通过 `datetime.date.format` 函数将日期时间格式化为 Pydantic 需要的格式，然后将其设置为 `datetime_format` 变量。

```py
   class Config:
       extra = "forbid"
```

在这里定义了一个名为 "Config" 的类，其中 `extra = "forbid"` 表示禁用任何额外配置，以使该类更简洁。

```py
   class TaskInfo(BaseModel):
       data_path: str
       is_regression: bool | None
       answer: str
       description: str
       category: List[str]
       task: str
```

在这里定义了一个名为 "TaskInfo" 的类，该类继承自 "BaseModel" 类。

```py
   class Config:
       extra = "forbid"
```

在这里定义了一个名为 "Config" 的类，其中 `extra = "forbid"` 表示禁用任何额外配置，以使该类更简洁。

```py
   class Benchmark(BaseModel):
       class Config:
           extra = "forbid"
       datetime_format: str = constr(datetime.date.format, "%d-%b-%Y %H:%M:%S %20:00:00")
```

在这里定义了一个名为 "Benchmark" 的类，该类继承自 "BaseModel" 类。通过 `datetime.date.format` 函数将日期时间格式化为 Pydantic 需要的格式，然后将其设置为 `datetime_format` 变量。最后将 `extra = "forbid"` 设置为 `Config` 的额外配置，以禁止任何额外的配置。

```py
   benchmarks: List[Benchmark] = []
```

在这里创建了一个名为 "benchmarks" 的列表，用于存储基准测试。

```py
   for task in TaskInfo.CATEGORIES:
       benchmarks.append(Benchmark(
           data_path=task_info.data_path,
           is_regression=task_info.is_regression,
           answer=task_info.answer,
           description=task_info.description,
           category=task_info.category,
           task=task
       ))
```

在这里遍历了 "TaskInfo" 类中定义的类别的所有元素，并使用 `Benchmark` 类创建了一个基准测试实例，然后将其添加到 "benchmarks" 列表中。

```py
   print(f"Total benchmarks: {len(benchmarks)}")
```

在这里输出基准测试的数量。


```py
from typing import Dict, List

datetime_format = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$"
from pydantic import BaseModel, constr


class BaseModelBenchmark(BaseModel):
    class Config:
        extra = "forbid"


class TaskInfo(BaseModelBenchmark):
    data_path: str
    is_regression: bool | None
    answer: str
    description: str
    category: List[str]
    task: str


```

这两段代码定义了两个类，RepositoryInfo和Metrics，它们都是属于BaseModelBenchmark的子类。

RepositoryInfo类包含以下字段：
- repo_url：仓库的URL，可能是None，表示没有提供该字段。
- team_name：团队的名称，可能是None，表示没有提供该字段。
- benchmark_git_commit_sha：用于标识唯一基准测试的Git提交 SHA，可能是None，表示没有提供该字段。
- agent_git_commit_sha：用于标识唯一虚拟机的Git提交 SHA，可能是None，表示没有提供该字段。

Metrics类包含以下字段：
- difficulty：测试的难度，可能是没有提供该字段。
- success：测试是否成功，如果是，值为True，如果不是，值为False。
- success_percentage：测试成功率，可能是没有提供该字段。
- run_time：运行时间，可能是没有提供该字段。
- fail_reason：失败的原因，可能是没有提供该字段。
- attempted：尝试运行测试，可能是没有提供该字段。
- cost：运行测试的成本，可能是没有提供该字段。

这两段代码定义了两个类，RepositoryInfo和Metrics，它们都是属于BaseModelBenchmark的子类。这些类用于存储基准测试的信息，包括测试的名称、描述、结果、运行时间、失败的原因等等。这些信息可以用于基准测试的结果统计和分析，例如计算成功率、失败率、运行成本等。


```py
class RepositoryInfo(BaseModelBenchmark):
    repo_url: str | None
    team_name: str | None
    benchmark_git_commit_sha: str | None
    agent_git_commit_sha: str | None


class Metrics(BaseModelBenchmark):
    difficulty: str | None
    success: bool
    success_percentage: float | None
    run_time: str | None
    fail_reason: str | None
    attempted: bool
    cost: float | None


```

这段代码定义了一个名为 `RunDetails` 的类，它是 `BaseModelBenchmark` 的子类。这个类包含了一些与基准测试相关的参数，包括 `test_name`、`run_id`、`command` 和 `completion_time`，这些参数在基准测试中用于记录测试的名称、运行的唯一ID、命令和完成时间。

接着定义了一个名为 `BenchmarkRun` 的类，也是 `BaseModelBenchmark` 的子类。这个类包含了一个 `RunDetails` 类型的变量 `run_details`，这个变量的作用类似于 `TestDetails` 类的 `test_name`、`run_id` 和 `command` 参数，但使用了不同的数据类型。`run_details` 类包含 `benchmark_start_time` 参数，这个参数是一个 `datetime_format` 类型的常量，用于记录基准测试开始的时间。然后定义了一个 `TaskInfo` 类型的变量 `task_info`，这个变量包含了一些与测试相关的信息，如测试的计划、预计的完成时间等。`TaskInfo` 类包含 `metrics` 参数，这个参数是一个 `Metrics` 类型的变量，用于记录测试的进度和结果。最后定义了一个 `ReputationInfo` 类型的变量 `repository_info`，这个变量包含了一些与仓库相关的信息，如仓库的名称、描述、URL 等。最后，定义了一个 `Dictionary` 类型的变量 `config`，这个变量包含了一些与运行时配置相关的信息，如环境变量、软件包等。


```py
class RunDetails(BaseModelBenchmark):
    test_name: str
    run_id: str | None
    command: str
    completion_time: str | None
    benchmark_start_time: constr(regex=datetime_format)


class BenchmarkRun(BaseModelBenchmark):
    repository_info: RepositoryInfo
    run_details: RunDetails
    task_info: TaskInfo
    metrics: Metrics
    reached_cutoff: bool | None
    config: Dict[str, str | dict[str, str]]

```

# `benchmark/agbenchmark/utils/challenge.py`

这段代码是一个Python脚本，它的作用是：

1. 导入一些必要的模块和函数，包括glob、math、os、subprocess、sys、ABC、pathlib、openai和pytest。
2. 从抽象基类ABC中继承一个名为run_api_agent的函数，这个函数可以接受两个参数，第一个参数是一个任何类型的代理类，第二个参数是接口名称。
3. 在run_api_agent函数中，使用openai创建一个智能代理，并使用pytest. fixture装饰器来模拟运行该代理的过程。
4. 使用agbenchmark.__main__.py中的OPTIONAL_CATEGORIES参数来控制是否要输出源代码。
5. 在OPTIONAL_CATEGORIES参数的值可以为'force'或者'no'，如果是'force'，则会输出该脚本作为openai的实验。
6. 在tests目录下创建一个名为temp_folder_abs_path的文件夹，用于保存实验数据。

具体来说，这段代码的作用是：运行agbenchmark的子程序，通过使用openai模拟一个智能代理，接受 ground 作为接口名称，在OPTIONAL_CATEGORIES参数的值为'force'时，输出该脚本作为openai的实验。


```py
import glob
import math
import os
import subprocess
import sys
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List

import openai
import pytest

from agbenchmark.__main__ import OPTIONAL_CATEGORIES, TEMP_FOLDER_ABS_PATH
from agbenchmark.agent_api_interface import run_api_agent
from agbenchmark.utils.data_types import ChallengeData, Ground
```

It looks like this is a simple pytest plugin that retrieves scores from a competition. The `OPTIONAL_CATEGORIES` list is defined in the ` competition.py` file, which is not included in the provided code snippet.

The plugin retrieves the scores of an agent from the competition by parsing


```py
from agbenchmark.utils.prompts import (
    END_PROMPT,
    FEW_SHOT_EXAMPLES,
    PROMPT_MAP,
    SCORING_MAP,
)
from agbenchmark.utils.utils import agent_eligibible_for_optional_categories


class Challenge(ABC):
    """The parent class to all specific challenges classes.
    Defines helper methods for running a challenge"""

    _data_cache: Dict[str, ChallengeData] = {}
    CHALLENGE_LOCATION: str = ""
    scores: dict[str, Any] = {}  # this is for suites

    @property
    def data(self) -> ChallengeData:
        if self.CHALLENGE_LOCATION not in self._data_cache:
            self._data_cache[self.CHALLENGE_LOCATION] = ChallengeData.deserialize(
                self.CHALLENGE_LOCATION
            )
        return self._data_cache[self.CHALLENGE_LOCATION]

    @property
    def task(self) -> str:
        return self.data.task

    @property
    def dependencies(self) -> list:
        return self.data.dependencies

    async def setup_challenge(self, config: Dict[str, Any], cutoff: int) -> None:
        from agbenchmark.agent_interface import copy_artifacts_into_temp_folder

        if not self.task:
            return

        print(
            f"\033[1;35m============Starting {self.data.name} challenge============\033[0m"
        )
        print(f"\033[1;30mTask: {self.task}\033[0m")

        await run_api_agent(self.data, config, self.ARTIFACTS_LOCATION, cutoff)

        # hidden files are added after the agent runs. Hidden files can be python test files.
        # We copy them in the temporary folder to make it easy to import the code produced by the agent
        artifact_paths = [
            self.ARTIFACTS_LOCATION,
            str(Path(self.CHALLENGE_LOCATION).parent),
        ]
        for path in artifact_paths:
            copy_artifacts_into_temp_folder(TEMP_FOLDER_ABS_PATH, "custom_python", path)

    def test_method(self, config: Dict[str, Any]) -> None:
        raise NotImplementedError

    def get_artifacts_out(
        self, workspace: str | dict[str, str], ground: Ground
    ) -> List[str]:
        if isinstance(workspace, dict):
            workspace = workspace["output"]

        script_dir = workspace
        files_contents = []

        for file_pattern in ground.files:
            # Check if it is a file extension
            if file_pattern.startswith("."):
                # Find all files with the given extension in the workspace
                matching_files = glob.glob(os.path.join(script_dir, "*" + file_pattern))
            else:
                # Otherwise, it is a specific file
                matching_files = [os.path.join(script_dir, file_pattern)]

            for file_path in matching_files:
                if ground.eval.type == "python":
                    result = subprocess.run(
                        [sys.executable, file_path],
                        cwd=os.path.abspath(workspace),
                        capture_output=True,
                        text=True,
                    )
                    if "error" in result.stderr or result.returncode != 0:
                        print(result.stderr)
                        assert False, result.stderr
                    files_contents.append(f"Output: {result.stdout}\n")
                else:
                    with open(file_path, "r") as f:
                        files_contents.append(f.read())
        else:
            if ground.eval.type == "pytest":
                result = subprocess.run(
                    [sys.executable, "-m", "pytest"],
                    cwd=TEMP_FOLDER_ABS_PATH,
                    capture_output=True,
                    text=True,
                )
                if "error" in result.stderr or result.returncode != 0:
                    print(result.stderr)
                    assert False, result.stderr
                files_contents.append(f"Output: {result.stdout}\n")

        return files_contents

    def scoring(self, config: Dict[str, Any], content: str, ground: Ground) -> float:
        print("\033[1;34mScoring content:\033[0m", content)
        if ground.should_contain:
            for should_contain_word in ground.should_contain:
                if not getattr(ground, 'case_sensitive', True):
                    should_contain_word = should_contain_word.lower()
                    content = content.lower()
                print_content = (
                    f"\033[1;34mWord that should exist\033[0m - {should_contain_word}:"
                )
                if should_contain_word not in content:
                    print(print_content, "False")
                    return 0.0
                else:
                    print(print_content, "True")

        if ground.should_not_contain:
            for should_not_contain_word in ground.should_not_contain:
                if not getattr(ground, 'case_sensitive', True):
                    should_not_contain_word = should_not_contain_word.lower()
                    content = content.lower()
                print_content = f"\033[1;34mWord that should not exist\033[0m - {should_not_contain_word}:"
                if should_not_contain_word in content:
                    print(print_content, "False")
                    return 0.0
                else:
                    print(print_content, "True")

        return 1.0

    def llm_eval(self, config: Dict[str, Any], content: str, ground: Ground) -> float:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if os.getenv("IS_MOCK"):
            return 1.0

        # the validation for this is done in the Eval BaseModel
        scoring = SCORING_MAP[ground.eval.scoring]  # type: ignore
        prompt = PROMPT_MAP[ground.eval.template].format(task=self.data.task, scoring=scoring, answer=ground.answer, response=content)  # type: ignore

        if ground.eval.examples:
            prompt += FEW_SHOT_EXAMPLES.format(examples=ground.eval.examples)

        prompt += END_PROMPT

        answer = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
            ],
        )

        return float(answer["choices"][0]["message"]["content"])  # type: ignore

    def get_scores(self, config: Dict[str, Any]) -> dict[str, Any]:
        scores = []
        scores_dict: Any = {}
        percentage = None
        answers = {}
        try:
            if self.data.task == "" and os.getenv("IS_MOCK"):
                scores = [1.0]
                answers = {"mock": "This is a mock answer"}
            elif isinstance(self.data.ground, Ground):
                files_contents = self.get_artifacts_out(
                    TEMP_FOLDER_ABS_PATH, self.data.ground
                )
                answers = {"answer": files_contents}
                for file_content in files_contents:
                    score = self.scoring(config, file_content, self.data.ground)
                    print("\033[1;32mYour score is:\033[0m", score)
                    scores.append(score)

                if self.data.ground.eval.type == "llm":
                    llm_eval = self.llm_eval(
                        config, "\n".join(files_contents), self.data.ground
                    )
                    if self.data.ground.eval.scoring == "percentage":
                        scores.append(math.ceil(llm_eval / 100))
                    elif self.data.ground.eval.scoring == "scale":
                        scores.append(math.ceil(llm_eval / 10))
                    print("\033[1;32mYour score is:\033[0m", llm_eval)

                    scores.append(llm_eval)
        except Exception as e:
            print("Error getting scores", e)

        scores_data = {
            "values": scores,
            "scores_obj": scores_dict,
            "percentage": percentage,
            "answers": answers,
        }

        self.scores[self.__class__.__name__] = scores_data

        return scores_data

    def get_dummy_scores(self, test_name: str, scores: dict[str, Any]) -> int | None:
        return 1  # remove this once this works
        if 1 in scores.get("scores_obj", {}).get(test_name, []):
            return 1

        return None

    def skip_optional_categories(self, config: Dict[str, Any]) -> None:
        challenge_category = self.data.category
        categories = [
            category
            for category in OPTIONAL_CATEGORIES
            if category in challenge_category
        ]
        if not agent_eligibible_for_optional_categories(
            categories, config.get("category", [])
        ):
            pytest.skip("Agent is not eligible for this category")

```

# `benchmark/agbenchmark/utils/data_types.py`

这段代码的作用是定义了一个名为 "DifficultyLevel" 的枚举类型，它用于表示计算机编程中的难度级别。这个枚举类型包含了 7 个不同的难度级别，分别对应于 "interface"、"basic"、"novice"、"intermediate"、"advanced" 和 "expert" 这些枚举类型。

此外，代码还引入了 datetime、json 和 sys 模块，以及从 datetime 和 enum 模块中定义的一些函数和类型。最后，代码导入了 pydantic 库，以便使用其 BaseModel 和 constr 函数来定义数据模型和验证函数。


```py
import datetime
import json
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, constr, validator


class DifficultyLevel(Enum):
    interface = "interface"
    basic = "basic"
    novice = "novice"
    intermediate = "intermediate"
    advanced = "advanced"
    expert = "expert"
    human = "human"


```

The `DifficultyLevel` class defines several levels, along with their corresponding difficulty numbers and descriptions.

The `STRING_DIFFICULTY_MAP` dictionary maps the difficulty numbers from the `DifficultyLevel` class to their corresponding descriptions.

You can use the `calculate_info_test_path` method to calculate the path to a directory where the test report will be saved, taking into account the `DifficultyLevel` and the `START_TIME` parameter.


```py
# map from enum to difficulty level (numeric)
DIFFICULTY_MAP = {
    DifficultyLevel.interface: 1,
    DifficultyLevel.basic: 2,
    DifficultyLevel.novice: 3,
    DifficultyLevel.intermediate: 4,
    DifficultyLevel.advanced: 5,
    DifficultyLevel.expert: 6,
    DifficultyLevel.human: 7,
}

STRING_DIFFICULTY_MAP = {e.value: DIFFICULTY_MAP[e] for e in DifficultyLevel}


def calculate_info_test_path(base_path: Path, benchmark_start_time: datetime) -> Path:
    """
    Calculates the path to the directory where the test report will be saved.
    """
    # Ensure the reports path exists
    base_path.mkdir(parents=True, exist_ok=True)

    # Get current UTC date-time stamp
    date_stamp = benchmark_start_time.strftime("%Y%m%dT%H%M%S")

    # Default run name
    run_name = "full_run"

    # Map command-line arguments to their respective labels
    arg_labels = {
        "--test": None,
        "--category": None,
        "--maintain": "maintain",
        "--improve": "improve",
        "--explore": "explore",
    }

    # Identify the relevant command-line argument
    for arg, label in arg_labels.items():
        if arg in sys.argv:
            test_arg = sys.argv[sys.argv.index(arg) + 1] if label is None else None
            run_name = arg.strip("--")
            if test_arg:
                run_name = f"{run_name}_{test_arg}"
            break

    # Create the full new directory path with ISO standard UTC date-time stamp
    report_path = base_path / f"{date_stamp}_{run_name}"

    # Ensure the new directory is created
    report_path.mkdir(exist_ok=True)
    return report_path


```

这段代码定义了一个名为 `AgentBenchmarkConfig` 的类，代表代理基准测试的配置。该类包含了一些与代理基准测试相关的属性，包括 `agent_benchmark_config_path`、`reports_folder` 和 `host`。

`get_reports_location` 方法用于获取基准测试报告的存储位置，如果 `reports_folder` 属性没有被设置，则默认值为 `agent_benchmark_config_path` 的 parent 目录中，名为 `reports` 的新目录。

`get_reports_path` 方法用于获取基准测试报告的路径，其中 `benchmark_start_time` 参数是基准测试的开始时间。该方法先调用 `get_reports_location` 方法获取报告存储位置，然后使用 `calculate_info_test_path` 方法计算报告路径，最后返回报告位置。

`get_regression_reports_path` 方法与 `get_reports_path` 类似，但该方法返回的是一个更具体的路径，即基准测试报告中 regression 部分的路径。

`get_success_rate_path` 方法与 `get_reports_path` 类似，但该方法返回的是一个更具体的路径，即基准测试报告中 success rate 部分的路径。

`get_agent_home_directory` 方法返回代理的 home directory 的路径。该 home directory 包含代理的所有相关文件和目录，例如日志文件、配置文件和基准测试的资料等。


```py
class AgentBenchmarkConfig(BaseModel):
    """
    This class represents the configuration for the Agent agbenchmark.
    It includes the following attributes:
    - agent_benchmark_config_path: The path to the agent benchmark config that this object was created from.
    - reports_folder: The path to the folder where the benchmark reports will be stored.
    - host: The host where the benchmark is run.
    """

    agent_benchmark_config_path: Path | None = None
    reports_folder: Path | None = None
    host: str | None

    def get_reports_location(self) -> Path:
        # if not self.reports_folder:
        #     self.reports_folder = (
        #         Path(self.agent_benchmark_config_path).parent / "reports"
        #     ).resolve()
        return Path.cwd() / "agbenchmark_config" / "reports"

    def get_reports_path(self, benchmark_start_time: datetime) -> Path:
        return calculate_info_test_path(
            self.get_reports_location(), benchmark_start_time
        )

    def get_regression_reports_path(self) -> Path:
        return self.get_reports_location() / "regression_tests.json"

    def get_success_rate_path(self) -> Path:
        return self.get_reports_location() / "success_rate.json"

    def get_agent_home_directory(self) -> Path:
        return Path(self.agent_benchmark_config_path).resolve().parent


```



这个代码定义了一个名为Info的类，它继承自名为BaseModel的类。Info类包含三个成员变量：difficulty,description和side_effects。其中difficulty成员变量是一个DifficultyLevel类型的变量，description是一个字符串类型的变量，side_effects是一个包含多个字符串类型的成员变量列表。

difficulty_to_enum方法是一个静态方法，它用于将一个字符串类型和一个DifficultyLevel类型之间的类型转换。这个方法接收两个参数：一个是cls，即Info类的实例，另一个是v，即需要转换的字符串类型。

在该方法中，首先检查v是否是DifficultyLevel类型，如果是，则返回v。如果不是DifficultyLevel类型，则需要将字符串转换为DifficultyLevel类型的实例。在这个过程中，会尝试将v转换为DifficultyLevel类型的实例，但是如果转换失败，则会引发出一个ValueError异常。

最后，在difficulty_to_enum方法中，还定义了一个difficulty_to_enum方法，用于将一个DifficultyLevel类型的对象转换为字符串类型。这个方法与difficulty_to_enum方法类似，但是它的参数是一个DifficultyLevel类型的对象，而不是需要转换的字符串类型。这个方法会尝试将DifficultyLevel对象转换为字符串类型，如果没有问题，则返回对象实例，否则会引发出一个KeyError异常。


```py
class Info(BaseModel):
    difficulty: DifficultyLevel
    description: constr(regex=r"^Tests if the agent can.*")
    side_effects: List[str]

    @validator("difficulty", pre=True)
    def difficulty_to_enum(cls: "Info", v: str | DifficultyLevel) -> DifficultyLevel:
        """Convert a string to an instance of DifficultyLevel."""
        if isinstance(v, DifficultyLevel):
            return v

        if isinstance(v, str):
            try:
                return DifficultyLevel(v.lower())
            except ValueError:
                pass

        raise ValueError(f"Cannot convert {v} to DifficultyLevel.")


```

这段代码定义了一个名为Eval的类，其继承自名为BaseModel的类。在这个类的定义中，定义了四个字段类型、 scoring、template和examples，分别表示评估的类型、评分模式、模板和示例。

在该类中，定义了一个名为validate_eval_fields的验证函数，用于检查type字段的值是否为'llm'，如果是，则验证函数将忽略eval_fields这个字段。否则，验证函数将raise ValueError。

还定义了一个名为validate_scoring的验证函数，用于检查scoring字段的值是否合法，如果无效，则验证函数将raise ValueError。

最后还定义了一个名为validate_template的验证函数，用于检查template字段的值是否合法，如果无效，则验证函数将raise ValueError。

总结起来，这段代码定义了一个用于评估的类，其中包含了评估的类型、评分模式、模板和示例。这些函数用于确保评估数据的正确性，如果出现错误，将引发特定的错误消息。


```py
class Eval(BaseModel):
    type: str
    scoring: Optional[str]
    template: Optional[str]
    examples: Optional[str]

    @validator("scoring", "template", always=True)
    def validate_eval_fields(cls, v, values, field):
        if "type" in values and values["type"] == "llm":
            if v is None:
                raise ValueError(f"{field.name} must be provided when type is 'llm'")
        else:
            if v is not None:
                raise ValueError(f"{field.name} should only exist when type is 'llm'")
        return v

    @validator("scoring")
    def validate_scoring(cls, v):
        if v is not None and v not in ["percentage", "scale", "binary"]:
            raise ValueError(
                "scoring must be either 'percentage', 'scale', or 'binary'"
            )
        return v

    @validator("template")
    def validate_template(cls, v):
        if v is not None and v not in ["rubric", "reference", "question", "custom"]:
            raise ValueError(
                "template must be either 'rubric', 'reference', 'question', or 'custom'"
            )
        return v


```

这段代码定义了一个名为Ground的类，其继承自类BaseModel。

在这个类中，定义了三个变量：answer、should_contain和should_not_contain，它们都是 optional 的列表类型变量，即可能不存在这些变量，并且可以有多个不同的值。

接下来，定义了三个变量：files,case_sensitive和eval，它们都是类的属性，eval使用了一个 Eval的函数。

接着，定义了一个名为Category的枚举类型，其值为DATA、GENERALIST和CODING，分别对应着三个不同的类别。

最后，没有定义任何方法，直接返回了Ground类的实例。


```py
class Ground(BaseModel):
    answer: str
    should_contain: Optional[List[str]] = None
    should_not_contain: Optional[List[str]] = None
    files: List[str]
    case_sensitive: Optional[bool] = True
    eval: Eval


class Category(str, Enum):
    DATA = "data"
    GENERALIST = "general"
    CODING = "coding"
    SCRAPE_SYNTHESIZE = "scrape_synthesize"


```

This is a Python class that defines a `ChallengeData` class for solving challenges based on a provided task and its associated data. It has methods for reading challenge data from a file, creating a challenge object from a given task and its associated data, and creating a challenge object from a given test data.

The `ChallengeData` class has several attributes, including `name`, `dependencies`, `category`, `task`, and `cutoff`, which correspond to the task and its associated data. It also has an attribute `info`, which is a dictionary of information about the task, such as the task description and any dependencies. Additionally, it has an attribute `ground`, which is a dictionary of the ground rules for the task, such as if the task can be done in parallel or not.

The class has two methods for reading challenge data from a file: `read_challenge_data()` and `read_shared_category()`, which are used to read the challenge data and the shared category, respectively.

The class also has two methods for creating challenge objects from a given task and its associated data: `create_challenge_object()` and `create_challenge_object_from_test_data()`, which are used to create the challenge object from a task and its associated data or a test data, respectively.

Overall, this class provides a convenient way to create and manage challenge objects for tasks and their associated data.


```py
class ChallengeData(BaseModel):
    name: str
    category: List[Category]
    task: str
    dependencies: List[str]
    cutoff: int
    ground: Ground | Dict[str, Ground]
    info: Info | Dict[str, Info]
    metadata: Optional[Dict[str, Any]] = None

    def serialize(self, path: str) -> None:
        with open(path, "w") as file:
            file.write(self.json())

    def get_data(self) -> dict:
        return self.dict()

    @staticmethod
    def get_json_from_path(json_path: Path | str) -> dict:
        path = Path(json_path).resolve()
        with open(path, "r") as file:
            data = json.load(file)
        return data

    @staticmethod
    def deserialize(path: str) -> "ChallengeData":
        # this script is in root/agbenchmark/utils/define_task_types.py
        script_dir = Path(__file__).resolve().parent.parent.parent
        json_path = script_dir / Path(path)

        with open(json_path, "r") as file:
            data = json.load(file)
        try:
            return ChallengeData(**data)
        except:
            test = "ok"

    def challenge_from_datum(self, file_datum: list[dict[str, Any]]) -> "ChallengeData":
        same_task_data = {
            "name": self.prefix,
            "dependencies": self.dependencies,
            "category": self.shared_category,
            "task": self.task,
            "cutoff": self.cutoff,
        }

        if not self.info:
            same_task_data["info"] = {
                datum["name"]: datum["info"] for datum in file_datum
            }
        else:
            same_task_data["info"] = self.info

        if not self.ground:
            same_task_data["ground"] = {
                datum["name"]: datum["ground"] for datum in file_datum
            }
        else:
            same_task_data["ground"] = self.ground

        return ChallengeData(**same_task_data)

    def challenge_from_test_data(self, data: dict[str, Any]) -> "ChallengeData":
        same_task_data = {
            "name": data["name"],
            "dependencies": data["dependencies"],
            "category": data["category"],
            "info": data["info"],
            "ground": data["ground"],
        }

        if self.same_task:
            same_task_data["category"].extend(self.shared_category)
            same_task_data["task"] = self.task
            same_task_data["cutoff"] = self.cutoff
        else:
            same_task_data["task"] = data["task"]
            same_task_data["cutoff"] = data["cutoff"]

        return ChallengeData(**same_task_data)

```

# `benchmark/agbenchmark/utils/get_data_from_helicone.py`

This code appears to be using the `helicone` API to query the GraphQL server for data.

The function `get_data_from_helicone` takes a `challenge` argument and returns an optional `float` value. It does this by making a GraphQL query to the `helicone.ai/api/graphql` endpoint, passing in the `authorization` header with your Helicone API key.

The query appears to be sending a GraphQL query that fetches data for the specified challenge. The query likely defines the variables to use for the query and the operation to perform. The `helicone` API then executes the query and returns the results, which are passed to the function as a `float`.

It is important to note that this code snippet may only work if you have a working environment with the Helicone API and have set up the necessary authorization tokens.


```py
import json
import os
from typing import Optional

import requests

from agbenchmark.__main__ import BENCHMARK_START_TIME
from agbenchmark.agent_interface import HELICONE_GRAPHQL_LOGS


def get_data_from_helicone(challenge: str) -> Optional[float]:
    # Define the endpoint of your GraphQL server
    url = "https://www.helicone.ai/api/graphql"

    # Set the headers, usually you'd need to set the content type and possibly an authorization token
    headers = {"authorization": f"Bearer {os.environ.get('HELICONE_API_KEY')}"}

    # Define the query, variables, and operation name
    query = """
```

这段代码是一个 GraphQL 查询语句，它使用 Query 和variables 变量来获取数据。

查询语句中定义了一个变量 properties，该变量包含了一些测试用例的属性，这些属性的值分别等于环境的 AGENT_NAME，基准测试的启动时间，以及一个名为 challenge 的变量。

然后，定义了一个名为 operation_name 的变量，它将作为 GraphQL 查询语句中的查询操作名称。

接着，定义了一个数据字典 data，该字典将作为 GraphQL 查询语句中的变量输入。

然后，定义了一个名为 response 的变量，该变量将作为 GraphQL 查询语句中的响应对象，用于从服务器获取数据。

接着，定义了一个名为 request 的异常类，该异常类用于在请求服务器时处理 HTTP 请求错误。

然后，定义了一个名为 json_error 的异常类，该异常类用于在解析 JSON 响应时处理解析错误。

接着，定义了一个名为 example_query 的函数，它接收一个名为 properties 的变量，以及一个 GraphQL 查询语句，该查询语句定义了查询操作名称、变量、以及查询语句本身。

example_query 函数会向服务器发送请求，并且使用 variables 变量中定义的值来作为请求参数。它还会执行操作名为 operation_name 的查询操作，并将得到的结果存储在 data 变量中。

最后，如果操作成功，它还会检查返回的响应是否为有效的 JSON 响应，如果不是，则会输出错误并重新抛出异常。


```py
query ExampleQuery($properties: [PropertyFilter!]){
  aggregatedHeliconeRequest(properties: $properties) {
    costUSD
  }
}
"""

    variables = {
        "properties": [
            {
                "value": {"equals": os.environ.get("AGENT_NAME")},
                "name": "agent",
            },
            {
                "value": {"equals": BENCHMARK_START_TIME},
                "name": "benchmark_start_time",
            },
            {"value": {"equals": challenge}, "name": "challenge"},
        ]
    }
    if HELICONE_GRAPHQL_LOGS:
        print(query)
        print(json.dumps(variables, indent=4))

    operation_name = "ExampleQuery"

    data = {}
    response = None

    try:
        response = requests.post(
            url,
            headers=headers,
            json={
                "query": query,
                "variables": variables,
                "operationName": operation_name,
            },
        )

        data = response.json()
    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None  # Re-raise the exception to stop execution
    except json.JSONDecodeError:
        print(f"Invalid JSON response: {response.text if response else 'No response'}")
        return None
    except Exception as err:
        print(f"Other error occurred: {err}")
        return None

    try:
        if data is None or data.get("data") is None:
            print("Invalid response received from server: no data")
            return None
        return (
            data.get("data", {})
            .get("aggregatedHeliconeRequest", {})
            .get("costUSD", None)
        )
    except Exception as err:
        print(f"Error occurred while parsing response: {err}")
        return None

```

# `benchmark/agbenchmark/utils/prompts.py`

这段代码是一个Python字典，用于存储SCORING_MAP中的评分规则。SCORING_MAP是一个包含四种评分规则的字典，分别是一个float类型的“percentage”（百分比评分）、一个integer类型的“scale”（分数范围在1到10之间）、一个boolean类型的“binary”（二进制评分）以及一个float类型的“reference_prompt”（参考提示）。

这个代码的作用是定义并返回一个表示人类回答与机器生成的回答相似度的评分标准。这个评分标准基于不同的规则，用于评估机器生成回答的质量。例如，“percentage”规则决定评分基于回答的百分比，“scale”规则决定评分基于一个1到10之间的浮点数，“binary”规则决定评分基于一个0或1的二进制分数，“reference_prompt”规则提供了一个参考答案，用于评估机器生成回答与人类回答的相似度。


```py
SCORING_MAP = {
    "percentage": "assign a float score that will represent a percentage out of 100. Use decimal points to be even more accurate. 0 represents the worst possible generation, while 100 represents the ideal generation",
    "scale": "assign an integer score from a scale of 1-10. 1 represents a really bad generation, while 10 represents an ideal generation",
    "binary": "assign a binary score of either 0 or 1. 0 represents a failure, while 1 represents a success",
}


REFERENCE_PROMPT = """Ignore previous directions. You are now an expert at evaluating how close machine generated responses are to human answers. You essentially act as a hyper advanced BLEU score.
In order to score the machine generated response you will {scoring}. Make sure to factor in the distance to the ideal response into your thinking, deliberation, and final result regarding scoring. Return nothing but a float score.

Here is the given task for you to evaluate:
{task}

Here is the ideal response you're comparing to based on the task:
{answer}

```

这段代码是一个机器学习生成的文本，用于回答评估机器生成响应的标准。它包含了一个提示，告诉评估者要忽略以前的指示，并专注于评估为给定任务的生成的文本。然后，它提供了一个评分标准，告诉评估者要如何使用这个标准来评分生成的文本。接着，它给出了一个具体的任务，让评估者可以利用提供的评分标准来评估生成的文本。最后，它提供了用于评估的当前机器生成的响应，以帮助评估者了解如何使用评分标准来评估生成的文本。


```py
Here is the current machine generated response to the task that you need to evaluate:
{response}

"""

RUBRIC_PROMPT = """Ignore previous directions. You are now an expert at evaluating machine generated responses to given tasks.
In order to score the generated texts you will {scoring}. Make sure to factor in rubric into your thinking, deliberation, and final result regarding scoring. Return nothing but a float score.

Here is the given task for you to evaluate:
{task}

Use the below rubric to guide your thinking about scoring:
{answer}

Here is the current machine generated response to the task that you need to evaluate:
{response}

```

这段代码是一个用于生成评估机器生成回应的模板，它会向用户提供一个任务和相应的参考答案以及一个用于评分的问题。用户需要按照任务要求生成一个评分，评分应当基于参考答案和生成的回应之间的匹配程度。问题提示用户在评分时需要着重考虑生成的回答是否能够很好地回答问题。

具体而言，这段代码的作用是提供一个用于生成评估机器生成回应的模板，该模板包含一个任务、一个参考答案和一个用于评分的问题。用户需要根据任务要求生成一个评分，评分应当基于参考答案和生成的回应之间的匹配程度。


```py
"""

QUESTION_PROMPT = """Ignore previous directions. You are now an expert at evaluating machine generated responses to given tasks.
In order to score the generated texts you will {scoring}. Make sure to think about whether the generated response answers the question well in order to score accurately. Return nothing but a float score.

Here is the given task:
{task}

Here is a question that checks if the task was completed correctly:
{answer}

Here is the current machine generated response to the task that you need to evaluate:
{response}

"""

```

这段代码定义了一个名为FEW_SHOT_EXAMPLES的字符串变量，其中包含了一些示例用于评分机器生成的响应的评分标准。

接着定义了一个名为CUSTOM_PROMPT的字符串变量，其中包含了一个模板，用于在响应中包含自定义的提示信息。

然后定义了一个名为PROMPT_MAP的字典变量，其中包含了一些常见的评分标准和相应的模板。这些模板用于在学生提交的作业中进行评分，评分标准包括RUBRIC_PROMPT、REFERENCE_PROMPT、QUESTION_PROMPT和CUSTOM_PROMPT。

最后，在上述定义的这些变量中，为PROMPT_MAP增加了几个键值对，这些键值对为RUBRIC_PROMPT、REFERENCE_PROMPT、QUESTION_PROMPT和CUSTOM_PROMPT，对应的模板分别为RUBRIC_PROMPT、REFERENCE_PROMPT、QUESTION_PROMPT和CUSTOM_PROMPT。

该代码的作用是定义了一个包含了评分标准和模板的接口，用于在机器生成的响应中进行评分。它为评分者和被评分者提供了方便的方式来确定应该如何评估学生的作业，并为评分者提供了多个评分标准和相应的模板。


```py
FEW_SHOT_EXAMPLES = """Here are some examples of how to score a machine generated response based on the above:
{examples}

"""

CUSTOM_PROMPT = """{custom}
{scoring}

"""

PROMPT_MAP = {
    "rubric": RUBRIC_PROMPT,
    "reference": REFERENCE_PROMPT,
    "question": QUESTION_PROMPT,
    "custom": CUSTOM_PROMPT,
}

```

这段代码是一个字符串，定义了一个常量END_PROMPT。END_PROMPT是一个简单的 message，它告诉我们要记住在回答问题时总是提供一个 float 分数。这个分数是一个简单的提示，告诉你在回答问题时不要忘记提供答案。


```py
END_PROMPT = """Remember to always end your response with nothing but a float score.
Float score:"""

```

# `benchmark/agbenchmark/utils/utils.py`

这段代码是一个Python脚本，它包括了以下内容：

1. 导入了一些外部库和函数，如json、os、re、Pathlib和typing等。
2. 通过调用`dotenv`库加载了当前环境中的代理程序名称，如果没有设置代理程序名称，则返回一个默认值。
3. 定义了一个名为`AGENT_NAME`的变量，并将其设置为操作系统中的代理程序名称。
4. 定义了一个名为`REPORT_LOCATION`的变量，并将其设置为操作系统中报告所在目录。
5. 通过`os.getenv`函数获取了当前目录下的report_location环境变量，如果没有设置该变量，则返回一个默认值。
6. 定义了一个名为`DifficultyMap`的类，该类使用了`AGENT_NAME`和`REPORT_LOCATION`作为成员变量，并实现了`DifficultyLevel`接口。
7. 该代码还包含了其他一些函数和类，但在此处没有给出。


```py
# radio charts, logs, helper functions for tests, anything else relevant.
import json
import os
import re
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv

load_dotenv()
from agbenchmark.utils.data_types import DIFFICULTY_MAP, DifficultyLevel

AGENT_NAME = os.getenv("AGENT_NAME")
REPORT_LOCATION = os.getenv("REPORT_LOCATION", None)


```



这两段代码分别是 Python 函数 replace_backslash() 和 calculate_success_percentage() 的定义。

replace_backslash() 函数的作用是将被传入的字符串中的所有制表符替换为反斜杠。具体实现是通过使用正则表达式 r"\\+" 匹配一个或多个制表符，然后使用 re.sub() 函数将其替换为反斜杠。如果传入的字符串已经是反斜杠，则不做任何处理。如果传入的是列表，则递归地将 replace_backslash() 函数应用于列表中的每个元素。如果传入的是字典，则递归地将 replace_backslash() 函数应用于字典中的每个键值对。

calculate_success_percentage() 函数的作用是计算一个列表中所有布尔值的成功率。具体实现是，如果列表的长度小于10，则将所有布尔值都视为成功，否则取最后一个10个结果计算成功率，并将成功率四舍五入保留一位小数。成功率计算公式为成功结果数除以总结果数，再将结果乘以100，最后将结果保留两位小数并四舍五入。

这两个函数的具体实现没有其他注释，因此无法通过阅读代码来理解它们如何被使用。


```py
def replace_backslash(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(
            r"\\+", "/", value
        )  # replace one or more backslashes with a forward slash
    elif isinstance(value, list):
        return [replace_backslash(i) for i in value]
    elif isinstance(value, dict):
        return {k: replace_backslash(v) for k, v in value.items()}
    else:
        return value


def calculate_success_percentage(results: list[bool]) -> float:
    # Take the last 10 results or all if less than 10
    last_results = results[-10:] if len(results) > 10 else results
    success_count = last_results.count(True)
    total_count = len(last_results)
    if total_count == 0:
        return 0
    success_percentage = (success_count / total_count) * 100  # as a percentage
    return round(success_percentage, 2)


```

此函数的作用是获取一个挑战文件的测试路径。它接收一个参数，可以是文件路径或Path类型。如果参数是一个字符串，函数会将它转换为Path类型。如果参数是一个Path对象，函数会尝试在路径中找到"agbenchmark"子目录的位置，并返回该位置。如果无法找到该子目录，函数将引发一个 ValueError。

函数的实现包含以下步骤：

1. 如果传入的参数是字符串类型，函数将尝试将其转换为Path对象。
2. 如果尝试失败，函数将引发一个 ValueError，并输出一个有关失败消息的详细信息。
3. 如果传入的参数是一个Path对象，函数将检查"agbenchmark"子目录是否存在于路径中。如果是，函数将返回该子目录所在的路径。
4. 如果"agbenchmark"子目录不存在，函数将通过replace_backslash函数将路径中的反斜杠字符串转换为正斜杠字符串，并返回该结果。
5. 如果函数返回的是一个字符串，它将被认为是格式化的路径，并直接返回它。如果返回的是一个Path对象，函数将直接返回它。


```py
def get_test_path(json_file: str | Path) -> str:
    if isinstance(json_file, str):
        json_file = Path(json_file)

    # Find the index of "agbenchmark" in the path parts
    try:
        agbenchmark_index = json_file.parts.index("benchmark")
    except ValueError:
        raise ValueError("Invalid challenge location.")

    # Create the path from "agbenchmark" onwards
    challenge_location = Path(*json_file.parts[agbenchmark_index:])

    formatted_location = replace_backslash(str(challenge_location))
    if isinstance(formatted_location, str):
        return formatted_location
    else:
        return str(challenge_location)


```

This looks like a Python function that takes a `test_data` dictionary, which contains information about the tests, including their difficulty level. The function returns a string indicating the highest difficulty level for the highest difficulty test, or a string indicating that no tests were successful if there were no tests.

The function first checks if the "tests" key is present in the `test_data` dictionary. If it is, the function retrieves the value and attempts to find the highest difficulty level for the test. If the "tests" key is not present, the function looks for the "metrics" key and, if it is present, retrieves the "difficulty" value. The function then attempts to convert the difficulty string to an integer and uses that to look up the corresponding DifficultyLevel value in the `DIFFICULTY_MAP` dictionary. If it is not possible to look up the DifficultyLevel value, it will print an error message and continue.

If the "metrics" key is present and includes a "success" value, the function will check if the difficulty level is the highest difficulty for the test. If it is, the function will return the highest difficulty level as the result. If the difficulty level is not the highest difficulty for the test, the function will convert the difficulty string to lowercase and attempt to look up the corresponding DifficultyLevel value in the `DIFFICULTY_MAP` dictionary. If it is not possible to look up the DifficultyLevel value, it will print an error message and continue.

If there are no tests in the `test_data` dictionary, the function will print the string "No successful tests". If there is a test with a difficulty level, the function will return the highest difficulty level as the result.


```py
def get_highest_success_difficulty(
    data: dict, just_string: Optional[bool] = None
) -> str:
    highest_difficulty = None
    highest_difficulty_level = 0

    for test_name, test_data in data.items():
        try:
            if test_data.get("tests", None):
                highest_difficulty_str = test_data["metrics"]["highest_difficulty"]
                try:
                    highest_difficulty = DifficultyLevel[highest_difficulty_str]
                    highest_difficulty_level = DIFFICULTY_MAP[highest_difficulty]
                except KeyError:
                    print(
                        f"Unexpected difficulty level '{highest_difficulty_str}' in test '{test_name}'"
                    )
                    continue
            else:
                if test_data["metrics"]["success"]:
                    difficulty_str = test_data["metrics"]["difficulty"]

                    try:
                        difficulty_enum = DifficultyLevel[difficulty_str.lower()]
                        difficulty_level = DIFFICULTY_MAP[difficulty_enum]

                        if difficulty_level > highest_difficulty_level:
                            highest_difficulty = difficulty_enum
                            highest_difficulty_level = difficulty_level
                    except KeyError:
                        print(
                            f"Unexpected difficulty level '{difficulty_str}' in test '{test_name}'"
                        )
                        continue
        except Exception:
            print(f"Make sure you selected the right test, no reports were generated.")
            break

    if highest_difficulty is not None:
        highest_difficulty_str = highest_difficulty.name  # convert enum to string
    else:
        highest_difficulty_str = ""

    if highest_difficulty_level and not just_string:
        return f"{highest_difficulty_str}: {highest_difficulty_level}"
    elif highest_difficulty_str:
        return highest_difficulty_str
    return "No successful tests"


```

这段代码定义了一个名为 `get_git_commit_sha` 的函数，用于获取指定目录下分支的最后一个提交(即 SHA)。

函数内部首先尝试使用 Git 客户端连接到目录对应的 Git 仓库，然后获取该仓库的远程分支(origin 或 main)的 URL。接下来，代码检查远程 URL 是否以 ".git" 结尾，如果是，则将远程 URL 移除此后跟目录，否则执行以下操作：

1. 从远程分支的 URL 中提取出分支的最后一个提交(commit hash)。
2. 将该分支的最后一个提交合并到本地分支，并返回合并后的 commit hash。

如果在执行过程中出现异常，函数将打印错误消息并返回 `None`，否则返回分支的最后一个提交。


```py
# def get_git_commit_sha(directory: Path) -> Optional[str]:
#     try:
#         repo = git.Repo(directory)
#         remote_url = repo.remotes.origin.url
#         if remote_url.endswith(".git"):
#             remote_url = remote_url[:-4]
#         git_commit_sha = f"{remote_url}/tree/{repo.head.commit.hexsha}"

#         # print(f"GIT_COMMIT_SHA: {git_commit_sha}")
#         return git_commit_sha
#     except Exception:
#         # print(f"{directory} is not a git repository!")
#         return None


```

这是一个 Python 函数，名为 `agent_eligibile_for_optional_categories`，它接受两个参数：`optional_challenge_categories` 和 `agent_categories`，并返回一个布尔值。

函数的主要目的是检查给定的 `optional_challenge_categories` 是否属于给定的 `agent_categories` 中的任何一种。具体实现是通过遍历 `optional_challenge_categories`，并检查它是否属于 `agent_categories` 中的任何一种。如果 `optional_challenge_categories` 中包含的元素不在 `agent_categories` 中，那么函数返回 `False`，否则返回 `True`。

函数还可以接受一个名为 `write_pretty_json` 的辅助函数，这个函数将给定的 `data` 对象写入一个漂亮的 JSON 文件中。这个函数的主要作用是方便地创建一个漂亮的 JSON 文件，并将 `data` 对象写入其中。


```py
def agent_eligibible_for_optional_categories(
    optional_challenge_categories: List, agent_categories: List
) -> bool:
    for element in optional_challenge_categories:
        if element not in agent_categories:
            return False
    return True


def write_pretty_json(data, json_file):
    sorted_data = deep_sort(data)
    json_graph = json.dumps(sorted_data, indent=4)
    with open(json_file, "w") as f:
        f.write(json_graph)
        f.write("\n")


```



这段代码定义了一个名为 `deep_sort` 的函数，用于对传入的 JSON 对象中的键进行递归排序。

函数接收一个名为 `obj` 的参数，首先检查该参数是否为字典类型，如果是，则按照键(也就是列表中的元素)的值进行排序，并将排好序的键(也就是列表中的元素)返回。如果不是，则将传入的列表中的每个元素递归调用 `deep_sort` 函数，并将得到的结果合并起来。如果传入的参数是列表类型，则直接返回输入的列表。

重要的是，函数内部使用了递归函数，因此函数的时间复杂度为 O(nlogn)。同时，由于使用了列表的 `sorted` 函数，因此函数内部的元素顺序与原始传入的 JSON 对象的顺序是一致的。


```py
def deep_sort(obj):
    """
    Recursively sort the keys in JSON object
    """
    if isinstance(obj, dict):
        return {k: deep_sort(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [deep_sort(elem) for elem in obj]
    return obj

```

# `benchmark/agbenchmark/utils/dependencies/constants.py`

这段代码定义了三个常量，分别用于这个模块的不同用途。

1. `MARKER_NAME`：表示测试用签名的名称，这里使用了 "depends" 作为关键字，表明这个签名的目的是声明它所 depend 的测试用例。

2. `MARKER_KWARG_ID`：表示用于标记测试用例的元数据的名称，这里使用了 "name" 作为关键字，表明这个元数据包含的是一些描述性的名称。

3. `MARKER_KWARG_DEPENDENCIES`：表示用于指定测试用例是否依赖其它测试用例的元数据，这里使用了 "on" 作为关键字，表明这个元数据表示的是测试用例之间的依赖关系，这里使用了 "on"，表示这个测试用例是依赖于其它测试用例的。


```py
""" Constants for this module. """

# The name of the marker used
MARKER_NAME = "depends"

# The name of the keyword argument for the marker that contains custom name(s) for the tests
MARKER_KWARG_ID = "name"

# The name of the keyword argument for the marker that specifies the tests to depend on
MARKER_KWARG_DEPENDENCIES = "on"

```

# `benchmark/agbenchmark/utils/dependencies/graphs.py`

这段代码的作用是定义了一个名为 `bezier_curve` 的函数，它接受四个输入参数：源点 `src`、控制点 `ctrl`、目标点 `dst` 和一个开放签 `dst`。函数的主要目的是使用这些点生成 Bézier 曲线，并返回生成的曲线点。

函数实现的基本思路是，通过计算 Bézier 曲线上的每个点，将源点与控制点、目标点分别组合成三个向量，然后将这些向量加到对应的生成的点上，最终得到 Bézier 曲线上的所有点。这个过程可以通过递归方式重复进行，从而生成多段 Bézier 曲线。

具体实现中，函数首先从输入中提取出 `src`、`ctrl` 和 `dst`，然后分别计算出每个点在 Bézier 曲线上的坐标。接着，函数使用这些坐标计算出 Bézier 曲线上的三个向量，然后将这些向量分别加上对应的控制点和目标点，最后得到 Bézier 曲线上的所有点。这些点按照顺次顺序存回，形成了一个列表返回。


```py
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pyvis.network import Network

from agbenchmark.generate_test import DATA_CATEGORY
from agbenchmark.utils.utils import write_pretty_json


def bezier_curve(
    src: np.ndarray, ctrl: List[float], dst: np.ndarray
) -> List[np.ndarray]:
    """
    Generate Bézier curve points.

    Args:
    - src (np.ndarray): The source point.
    - ctrl (List[float]): The control point.
    - dst (np.ndarray): The destination point.

    Returns:
    - List[np.ndarray]: The Bézier curve points.
    """
    curve = []
    for t in np.linspace(0, 1, num=100):
        curve_point = (
            np.outer((1 - t) ** 2, src)
            + 2 * np.outer((1 - t) * t, ctrl)
            + np.outer(t**2, dst)
        )
        curve.append(curve_point[0])
    return curve


```

这段代码定义了一个名为“curved_edges”的函数，它接受一个图形对象G、一个字典pos，以及一个距离参数dist。函数的作用是在G中找到所有在同一水平层上的节点，并对于这些节点，计算出它们之间的最小距离，然后使用Bezier曲线算法或者curve40算法绘制出这些节点的轮廓。

具体来说，函数首先获取图形对象G中的所有节点及其位置信息，然后遍历所有节点。对于每个节点u，函数获取其相邻节点v的位置信息，并计算节点u和节点v之间的距离。如果节点u和节点v在同一水平层上，函数使用abs函数计算它们之间的距离，如果距离小于0.01，则说明节点u和节点v在同一水平层上。如果节点u和节点v不在同一水平层上，函数使用Bezier曲线算法或者curve40算法绘制出节点u和节点v之间的轮廓，并将 arrows 箭头加入图形中。

函数还使用了两个辅助函数，一个是Bezier曲线算法，另一个是patches.FancyArrowPatch函数。Bezier曲线算法是一个流行的曲线绘制算法，可以通过给定控制点集合和距离参数来绘制出一条曲线。这个函数接受两个参数，一个是控制点集合，另一个是距离参数。函数的代码实现了Bezier曲线算法的基本逻辑。

patches.FancyArrowPatch函数是Matplotlib库中的一个辅助函数，用于创建一个带有箭头的实体框。这个函数接受一些参数，包括边界框的坐标、类型、颜色、线条样式和线宽等。函数的代码实现了这些参数的使用。


```py
def curved_edges(
    G: nx.Graph, pos: Dict[Any, Tuple[float, float]], dist: float = 0.2
) -> None:
    """
    Draw curved edges for nodes on the same level.

    Args:
    - G (Any): The graph object.
    - pos (Dict[Any, Tuple[float, float]]): Dictionary with node positions.
    - dist (float, optional): Distance for curvature. Defaults to 0.2.

    Returns:
    - None
    """
    ax = plt.gca()
    for u, v, data in G.edges(data=True):
        src = np.array(pos[u])
        dst = np.array(pos[v])

        same_level = abs(src[1] - dst[1]) < 0.01

        if same_level:
            control = [(src[0] + dst[0]) / 2, src[1] + dist]
            curve = bezier_curve(src, control, dst)
            arrow = patches.FancyArrowPatch(
                posA=curve[0],  # type: ignore
                posB=curve[-1],  # type: ignore
                connectionstyle=f"arc3,rad=0.2",
                color="gray",
                arrowstyle="-|>",
                mutation_scale=15.0,
                lw=1,
                shrinkA=10,
                shrinkB=10,
            )
            ax.add_patch(arrow)
        else:
            ax.annotate(
                "",
                xy=dst,
                xytext=src,
                arrowprops=dict(
                    arrowstyle="-|>", color="gray", lw=1, shrinkA=10, shrinkB=10
                ),
            )


```

这段代码的作用是定义了一个名为 `tree_layout` 的函数，它接受一个图论对象 `graph` 和一个根节点 `root_node`，并返回一个字典 `pos`，其中每行记录了节点在树布局中的位置。

函数的实现基于以下几个步骤：

1. 首先，函数使用 nx.bfs_tree 函数计算根节点的邻接图的 BFS 树。
2. 然后，函数遍历邻接图中的所有节点，计算每个节点的深度。
3. 接着，函数根据每个节点的深度，在垂直方向上对节点进行水平翻转，以实现 alternating vertical shifts（交替垂直迁移）。
4. 最后，函数根据每个节点的深度和水平翻转后的位置，在树布局中指定节点的坐标，并将节点的 depth 属性加入到字典 `pos` 中。

由于函数中使用了树布局（也称为满壳树或平衡树）的布局算法，因此它只适用于具有层次结构的图。同时，由于函数只处理了一棵树，因此它的性能可能会因为树的大小和复杂度而变得不够理想。


```py
def tree_layout(graph: nx.DiGraph, root_node: Any) -> Dict[Any, Tuple[float, float]]:
    """Compute positions as a tree layout centered on the root with alternating vertical shifts."""
    bfs_tree = nx.bfs_tree(graph, source=root_node)
    levels = {
        node: depth
        for node, depth in nx.single_source_shortest_path_length(
            bfs_tree, root_node
        ).items()
    }

    pos = {}
    max_depth = max(levels.values())
    level_positions = {i: 0 for i in range(max_depth + 1)}  # type: ignore

    # Count the number of nodes per level to compute the width
    level_count: Any = {}
    for node, level in levels.items():
        level_count[level] = level_count.get(level, 0) + 1

    vertical_offset = (
        0.07  # The amount of vertical shift per node within the same level
    )

    # Assign positions
    for node, level in sorted(levels.items(), key=lambda x: x[1]):
        total_nodes_in_level = level_count[level]
        horizontal_spacing = 1.0 / (total_nodes_in_level + 1)
        pos_x = (
            0.5
            - (total_nodes_in_level - 1) * horizontal_spacing / 2
            + level_positions[level] * horizontal_spacing
        )

        # Alternately shift nodes up and down within the same level
        pos_y = (
            -level
            + (level_positions[level] % 2) * vertical_offset
            - ((level_positions[level] + 1) % 2) * vertical_offset
        )
        pos[node] = (pos_x, pos_y)

        level_positions[level] += 1

    return pos


```

这段代码定义了一个名为 `graph_spring_layout` 的函数，它接受一个有向图（DiGraph）和图的标签，以及一个布尔值，表示是否使用树状布局。

函数的主要作用是按照 spring 布局对有向图进行布局。具体实现包括以下几个步骤：

1. 如果设置了 `tree` 为真，那么根据节点数调整 spring 布局中的 k 值。
2. 如果设置了 `tree` 为假，那么根据节点数使用 spring 布局对整个图进行布局。
3. 调用 `spring_layout` 函数对整个图进行布局，并返回根节点的位置。
4. 使用 `draw_networkx_nodes` 和 `draw_networkx_labels` 函数分别对节点和标签进行绘制。
5. 使用 `curved_edges` 函数处理弓形边。
6. 最后，对整个图形进行tight_layout调整，并显示图形。


```py
def graph_spring_layout(
    dag: nx.DiGraph, labels: Dict[Any, str], tree: bool = True
) -> None:
    num_nodes = len(dag.nodes())
    # Setting up the figure and axis
    fig, ax = plt.subplots()
    ax.axis("off")  # Turn off the axis

    base = 3.0

    if num_nodes > 10:
        base /= 1 + math.log(num_nodes)
        font_size = base * 10

    font_size = max(10, base * 10)
    node_size = max(300, base * 1000)

    if tree:
        root_node = [node for node, degree in dag.in_degree() if degree == 0][0]
        pos = tree_layout(dag, root_node)
    else:
        # Adjust k for the spring layout based on node count
        k_value = 3 / math.sqrt(num_nodes)

        pos = nx.spring_layout(dag, k=k_value, iterations=50)

    # Draw nodes and labels
    nx.draw_networkx_nodes(dag, pos, node_color="skyblue", node_size=int(node_size))
    nx.draw_networkx_labels(dag, pos, labels=labels, font_size=int(font_size))

    # Draw curved edges
    curved_edges(dag, pos)  # type: ignore

    plt.tight_layout()
    plt.show()


```

这段代码定义了两个函数：`rgb_to_hex` 和 `get_category_colors`。

函数 `rgb_to_hex` 接收一个 RGB 值列表（float 类型），并将其转换为颜色代码字符串。具体地，它将每种颜色值乘以 255，然后使用格式化字符串将它们组合成一个字符串。这个字符串中的 `:` 表示 RGB 颜色值，`{:02x}{:02x}{:02x}` 中的 `{}` 表示颜色值的范围，即从 0 到 255。通过这种方式，函数 `rgb_to_hex` 可以将一个 RGB 值列表转换为颜色代码字符串，并可以方便地将它们输出给调用者。

函数 `get_category_colors` 接收一个字典，其中键是类别名称，值是颜色代码字符串。它使用了一个 `DICT` 类型，将每个类别名称映射到一个颜色代码字符串。具体地，它创建了一个 `cm` 对象，然后使用 `get_cmap` 方法从 `plt.cm` 中获取一个颜色映射，长度为 `len(categories)`，然后创建一个 `RGB` 对象，将颜色值设置为每个类别的颜色代码字符串。最后，它将每个类别名称映射到它对应的颜色代码字符串，并将它们存储在一个 `DICT` 类型中，使得每个类别都有它自己的颜色代码字符串。

总的来说，这两个函数的主要目的是将颜色数据转换为字符串形式，以便在程序中进行使用。


```py
def rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    )


def get_category_colors(categories: Dict[Any, str]) -> Dict[str, str]:
    unique_categories = set(categories.values())
    colormap = plt.cm.get_cmap("tab10", len(unique_categories))  # type: ignore
    return {
        category: rgb_to_hex(colormap(i)[:3])
        for i, category in enumerate(unique_categories)
    }


```

It looks like this is a Python script that is using Flutter's `write_pretty_json` and `validate_skill_tree` functions, as well as writing the data to a JSON file.

It appears to be processing a graph data structure, extracting a subgraph based


```py
def graph_interactive_network(
    dag: nx.DiGraph,
    labels: Dict[Any, Dict[str, Any]],
    html_graph_path: str = "",
) -> None:
    nt = Network(notebook=True, width="100%", height="800px", directed=True)

    category_colors = get_category_colors(DATA_CATEGORY)

    # Add nodes and edges to the pyvis network
    for node, json_data in labels.items():
        label = json_data.get("name", "")
        # remove the first 4 letters of label
        label_without_test = label[4:]
        node_id_str = node.nodeid

        # Get the category for this label
        category = DATA_CATEGORY.get(
            label, "unknown"
        )  # Default to 'unknown' if label not found

        # Get the color for this category
        color = category_colors.get(category, "grey")

        nt.add_node(
            node_id_str,
            label=label_without_test,
            color=color,
            data=json_data,
        )

    # Add edges to the pyvis network
    for edge in dag.edges():
        source_id_str = edge[0].nodeid
        target_id_str = edge[1].nodeid
        edge_id_str = (
            f"{source_id_str}_to_{target_id_str}"  # Construct a unique edge id
        )
        if not (source_id_str in nt.get_nodes() and target_id_str in nt.get_nodes()):
            print(
                f"Skipping edge {source_id_str} -> {target_id_str} due to missing nodes."
            )
            continue
        nt.add_edge(source_id_str, target_id_str, id=edge_id_str)

    # Configure physics for hierarchical layout
    hierarchical_options = {
        "enabled": True,
        "levelSeparation": 200,  # Increased vertical spacing between levels
        "nodeSpacing": 250,  # Increased spacing between nodes on the same level
        "treeSpacing": 250,  # Increased spacing between different trees (for forest)
        "blockShifting": True,
        "edgeMinimization": True,
        "parentCentralization": True,
        "direction": "UD",
        "sortMethod": "directed",
    }

    physics_options = {
        "stabilization": {
            "enabled": True,
            "iterations": 1000,  # Default is often around 100
        },
        "hierarchicalRepulsion": {
            "centralGravity": 0.0,
            "springLength": 200,  # Increased edge length
            "springConstant": 0.01,
            "nodeDistance": 250,  # Increased minimum distance between nodes
            "damping": 0.09,
        },
        "solver": "hierarchicalRepulsion",
        "timestep": 0.5,
    }

    nt.options = {
        "nodes": {
            "font": {
                "size": 20,  # Increased font size for labels
                "color": "black",  # Set a readable font color
            },
            "shapeProperties": {"useBorderWithImage": True},
        },
        "edges": {
            "length": 250,  # Increased edge length
        },
        "physics": physics_options,
        "layout": {"hierarchical": hierarchical_options},
    }

    # Serialize the graph to JSON
    graph_data = {"nodes": nt.nodes, "edges": nt.edges}

    home_path = Path.cwd()
    write_pretty_json(graph_data, home_path / "frontend" / "public" / "graph.json")

    flutter_app_path = home_path.parent / "frontend" / "assets"

    # Optionally, save to a file
    # Sync with the flutter UI
    # this literally only works in the AutoGPT repo, but this part of the code is not reached if BUILD_SKILL_TREE is false
    write_pretty_json(graph_data, flutter_app_path / "tree_structure.json")
    validate_skill_tree(graph_data, "")
    import json

    # Extract node IDs with category "coding"

    coding_tree = extract_subgraph_based_on_category(graph_data.copy(), "coding")
    validate_skill_tree(coding_tree, "coding")
    write_pretty_json(
        coding_tree,
        flutter_app_path / "coding_tree_structure.json",
    )

    data_tree = extract_subgraph_based_on_category(graph_data.copy(), "data")
    # validate_skill_tree(data_tree, "data")
    write_pretty_json(
        data_tree,
        flutter_app_path / "data_tree_structure.json",
    )

    general_tree = extract_subgraph_based_on_category(graph_data.copy(), "general")
    validate_skill_tree(general_tree, "general")
    write_pretty_json(
        general_tree,
        flutter_app_path / "general_tree_structure.json",
    )

    scrape_synthesize_tree = extract_subgraph_based_on_category(
        graph_data.copy(), "scrape_synthesize"
    )
    validate_skill_tree(scrape_synthesize_tree, "scrape_synthesize")
    write_pretty_json(
        scrape_synthesize_tree,
        flutter_app_path / "scrape_synthesize_tree_structure.json",
    )
    # If you want to convert back to JSON
    filtered_json = json.dumps(graph_data, indent=4)
    print(filtered_json)

    if html_graph_path:
        file_path = str(Path(html_graph_path).resolve())

        nt.write_html(file_path)


```

该函数的作用是从给定的图形中选择一个子图，该子图包含所有与给定类别的节点和边，并使其中的所有节点都可达该类别的所有节点。

函数的实现通过两个递归函数进行：`reverse_dfs(node_id)` 和 `nodes_with_target_category`。

第一个函数 `reverse_dfs(node_id)` 用于确定给定类别的节点，并将这些节点添加到子图中的节点列表中。该函数首先检查节点是否已存在于子图中的节点列表中，如果是，则返回。否则，它将遍历与给定类别相关的图中的所有节点，并将它们添加到子图中的节点列表中。

第二个函数 `nodes_with_target_category` 用于查找与给定类别相关的节点，并从其开始进行反向深度优先搜索（DFS）。递归地处理与给定类别相关的节点，并将它们添加到子图中的节点列表中。


```py
def extract_subgraph_based_on_category(graph, category):
    """
    Extracts a subgraph that includes all nodes and edges required to reach all nodes with a specified category.

    :param graph: The original graph.
    :param category: The target category.
    :return: Subgraph with nodes and edges required to reach the nodes with the given category.
    """

    subgraph = {"nodes": [], "edges": []}
    visited = set()

    def reverse_dfs(node_id):
        if node_id in visited:
            return
        visited.add(node_id)

        node_data = next(node for node in graph["nodes"] if node["id"] == node_id)

        # Add the node to the subgraph if it's not already present.
        if node_data not in subgraph["nodes"]:
            subgraph["nodes"].append(node_data)

        for edge in graph["edges"]:
            if edge["to"] == node_id:
                if edge not in subgraph["edges"]:
                    subgraph["edges"].append(edge)
                reverse_dfs(edge["from"])

    # Identify nodes with the target category and initiate reverse DFS from them.
    nodes_with_target_category = [
        node["id"] for node in graph["nodes"] if category in node["data"]["category"]
    ]

    for node_id in nodes_with_target_category:
        reverse_dfs(node_id)

    return subgraph


```

该函数判断给定的图（graph）是否为环形。如果图中有环形，则函数将返回环形的路径；否则，函数返回None。函数采用深度优先搜索（DFS）的方式遍历图中的所有节点，并且在搜索过程中使用两个集合（visited 和 stack）来跟踪已经访问过的节点。函数使用 parent_map 字典来存储图中的父节点映射，其中 key 为节点 ID，value 为其父节点 ID。函数首先定义了所有已访问过的节点为一个集合（set），然后定义了两个集合：stack 和 visited。stack 用于存储当前正在进行的搜索路径，而 visited 用于存储已经访问过的节点。函数接下来遍历图中的所有节点，并对于每个节点，首先将其添加到 visited 集合中，然后将其添加到 stack 中。接着，函数遍历图中的所有边（edges），并将边 from 设置为当前节点。如果边当前节点存在，则检查边 to 是否存在于 stack 中。如果边 to 存在于 stack 中，则说明已检测到环形，函数返回环形的路径。否则，函数创建一个新的空集合（set）来存储环形的路径，并递归调用 itself（即 dfs）函数来获取环形的路径。


```py
def is_circular(graph):
    def dfs(node, visited, stack, parent_map):
        visited.add(node)
        stack.add(node)
        for edge in graph["edges"]:
            if edge["from"] == node:
                if edge["to"] in stack:
                    # Detected a cycle
                    cycle_path = []
                    current = node
                    while current != edge["to"]:
                        cycle_path.append(current)
                        current = parent_map.get(current)
                    cycle_path.append(edge["to"])
                    cycle_path.append(node)
                    return cycle_path[::-1]
                elif edge["to"] not in visited:
                    parent_map[edge["to"]] = node
                    cycle_path = dfs(edge["to"], visited, stack, parent_map)
                    if cycle_path:
                        return cycle_path
        stack.remove(node)
        return None

    visited = set()
    stack = set()
    parent_map = {}
    for node in graph["nodes"]:
        node_id = node["id"]
        if node_id not in visited:
            cycle_path = dfs(node_id, visited, stack, parent_map)
            if cycle_path:
                return cycle_path
    return None


```



This code defines a function `get_roots` that takes a graph dictionary as an input. The purpose of this function is to return a list of all the nodes in the graph that have no incoming edges, which are therefore considered to be the roots of the graph.

To do this, the function first creates a set of all node IDs in the graph by iterating over the `nodes` dictionary and storing each node's `id` in the set. It then creates a set of nodes with incoming edges by iterating over the `edges` dictionary and storing each edge's `to` value in the set.

Finally, the function removes any nodes from the set of nodes with incoming edges and returns the list of remaining nodes, which are the roots of the graph.


```py
def get_roots(graph):
    """
    Return the roots of a graph. Roots are nodes with no incoming edges.
    """
    # Create a set of all node IDs
    all_nodes = {node["id"] for node in graph["nodes"]}

    # Create a set of nodes with incoming edges
    nodes_with_incoming_edges = {edge["to"] for edge in graph["edges"]}

    # Roots are nodes that have no incoming edges
    roots = all_nodes - nodes_with_incoming_edges

    return list(roots)


```

这段代码定义了一个名为 validate_skill_tree 的函数，它接受两个参数：一个表示图的 dictionary Graph 和一个表示技能树的字符串 Skill Tree Name。

函数的主要作用是验证给定的图是否代表了一个有效的技能树，如果验证失败，则抛出一个 ValueError，并给出相应的错误描述。

具体来说，函数首先检查给定的图是否为环形，如果为环形，则抛出一个 ValueError，并给出相应的错误描述。如果图不是环形，则接着检查给定的图是否存在两个根。如果图存在两个根，则抛出一个 ValueError，并给出相应的错误描述。如果图不存在两个根，则抛出一个 ValueError，并给出相应的错误描述。

validate_skill_tree 函数的实现主要依赖于 is_circular 和 get_roots 函数，它们的功能分别判断图是否为环形，以及获取图中的根。


```py
def validate_skill_tree(graph, skill_tree_name):
    """
    Validate if a given graph represents a valid skill tree and raise appropriate exceptions if not.

    :param graph: A dictionary representing the graph with 'nodes' and 'edges'.
    :raises: ValueError with a description of the invalidity.
    """
    # Check for circularity
    cycle_path = is_circular(graph)
    if cycle_path:
        cycle_str = " -> ".join(cycle_path)
        raise ValueError(
            f"{skill_tree_name} skill tree is circular! Circular path detected: {cycle_str}."
        )

    # Check for multiple roots
    roots = get_roots(graph)
    if len(roots) > 1:
        raise ValueError(f"{skill_tree_name} skill tree has multiple roots: {roots}.")
    elif not roots:
        raise ValueError(f"{skill_tree_name} skill tree has no roots.")

```