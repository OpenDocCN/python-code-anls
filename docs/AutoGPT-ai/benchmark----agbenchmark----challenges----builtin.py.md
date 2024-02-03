# `.\AutoGPT\benchmark\agbenchmark\challenges\builtin.py`

```py
# 导入所需的模块
from collections import deque
import glob
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, ClassVar, Iterator, Literal, Optional

import pytest
from agent_protocol_client import AgentApi, ApiClient, Configuration as ClientConfig
from colorama import Fore, Style
from openai import _load_client as get_openai_client
from pydantic import BaseModel, constr, Field, validator

# 导入自定义模块
from agbenchmark.agent_api_interface import download_agent_artifacts_into_folder
from agbenchmark.agent_interface import copy_challenge_artifacts_into_workspace
from agbenchmark.config import AgentBenchmarkConfig
from agbenchmark.utils.data_types import Category, DifficultyLevel, EvalResult
from agbenchmark.utils.prompts import (
    END_PROMPT,
    FEW_SHOT_EXAMPLES,
    PROMPT_MAP,
    SCORING_MAP,
)

# 设置日志记录器
logger = logging.getLogger(__name__)

# 读取可选类别的 JSON 文件
with open(Path(__file__).parent / "optional_categories.json") as f:
    # 加载 JSON 文件中的可选类别列表
    OPTIONAL_CATEGORIES: list[str] = json.load(f)["optional_categories"]

# 定义内置挑战规范类
class BuiltinChallengeSpec(BaseModel):
    # 挑战评估 ID
    eval_id: str = ""
    # 挑战名称
    name: str
    # 挑战任务
    task: str
    # 挑战类别列表
    category: list[Category]
    # 挑战依赖列表
    dependencies: list[str]
    # 挑战截止时间
    cutoff: int

    # 挑战信息类
    class Info(BaseModel):
        # 挑战难度级别
        difficulty: DifficultyLevel
        # 挑战描述
        description: constr(regex=r"^Tests if the agent can.*")
        # 副作用列表
        side_effects: list[str] = Field(default_factory=list)

    # 挑战信息
    info: Info
    # 定义 Ground 类，继承自 BaseModel
    class Ground(BaseModel):
        # 答案字段，类型为字符串
        answer: str
        # 应包含的内容列表，可选
        should_contain: Optional[list[str]] = None
        # 不应包含的内容列表，可选
        should_not_contain: Optional[list[str]] = None
        # 文件列表
        files: list[str]
        # 是否区分大小写，可选，默认为 True

        # 定义 Eval 类，嵌套在 Ground 类中
        class Eval(BaseModel):
            # 类型字段，字符串类型
            type: str
            # 评分字段，可选，取值为 "percentage", "scale", "binary" 中的一个
            scoring: Optional[Literal["percentage", "scale", "binary"]]
            # 模板字段，可选，取值为 "rubric", "reference", "question", "custom" 中的一个
            template: Optional[Literal["rubric", "reference", "question", "custom"]]
            # 示例字段，可选
            examples: Optional[str]

            # 验证器函数，验证 scoring 和 template 字段
            @validator("scoring", "template", always=True)
            def validate_eval_fields(cls, v, values, field):
                # 如果 type 字段为 'llm'，则 scoring 字段必须提供
                if "type" in values and values["type"] == "llm":
                    if v is None:
                        raise ValueError(
                            f"{field.name} must be provided when eval type is 'llm'"
                        )
                else:
                    # 如果 type 字段不为 'llm'，则 scoring 字段不应该存在
                    if v is not None:
                        raise ValueError(
                            f"{field.name} should only exist when eval type is 'llm'"
                        )
                return v

        # eval 字段，类型为 Eval 类
        eval: Eval

    # ground 对象，类型为 Ground 类
    ground: Ground

    # 元数据字段，可选，字典类型
    metadata: Optional[dict[str, Any]] = None
    # 规范文件字段，路径类型或者 None，不包含在模型中
    spec_file: Path | None = Field(None, exclude=True)
# 定义一个名为 BuiltinChallenge 的类，作为 AGBenchmark 内置挑战的基类
class BuiltinChallenge(BaseChallenge):
    """
    Base class for AGBenchmark's built-in challenges (challenges/**/*.json).

    All of the logic is present in this class. Individual challenges are created as
    subclasses of `BuiltinChallenge` with challenge-specific values assigned to the
    ClassVars `_spec` etc.

    Dynamically constructing subclasses rather than class instances for the individual
    challenges makes them suitable for collection by Pytest, which will run their
    `test_method` like any regular test item.
    """

    # 类变量 _spec，用于存储 BuiltinChallengeSpec 类型的值
    _spec: ClassVar[BuiltinChallengeSpec]
    # 类变量 CHALLENGE_LOCATION，用于存储挑战的位置信息
    CHALLENGE_LOCATION: ClassVar[str]
    # 类变量 ARTIFACTS_LOCATION，用于存储挑战相关文件的位置信息

    # 类变量 SOURCE_URI_PREFIX，用于存储挑战源文件的前缀
    SOURCE_URI_PREFIX = "__BUILTIN__"

    # 类方法，根据挑战规范创建 BuiltinChallenge 类的实例
    @classmethod
    def from_challenge_spec(
        cls, spec: BuiltinChallengeSpec
    ) -> type["BuiltinChallenge"]:
        # 如果挑战规范中未定义 spec_file，则抛出 ValueError 异常
        if not spec.spec_file:
            raise ValueError("spec.spec_file not defined")

        # 创建 ChallengeInfo 对象，存储挑战相关信息
        challenge_info = ChallengeInfo(
            eval_id=spec.eval_id,
            name=spec.name,
            task=spec.task,
            task_artifacts_dir=spec.spec_file.parent,
            category=spec.category,
            difficulty=spec.info.difficulty,
            description=spec.info.description,
            dependencies=spec.dependencies,
            reference_answer=spec.ground.answer,
            source_uri=(
                f"__BUILTIN__/{spec.spec_file.relative_to(Path(__file__).parent)}"
            ),
        )

        # 构建挑战类的名称
        challenge_class_name = f"Test{challenge_info.name}"
        logger.debug(f"Creating {challenge_class_name} from spec: {spec.spec_file}")
        # 动态创建挑战类，并返回
        return type(
            challenge_class_name,
            (BuiltinChallenge,),
            {
                "info": challenge_info,
                "_spec": spec,
                "CHALLENGE_LOCATION": str(spec.spec_file),
                "ARTIFACTS_LOCATION": str(spec.spec_file.resolve().parent),
            },
        )

    # 类方法
    @classmethod
    # 从挑战规范文件中创建内置挑战类的实例
    def from_challenge_spec_file(cls, spec_file: Path) -> type["BuiltinChallenge"]:
        # 解析挑战规范文件
        challenge_spec = BuiltinChallengeSpec.parse_file(spec_file)
        # 设置挑战规范文件路径
        challenge_spec.spec_file = spec_file
        # 调用类方法从挑战规范创建内置挑战类实例
        return cls.from_challenge_spec(challenge_spec)

    # 从源 URI 创建内置挑战类的实例
    @classmethod
    def from_source_uri(cls, source_uri: str) -> type["BuiltinChallenge"]:
        # 检查源 URI 是否以指定前缀开头
        if not source_uri.startswith(cls.SOURCE_URI_PREFIX):
            raise ValueError(f"Invalid source_uri for BuiltinChallenge: {source_uri}")

        # 从源 URI 中提取路径
        path = source_uri.split("/", 1)[1]
        # 构建挑战规范文件的路径
        spec_file = Path(__file__).parent / path
        # 调用类方法从挑战规范文件创建内置挑战类实例
        return cls.from_challenge_spec_file(spec_file)

    # 异步测试方法
    @pytest.mark.asyncio
    async def test_method(
        self,
        config: AgentBenchmarkConfig,
        request: pytest.FixtureRequest,
        i_attempt: int,
    # 定义一个方法，接受一个参数并不返回任何内容
    ) -> None:
        # 检查环境变量中是否存在名为 "HELICONE_API_KEY" 的值
        if os.environ.get("HELICONE_API_KEY"):
            # 导入 HeliconeLockManager 类
            from helicone.lock import HeliconeLockManager

            # 将挑战名称写入自定义属性 "challenge"
            HeliconeLockManager.write_custom_property("challenge", self.info.name)

        # 设置超时时间，默认为 60 秒
        timeout = self._spec.cutoff or 60

        # 如果命令行参数中包含 "--nc"，则将超时时间设置为 100000 秒
        if request.config.getoption("--nc"):
            timeout = 100000
        # 如果命令行参数中包含 "--cutoff"，则将超时时间设置为对应值
        elif cutoff := request.config.getoption("--cutoff"):
            timeout = int(cutoff)  # type: ignore

        # 初始化任务 ID
        task_id = ""
        timed_out = None
        try:
            # 异步循环运行挑战步骤
            async for step in self.run_challenge(config, timeout):
                # 如果任务 ID 为空，则设置为当前步骤的任务 ID
                if not task_id:
                    task_id = step.task_id
                # 如果命令行参数中包含 "--mock"，则只运行一步挑战
                if request.config.getoption("--mock"):
                    # 跳出循环
                    break
            timed_out = False
        except TimeoutError:
            # 捕获超时错误
            timed_out = True
        # 将超时状态添加到测试节点的用户属性中
        request.node.user_properties.append(("timed_out", timed_out))

        # 配置代理客户端
        agent_client_config = ClientConfig(host=config.host)
        # 异步创建 API 客户端
        async with ApiClient(agent_client_config) as api_client:
            api_instance = AgentApi(api_client)
            # 获取任务状态的评估结果
            eval_results = await self.evaluate_task_state(api_instance, task_id)

        # 如果没有评估结果
        if not eval_results:
            # 如果超时，则抛出超时错误
            if timed_out:
                raise TimeoutError("Timed out, no results to evaluate")
            else:
                raise ValueError("No results to evaluate")

        # 将评估结果添加到测试节点的用户属性中
        request.node.user_properties.append(
            (
                "answers",
                [r.result for r in eval_results]
                if request.config.getoption("--keep-answers")
                else None,
            )
        )
        # 将分数添加到测试节点的用户属性中
        request.node.user_properties.append(("scores", [r.score for r in eval_results]))

        # 检查是否存在任何通过的评估结果，如果没有则引发异常
        assert any(r.passed for r in eval_results), (
            f"No passed evals: {eval_results}"
            if not timed_out
            else f"Timed out; no passed evals: {eval_results}"
        )

    # 定义一个类方法
    @classmethod
    # 异步函数，用于评估任务状态并返回评估结果列表
    async def evaluate_task_state(
        cls, agent: AgentApi, task_id: str
    ) -> list[EvalResult]:
        # 创建临时目录作为工作空间
        with tempfile.TemporaryDirectory() as workspace:
            # 将临时目录路径转换为 Path 对象
            workspace = Path(workspace)
            # 下载代理工件到工作空间
            await download_agent_artifacts_into_folder(agent, task_id, workspace)
            # 如果存在任务工件目录
            if cls.info.task_artifacts_dir:
                # 将挑战工件复制到工作空间中的自定义 Python 目录
                copy_challenge_artifacts_into_workspace(
                    cls.info.task_artifacts_dir, "custom_python", workspace
                )

            # 返回工作空间内容的评估结果列表
            return list(cls.evaluate_workspace_content(workspace))

    # 类方法
    @classmethod
    # 评估工作区内容，返回评估结果的迭代器
    def evaluate_workspace_content(cls, workspace: Path) -> Iterator[EvalResult]:
        # 如果任务为空且环境变量 IS_MOCK 存在
        if cls._spec.task == "" and os.getenv("IS_MOCK"):
            # 返回一个模拟答案的评估结果
            yield EvalResult(
                result="This is a mock answer",
                result_source="step_output",
                score=1.0,
                passed=True,
            )
            return

        # 获取评估结果的 ground 属性
        result_ground = cls._spec.ground
        # 获取用于评估的输出内容
        outputs_for_eval = cls.get_outputs_for_eval(workspace, result_ground)

        # 如果评估结果应包含或不应包含内容
        if result_ground.should_contain or result_ground.should_not_contain:
            # 遍历输出内容，计算得分并返回评估结果
            for source, content in outputs_for_eval:
                score = cls.score_result(content, result_ground)
                if score is not None:
                    # 打印得分
                    print(f"{Fore.GREEN}Your score is:{Style.RESET_ALL}", score)
                    yield EvalResult(
                        result=content,
                        result_source=str(source),
                        score=score,
                        passed=score > 0.9,  # FIXME: arbitrary threshold
                    )

        # 如果评估结果的类型为 "llm"
        if result_ground.eval.type == "llm":
            # 将所有输出内容组合成一个字符串
            combined_results = "\n".join(output[1] for output in outputs_for_eval)
            # 使用 llm 方法评估结果
            llm_eval = cls.score_result_with_llm(combined_results, result_ground)
            # 打印 llm 得分
            print(f"{Fore.GREEN}Your score is:{Style.RESET_ALL}", llm_eval)
            # 根据评估类型计算得分
            if result_ground.eval.scoring == "percentage":
                score = llm_eval / 100
            elif result_ground.eval.scoring == "scale":
                score = llm_eval / 10
            else:
                score = llm_eval

            yield EvalResult(
                result=combined_results,
                result_source=", ".join(str(res[0]) for res in outputs_for_eval),
                score=score,
                passed=score > 0.9,  # FIXME: arbitrary threshold
            )

    @staticmethod
    # 获取用于评估的输出内容
    def get_outputs_for_eval(
        workspace: str | Path | dict[str, str], ground: BuiltinChallengeSpec.Ground
    # 定义一个生成器函数，返回一个迭代器，每次迭代返回一个元组，包含文件路径和内容
    ) -> Iterator[tuple[str | Path, str]]:
        # 如果 workspace 是一个字典，则将其替换为字典中的 "output" 键对应的值
        if isinstance(workspace, dict):
            workspace = workspace["output"]

        # 将 script_dir 设置为 workspace
        script_dir = workspace

        # 遍历 ground.files 中的每个文件模式
        for file_pattern in ground.files:
            # 检查文件模式是否以点号开头，表示文件扩展名
            if file_pattern.startswith("."):
                # 在 workspace 中查找所有具有给定扩展名的文件
                matching_files = glob.glob(os.path.join(script_dir, "*" + file_pattern))
            else:
                # 否则，文件模式是一个具体的文件路径
                matching_files = [os.path.join(script_dir, file_pattern)]

            # 遍历匹配的文件路径
            for file_path in matching_files:
                # 如果 ground.eval.type 是 "python"
                if ground.eval.type == "python":
                    # 运行指定文件路径的 Python 脚本
                    result = subprocess.run(
                        [sys.executable, file_path],
                        cwd=os.path.abspath(workspace),
                        capture_output=True,
                        text=True,
                    )
                    # 如果标准错误中包含 "error" 或返回码不为 0
                    if "error" in result.stderr or result.returncode != 0:
                        # 打印标准错误信息
                        print(result.stderr)
                        # 断言失败，输出标准错误信息
                        assert False, result.stderr
                    # 生成一个元组，包含文件路径的相对路径和输出内容
                    yield (
                        Path(file_path).relative_to(workspace),
                        f"Output: {result.stdout}\n",
                    )
                else:
                    # 否则，使用 with 语句打开文件
                    with open(file_path, "r") as f:
                        # 生成一个元组，包含文件路径的相对路径和文件内容
                        yield Path(file_path).relative_to(workspace), f.read()
        else:
            # 如果 ground.eval.type 是 "pytest"
            if ground.eval.type == "pytest":
                # 运行 pytest 命令
                result = subprocess.run(
                    [sys.executable, "-m", "pytest"],
                    cwd=os.path.abspath(workspace),
                    capture_output=True,
                    text=True,
                )
                # 如果标准错误中包含 "error" 或返回码不为 0
                if "error" in result.stderr or result.returncode != 0:
                    # 打印标准错误信息
                    print(result.stderr)
                    # 断言失败，输出标准错误信息
                    assert False, result.stderr
                # 生成一个元组，包含 "pytest" 和输出内容
                yield "pytest", f"Output: {result.stdout}\n"

    @staticmethod
    # 定义一个函数，用于评分给定内容和标准
    def score_result(content: str, ground: BuiltinChallengeSpec.Ground) -> float | None:
        # 打印评分内容
        print(f"{Fore.BLUE}Scoring content:{Style.RESET_ALL}", content)
        
        # 如果标准中包含应该存在的单词
        if ground.should_contain:
            # 遍历每个应该存在的单词
            for should_contain_word in ground.should_contain:
                # 如果不区分大小写
                if not ground.case_sensitive:
                    # 将单词和内容转换为小写
                    should_contain_word = should_contain_word.lower()
                    content = content.lower()
                # 打印应该存在的单词
                print_content = (
                    f"{Fore.BLUE}Word that should exist{Style.RESET_ALL}"
                    f" - {should_contain_word}:"
                )
                # 如果单词不在内容中
                if should_contain_word not in content:
                    # 打印结果为 False，返回 0.0 分
                    print(print_content, "False")
                    return 0.0
                else:
                    # 打印结果为 True，返回 1.0 分
                    print(print_content, "True")
                    return 1.0

        # 如果标准中包含不应该存在的单词
        if ground.should_not_contain:
            # 遍历每个不应该存在的单词
            for should_not_contain_word in ground.should_not_contain:
                # 如果不区分大小写
                if not ground.case_sensitive:
                    # 将单词和内容转换为小写
                    should_not_contain_word = should_not_contain_word.lower()
                    content = content.lower()
                # 打印不应该存在的单词
                print_content = (
                    f"{Fore.BLUE}Word that should not exist{Style.RESET_ALL}"
                    f" - {should_not_contain_word}:"
                )
                # 如果单词在内容中
                if should_not_contain_word in content:
                    # 打印结果为 False，返回 0.0 分
                    print(print_content, "False")
                    return 0.0
                else:
                    # 打印结果为 True，返回 1.0 分
                    print(print_content, "True")
                    return 1.0

    # 定义一个类方法，用于评分给定内容和标准
    @classmethod
    def score_result_with_llm(
        cls, content: str, ground: BuiltinChallengeSpec.Ground
    # 定义函数返回类型为浮点数
    ) -> float:
        # 如果环境变量 IS_MOCK 存在，则返回 1.0
        if os.getenv("IS_MOCK"):
            return 1.0

        # 从 SCORING_MAP 中获取评分方式
        scoring = SCORING_MAP[ground.eval.scoring]  # type: ignore
        # 根据 ground.eval.template 选择对应的提示模板，并填充相关信息
        prompt = PROMPT_MAP[ground.eval.template].format(  # type: ignore
            task=cls._spec.task, scoring=scoring, answer=ground.answer, response=content
        )

        # 如果存在 few-shot examples，则添加到提示中
        if ground.eval.examples:
            prompt += FEW_SHOT_EXAMPLES.format(examples=ground.eval.examples)

        # 添加结束提示
        prompt += END_PROMPT

        # 使用 OpenAI 客户端创建对话完成请求
        answer = get_openai_client().chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
            ],
        )

        # 返回对话完成结果中的第一个消息内容，并转换为浮点数
        return float(answer.choices[0].message.content)  # type: ignore
# 加载内置挑战，返回挑战类型的迭代器
def load_builtin_challenges() -> Iterator[type[BuiltinChallenge]]:
    # 记录日志信息，表示正在加载内置挑战
    logger.info("Loading built-in challenges...")

    # 获取挑战文件所在路径
    challenges_path = os.path.dirname(__file__)
    # 记录调试信息，表示在挑战路径中查找挑战规范文件
    logger.debug(f"Looking for challenge spec files in {challenges_path}...")

    # 使用递归方式查找挑战路径下所有的data.json文件，并存储在双端队列中
    json_files = deque(
        glob.glob(
            f"{challenges_path}/**/data.json",
            recursive=True,
        )
    )

    # 记录调试信息，表示找到了多少个内置挑战
    logger.debug(f"Found {len(json_files)} built-in challenges.")

    loaded, ignored = 0, 0
    # 遍历json_files
    while json_files:
        # 从json_files中取出并移除第一个元素
        json_file = json_files.popleft()
        # 判断是否应该忽略该挑战
        if _challenge_should_be_ignored(json_file):
            ignored += 1
            continue

        # 从挑战规范文件生成挑战对象
        challenge = BuiltinChallenge.from_challenge_spec_file(Path(json_file))
        # 记录调试信息，表示为挑战生成了测试
        logger.debug(f"Generated test for {challenge.info.name}")
        # 返回挑战对象
        yield challenge

        loaded += 1

    # 记录日志信息，表示内置挑战加载完成，加载了多少个，忽略了多少个
    logger.info(
        f"Loading built-in challenges complete: loaded {loaded}, ignored {ignored}."
    )


# 判断是否应该忽略挑战
def _challenge_should_be_ignored(json_file_path: str):
    return (
        "challenges/deprecated" in json_file_path
        or "challenges/library" in json_file_path
    )
```