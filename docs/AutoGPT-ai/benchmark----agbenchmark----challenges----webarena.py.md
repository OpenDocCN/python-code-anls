# `.\AutoGPT\benchmark\agbenchmark\challenges\webarena.py`

```py
# 导入所需的库
import logging
import os
from abc import ABC, abstractmethod
from typing import ClassVar, Iterator, Literal

import pytest
import requests
from agent_protocol_client import AgentApi, Step
from pydantic import BaseModel, validator, ValidationError

from agbenchmark.config import AgentBenchmarkConfig
from agbenchmark.utils.data_types import Category, EvalResult

# 导入自定义的模块
from .base import BaseChallenge, ChallengeInfo

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 定义字面量类型
EvalType = Literal["string_match", "url_match", "program_html"]
WebArenaSite = Literal[
    "gitlab", "map", "reddit", "shopping", "shopping_admin", "wikipedia"
]
ReferenceAnswerType = Literal["exact_match", "fuzzy_match", "must_include"]

# 定义 WebArena 站点信息的数据模型
class WebArenaSiteInfo(BaseModel):
    base_url: str
    available: bool = True
    additional_info: str = ""
    unavailable_reason: str = ""

# 从环境变量中获取 Git 用户名和密码
_git_user, _git_password = os.getenv("WEBARENA_GIT_CREDENTIALS", ":").split(":")

# 创建 WebArena 站点信息的字典
site_info_map: dict[WebArenaSite, WebArenaSiteInfo] = {
    "gitlab": WebArenaSiteInfo(
        base_url="http://git.junglegym.ai",
        available=bool(_git_user and _git_password),
        additional_info=(
            f"To log in, use the username '{_git_user}' and password '{_git_password}'."
        ),
        unavailable_reason=(
            "WEBARENA_GIT_CREDENTIALS not set (correctly): "
            f"'{os.getenv('WEBARENA_GIT_CREDENTIALS', '')}', "
            "should be USERNAME:PASSWORD."
        ),
    ),
    "map": WebArenaSiteInfo(
        base_url="http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000/"
    ),
    "reddit": WebArenaSiteInfo(base_url="http://forum.junglegym.ai"),
    "shopping": WebArenaSiteInfo(base_url="http://shop.junglegym.ai"),
    "shopping_admin": WebArenaSiteInfo(
        base_url="http://cms.junglegym.ai/admin",
        additional_info="To log in, use the username 'admin' and password 'admin1234'.",
    ),
    "wikipedia": WebArenaSiteInfo(base_url="http://wiki.junglegym.ai"),
}
# 获取网站的 URL 地址
def get_site_url(site: WebArenaSite) -> str:
    # 如果网站不在 site_info_map 中，则抛出异常
    if site not in site_info_map:
        raise ValueError(f"JungleGym site '{site}' unknown, cannot resolve URL")
    # 返回对应网站的基本 URL 地址
    return site_info_map[site].base_url


# 解析 URI，将带有模拟主机的 URI，如 `__WIKI__/wiki/Octopus`，替换为对应的 JungleGym 站点镜像主机
def resolve_uri(uri: str) -> str:
    """
    Resolves URIs with mock hosts, like `__WIKI__/wiki/Octopus`, with the corresponding
    JungleGym site mirror host.
    """
    # 拆分 URI，以 "__" 为分隔符
    segments = uri.split("__")
    # 如果拆分后的段数大于 2，并且 site 在 site_info_map 中
    if len(segments) > 2 and (site := segments[1]).lower() in site_info_map:
        # 替换 URI 中的 "__site__" 为对应网站的 URL 地址
        return uri.replace(f"__{site}__", get_site_url(site.lower()))  # type: ignore
    return uri


# 定义抽象类 Eval
class Eval(ABC):
    @abstractmethod
    def evaluate(self, string: str) -> bool:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...


# 继承 BaseModel 和 Eval 类的 StringEval 类
class StringEval(BaseModel, Eval):
    type: ReferenceAnswerType


# 继承 StringEval 类的 ExactStringMatchEval 类
class ExactStringMatchEval(StringEval):
    type: Literal["exact_match"] = "exact_match"
    reference_answer: str

    @property
    def description(self) -> str:
        return f"Answer must be '{self.reference_answer}'"

    def evaluate(self, string: str) -> bool:
        return string == self.reference_answer


# 继承 StringEval 类的 FuzzyStringMatchEval 类
class FuzzyStringMatchEval(StringEval):
    type: Literal["fuzzy_match"] = "fuzzy_match"
    reference_answer: str

    @property
    def description(self) -> str:
        return f"Answer must contain something like '{self.reference_answer}'"

    def evaluate(self, string: str) -> bool:
        # TODO: use LLM for matching (or something else that's flexible/robust)
        return self.reference_answer.lower() in string.lower()


# 继承 StringEval 类的 MustIncludeStringEval 类
class MustIncludeStringEval(StringEval):
    type: Literal["must_include"] = "must_include"
    reference_answer: str

    @property
    def description(self) -> str:
        return f"Answer must include '{self.reference_answer}'"

    def evaluate(self, string: str) -> bool:
        return self.reference_answer.lower() in string.lower()


# 继承 BaseModel 和 Eval 类的 UrlMatchEval 类
class UrlMatchEval(BaseModel, Eval):
    url: str
    """Example: `"__WIKI__/wiki/Octopus"`"""
    # 定义一个描述属性，返回一个字符串，描述代理必须导航到的 URL
    @property
    def description(self) -> str:
        return f"Agent must navigate to '{self.url}'"

    # 定义一个评估方法，接受一个 URL 参数，返回一个布尔值
    def evaluate(self, url: str) -> bool:
        # 检查传入的 URL 是否等于指定的 URL，使用 resolve_uri 函数解析 URL
        return url == resolve_uri(self.url)
# 定义一个类 ProgramHtmlEval，继承自BaseModel，用于表示一个需要评估的网页程序
class ProgramHtmlEval(BaseModel):
    # 网页的URL
    url: str
    # 定位器，用于定位需要检查的元素
    locator: str
    # 需要检查的内容，是一个 JavaScript 代码，用于返回需要检查的值
    """JavaScript code that returns the value to check"""
    required_contents: str

    # 返回描述信息，描述需要检查的内容
    @property
    def description(self) -> str:
        return (
            f"On the webpage {self.url}, "
            f"`{self.locator}` should contain '{self.required_contents}'"
        )

    # 评估函数，用于判断网页中是否包含指定内容
    def evaluate(self, selenium_instance) -> bool:
        # 执行 JavaScript 代码，获取需要检查的值
        result = selenium_instance.execute_script(
            self.locator or "return document.body.innerHTML;"
        )
        # 判断需要检查的内容是否在获取的值中
        return self.required_contents in result


# 定义一个类型别名 _Eval，包含了三种评估方式：StringEval、UrlMatchEval、ProgramHtmlEval
_Eval = StringEval | UrlMatchEval | ProgramHtmlEval


# 定义一个类 WebArenaChallengeSpec，继承自BaseModel，用于表示 Web 竞技场挑战的规范
class WebArenaChallengeSpec(BaseModel):
    # 任务ID
    task_id: int
    # 需要完成任务的网站列表
    sites: list[WebArenaSite]
    """The sites needed to complete the task"""
    # 开始的完整 URL
    start_url: str
    """The full URL at which to start"""
    # JungleGym 站点的基本 URL
    start_url_junglegym: str
    """The JungleGym site (base URL) at which to start"""
    # 是否需要登录
    require_login: bool
    # 是否需要重置
    require_reset: bool
    # 存储状态
    storage_state: str | None

    # 意图
    intent: str
    # 意图模板
    intent_template: str
    # 意图模板ID
    intent_template_id: int
    # 实例化字典
    instantiation_dict: dict[str, str | list[str]]

    # 评估集合，用于评估代理的表现
    eval: EvalSet
    """Evaluation criteria by which to judge the agent's performance"""

    # 返回代理的任务描述
    @property
    def assignment_for_agent(self):
        # 获取所有网站的 URL
        sites = [get_site_url(s) for s in self.sites]
        # 导航约束
        nav_constraint = (
            f"You are ONLY allowed to access URLs in {' and '.join(sites)}."
        )

        return (
            f"First of all, go to {self.start_url}. "
            f"{self.intent.rstrip('.')}.\n"
            f"{nav_constraint}"
        )


# 定义一个类 WebArenaChallenge，继承自 BaseChallenge，用于表示 Web 竞技场挑战
class WebArenaChallenge(BaseChallenge):
    # 规范
    _spec: ClassVar[WebArenaChallengeSpec]

    # 源 URI 前缀
    SOURCE_URI_PREFIX = "__JUNGLEGYM__/webarena/tasks/"
    # 源 URI 模板
    SOURCE_URI_TEMPLATE = f"{SOURCE_URI_PREFIX}{{task_id}}"

    @classmethod
    # 从给定的源 URI 创建 WebArenaChallenge 实例
    def from_source_uri(cls, source_uri: str) -> type["WebArenaChallenge"]:
        # 检查源 URI 是否以指定前缀开头，如果不是则抛出数值错误
        if not source_uri.startswith(cls.SOURCE_URI_PREFIX):
            raise ValueError(f"Invalid source_uri for WebArenaChallenge: {source_uri}")

        # 替换源 URI 的前缀，获取完整的源 URL
        source_url = source_uri.replace(
            cls.SOURCE_URI_PREFIX,
            "https://api.junglegym.ai/get_webarena_by_task_id?task_id=",
        )
        # 发送 GET 请求获取数据，并将结果转换为 JSON 格式
        results = requests.get(source_url).json()["data"]
        # 如果结果为空，则抛出数值错误
        if not results:
            raise ValueError(f"Could not fetch challenge {source_uri}")
        # 根据获取的结果创建 WebArenaChallenge 实例
        return cls.from_challenge_spec(WebArenaChallengeSpec.parse_obj(results[0]))

    # 根据给定的 WebArenaChallengeSpec 创建 WebArenaChallenge 实例
    @classmethod
    def from_challenge_spec(
        cls, spec: WebArenaChallengeSpec
    ) -> type["WebArenaChallenge"]:
        # 创建挑战信息对象
        challenge_info = ChallengeInfo(
            eval_id=f"junglegym-webarena-{spec.task_id}",
            name=f"WebArenaTask_{spec.task_id}",
            task=spec.assignment_for_agent,
            category=[
                Category.GENERALIST,
                Category.WEB,
            ],  # TODO: make categories more specific
            reference_answer=spec.eval.reference_answer_raw_annotation,
            source_uri=cls.SOURCE_URI_TEMPLATE.format(task_id=spec.task_id),
        )
        # 创建 WebArenaChallenge 类型实例
        return type(
            f"Test{challenge_info.name}",
            (WebArenaChallenge,),
            {
                "info": challenge_info,
                "_spec": spec,
            },
        )

    # 类方法，用于创建 WebArenaChallenge 实例
    @classmethod
    # 评估答案的方法，返回评估结果列表
    def evaluate_answer(cls, answer: str) -> list[tuple[_Eval, EvalResult]]:
        # 初始化结果列表
        results: list[tuple[_Eval, EvalResult]] = []
        # 遍历评估器列表
        for evaluator in cls._spec.eval.evaluators:
            # 如果评估器是 StringEval 类型
            if isinstance(evaluator, StringEval):  # string_match
                # 将评估结果添加到结果列表中
                results.append(
                    (
                        evaluator,
                        EvalResult(
                            result=answer,
                            result_source="step_output",
                            score=evaluator.evaluate(answer),
                            passed=evaluator.evaluate(answer),
                        ),
                    )
                )
        # 返回结果列表
        return results

    # 评估步骤结果的方法，返回评估结果列表
    @classmethod
    def evaluate_step_result(cls, step: Step) -> list[tuple[_Eval, EvalResult]]:
        # 断言步骤有输出
        assert step.output
        # 评估答案并获取评估结果列表
        eval_results = cls.evaluate_answer(step.output)
        # 遍历评估器列表
        for eval in cls._spec.eval.evaluators:
            # 如果评估器是 UrlMatchEval 类型
            if isinstance(eval, UrlMatchEval):
                # 判断 URL 是否在步骤输出中，生成 passed 变量
                passed = resolve_uri(eval.url) in step.output  # HACK: url_match bodge
                # 将评估结果添加到结果列表中
                eval_results.append(
                    (
                        eval,
                        EvalResult(
                            result=step.output,
                            result_source="step_output",
                            score=1.0 if passed else 0.0,
                            passed=passed,
                        ),
                    )
                )
            # TODO: add support for program_html evals
        # 返回结果列表
        return eval_results

    # 评估任务状态的异步方法
    @classmethod
    async def evaluate_task_state(
        cls, agent: AgentApi, task_id: str
    # 定义一个方法，接收一个任务ID并返回一个评估结果列表
    ) -> list[EvalResult]:
        # 获取任务的步骤列表
        steps: list[Step] = (await agent.list_agent_task_steps(task_id)).steps

        # 对每个步骤的结果进行评估
        eval_results_per_step = [cls.evaluate_step_result(step) for step in steps]
        
        # 从每个步骤的评估结果矩阵中获取每个评估的最高分的 EvalResult
        return [
            max(step_results_for_eval, key=lambda r: r[1].score)[1]
            for step_results_for_eval in zip(*eval_results_per_step)
        ]

    # 标记为异步测试方法
    @pytest.mark.asyncio
    async def test_method(
        self,
        config: AgentBenchmarkConfig,
        request: pytest.FixtureRequest,
        i_attempt: int,
# 加载 WebArena 挑战，返回挑战类型的迭代器
def load_webarena_challenges() -> Iterator[type[WebArenaChallenge]]:
    # 记录日志信息，表示正在加载 WebArena 挑战
    logger.info("Loading WebArena challenges...")

    # 遍历站点信息映射中的每个站点及其信息
    for site, info in site_info_map.items():
        # 如果站点不可用
        if not info.available:
            # 记录警告日志，表示 JungleGym 站点不可用，跳过使用该站点的所有挑战
            logger.warning(
                f"JungleGym site '{site}' is not available: {info.unavailable_reason} "
                "Skipping all challenges which use this site."
            )

    # 从指定 URL 获取 WebArena 数据集
    # response = requests.get("https://api.junglegym.ai/get_full_webarena_dataset")
    # challenge_dicts = response.json()["data"]

    # 当完整的 WebArena 挑战集不受支持时，使用手动选择的挑战
    import json
    from pathlib import Path

    # 从本地文件中加载挑战数据集
    challenge_dicts = json.loads(
        (Path(__file__).parent / "webarena_selection.json").read_bytes()
    )

    # 记录调试日志，表示已获取 WebArena 数据集，正在构建指定数量的 WebArenaChallenges
    logger.debug(
        "Fetched WebArena dataset. "
        f"Constructing {len(challenge_dicts)} WebArenaChallenges..."
    )
    # 初始化加载成功、加载失败和跳过的挑战数量
    loaded = 0
    failed = 0
    skipped = 0
    # 遍历挑战字典列表中的每个挑战
    for entry in challenge_dicts:
        try:
            # 尝试解析挑战规范对象
            challenge_spec = WebArenaChallengeSpec.parse_obj(entry)
            # 遍历挑战规范对象中的站点列表
            for site in challenge_spec.sites:
                # 获取站点信息
                site_info = site_info_map.get(site)
                # 如果站点信息为空
                if site_info is None:
                    # 记录警告日志，跳过当前站点
                    logger.warning(
                        f"WebArena task {challenge_spec.task_id} requires unknown site "
                        f"'{site}'; skipping..."
                    )
                    break
                # 如果站点不可用
                if not site_info.available:
                    # 记录调试日志，跳过当前站点
                    logger.debug(
                        f"WebArena task {challenge_spec.task_id} requires unavailable "
                        f"site '{site}'; skipping..."
                    )
                    break
            else:
                # 如果所有站点都可用，生成 WebArena 挑战对象并返回
                yield WebArenaChallenge.from_challenge_spec(challenge_spec)
                loaded += 1
                continue
            # 如果出现验证错误，记录失败次数并输出错误信息
            skipped += 1
        except ValidationError as e:
            failed += 1
            logger.warning(f"Error validating WebArena challenge entry: {entry}")
            logger.warning(f"Error details: {e}")
    # 记录信息日志，显示加载挑战的结果
    logger.info(
        "Loading WebArena challenges complete: "
        f"loaded {loaded}, skipped {skipped}. {failed} challenge failed to load."
    )
```