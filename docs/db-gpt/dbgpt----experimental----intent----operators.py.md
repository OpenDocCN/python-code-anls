# `.\DB-GPT-src\dbgpt\experimental\intent\operators.py`

```py
# 导入所需模块和类型声明
from typing import Dict, List, Optional, cast

from dbgpt.core import ModelMessage, ModelRequest, ModelRequestContext
from dbgpt.core.awel import BranchFunc, BranchOperator, BranchTaskType, MapOperator
from dbgpt.model.operators.llm_operator import MixinLLMOperator

# 导入基类和响应类
from .base import BaseIntentDetection, IntentDetectionResponse

# 定义意图检测操作符类，继承自MixinLLMOperator、BaseIntentDetection和MapOperator
class IntentDetectionOperator(
    MixinLLMOperator, BaseIntentDetection, MapOperator[ModelRequest, ModelRequest]
):
    """The intent detection operator."""

    # 初始化方法，接收意图定义、提示模板、响应格式和示例等参数
    def __init__(
        self,
        intent_definitions: str,
        prompt_template: Optional[str] = None,
        response_format: Optional[str] = None,
        examples: Optional[str] = None,
        **kwargs
    ):
        """Create the intent detection operator."""
        # 调用父类初始化方法
        MixinLLMOperator.__init__(self)
        MapOperator.__init__(self, **kwargs)
        BaseIntentDetection.__init__(
            self,
            intent_definitions=intent_definitions,
            prompt_template=prompt_template,
            response_format=response_format,
            examples=examples,
        )

    # 异步方法，将意图检测结果合并到上下文中
    async def map(self, input_value: ModelRequest) -> ModelRequest:
        """Detect the intent.

        Merge the intent detection result into the context.
        """
        # 默认语言为英语
        language = "en"
        # 如果存在系统应用，获取当前语言配置
        if self.system_app:
            language = self.system_app.config.get_current_lang()
        # 解析输入值中的消息
        messages = self.parse_messages(input_value)
        # 检测意图
        ic = await self.detect_intent(
            messages,
            input_value.model,
            language=language,
        )
        # 如果输入值的上下文为空，创建一个新的上下文对象
        if not input_value.context:
            input_value.context = ModelRequestContext()
        # 如果输入值的额外信息为空，创建一个空字典
        if not input_value.context.extra:
            input_value.context.extra = {}
        # 将意图检测结果存储到输入值的额外信息中的intent_detection字段
        input_value.context.extra["intent_detection"] = ic
        # 返回处理后的输入值
        return input_value

    # 解析请求中的消息
    def parse_messages(self, request: ModelRequest) -> List[ModelMessage]:
        """Parse the messages from the request."""
        return request.get_messages()


# 定义意图检测分支操作符类，继承自BranchOperator，操作ModelRequest到ModelRequest的映射
class IntentDetectionBranchOperator(BranchOperator[ModelRequest, ModelRequest]):
    """The intent detection branch operator."""

    # 初始化方法，接收结束任务名称和其他关键字参数
    def __init__(self, end_task_name: str, **kwargs):
        """Create the intent detection branch operator."""
        # 调用父类初始化方法
        super().__init__(**kwargs)
        # 设置结束任务名称属性
        self._end_task_name = end_task_name

    # 异步方法，定义分支逻辑
    async def branches(
        self,
        ) -> Dict[BranchFunc[ModelRequest], BranchTaskType]:
        """Branch the intent detection result to different tasks."""
        # 获取所有下游任务的名称集合
        download_task_names = set(task.node_name for task in self.downstream)  # noqa
        # 创建一个空的分支函数映射字典
        branch_func_map = {}
        # 遍历每个任务名称
        for task_name in download_task_names:

            # 定义一个检查函数，用于判断是否分支到当前任务
            def check(r: ModelRequest, outer_task_name=task_name):
                # 如果请求上下文为空或额外信息为空，则返回 False
                if not r.context or not r.context.extra:
                    return False
                # 获取意图检测结果
                ic_result = r.context.extra.get("intent_detection")
                if not ic_result:
                    return False
                # 将意图检测结果转换为 IntentDetectionResponse 类型
                ic: IntentDetectionResponse = cast(IntentDetectionResponse, ic_result)
                # 如果意图检测结果中存在空槽位，则判断是否结束当前任务
                if ic.has_empty_slot():
                    return self._end_task_name == outer_task_name
                else:
                    # 否则判断是否分支到当前任务
                    return outer_task_name == ic.task_name

            # 将检查函数和任务名称添加到分支函数映射字典中
            branch_func_map[check] = task_name

        # 返回分支函数映射字典
        return branch_func_map  # type: ignore
```