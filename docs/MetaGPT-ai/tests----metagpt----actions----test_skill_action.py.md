# `MetaGPT\tests\metagpt\actions\test_skill_action.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/19
@Author  : mashenquan
@File    : test_skill_action.py
@Desc    : Unit tests.
"""
# 导入 pytest 模块
import pytest

# 导入需要测试的类和方法
from metagpt.actions.skill_action import ArgumentsParingAction, SkillAction
from metagpt.learn.skill_loader import Example, Parameter, Returns, Skill

# 定义测试类
class TestSkillAction:
    # 定义一个技能对象
    skill = Skill(
        name="text_to_image",
        description="Create a drawing based on the text.",
        id="text_to_image.text_to_image",
        x_prerequisite={
            "configurations": {
                "OPENAI_API_KEY": {
                    "type": "string",
                    "description": "OpenAI API key, For more details, checkout: `https://platform.openai.com/account/api-keys`",
                },
                "METAGPT_TEXT_TO_IMAGE_MODEL_URL": {"type": "string", "description": "Model url."},
            },
            "required": {"oneOf": ["OPENAI_API_KEY", "METAGPT_TEXT_TO_IMAGE_MODEL_URL"]},
        },
        parameters={
            "text": Parameter(type="string", description="The text used for image conversion."),
            "size_type": Parameter(type="string", description="size type"),
        },
        examples=[
            Example(ask="Draw a girl", answer='text_to_image(text="Draw a girl", size_type="512x512")'),
            Example(ask="Draw an apple", answer='text_to_image(text="Draw an apple", size_type="512x512")'),
        ],
        returns=Returns(type="string", format="base64"),
    )

    # 测试参数解析方法
    @pytest.mark.asyncio
    async def test_parser(self):
        args = ArgumentsParingAction.parse_arguments(
            skill_name="text_to_image", txt='`text_to_image(text="Draw an apple", size_type="512x512")`'
        )
        assert args.get("text") == "Draw an apple"
        assert args.get("size_type") == "512x512"

    # 测试参数解析动作方法
    @pytest.mark.asyncio
    async def test_parser_action(self):
        parser_action = ArgumentsParingAction(skill=self.skill, ask="Draw an apple")
        rsp = await parser_action.run()
        assert rsp
        assert parser_action.args
        assert parser_action.args.get("text") == "Draw an apple"
        assert parser_action.args.get("size_type") == "512x512"

        action = SkillAction(skill=self.skill, args=parser_action.args)
        rsp = await action.run()
        assert rsp
        assert "image/png;base64," in rsp.content or "http" in rsp.content

    # 参数化测试方法
    @pytest.mark.parametrize(
        ("skill_name", "txt", "want"),
        [
            ("skill1", 'skill1(a="1", b="2")', {"a": "1", "b": "2"}),
            ("skill1", '(a="1", b="2")', None),
            ("skill1", 'skill1(a="1", b="2"', None),
        ],
    )
    def test_parse_arguments(self, skill_name, txt, want):
        args = ArgumentsParingAction.parse_arguments(skill_name, txt)
        assert args == want

    # 测试查找和调用函数出错的方法
    @pytest.mark.asyncio
    async def test_find_and_call_function_error(self):
        with pytest.raises(ValueError):
            await SkillAction.find_and_call_function("dummy_call", {"a": 1})

    # 测试技能动作出错的方法
    @pytest.mark.asyncio
    async def test_skill_action_error(self):
        action = SkillAction(skill=self.skill, args={})
        await action.run()


if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```