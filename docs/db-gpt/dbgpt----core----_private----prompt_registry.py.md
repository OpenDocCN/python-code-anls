# `.\DB-GPT-src\dbgpt\core\_private\prompt_registry.py`

```py
"""
Prompt template registry.

This module is deprecated. we will remove it in the future.
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
from typing import Dict, List, Optional

_DEFAULT_MODEL_KEY = "___default_prompt_template_model_key__"
_DEFUALT_LANGUAGE_KEY = "___default_prompt_template_language_key__"


class PromptTemplateRegistry:
    """
    The PromptTemplateRegistry class is a manager of prompt template of all scenes.
    """

    def __init__(self) -> None:
        self.registry = defaultdict(dict)  # type: ignore

    def register(
        self,
        prompt_template,
        language: str = "en",
        is_default: bool = False,
        model_names: Optional[List[str]] = None,
        scene_name: Optional[str] = None,
    ) -> None:
        """
        Register prompt template with scene name, language
        registry dict format:
        {
            "<scene_name>": {
                _DEFAULT_MODEL_KEY: {
                    _DEFUALT_LANGUAGE_KEY: <prompt_template>,
                    "<language>": <prompt_template>
                },
                "<model_name>": {
                    "<language>": <prompt_template>
                }
            }
        }
        """
        # If scene_name is not provided, use prompt_template's template_scene
        if not scene_name:
            scene_name = prompt_template.template_scene
        # Raise an error if scene_name is empty
        if not scene_name:
            raise ValueError("Prompt template scene name cannot be empty")
        # If model_names is not provided, use _DEFAULT_MODEL_KEY
        if not model_names:
            model_names = [_DEFAULT_MODEL_KEY]
        # Access or initialize the registry entry for the scene_name
        scene_registry = self.registry[scene_name]
        # Call helper function to register prompt_template in the scene_registry
        _register_scene_prompt_template(
            scene_registry, prompt_template, language, model_names
        )
        # If is_default flag is set, also register under _DEFAULT_MODEL_KEY
        if is_default:
            _register_scene_prompt_template(
                scene_registry,
                prompt_template,
                _DEFUALT_LANGUAGE_KEY,
                [_DEFAULT_MODEL_KEY],
            )
            _register_scene_prompt_template(
                scene_registry, prompt_template, language, [_DEFAULT_MODEL_KEY]
            )

    def get_prompt_template(
        self,
        scene_name: str,
        language: str,
        model_name: str,
        proxyllm_backend: Optional[str] = None,
    ) -> Optional[str]:
        # Implementation of get_prompt_template method is missing
        pass


def _register_scene_prompt_template(
    scene_registry: dict,
    prompt_template,
    language: str,
    model_names: List[str],
) -> None:
    """
    Helper function to register prompt_template in the scene_registry
    """
    pass
    ):
        """获取带有场景名称、语言和模型名称的提示模板
        proxyllm_backend: 查看CFG.PROXYLLM_BACKEND
        """
        # 获取场景名称对应的注册表条目
        scene_registry = self.registry[scene_name]

        # 打印提示信息，显示正在获取场景名称、模型名称、代理LLM后端和语言的提示模板
        print(
            f"获取场景名称为: {scene_name} 的提示模板，模型名称为: {model_name}，代理LLM后端为: {proxyllm_backend}，语言为: {language}"
        )
        
        # 初始化注册表为None
        registry = None

        # 如果proxyllm_backend存在，则尝试从场景注册表中获取对应条目
        if proxyllm_backend:
            registry = scene_registry.get(proxyllm_backend)
        
        # 如果未找到对应条目，则尝试从场景注册表中获取模型名称对应的条目
        if not registry:
            registry = scene_registry.get(model_name)
        
        # 如果还未找到对应条目，则尝试从场景注册表中获取默认模型名称对应的条目
        if not registry:
            registry = scene_registry.get(_DEFAULT_MODEL_KEY)
            
            # 如果仍未找到对应条目，则抛出数值错误异常
            if not registry:
                raise ValueError(
                    f"场景名称为 {scene_name}，模型名称为 {model_name}，语言为 {language} 的模板不存在"
                )
        else:
            # 否则，打印提示信息，显示场景名称有自定义模型名称和语言的提示模板
            print(
                f"场景: {scene_name} 具有模型: {model_name}，语言: {language} 的自定义提示模板"
            )
        
        # 获取语言对应的提示模板
        prompt_template = registry.get(language)
        
        # 如果未找到对应语言的提示模板，则尝试获取默认语言对应的提示模板
        if not prompt_template:
            prompt_template = registry.get(_DEFUALT_LANGUAGE_KEY)
        
        # 返回获取到的提示模板
        return prompt_template
# 注册场景提示模板到场景注册表中
def _register_scene_prompt_template(
    # 场景注册表，字典类型，键为字符串，值为字典
    scene_registry: Dict[str, Dict],
    # 提示模板，可以是任意类型的输入
    prompt_template,
    # 语言名称，字符串类型
    language: str,
    # 模型名称列表，每个元素是字符串类型
    model_names: List[str],
):
    # 遍历模型名称列表
    for model_name in model_names:
        # 如果模型名称不在场景注册表中，添加空字典作为其值
        if model_name not in scene_registry:
            scene_registry[model_name] = dict()
        # 获取当前模型名称对应的注册表
        registry = scene_registry[model_name]
        # 将语言名称作为键，提示模板作为值，存入注册表中
        registry[language] = prompt_template
```