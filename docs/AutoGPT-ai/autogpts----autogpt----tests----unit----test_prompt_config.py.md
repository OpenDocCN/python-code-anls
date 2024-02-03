# `.\AutoGPT\autogpts\autogpt\tests\unit\test_prompt_config.py`

```py
# 导入 AIDirectives 类，该类用于处理 AI 指令的配置
from autogpt.config.ai_directives import AIDirectives

# 测试 PromptConfig 类的测试用例，该类负责从 YAML 文件加载提示配置设置
def test_prompt_config_loading(tmp_path):
    # 测试提示配置是否正确加载

    # 定义 YAML 内容
    yaml_content = """
    constraints:
    - A test constraint
    - Another test constraint
    - A third test constraint
    resources:
    - A test resource
    - Another test resource
    - A third test resource
    best_practices:
    - A test best-practice
    - Another test best-practice
    - A third test best-practice
    """
    # 创建临时文件路径并写入 YAML 内容
    prompt_settings_file = tmp_path / "test_prompt_settings.yaml"
    prompt_settings_file.write_text(yaml_content)

    # 从文件中加载 AIDirectives 对象
    prompt_config = AIDirectives.from_file(prompt_settings_file)

    # 断言提示配置中约束的数量为 3
    assert len(prompt_config.constraints) == 3
    # 断言第一个约束的内容
    assert prompt_config.constraints[0] == "A test constraint"
    # 断言第二个约束的内容
    assert prompt_config.constraints[1] == "Another test constraint"
    # 断言第三个约束的内容
    assert prompt_config.constraints[2] == "A third test constraint"
    # 断言资源的数量为 3
    assert len(prompt_config.resources) == 3
    # 断言第一个资源的内容
    assert prompt_config.resources[0] == "A test resource"
    # 断言第二个资源的内容
    assert prompt_config.resources[1] == "Another test resource"
    # 断言第三个资源的内容
    assert prompt_config.resources[2] == "A third test resource"
    # 断言最佳实践的数量为 3
    assert len(prompt_config.best_practices) == 3
    # 断言第一个最佳实践的内容
    assert prompt_config.best_practices[0] == "A test best-practice"
    # 断言第二个最佳实践的内容
    assert prompt_config.best_practices[1] == "Another test best-practice"
    # 断言第三个最佳实践的内容
    assert prompt_config.best_practices[2] == "A third test best-practice"
```