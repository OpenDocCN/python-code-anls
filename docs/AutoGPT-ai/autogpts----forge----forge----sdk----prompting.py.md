# `.\AutoGPT\autogpts\forge\forge\sdk\prompting.py`

```py
# 导入所需的模块
import glob
import os
from difflib import get_close_matches
from typing import List

# 导入 Jinja2 相关模块
from jinja2 import Environment, FileSystemLoader

# 导入自定义的 ForgeLogger 类
from .forge_log import ForgeLogger

# 创建一个 ForgeLogger 实例，用于记录日志
LOG = ForgeLogger(__name__)

# 定义 PromptEngine 类，用于处理加载和填充提示的 Jinja2 模板
class PromptEngine:
    """
    Class to handle loading and populating Jinja2 templates for prompts.
    """
    # 初始化 PromptEngine 类，指定模型和是否启用调试日志
    def __init__(self, model: str, debug_enabled: bool = False):
        """
        Initialize the PromptEngine with the specified model.

        Args:
            model (str): The model to use for loading prompts.
            debug_enabled (bool): Enable or disable debug logging.
        """
        # 设置模型和调试日志是否启用
        self.model = model
        self.debug_enabled = debug_enabled
        # 如果启用调试日志，则记录初始化信息
        if self.debug_enabled:
            LOG.debug(f"Initializing PromptEngine for model: {model}")

        try:
            # 获取所有模型目录的列表
            models_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../prompts")
            )
            # 获取模型目录下的所有子目录名称作为模型名称列表
            model_names = [
                os.path.basename(os.path.normpath(d))
                for d in glob.glob(os.path.join(models_dir, "*/"))
                if os.path.isdir(d) and "techniques" not in d
            ]

            # 获取与指定模型最接近的模型名称
            self.model = self.get_closest_match(self.model, model_names)

            # 如果启用调试日志，则记录使用的最接近匹配的模型
            if self.debug_enabled:
                LOG.debug(f"Using the closest match model for prompts: {self.model}")

            # 设置模板环境的加载路径为模型目录
            self.env = Environment(loader=FileSystemLoader(models_dir))
        except Exception as e:
            # 如果出现异常，则记录错误信息并抛出异常
            LOG.error(f"Error initializing Environment: {e}")
            raise

    @staticmethod
    def get_closest_match(target: str, model_dirs: List[str]) -> str:
        """
        Find the closest match to the target in the list of model directories.

        Args:
            target (str): The target model.
            model_dirs (list): The list of available model directories.

        Returns:
            str: The closest match to the target.
        """
        try:
            # 获取与目标最接近的匹配项
            matches = get_close_matches(target, model_dirs, n=1, cutoff=0.1)
            if matches:
                # 将匹配项转换为字符串并打印调试信息
                matches_str = ", ".join(matches)
                LOG.debug(matches_str)
            # 遍历匹配项并打印信息
            for m in matches:
                LOG.info(m)
            # 返回第一个匹配项
            return matches[0]
        except Exception as e:
            # 打印错误信息并抛出异常
            LOG.error(f"Error finding closest match: {e}")
            raise

    def load_prompt(self, template: str, **kwargs) -> str:
        """
        Load and populate the specified template.

        Args:
            template (str): The name of the template to load.
            **kwargs: The arguments to populate the template with.

        Returns:
            str: The populated template.
        """
        try:
            # 拼接模板路径
            template = os.path.join(self.model, template)
            if self.debug_enabled:
                # 如果调试模式开启，打印加载模板信息
                LOG.debug(f"Loading template: {template}")
            # 加载模板文件
            template = self.env.get_template(f"{template}.j2")
            if self.debug_enabled:
                # 如果调试模式开启，打印渲染模板信息和参数
                LOG.debug(f"Rendering template: {template} with args: {kwargs}")
            # 渲染模板并返回结果
            return template.render(**kwargs)
        except Exception as e:
            # 打印错误信息并抛出异常
            LOG.error(f"Error loading or rendering template: {e}")
            raise
```