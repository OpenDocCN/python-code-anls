# AutoGPT源码解析 7

# `autogpts/autogpt/autogpt/config/ai_profile.py`

This is a class called `AIProfile` that uses a config file (typically stored as a YAML file) to define an AI with a name, role, and goals. It also includes an API budget, which is initially set to 0.0.

The class has two methods for loading and saving settings: `load` and `save`. The `load` method reads the AI settings from the specified config file and returns an instance of the `AIProfile` class with the defined parameters. The `save` method takes an example `AIProfile` object and saves its settings to the specified config file.

Note that the class assumes that the config file contains a list of goals in the format `{goal: str}`, which is then converted to a list of strings for easier handling in the `AIProfile` class.


```py
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AIProfile(BaseModel):
    """
    Object to hold the AI's personality.

    Attributes:
        ai_name (str): The name of the AI.
        ai_role (str): The description of the AI's role.
        ai_goals (list): The list of objectives the AI is supposed to complete.
        api_budget (float): The maximum dollar value for API calls (0.0 means infinite)
    """

    ai_name: str = ""
    ai_role: str = ""
    ai_goals: list[str] = Field(default_factory=list[str])
    api_budget: float = 0.0

    @staticmethod
    def load(ai_settings_file: str | Path) -> "AIProfile":
        """
        Returns class object with parameters (ai_name, ai_role, ai_goals, api_budget)
        loaded from yaml file if yaml file exists, else returns class with no parameters.

        Parameters:
            ai_settings_file (Path): The path to the config yaml file.

        Returns:
            cls (object): An instance of given cls object
        """

        try:
            with open(ai_settings_file, encoding="utf-8") as file:
                config_params = yaml.load(file, Loader=yaml.FullLoader) or {}
        except FileNotFoundError:
            config_params = {}

        ai_name = config_params.get("ai_name", "")
        ai_role = config_params.get("ai_role", "")
        ai_goals = [
            str(goal).strip("{}").replace("'", "").replace('"', "")
            if isinstance(goal, dict)
            else str(goal)
            for goal in config_params.get("ai_goals", [])
        ]
        api_budget = config_params.get("api_budget", 0.0)

        return AIProfile(
            ai_name=ai_name, ai_role=ai_role, ai_goals=ai_goals, api_budget=api_budget
        )

    def save(self, ai_settings_file: str | Path) -> None:
        """
        Saves the class parameters to the specified file yaml file path as a yaml file.

        Parameters:
            ai_settings_file (Path): The path to the config yaml file.

        Returns:
            None
        """

        with open(ai_settings_file, "w", encoding="utf-8") as file:
            yaml.dump(self.dict(), file, allow_unicode=True)

```

# `autogpts/autogpt/autogpt/config/config.py`

这是一个名为`Configuration`的类，用于存储不同脚本访问时布尔状态的配置信息。它从`__future__`导入了一个`annotations`注解，用于标记这个类的来源。

接着，它导入了`contextlib`、`os`、`re`、`pathlib.Path`、`typing.Any`、`typing.Dict`、`typing.Optional`和`typing.Union`。

然后，它从`typing.Union`中定义了一个名为`True`的类，用于表示布尔值的多种类型。

接下来，它定义了一个`from typing import Any, Dict, Optional, Union`，用于定义输入参数的类型。

接着，它定义了一个名为`Configuration`的类，它包含一个名为`configuration`的属性，用于存储布尔状态的配置信息。

然后，它定义了一个名为`AutoGPTPluginTemplate`的类，它继承了自定义的`AutoGPTPluginTemplate`类，用于自动生成GPT的模板。

接着，它定义了一个名为`Colorama`的类，它使用了`colorama`库，用于打印出颜色标记的输出信息。

然后，它定义了一个名为`Fore`的类，它使用了`colorama`库，用于打印出颜色标记的输出信息。

接着，它定义了一个名为`PathLike`的类，它使用了`pathlib`库，用于处理文件路径类似于`!`的情况。

接着，它定义了一个名为`Re``class` like Regex for file and directory paths.`Re`类用于处理文件和目录路径，类似于`re`库中的`re.Pattern`类。

接着，它定义了一个名为`Path`的类，它使用了`pathlib`库，用于处理文件和目录路径。

然后，它定义了一个名为`自动生成模板`的类，它实现了`AutoGPTPluginTemplate`接口，用于生成GPT的模板。

接着，它定义了一个名为`颜色化`的函数，它接收一个`模板`参数，用于生成GPT的模板，并使用`colorama`库将其颜色标记的输出信息打印出来。

最后，它定义了一个名为`配置`的函数，它接收一个`模板`参数和一个`配置`参数，用于根据`模板`生成相应的配置信息，并将其存储在`配置`函数返回的`configuration`属性中。

综上所述，这段代码定义了一个用于存储不同脚本访问时布尔状态的配置信息的`Configuration`类，实现了`AutoGPTPluginTemplate`接口的`Colorative`自动生成模板`AutoGPTPluginTemplate`，以及定义了一系列辅助函数和类用于生成GPT的模板，并将其颜色标记的输出信息打印出来，最终实现了`Configuration`类中定义的`configuration`属性。


```py
"""Configuration class to store the state of bools for different scripts access."""
from __future__ import annotations

import contextlib
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from auto_gpt_plugin_template import AutoGPTPluginTemplate
from colorama import Fore
from pydantic import Field, validator

import autogpt
```

这段代码是一个自定义的 Python 程序，从多个文件中读取配置信息，并尝试创建或加载与人工智能相关的服务。具体来说，它包括：

1. 从 `autogpt.core.configuration.schema` 导入一个名为 `Configurable` 的类，该类定义了程序如何设置或获取配置信息。
2. 从 `autogpt.core.resource.model_providers.openai` 中导入一个名为 `OPEN_AI_CHAT_MODELS` 的类，它用于加载 OpenAI 模型。
3. 从 `autogpt.plugins.plugins_config` 中导入一个名为 `PluginsConfig` 的类，它用于加载插件的配置信息。
4. 从 `autogpt.speech` 中导入一个名为 `TTSConfig` 的类，它用于加载文本到语音的配置信息。
5. 从 `pathlib` 库中导入了几个常用的路径，如 `PROJECT_ROOT`（项目根目录）和一些配置文件。
6. 在程序的主要部分中，定义了一些变量，包括 GPT 4 和 GPT 3 的模型，以用于加载预训练的模型。


```py
from autogpt.core.configuration.schema import Configurable, SystemSettings
from autogpt.core.resource.model_providers.openai import OPEN_AI_CHAT_MODELS
from autogpt.plugins.plugins_config import PluginsConfig
from autogpt.speech import TTSConfig

PROJECT_ROOT = Path(autogpt.__file__).parent.parent
AI_SETTINGS_FILE = Path("ai_settings.yaml")
AZURE_CONFIG_FILE = Path("azure.yaml")
PLUGINS_CONFIG_FILE = Path("plugins_config.yaml")
PROMPT_SETTINGS_FILE = Path("prompt_settings.yaml")

GPT_4_MODEL = "gpt-4"
GPT_3_MODEL = "gpt-3.5-turbo"


```

This is a Python implementation of a function that builds an OpenAI model deployment. The function takes in several parameters and returns a dictionary of keyword arguments that can be passed to the `OpenAI` class to create the model deployment.

The function supports two modes for building the model deployment: fast_llm and smart_llm. The fast_llm mode uses Azure as the backing engine, while the smart_llm mode uses the Azure-暴露的GPT模型。

When using the fast_llm mode, the function checks if the model deployment already exists with the key `GPT_4_MODEL`. If it does, the function returns a tuple of the deployment ID and the name of the deployment model. If it does not exist, the function builds the deployment model and returns the deployment ID.

When using the smart_llm mode, the function checks if the model deployment already exists with the key `GPT_3_MODEL`. If it does, the function returns a tuple of the deployment ID and the name of the deployment model. If it does not exist, the function builds the deployment model and returns the deployment ID.

The function also supports a third mode for building the model deployment, which is the embedding model. In this mode, the function uses the same key `GPT_3_MODEL` as in the smart_llm mode.

The function uses the `azure-model-to-deployment-id-map` to map the Azure deployment IDs to the OpenAI deployment model keys.

This implementation is for informational purposes only and should not be used as the sole basis for building a production-ready model deployment.


```py
class Config(SystemSettings, arbitrary_types_allowed=True):
    name: str = "Auto-GPT configuration"
    description: str = "Default configuration for the Auto-GPT application."
    ########################
    # Application Settings #
    ########################
    project_root: Path = PROJECT_ROOT
    app_data_dir: Path = project_root / "data"
    skip_news: bool = False
    skip_reprompt: bool = False
    authorise_key: str = "y"
    exit_key: str = "n"
    debug_mode: bool = False
    plain_output: bool = False
    noninteractive_mode: bool = False
    chat_messages_enabled: bool = True
    # TTS configuration
    tts_config: TTSConfig = TTSConfig()

    ##########################
    # Agent Control Settings #
    ##########################
    # Paths
    ai_settings_file: Path = project_root / AI_SETTINGS_FILE
    prompt_settings_file: Path = project_root / PROMPT_SETTINGS_FILE
    # Model configuration
    fast_llm: str = "gpt-3.5-turbo-16k"
    smart_llm: str = "gpt-4-0314"
    temperature: float = 0
    openai_functions: bool = False
    embedding_model: str = "text-embedding-ada-002"
    browse_spacy_language_model: str = "en_core_web_sm"
    # Run loop configuration
    continuous_mode: bool = False
    continuous_limit: int = 0

    ##########
    # Memory #
    ##########
    memory_backend: str = "json_file"
    memory_index: str = "auto-gpt-memory"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    wipe_redis_on_start: bool = True

    ############
    # Commands #
    ############
    # General
    disabled_command_categories: list[str] = Field(default_factory=list)
    # File ops
    restrict_to_workspace: bool = True
    allow_downloads: bool = False
    # Shell commands
    shell_command_control: str = "denylist"
    execute_local_commands: bool = False
    shell_denylist: list[str] = Field(default_factory=lambda: ["sudo", "su"])
    shell_allowlist: list[str] = Field(default_factory=list)
    # Text to image
    image_provider: Optional[str] = None
    huggingface_image_model: str = "CompVis/stable-diffusion-v1-4"
    sd_webui_url: Optional[str] = "http://localhost:7860"
    image_size: int = 256
    # Audio to text
    audio_to_text_provider: str = "huggingface"
    huggingface_audio_to_text_model: Optional[str] = None
    # Web browsing
    selenium_web_browser: str = "chrome"
    selenium_headless: bool = True
    user_agent: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"

    ###################
    # Plugin Settings #
    ###################
    plugins_dir: str = "plugins"
    plugins_config_file: Path = project_root / PLUGINS_CONFIG_FILE
    plugins_config: PluginsConfig = Field(
        default_factory=lambda: PluginsConfig(plugins={})
    )
    plugins: list[AutoGPTPluginTemplate] = Field(default_factory=list, exclude=True)
    plugins_allowlist: list[str] = Field(default_factory=list)
    plugins_denylist: list[str] = Field(default_factory=list)
    plugins_openai: list[str] = Field(default_factory=list)

    ###############
    # Credentials #
    ###############
    # OpenAI
    openai_api_key: Optional[str] = None
    openai_api_type: Optional[str] = None
    openai_api_base: Optional[str] = None
    openai_api_version: Optional[str] = None
    openai_organization: Optional[str] = None
    use_azure: bool = False
    azure_config_file: Optional[Path] = project_root / AZURE_CONFIG_FILE
    azure_model_to_deployment_id_map: Optional[Dict[str, str]] = None
    # Github
    github_api_key: Optional[str] = None
    github_username: Optional[str] = None
    # Google
    google_api_key: Optional[str] = None
    google_custom_search_engine_id: Optional[str] = None
    # Huggingface
    huggingface_api_token: Optional[str] = None
    # Stable Diffusion
    sd_webui_auth: Optional[str] = None

    @validator("plugins", each_item=True)
    def validate_plugins(cls, p: AutoGPTPluginTemplate | Any):
        assert issubclass(
            p.__class__, AutoGPTPluginTemplate
        ), f"{p} does not subclass AutoGPTPluginTemplate"
        assert (
            p.__class__.__name__ != "AutoGPTPluginTemplate"
        ), f"Plugins must subclass AutoGPTPluginTemplate; {p} is a template instance"
        return p

    @validator("openai_functions")
    def validate_openai_functions(cls, v: bool, values: dict[str, Any]):
        if v:
            smart_llm = values["smart_llm"]
            assert OPEN_AI_CHAT_MODELS[smart_llm].has_function_call_api, (
                f"Model {smart_llm} does not support OpenAI Functions. "
                "Please disable OPENAI_FUNCTIONS or choose a suitable model."
            )
        return v

    def get_openai_credentials(self, model: str) -> dict[str, str]:
        credentials = {
            "api_key": self.openai_api_key,
            "api_base": self.openai_api_base,
            "organization": self.openai_organization,
        }
        if self.use_azure:
            azure_credentials = self.get_azure_credentials(model)
            credentials.update(azure_credentials)
        return credentials

    def get_azure_credentials(self, model: str) -> dict[str, str]:
        """Get the kwargs for the Azure API."""

        # Fix --gpt3only and --gpt4only in combination with Azure
        fast_llm = (
            self.fast_llm
            if not (
                self.fast_llm == self.smart_llm
                and self.fast_llm.startswith(GPT_4_MODEL)
            )
            else f"not_{self.fast_llm}"
        )
        smart_llm = (
            self.smart_llm
            if not (
                self.smart_llm == self.fast_llm
                and self.smart_llm.startswith(GPT_3_MODEL)
            )
            else f"not_{self.smart_llm}"
        )

        deployment_id = {
            fast_llm: self.azure_model_to_deployment_id_map.get(
                "fast_llm_deployment_id",
                self.azure_model_to_deployment_id_map.get(
                    "fast_llm_model_deployment_id"  # backwards compatibility
                ),
            ),
            smart_llm: self.azure_model_to_deployment_id_map.get(
                "smart_llm_deployment_id",
                self.azure_model_to_deployment_id_map.get(
                    "smart_llm_model_deployment_id"  # backwards compatibility
                ),
            ),
            self.embedding_model: self.azure_model_to_deployment_id_map.get(
                "embedding_model_deployment_id"
            ),
        }.get(model, None)

        kwargs = {
            "api_type": self.openai_api_type,
            "api_base": self.openai_api_base,
            "api_version": self.openai_api_version,
        }
        if model == self.embedding_model:
            kwargs["engine"] = deployment_id
        else:
            kwargs["deployment_id"] = deployment_id
        return kwargs


```

如果你想使用 Azure 存储桶来托管 OpenAI 模型，你需要进行以下步骤：

1. 创建一个 Azure 订阅并获取一个 SubscriptionID 和一个 ResourceID。
2. 在 Azure 门户中订阅你的 Azure 服务并创建一个资源。
3. 创建一个或使用一个已有的包含用于托管 OpenAI 模型的环境的工作区。
4. 创建一个或使用一个已有的包含用于托管 OpenAI 模型的服务，并将它指定为你的环境。
5. 下载你的模型并将其上传到 Azure 存储桶中。
6. 更新你的环境以使用新的位置和存储。

在上述步骤中，你需要确保你的环境能够访问 Azure 存储桶。这 可以 通过将包含存储桶的 region 添加到环境配置文件中来实现。你还需要使用正确的身份验证方法来保护你的环境。

具体来说，以下是将这些步骤转化为 Python 代码的示例：

```py
import yaml

class AzureConfig:
   @classmethod
   def load_azure_config(cls, config_file: Path) -> Dict[str, str]:
       """
       Loads the configuration parameters for Azure hosting from the specified file
         path as a yaml file.

       Parameters:
           config_file (Path): The path to the config yaml file.

       Returns:
           Dict
       """
       with open(config_file) as file:
           config_params = yaml.load(file, Loader=yaml.FullLoader) or {}

       return {
           "openai_api_type": config_params.get("azure_api_type", "azure"),
           "openai_api_base": config_params.get("azure_api_base", ""),
           "openai_api_version": config_params.get(
               "azure_api_version", "2023-03-15-preview"
           ),
           "azure_model_to_deployment_id_map": config_params.get(
               "azure_model_map", {}
           ),
       }

   @classmethod
   def update_azure_config(cls, config: Dict[str, str]):
       """
       Updates the Azure config with the current environment configuration.

       Parameters:
           config (Dict[str, str]): The current Azure config.

       Returns:
           Dict[str, str]
       """
       updated_config = {}

       # Update the environment ID
       updated_config["environment_id"] = config["environment_id"]

       # Update the environment location
       updated_config["location"] = config["location"]

       # Update the environment name and version
       updated_config["name"] = config["name"]
       updated_config["version"] = config["version"]

       # Update the environment
       updated_config["azure_api_type"] = config["azure_api_type"]
       updated_config["azure_api_base"] = config["azure_api_base"]
       updated_config["azure_api_version"] = config["azure_api_version"]

       updated_config["azure_model_to_deployment_id_map"] = config["azure_model_map"]

       return updated_config
```


```py
class ConfigBuilder(Configurable[Config]):
    default_settings = Config()

    @classmethod
    def build_config_from_env(cls, project_root: Path = PROJECT_ROOT) -> Config:
        """Initialize the Config class"""
        config_dict = {
            "project_root": project_root,
            "authorise_key": os.getenv("AUTHORISE_COMMAND_KEY"),
            "exit_key": os.getenv("EXIT_KEY"),
            "plain_output": os.getenv("PLAIN_OUTPUT", "False") == "True",
            "shell_command_control": os.getenv("SHELL_COMMAND_CONTROL"),
            "ai_settings_file": project_root
            / Path(os.getenv("AI_SETTINGS_FILE", AI_SETTINGS_FILE)),
            "prompt_settings_file": project_root
            / Path(os.getenv("PROMPT_SETTINGS_FILE", PROMPT_SETTINGS_FILE)),
            "fast_llm": os.getenv("FAST_LLM", os.getenv("FAST_LLM_MODEL")),
            "smart_llm": os.getenv("SMART_LLM", os.getenv("SMART_LLM_MODEL")),
            "embedding_model": os.getenv("EMBEDDING_MODEL"),
            "browse_spacy_language_model": os.getenv("BROWSE_SPACY_LANGUAGE_MODEL"),
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "use_azure": os.getenv("USE_AZURE") == "True",
            "azure_config_file": project_root
            / Path(os.getenv("AZURE_CONFIG_FILE", AZURE_CONFIG_FILE)),
            "execute_local_commands": os.getenv("EXECUTE_LOCAL_COMMANDS", "False")
            == "True",
            "restrict_to_workspace": os.getenv("RESTRICT_TO_WORKSPACE", "True")
            == "True",
            "openai_functions": os.getenv("OPENAI_FUNCTIONS", "False") == "True",
            "tts_config": {
                "provider": os.getenv("TEXT_TO_SPEECH_PROVIDER"),
            },
            "github_api_key": os.getenv("GITHUB_API_KEY"),
            "github_username": os.getenv("GITHUB_USERNAME"),
            "google_api_key": os.getenv("GOOGLE_API_KEY"),
            "image_provider": os.getenv("IMAGE_PROVIDER"),
            "huggingface_api_token": os.getenv("HUGGINGFACE_API_TOKEN"),
            "huggingface_image_model": os.getenv("HUGGINGFACE_IMAGE_MODEL"),
            "audio_to_text_provider": os.getenv("AUDIO_TO_TEXT_PROVIDER"),
            "huggingface_audio_to_text_model": os.getenv(
                "HUGGINGFACE_AUDIO_TO_TEXT_MODEL"
            ),
            "sd_webui_url": os.getenv("SD_WEBUI_URL"),
            "sd_webui_auth": os.getenv("SD_WEBUI_AUTH"),
            "selenium_web_browser": os.getenv("USE_WEB_BROWSER"),
            "selenium_headless": os.getenv("HEADLESS_BROWSER", "True") == "True",
            "user_agent": os.getenv("USER_AGENT"),
            "memory_backend": os.getenv("MEMORY_BACKEND"),
            "memory_index": os.getenv("MEMORY_INDEX"),
            "redis_host": os.getenv("REDIS_HOST"),
            "redis_password": os.getenv("REDIS_PASSWORD"),
            "wipe_redis_on_start": os.getenv("WIPE_REDIS_ON_START", "True") == "True",
            "plugins_dir": os.getenv("PLUGINS_DIR"),
            "plugins_config_file": project_root
            / Path(os.getenv("PLUGINS_CONFIG_FILE", PLUGINS_CONFIG_FILE)),
            "chat_messages_enabled": os.getenv("CHAT_MESSAGES_ENABLED") == "True",
        }

        config_dict["disabled_command_categories"] = _safe_split(
            os.getenv("DISABLED_COMMAND_CATEGORIES")
        )

        config_dict["shell_denylist"] = _safe_split(
            os.getenv("SHELL_DENYLIST", os.getenv("DENY_COMMANDS"))
        )
        config_dict["shell_allowlist"] = _safe_split(
            os.getenv("SHELL_ALLOWLIST", os.getenv("ALLOW_COMMANDS"))
        )

        config_dict["google_custom_search_engine_id"] = os.getenv(
            "GOOGLE_CUSTOM_SEARCH_ENGINE_ID", os.getenv("CUSTOM_SEARCH_ENGINE_ID")
        )

        if os.getenv("ELEVENLABS_API_KEY"):
            config_dict["tts_config"]["elevenlabs"] = {
                "api_key": os.getenv("ELEVENLABS_API_KEY"),
                "voice_id": os.getenv("ELEVENLABS_VOICE_ID", ""),
            }
        if os.getenv("STREAMELEMENTS_VOICE"):
            config_dict["tts_config"]["streamelements"] = {
                "voice": os.getenv("STREAMELEMENTS_VOICE"),
            }

        if not config_dict["tts_config"]["provider"]:
            if os.getenv("USE_MAC_OS_TTS"):
                default_tts_provider = "macos"
            elif "elevenlabs" in config_dict["tts_config"]:
                default_tts_provider = "elevenlabs"
            elif os.getenv("USE_BRIAN_TTS"):
                default_tts_provider = "streamelements"
            else:
                default_tts_provider = "gtts"
            config_dict["tts_config"]["provider"] = default_tts_provider

        config_dict["plugins_allowlist"] = _safe_split(os.getenv("ALLOWLISTED_PLUGINS"))
        config_dict["plugins_denylist"] = _safe_split(os.getenv("DENYLISTED_PLUGINS"))

        with contextlib.suppress(TypeError):
            config_dict["image_size"] = int(os.getenv("IMAGE_SIZE"))
        with contextlib.suppress(TypeError):
            config_dict["redis_port"] = int(os.getenv("REDIS_PORT"))
        with contextlib.suppress(TypeError):
            config_dict["temperature"] = float(os.getenv("TEMPERATURE"))

        if config_dict["use_azure"]:
            azure_config = cls.load_azure_config(
                project_root / config_dict["azure_config_file"]
            )
            config_dict.update(azure_config)

        elif os.getenv("OPENAI_API_BASE_URL"):
            config_dict["openai_api_base"] = os.getenv("OPENAI_API_BASE_URL")

        openai_organization = os.getenv("OPENAI_ORGANIZATION")
        if openai_organization is not None:
            config_dict["openai_organization"] = openai_organization

        config_dict_without_none_values = {
            k: v for k, v in config_dict.items() if v is not None
        }

        config = cls.build_agent_configuration(config_dict_without_none_values)

        # Set secondary config variables (that depend on other config variables)

        config.plugins_config = PluginsConfig.load_config(
            config.plugins_config_file,
            config.plugins_denylist,
            config.plugins_allowlist,
        )

        return config

    @classmethod
    def load_azure_config(cls, config_file: Path) -> Dict[str, str]:
        """
        Loads the configuration parameters for Azure hosting from the specified file
          path as a yaml file.

        Parameters:
            config_file (Path): The path to the config yaml file.

        Returns:
            Dict
        """
        with open(config_file) as file:
            config_params = yaml.load(file, Loader=yaml.FullLoader) or {}

        return {
            "openai_api_type": config_params.get("azure_api_type", "azure"),
            "openai_api_base": config_params.get("azure_api_base", ""),
            "openai_api_version": config_params.get(
                "azure_api_version", "2023-03-15-preview"
            ),
            "azure_model_to_deployment_id_map": config_params.get(
                "azure_model_map", {}
            ),
        }


```

这段代码是一个函数 `assert_config_has_openai_api_key`，它用于检查 OpenAI API 密钥是否在配置文件 `config.py` 中设置，或者是否通过环境变量设置。如果没有设置 API 密钥，函数会输出一条消息并提示用户设置，然后询问用户输入他们的 API 密钥。如果用户设置了一个有效的 API 密钥，函数会将其存储为 `openai_api_key` 环境变量，并更新配置文件中的 `openai_api_key` 变量。如果设置的 API 密钥无效，函数会输出一条消息并退出程序。


```py
def assert_config_has_openai_api_key(config: Config) -> None:
    """Check if the OpenAI API key is set in config.py or as an environment variable."""
    if not config.openai_api_key:
        print(
            Fore.RED
            + "Please set your OpenAI API key in .env or as an environment variable."
            + Fore.RESET
        )
        print("You can get your key from https://platform.openai.com/account/api-keys")
        openai_api_key = input(
            "If you do have the key, please enter your OpenAI API key now:\n"
        )
        key_pattern = r"^sk-\w{48}"
        openai_api_key = openai_api_key.strip()
        if re.search(key_pattern, openai_api_key):
            os.environ["OPENAI_API_KEY"] = openai_api_key
            config.openai_api_key = openai_api_key
            print(
                Fore.GREEN
                + "OpenAI API key successfully set!\n"
                + Fore.YELLOW
                + "NOTE: The API key you've set is only temporary.\n"
                + "For longer sessions, please set it in .env file"
                + Fore.RESET
            )
        else:
            print("Invalid OpenAI API key!")
            exit(1)


```

这段代码定义了一个名为 `_safe_split` 的函数，它接收一个字符串参数 `s` 和一个分隔符参数 `sep`，并返回一个字符串列表。

函数的实现采用了一种安全的方式，即使用 `Union` 类型来处理可能存在的 `None` 参数。在函数内部，首先检查 `s` 是否为 `None`，如果是，则返回一个空列表 `[]`。否则，函数将使用给定的分隔符 `sep` 将字符串 `s` 分割成一个新的字符串列表，并将结果返回。

例如，如果使用以下代码调用 `_safe_split` 函数：

```py
from typing import Union, List

s1 = "hello"
s2 = "world"
sep = " "

result = _safe_split(s1, sep)
print(result)  # 输出： ["hello", "world"]
```

结果为 `["hello", "world"]`，说明 `s1` 和 `sep` 都被正确地分成了两个字符串。


```py
def _safe_split(s: Union[str, None], sep: str = ",") -> list[str]:
    """Split a string by a separator. Return an empty list if the string is None."""
    if s is None:
        return []
    return s.split(sep)

```

# `autogpts/autogpt/autogpt/config/__init__.py`

这段代码定义了一个名为 `AutoGPTConfig` 的类，包含了自动语言处理 (AutoGPT) 的配置类。这个模块包含了一些继承自 `AIDirectives` 和 `AIProfile` 的类，用于定义和设置 AutoGPT的一些参数。

具体来说，这个模块中的代码会创建一个 `Config` 类，它是 `ConfigBuilder` 类的子类，用于定义和构建一个或多个子配置。这些子配置可能包括来自用户或环境的配置，以及来自 OpenAI 服务的 API 密钥等。

为了确保 `AutoGPTConfig` 类能够使用依赖于 OpenAI 的组件，代码中使用了 `assert_config_has_openai_api_key` 函数，用于检查 OpenAI API 是否与本地环境中的 API 密钥匹配。如果没有匹配的 API 密钥，函数将抛出一个异常。

此外，代码还定义了一系列函数，用于帮助用户创建和设置 AutoGPT的配置。例如，`AIDirectives` 类可能包含与指定模型相关的指令，`ConfigBuilder` 类可能包含用于创建和设置子配置的函数，而 `assert_config_has_openai_api_key` 函数则用于检查用户输入的 API 密钥是否正确。


```py
"""
This module contains the configuration classes for AutoGPT.
"""
from .ai_directives import AIDirectives
from .ai_profile import AIProfile
from .config import Config, ConfigBuilder, assert_config_has_openai_api_key

__all__ = [
    "assert_config_has_openai_api_key",
    "AIProfile",
    "AIDirectives",
    "Config",
    "ConfigBuilder",
]

```

# Re-architecture Notes

## Key Documents

- [Planned Agent Workflow](https://whimsical.com/agent-workflow-v2-NmnTQ8R7sVo7M3S43XgXmZ)
- [Original Architecture Diagram](https://www.figma.com/file/fwdj44tPR7ArYtnGGUKknw/Modular-Architecture?type=whiteboard&node-id=0-1) - This is sadly well out of date at this point.
- [Kanban](https://github.com/orgs/Significant-Gravitas/projects/1/views/1?filterQuery=label%3Are-arch)

## The Motivation

The `master` branch of AutoGPT is an organically grown amalgamation of many thoughts 
and ideas about agent-driven autonomous systems.  It lacks clear abstraction boundaries, 
has issues of global state and poorly encapsulated state, and is generally just hard to 
make effective changes to.  Mainly it's just a system that's hard to make changes to.  
And research in the field is moving fast, so we want to be able to try new ideas 
quickly.  

## Initial Planning

A large group of maintainers and contributors met do discuss the architectural 
challenges associated with the existing codebase. Many much-desired features (building 
new user interfaces, enabling project-specific agents, enabling multi-agent systems) 
are bottlenecked by the global state in the system. We discussed the tradeoffs between 
an incremental system transition and a big breaking version change and decided to go 
for the breaking version change. We justified this by saying:

- We can maintain, in essence, the same user experience as now even with a radical 
  restructuring of the codebase
- Our developer audience is struggling to use the existing codebase to build 
  applications and libraries of their own, so this breaking change will largely be 
  welcome.

## Primary Goals

- Separate the AutoGPT application code from the library code.
- Remove global state from the system
- Allow for multiple agents per user (with facilities for running simultaneously)
- Create a serializable representation of an Agent
- Encapsulate the core systems in abstractions with clear boundaries.

## Secondary goals

- Use existing tools to ditch any unneccesary cruft in the codebase (document loading, 
  json parsing, anything easier to replace than to port).
- Bring in the [core agent loop updates](https://whimsical.com/agent-workflow-v2-NmnTQ8R7sVo7M3S43XgXmZ)
  being developed simultaneously by @Pwuts 

# The Agent Subsystems

## Configuration

We want a lot of things from a configuration system. We lean heavily on it in the 
`master` branch to allow several parts of the system to communicate with each other.  
[Recent work](https://github.com/Significant-Gravitas/AutoGPT/pull/4737) has made it 
so that the config is no longer a singleton object that is materialized from the import 
state, but it's still treated as a 
[god object](https://en.wikipedia.org/wiki/God_object) containing all information about
the system and _critically_ allowing any system to reference configuration information 
about other parts of the system.  

### What we want

- It should still be reasonable to collate the entire system configuration in a 
  sensible way.
- The configuration should be validatable and validated.
- The system configuration should be a _serializable_ representation of an `Agent`.
- The configuration system should provide a clear (albeit very low-level) contract 
  about user-configurable aspects of the system.
- The configuration should reasonably manage default values and user-provided overrides.
- The configuration system needs to handle credentials in a reasonable way.
- The configuration should be the representation of some amount of system state, like 
  api budgets and resource usage.  These aspects are recorded in the configuration and 
  updated by the system itself.
- Agent systems should have encapsulated views of the configuration.  E.g. the memory 
  system should know about memory configuration but nothing about command configuration.

## Workspace

There are two ways to think about the workspace:

- The workspace is a scratch space for an agent where it can store files, write code, 
  and do pretty much whatever else it likes.
- The workspace is, at any given point in time, the single source of truth for what an 
  agent is.  It contains the serializable state (the configuration) as well as all 
  other working state (stored files, databases, memories, custom code).  

In the existing system there is **one** workspace.  And because the workspace holds so 
much agent state, that means a user can only work with one agent at a time.

## Memory

The memory system has been under extremely active development. 
See [#3536](https://github.com/Significant-Gravitas/AutoGPT/issues/3536) and 
[#4208](https://github.com/Significant-Gravitas/AutoGPT/pull/4208) for discussion and 
work in the `master` branch.  The TL;DR is 
that we noticed a couple of months ago that the `Agent` performed **worse** with 
permanent memory than without it.  Since then the knowledge storage and retrieval 
system has been [redesigned](https://whimsical.com/memory-system-8Ae6x6QkjDwQAUe9eVJ6w1) 
and partially implemented in the `master` branch.

## Planning/Prompt-Engineering

The planning system is the system that translates user desires/agent intentions into
language model prompts.  In the course of development, it has become pretty clear 
that `Planning` is the wrong name for this system

### What we want

- It should be incredibly obvious what's being passed to a language model, when it's
  being passed, and what the language model response is. The landscape of language 
  model research is developing very rapidly, so building complex abstractions between 
  users/contributors and the language model interactions is going to make it very 
  difficult for us to nimbly respond to new research developments.
- Prompt-engineering should ideally be exposed in a parameterizeable way to users. 
- We should, where possible, leverage OpenAI's new  
  [function calling api](https://openai.com/blog/function-calling-and-other-api-updates) 
  to get outputs in a standard machine-readable format and avoid the deep pit of 
  parsing json (and fixing unparsable json).

### Planning Strategies

The [new agent workflow](https://whimsical.com/agent-workflow-v2-NmnTQ8R7sVo7M3S43XgXmZ) 
has many, many interaction points for language models.  We really would like to not 
distribute prompt templates and raw strings all through the system. The re-arch solution 
is to encapsulate language model interactions into planning strategies. 
These strategies are defined by 

- The `LanguageModelClassification` they use (`FAST` or `SMART`)
- A function `build_prompt` that takes strategy specific arguments and constructs a 
  `LanguageModelPrompt` (a simple container for lists of messages and functions to
  pass to the language model)
- A function `parse_content` that parses the response content (a dict) into a better 
  formatted dict.  Contracts here are intentionally loose and will tighten once we have 
  at least one other language model provider.

## Resources

Resources are kinds of services we consume from external APIs.  They may have associated 
credentials and costs we need to manage.  Management of those credentials is implemented 
as manipulation of the resource configuration.  We have two categories of resources 
currently

- AI/ML model providers (including language model providers and embedding model providers, ie OpenAI)
- Memory providers (e.g. Pinecone, Weaviate, ChromaDB, etc.)

### What we want

- Resource abstractions should provide a common interface to different service providers 
  for a particular kind of service.  
- Resource abstractions should manipulate the configuration to manage their credentials 
  and budget/accounting.
- Resource abstractions should be composable over an API (e.g. I should be able to make 
  an OpenAI provider that is both a LanguageModelProvider and an EmbeddingModelProvider
  and use it wherever I need those services).

## Abilities

Along with planning and memory usage, abilities are one of the major augmentations of 
augmented language models.  They allow us to expand the scope of what language models
can do by hooking them up to code they can execute to obtain new knowledge or influence
the world.  

### What we want

- Abilities should have an extremely clear interface that users can write to.
- Abilities should have an extremely clear interface that a language model can 
  understand
- Abilities should be declarative about their dependencies so the system can inject them
- Abilities should be executable (where sensible) in an async run loop.
- Abilities should be not have side effects unless those side effects are clear in 
  their representation to an agent (e.g. the BrowseWeb ability shouldn't write a file,
  but the WriteFile ability can).

## Plugins

Users want to add lots of features that we don't want to support as first-party. 
Or solution to this is a plugin system to allow users to plug in their functionality or
to construct their agent from a public plugin marketplace.  Our primary concern in the
re-arch is to build a stateless plugin service interface and a simple implementation 
that can load plugins from installed packages or from zip files.  Future efforts will 
expand this system to allow plugins to load from a marketplace or some other kind 
of service.

### What is a Plugin

Plugins are a kind of garbage term.  They refer to a number of things.

- New commands for the agent to execute.  This is the most common usage.
- Replacements for entire subsystems like memory or language model providers
- Application plugins that do things like send emails or communicate via whatsapp
- The repositories contributors create that may themselves have multiple plugins in them.

### Usage in the existing system

The current plugin system is _hook-based_.  This means plugins don't correspond to 
kinds of objects in the system, but rather to times in the system at which we defer 
execution to them.  The main advantage of this setup is that user code can hijack 
pretty much any behavior of the agent by injecting code that supercedes the normal 
agent execution.  The disadvantages to this approach are numerous:

- We have absolutely no mechanisms to enforce any security measures because the threat 
  surface is everything.
- We cannot reason about agent behavior in a cohesive way because control flow can be
  ceded to user code at pretty much any point and arbitrarily change or break the
  agent behavior
- The interface for designing a plugin is kind of terrible and difficult to standardize
- The hook based implementation means we couple ourselves to a particular flow of 
  control (or otherwise risk breaking plugin behavior).  E.g. many of the hook targets
  in the [old workflow](https://whimsical.com/agent-workflow-VAzeKcup3SR7awpNZJKTyK) 
  are not present or mean something entirely different in the 
  [new workflow](https://whimsical.com/agent-workflow-v2-NmnTQ8R7sVo7M3S43XgXmZ).
- Etc.

### What we want

- A concrete definition of a plugin that is narrow enough in scope that we can define 
  it well and reason about how it will work in the system.
- A set of abstractions that let us define a plugin by its storage format and location 
- A service interface that knows how to parse the plugin abstractions and turn them 
  into concrete classes and objects.


## Some Notes on how and why we'll use OO in this project

First and foremost, Python itself is an object-oriented language. It's 
underlying [data model](https://docs.python.org/3/reference/datamodel.html) is built 
with object-oriented programming in mind. It offers useful tools like abstract base 
classes to communicate interfaces to developers who want to, e.g., write plugins, or 
help work on implementations. If we were working in a different language that offered 
different tools, we'd use a different paradigm.

While many things are classes in the re-arch, they are not classes in the same way. 
There are three kinds of things (roughly) that are written as classes in the re-arch:
1.  **Configuration**:  AutoGPT has *a lot* of configuration.  This configuration 
    is *data* and we use **[Pydantic](https://docs.pydantic.dev/latest/)** to manage it as 
    pydantic is basically industry standard for this stuff. It provides runtime validation 
    for all the configuration and allows us to easily serialize configuration to both basic 
    python types (dicts, lists, and primatives) as well as serialize to json, which is 
    important for us being able to put representations of agents 
    [on the wire](https://en.wikipedia.org/wiki/Wire_protocol) for web applications and 
    agent-to-agent communication. *These are essentially 
    [structs](https://en.wikipedia.org/wiki/Struct_(C_programming_language)) rather than 
    traditional classes.*
2.  **Internal Data**: Very similar to configuration, AutoGPT passes around boatloads 
    of internal data.  We are interacting with language models and language model APIs 
    which means we are handling lots of *structured* but *raw* text.  Here we also 
    leverage **pydantic** to both *parse* and *validate* the internal data and also to 
    give us concrete types which we can use static type checkers to validate against 
    and discover problems before they show up as bugs at runtime. *These are 
    essentially [structs](https://en.wikipedia.org/wiki/Struct_(C_programming_language)) 
    rather than traditional classes.*
3.  **System Interfaces**: This is our primary traditional use of classes in the 
    re-arch.  We have a bunch of systems. We want many of those systems to have 
    alternative implementations (e.g. via plugins). We use abstract base classes to 
    define interfaces to communicate with people who might want to provide those 
    plugins. We provide a single concrete implementation of most of those systems as a 
    subclass of the interface. This should not be controversial.

The approach is consistent with 
[prior](https://github.com/Significant-Gravitas/AutoGPT/issues/2458)
[work](https://github.com/Significant-Gravitas/AutoGPT/pull/2442) done by other 
maintainers in this direction.

From an organization standpoint, OO programming is by far the most popular programming 
paradigm (especially for Python). It's the one most often taught in programming classes
and the one with the most available online training for people interested in 
contributing.   

Finally, and importantly, we scoped the plan and initial design of the re-arch as a 
large group of maintainers and collaborators early on. This is consistent with the 
design we chose and no-one offered alternatives.


# AutoGPT Core

This subpackage contains the ongoing work for the 
[AutoGPT Re-arch](https://github.com/Significant-Gravitas/AutoGPT/issues/4770). It is 
a work in progress and is not yet feature complete.  In particular, it does not yet
have many of the AutoGPT commands implemented and is pending ongoing work to 
[re-incorporate vector-based memory and knowledge retrieval](https://github.com/Significant-Gravitas/AutoGPT/issues/3536).

## [Overview](ARCHITECTURE_NOTES.md)

The AutoGPT Re-arch is a re-implementation of the AutoGPT agent that is designed to be more modular,
more extensible, and more maintainable than the original AutoGPT agent.  It is also designed to be
more accessible to new developers and to be easier to contribute to. The re-arch is a work in progress
and is not yet feature complete.  It is also not yet ready for production use.

## Running the Re-arch Code

1. Open the `autogpt/core` folder in a terminal

2. Set up a dedicated virtual environment:  
   `python -m venv .venv`

3. Install everything needed to run the project:  
   `poetry install`


## CLI Application

There are two client applications for AutoGPT included.

:star2: **This is the reference application I'm working with for now** :star2: 

The first app is a straight CLI application.  I have not done anything yet to port all the friendly display stuff from the ~~`logger.typewriter_log`~~`user_friendly_output` logic.  

- [Entry Point](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpts/autogpt/autogpt/core/runner/cli_app/cli.py)
- [Client Application](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpts/autogpt/autogpt/core/runner/cli_app/main.py)

You'll then need a settings file.  Run

```py
poetry run cli make-settings
```

This will write a file called `default_agent_settings.yaml` with all the user-modifiable 
configuration keys to `~/auto-gpt/default_agent_settings.yml` and make the `auto-gpt` directory 
in your user directory if it doesn't exist). Your user directory is located in different places 
depending on your operating system:

- On Linux, it's `/home/USERNAME`
- On Windows, it's `C:\Users\USERNAME`
- On Mac, it's `/Users/USERNAME`

At a bare minimum, you'll need to set `openai.credentials.api_key` to your OpenAI API Key to run 
the model.

You can then run AutoGPT with 

```py
poetry run cli run
```

to launch the interaction loop.

### CLI Web App

:warning: I am not actively developing this application.  I am primarily working with the traditional CLI app
described above.  It is a very good place to get involved if you have web application design experience and are 
looking to get involved in the re-arch.

The second app is still a CLI, but it sets up a local webserver that the client application talks to
rather than invoking calls to the Agent library code directly.  This application is essentially a sketch 
at this point as the folks who were driving it have had less time (and likely not enough clarity) to proceed.

- [Entry Point](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpts/autogpt/autogpt/core/runner/cli_web_app/cli.py)
- [Client Application](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpts/autogpt/autogpt/core/runner/cli_web_app/client/client.py)
- [Server API](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpts/autogpt/autogpt/core/runner/cli_web_app/server/api.py)

To run, you still need to generate a default configuration.  You can do 

```py
poetry run cli-web make-settings
```

It invokes the same command as the bare CLI app, so follow the instructions above about setting your API key.

To run, do 

```py
poetry run cli-web client
```

This will launch a webserver and then start the client cli application to communicate with it.


# `autogpts/autogpt/autogpt/core/__init__.py`

我需要更具体的上下文来回答你的问题。可以请你提供更多上下文，让我更好地解释代码的作用。


```py

```

# `autogpts/autogpt/autogpt/core/ability/base.py`

这段代码定义了一个自定义的ABC类，包含两个静态成员变量：

```pypython
class AbilityResult:
   pass
```

```pypython
class P做了Slash() {
   // There's a reference to the `typing.Tuple` type in the
   // `T` parameter, but the `Any` type is used in the `type`
   // argument. This is because the `T` parameter is expected to
   // return an instance of `typing.Tuple[Any, Any]`, but the
   // `type` argument for the `AbilityResult` class expects
   // `Any` and `Any` as its arguments.
   pass

   // The `description` field is a potted validation string,
   // but it needs to be interpreted as a `str` because it is passed
   // as an argument to the `SystemConfiguration` class.
   description = "This is an example of an AbilityResult."
}
```

这两段代码定义了一个自定义的`AbilityResult`类，该类包含一个静态成员变量`status`，一个静态成员函数`generate_completion`和一个私有化构造函数`__init__`。

此外，代码还导入了一个`inflection`库，以及`pprint`库和`typing`库。


```py
import abc
from pprint import pformat
from typing import Any, ClassVar

import inflection
from pydantic import Field

from autogpt.core.configuration import SystemConfiguration
from autogpt.core.planning.simple import LanguageModelConfiguration
from autogpt.core.plugin.base import PluginLocation
from autogpt.core.resource.model_providers import CompletionModelFunction
from autogpt.core.utils.json_schema import JSONSchema

from .schema import AbilityResult


```

这段代码定义了一个名为AbilityConfiguration的类，它是一个模型配置结构体。这个结构体包含了一些通用的字段，如位置、所需软件包、语言模型配置、内存提供程序配置、工作区配置等。同时，它还继承了抽象类Abc，以及两个方法：name、description和parameters。

Ability是一个类，它继承自AbilityConfiguration，并重写了这些方法和一个名为__call__的抽象方法。这个抽象方法定义了如何初始化和返回一个Ability对象，同时处理参数和返回一个AbilityResult。

另外，这段代码还定义了一个CompletionModelFunction，它是CompletionModelFunction的实例，用于处理Ability对象的完成状态。


```py
class AbilityConfiguration(SystemConfiguration):
    """Struct for model configuration."""

    location: PluginLocation
    packages_required: list[str] = Field(default_factory=list)
    language_model_required: LanguageModelConfiguration = None
    memory_provider_required: bool = False
    workspace_required: bool = False


class Ability(abc.ABC):
    """A class representing an agent ability."""

    default_configuration: ClassVar[AbilityConfiguration]

    @classmethod
    def name(cls) -> str:
        """The name of the ability."""
        return inflection.underscore(cls.__name__)

    @property
    @classmethod
    @abc.abstractmethod
    def description(cls) -> str:
        """A detailed description of what the ability does."""
        ...

    @property
    @classmethod
    @abc.abstractmethod
    def parameters(cls) -> dict[str, JSONSchema]:
        ...

    @abc.abstractmethod
    async def __call__(self, *args: Any, **kwargs: Any) -> AbilityResult:
        ...

    def __str__(self) -> str:
        return pformat(self.spec)

    @property
    @classmethod
    def spec(cls) -> CompletionModelFunction:
        return CompletionModelFunction(
            name=cls.name(),
            description=cls.description,
            parameters=cls.parameters,
        )


```

该代码定义了一个名为 "AbilityRegistry" 的类，该类实现了 "abc.ABC"（抽象工厂设计模式）的接口。

在这个类的 "register\_ability" 方法中，通过实现 "abc.abstractmethod"（抽象方法）和 "abc.abstractmethod"（抽象方法）抽象方法，定义了一个能力注册的接口。该接口允许在注册能力时传递 "ability\_name" 和 "ability\_configuration" 两个参数。

在 "list\_abilities" 方法中，通过实现 "abc.abstractmethod" 抽象方法，定义了一个打印所有能力的接口。允许在调用此方法时传递一个列表参数，该列表将包含所有注册的能力名称。

在 "dump\_abilities" 方法中，通过实现 "abc.abstractmethod" 抽象方法，定义了一个将所有注册能力名称存储在 "abilities" 字典中的接口。允许在调用此方法时传递一个字典参数，该字典将包含所有注册的能力名称。

在 "get\_ability" 方法中，通过实现 "abc.abstractmethod" 抽象方法，定义了一个获取指定能力名称的 "Ability" 接口。允许在调用此方法时传递 "ability\_name" 参数，该参数将用于查找并返回具有该名称的能力对象。

在 "perform" 方法中，通过实现 "abc.abstractmethod" 抽象方法，定义了一个执行注册或加载能力的 "AbilityResult" 接口。允许在调用此方法时传递一个 "ability\_name" 和任意数量的参数，该参数将被用于执行相应的操作。


```py
class AbilityRegistry(abc.ABC):
    @abc.abstractmethod
    def register_ability(
        self, ability_name: str, ability_configuration: AbilityConfiguration
    ) -> None:
        ...

    @abc.abstractmethod
    def list_abilities(self) -> list[str]:
        ...

    @abc.abstractmethod
    def dump_abilities(self) -> list[CompletionModelFunction]:
        ...

    @abc.abstractmethod
    def get_ability(self, ability_name: str) -> Ability:
        ...

    @abc.abstractmethod
    async def perform(self, ability_name: str, **kwargs: Any) -> AbilityResult:
        ...

```

# `autogpts/autogpt/autogpt/core/ability/schema.py`

这段代码定义了一个名为`ContentType`的枚举类型和一个名为`Knowledge`的类。`ContentType`枚举类型定义了两种可能的值，分别为`TEXT`和`CODE`。`Knowledge`类使用了`typing.Any`类型，这意味着它可以接受任何类型的数据作为它的属性。

`Knowledge`类有一个`content`属性，它的类型是一个字符串（`str`）。此外，它还有一个`content_type`属性，它的类型是一个名为`ContentType`的枚举类型。最后，它还有一个`content_metadata`属性，它的类型是一个字典（`dict`）。

这段代码的主要目的是定义了两个类型：`ContentType`和`Knowledge`。`ContentType`用于定义了两种可能的枚举类型，`Knowledge`用于定义了一个类，该类包含了一个字符串属性和一个字典类型的属性。


```py
import enum
from typing import Any

from pydantic import BaseModel


class ContentType(str, enum.Enum):
    # TBD what these actually are.
    TEXT = "text"
    CODE = "code"


class Knowledge(BaseModel):
    content: str
    content_type: ContentType
    content_metadata: dict[str, Any]


```

这段代码定义了一个名为AbilityResult的类，继承自Model类(AbstractModel类)，用于表示一个能力的结果。

该类包含以下成员变量：

- `ability_name:` 该成员变量表示该能力的名称。
- `ability_args:` 该成员变量表示该能力所需要传递给的参数，参数类型为字典，键为参数名称，值为参数描述。
- `success:` 该成员变量表示该能力是否成功，如果成功则值为True，否则值为False。
- `message:` 该成员变量表示该能力成功后所输出的信息。
- `new_knowledge:` 该成员变量表示在该能力成功后获得的新知识。

该类还定义了一个名为`summary`的静态方法，用于将AbilityResult对象转换为字符串并返回。

该类的实例可以用来表示一个具有指定能力的成功或失败的结果，例如：

```py
AbilityResult(
    ability_name='Coverage',
    ability_args={
        'accuracy': '0.8',
        'freq': '200',
        'threshold': '0.1'
    },
    success=True,
    message='Your coverage has been increased!',
    new_knowledge=Knowledge(accuracy=0.8, freq='200', threshold=0.1)
)
```

该代码将创建一个AbilityResult实例，表示覆盖能力成功，能力参数为准确度为80%，频率为200，阈值为100。成功值为True，消息为"你的覆盖率已经提高！"。新知识为 Accuracy=0.8,Freq='200',Threshold=0.1。


```py
class AbilityResult(BaseModel):
    """The AbilityResult is a standard response struct for an ability."""

    ability_name: str
    ability_args: dict[str, str]
    success: bool
    message: str
    new_knowledge: Knowledge = None

    def summary(self) -> str:
        kwargs = ", ".join(f"{k}={v}" for k, v in self.ability_args.items())
        return f"{self.ability_name}({kwargs}): {self.message}"

```

# `autogpts/autogpt/autogpt/core/ability/simple.py`

这段代码是一个自定义的 Python 库，它实现了基于人工智能技术的能力引擎。下面是这段代码的一些关键部分和功能：

1. 引入logging库，用于记录一些重要的信息，如错误、警告等。

2. 从autogpt.core.ability.base类开始导入，这个类是ability引擎的基类，从这里开始我们可以使用一些核心的API。

3. 从autogpt.core.ability.builtins类开始导入，这个类包含了一些 built-in的能力，可以作为我们能力引擎的一部分。

4. 从autogpt.core.ability.schema类开始导入，这个类定义了能力的一些元数据和结构，也是我们能力引擎的一部分。

5. 从autogpt.core.configuration类开始导入，这个类允许我们在配置文件中设置一些关键的设置，如秘钥、日志等级等。

6. 从autogpt.core.memory.base类开始导入，这个类管理我们的内存，包括retrieve和store方法，用于我们在需要时动态地分配和释放内存。

7. 从autogpt.core.plugin.simple类开始导入，这个类实现了一些简单的插件服务，用于在一些特定的场景下进行一些额外的操作。

8. 从autogpt.core.resource.model_providers类开始导入，这个类定义了如何注册和获取模型提供商，以及如何获取模型和完成模型的信息。

9. 从autogpt.core.workspace.base类开始导入，这个类管理我们的工作区，包括获取和设置工作区的方法。

10. 在import的最后，定义了一个AbilityRegistry，这个类是用来注册和管理我们能力引擎中的所有能力的。

综上所述，这段代码实现了一个基于人工智能技术的能力引擎，允许用户通过配置文件来注册和管理不同的能力，并在需要的时候动态地分配和释放内存。


```py
import logging

from autogpt.core.ability.base import Ability, AbilityConfiguration, AbilityRegistry
from autogpt.core.ability.builtins import BUILTIN_ABILITIES
from autogpt.core.ability.schema import AbilityResult
from autogpt.core.configuration import Configurable, SystemConfiguration, SystemSettings
from autogpt.core.memory.base import Memory
from autogpt.core.plugin.simple import SimplePluginService
from autogpt.core.resource.model_providers import (
    ChatModelProvider,
    CompletionModelFunction,
    ModelProviderName,
)
from autogpt.core.workspace.base import Workspace


```

This is a Python class that defines a `SimplePluginService` implementation for a language model. The language model is configured through a dictionary and this class provides methods to register, list, and dump the abilities of the language model.

The `register_ability` method is used to register an ability, given its name and configuration. The `list_abilities` method returns a list of all the available abilities. The `dump_abilities` method returns the ability configuration for each ability in the language model.

The `get_ability` method is used to retrieve an ability by name. If the ability is not found, it raises a `ValueError`.

The `perform` method is used to invoke the language model. It takes an ability name as an argument and passes it to the `get_ability` method to retrieve the language model associated with that ability. It then passes the keyword arguments to the language model for performance.


```py
class AbilityRegistryConfiguration(SystemConfiguration):
    """Configuration for the AbilityRegistry subsystem."""

    abilities: dict[str, AbilityConfiguration]


class AbilityRegistrySettings(SystemSettings):
    configuration: AbilityRegistryConfiguration


class SimpleAbilityRegistry(AbilityRegistry, Configurable):
    default_settings = AbilityRegistrySettings(
        name="simple_ability_registry",
        description="A simple ability registry.",
        configuration=AbilityRegistryConfiguration(
            abilities={
                ability_name: ability.default_configuration
                for ability_name, ability in BUILTIN_ABILITIES.items()
            },
        ),
    )

    def __init__(
        self,
        settings: AbilityRegistrySettings,
        logger: logging.Logger,
        memory: Memory,
        workspace: Workspace,
        model_providers: dict[ModelProviderName, ChatModelProvider],
    ):
        self._configuration = settings.configuration
        self._logger = logger
        self._memory = memory
        self._workspace = workspace
        self._model_providers = model_providers
        self._abilities: list[Ability] = []
        for (
            ability_name,
            ability_configuration,
        ) in self._configuration.abilities.items():
            self.register_ability(ability_name, ability_configuration)

    def register_ability(
        self, ability_name: str, ability_configuration: AbilityConfiguration
    ) -> None:
        ability_class = SimplePluginService.get_plugin(ability_configuration.location)
        ability_args = {
            "logger": self._logger.getChild(ability_name),
            "configuration": ability_configuration,
        }
        if ability_configuration.packages_required:
            # TODO: Check packages are installed and maybe install them.
            pass
        if ability_configuration.memory_provider_required:
            ability_args["memory"] = self._memory
        if ability_configuration.workspace_required:
            ability_args["workspace"] = self._workspace
        if ability_configuration.language_model_required:
            ability_args["language_model_provider"] = self._model_providers[
                ability_configuration.language_model_required.provider_name
            ]
        ability = ability_class(**ability_args)
        self._abilities.append(ability)

    def list_abilities(self) -> list[str]:
        return [
            f"{ability.name()}: {ability.description}" for ability in self._abilities
        ]

    def dump_abilities(self) -> list[CompletionModelFunction]:
        return [ability.spec for ability in self._abilities]

    def get_ability(self, ability_name: str) -> Ability:
        for ability in self._abilities:
            if ability.name() == ability_name:
                return ability
        raise ValueError(f"Ability '{ability_name}' not found.")

    async def perform(self, ability_name: str, **kwargs) -> AbilityResult:
        ability = self.get_ability(ability_name)
        return await ability(**kwargs)

```

# `autogpts/autogpt/autogpt/core/ability/__init__.py`

这段代码定义了一个命令系统，用于扩展人工智能代理的功能。它包含以下几个主要部分：

1. 从 `autogpt.core.ability.base` 导入 `Ability`、`AbilityConfiguration` 和 `AbilityRegistry` 类。这些类提供了ability的定义、配置和管理功能。

2. 从 `autogpt.core.ability.schema` 导入 `AbilityResult` 类。这个类定义了ability的结果，即代理在完成某个任务后可能获得的奖励或状态。

3. 从 `autogpt.core.ability.simple` 导入 `AbilityRegistryConfiguration`、`AbilityRegistrySettings` 和 `SimpleAbilityRegistry` 类。这些类提供了ability的具体实现，包括registry配置、settings和registry实例。

4. 在函数内部，通过 `AbilityRegistry` 创建一个registry实例，并使用 `AbilityRegistryConfiguration` 进行配置。然后，使用 `SimpleAbilityRegistry` 加载定义好的ability。

5. 在ability内部，当代理完成任务时，会使用 `AbilityRegistry` 的 `AbilityResult` 获取结果，并返回给调用方。


```py
"""The command system provides a way to extend the functionality of the AI agent."""
from autogpt.core.ability.base import Ability, AbilityConfiguration, AbilityRegistry
from autogpt.core.ability.schema import AbilityResult
from autogpt.core.ability.simple import (
    AbilityRegistryConfiguration,
    AbilityRegistrySettings,
    SimpleAbilityRegistry,
)

```

# `autogpts/autogpt/autogpt/core/ability/builtins/create_new_ability.py`

`AbilityResult` is a class that represents the result of an `Ability` capability. It has several properties and methods to configure and execute the ability.

The `ability_name` property is a required string that specifies the name of the ability.

The `description` property is a required string that specifies a detailed description of the ability.

The `arguments` property is a required list of objects that specifies the arguments that the ability will accept. Each argument object has several properties, including `name`, `type`, and `description`.

The `required_arguments` property is a required list of strings that specifies the names of the arguments that are required to execute the ability.

The `package_requirements` property is a required list of strings that specifies the names of the Python packages that are required to execute the ability.

The `code` property is a required string that specifies the Python code that will be executed when the ability is called.

The `__call__` method is implemented to allow calling the `Ability` capability with the required arguments and options.


```py
import logging
from typing import ClassVar

from autogpt.core.ability.base import Ability, AbilityConfiguration
from autogpt.core.ability.schema import AbilityResult
from autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat
from autogpt.core.utils.json_schema import JSONSchema


class CreateNewAbility(Ability):
    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.CreateNewAbility",
        ),
    )

    def __init__(
        self,
        logger: logging.Logger,
        configuration: AbilityConfiguration,
    ):
        self._logger = logger
        self._configuration = configuration

    description: ClassVar[str] = "Create a new ability by writing python code."

    parameters: ClassVar[dict[str, JSONSchema]] = {
        "ability_name": JSONSchema(
            description="A meaningful and concise name for the new ability.",
            type=JSONSchema.Type.STRING,
            required=True,
        ),
        "description": JSONSchema(
            description="A detailed description of the ability and its uses, including any limitations.",
            type=JSONSchema.Type.STRING,
            required=True,
        ),
        "arguments": JSONSchema(
            description="A list of arguments that the ability will accept.",
            type=JSONSchema.Type.ARRAY,
            items=JSONSchema(
                type=JSONSchema.Type.OBJECT,
                properties={
                    "name": JSONSchema(
                        description="The name of the argument.",
                        type=JSONSchema.Type.STRING,
                    ),
                    "type": JSONSchema(
                        description="The type of the argument. Must be a standard json schema type.",
                        type=JSONSchema.Type.STRING,
                    ),
                    "description": JSONSchema(
                        description="A detailed description of the argument and its uses.",
                        type=JSONSchema.Type.STRING,
                    ),
                },
            ),
        ),
        "required_arguments": JSONSchema(
            description="A list of the names of the arguments that are required.",
            type=JSONSchema.Type.ARRAY,
            items=JSONSchema(
                description="The names of the arguments that are required.",
                type=JSONSchema.Type.STRING,
            ),
        ),
        "package_requirements": JSONSchema(
            description="A list of the names of the Python packages that are required to execute the ability.",
            type=JSONSchema.Type.ARRAY,
            items=JSONSchema(
                description="The of the Python package that is required to execute the ability.",
                type=JSONSchema.Type.STRING,
            ),
        ),
        "code": JSONSchema(
            description="The Python code that will be executed when the ability is called.",
            type=JSONSchema.Type.STRING,
            required=True,
        ),
    }

    async def __call__(
        self,
        ability_name: str,
        description: str,
        arguments: list[dict],
        required_arguments: list[str],
        package_requirements: list[str],
        code: str,
    ) -> AbilityResult:
        raise NotImplementedError

```