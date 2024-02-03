# `.\AutoGPT\autogpts\autogpt\autogpt\config\config.py`

```py
# 导入必要的库和模块
"""Configuration class to store the state of bools for different scripts access."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional, Union

from auto_gpt_plugin_template import AutoGPTPluginTemplate
from colorama import Fore
from pydantic import Field, SecretStr, validator

import autogpt
from autogpt.core.configuration.schema import (
    Configurable,
    SystemSettings,
    UserConfigurable,
)
from autogpt.core.resource.model_providers.openai import (
    OPEN_AI_CHAT_MODELS,
    OpenAICredentials,
)
from autogpt.file_workspace import FileWorkspaceBackendName
from autogpt.logs.config import LoggingConfig
from autogpt.plugins.plugins_config import PluginsConfig
from autogpt.speech import TTSConfig

# 获取AutoGPT的根目录
PROJECT_ROOT = Path(autogpt.__file__).parent.parent
# 定义文件路径常量
AI_SETTINGS_FILE = Path("ai_settings.yaml")
AZURE_CONFIG_FILE = Path("azure.yaml")
PLUGINS_CONFIG_FILE = Path("plugins_config.yaml")
PROMPT_SETTINGS_FILE = Path("prompt_settings.yaml")

# 定义GPT模型常量
GPT_4_MODEL = "gpt-4"
GPT_3_MODEL = "gpt-3.5-turbo"

# 定义配置类
class Config(SystemSettings, arbitrary_types_allowed=True):
    name: str = "Auto-GPT configuration"
    description: str = "Default configuration for the Auto-GPT application."

    ########################
    # Application Settings #
    ########################
    # 设置应用程序的根目录和数据目录
    project_root: Path = PROJECT_ROOT
    app_data_dir: Path = project_root / "data"
    # 设置是否跳过新闻和重新提示
    skip_news: bool = False
    skip_reprompt: bool = False
    # 设置授权和退出键
    authorise_key: str = UserConfigurable(default="y", from_env="AUTHORISE_COMMAND_KEY")
    exit_key: str = UserConfigurable(default="n", from_env="EXIT_KEY")
    # 设置是否为非交互模式和是否启用聊天消息
    noninteractive_mode: bool = False
    chat_messages_enabled: bool = UserConfigurable(
        default=True, from_env=lambda: os.getenv("CHAT_MESSAGES_ENABLED") == "True"
    )

    # TTS配置
    tts_config: TTSConfig = TTSConfig()
    logging: LoggingConfig = LoggingConfig()

    # Workspace
    # 定义工作空间后端的名称，默认为本地
    workspace_backend: FileWorkspaceBackendName = UserConfigurable(
        default=FileWorkspaceBackendName.LOCAL,
        from_env=lambda: FileWorkspaceBackendName(v)
        if (v := os.getenv("WORKSPACE_BACKEND"))
        else None,
    )

    ##########################
    # Agent Control Settings #
    ##########################
    # 定义 AI 设置文件的路径，默认为 AI_SETTINGS_FILE
    ai_settings_file: Path = UserConfigurable(
        default=AI_SETTINGS_FILE,
        from_env=lambda: Path(f) if (f := os.getenv("AI_SETTINGS_FILE")) else None,
    )
    # 定义提示设置文件的路径，默认为 PROMPT_SETTINGS_FILE
    prompt_settings_file: Path = UserConfigurable(
        default=PROMPT_SETTINGS_FILE,
        from_env=lambda: Path(f) if (f := os.getenv("PROMPT_SETTINGS_FILE")) else None,
    )

    # 定义快速 LLM 模型，默认为 "gpt-3.5-turbo-16k"
    fast_llm: str = UserConfigurable(
        default="gpt-3.5-turbo-16k",
        from_env=lambda: os.getenv("FAST_LLM"),
    )
    # 定义智能 LLM 模型，默认为 "gpt-4"
    smart_llm: str = UserConfigurable(
        default="gpt-4",
        from_env=lambda: os.getenv("SMART_LLM"),
    )
    # 定义温度参数，默认为 0
    temperature: float = UserConfigurable(
        default=0,
        from_env=lambda: float(v) if (v := os.getenv("TEMPERATURE")) else None,
    )
    # 定义是否启用 OpenAI 函数，默认为 False
    openai_functions: bool = UserConfigurable(
        default=False, from_env=lambda: os.getenv("OPENAI_FUNCTIONS", "False") == "True"
    )
    # 定义嵌入模型，默认为 "text-embedding-ada-002"
    embedding_model: str = UserConfigurable(
        default="text-embedding-ada-002", from_env="EMBEDDING_MODEL"
    )
    # 定义浏览 Spacy 语言模型，默认为 "en_core_web_sm"
    browse_spacy_language_model: str = UserConfigurable(
        default="en_core_web_sm", from_env="BROWSE_SPACY_LANGUAGE_MODEL"
    )

    # 定义运行循环配置
    continuous_mode: bool = False
    continuous_limit: int = 0

    ##########
    # Memory #
    ##########
    # 定义内存后端，默认为 "json_file"
    memory_backend: str = UserConfigurable("json_file", from_env="MEMORY_BACKEND")
    # 定义内存索引，默认为 "auto-gpt-memory"
    memory_index: str = UserConfigurable("auto-gpt-memory", from_env="MEMORY_INDEX")
    # 定义 Redis 主机，默认为 "localhost"
    redis_host: str = UserConfigurable("localhost", from_env="REDIS_HOST")
    # 设置 Redis 服务器的端口号，默认为 6379，可以从环境变量中获取
    redis_port: int = UserConfigurable(
        default=6379,
        from_env=lambda: int(v) if (v := os.getenv("REDIS_PORT")) else None,
    )
    # 设置 Redis 服务器的密码，默认为空字符串，可以从环境变量中获取
    redis_password: str = UserConfigurable("", from_env="REDIS_PASSWORD")
    # 在启动时是否清空 Redis 数据，默认为 True，可以从环境变量中获取
    wipe_redis_on_start: bool = UserConfigurable(
        default=True,
        from_env=lambda: os.getenv("WIPE_REDIS_ON_START", "True") == "True",
    )

    ############
    # Commands #
    ############
    # 禁用的命令类别列表，默认为空列表，可以从环境变量中获取并安全拆分
    disabled_command_categories: list[str] = UserConfigurable(
        default_factory=list,
        from_env=lambda: _safe_split(os.getenv("DISABLED_COMMAND_CATEGORIES")),
    )

    # 文件操作
    # 是否限制在工作空间内，默认为 True，可以从环境变量中获取
    restrict_to_workspace: bool = UserConfigurable(
        default=True,
        from_env=lambda: os.getenv("RESTRICT_TO_WORKSPACE", "True") == "True",
    )
    # 是否允许下载文件，默认为 False
    allow_downloads: bool = False

    # Shell 命令
    # Shell 命令控制方式，默认为 "denylist"，可以从环境变量中获取
    shell_command_control: str = UserConfigurable(
        default="denylist", from_env="SHELL_COMMAND_CONTROL"
    )
    # 是否执行本地命令，默认为 False，可以从环境变量中获取
    execute_local_commands: bool = UserConfigurable(
        default=False,
        from_env=lambda: os.getenv("EXECUTE_LOCAL_COMMANDS", "False") == "True",
    )
    # Shell 命令拒绝列表，默认为 ["sudo", "su"]，可以从环境变量中获取并安全拆分
    shell_denylist: list[str] = UserConfigurable(
        default_factory=lambda: ["sudo", "su"],
        from_env=lambda: _safe_split(
            os.getenv("SHELL_DENYLIST", os.getenv("DENY_COMMANDS"))
        ),
    )
    # Shell 命令允许列表，默认为空列表，可以从环境变量中获取并安全拆分
    shell_allowlist: list[str] = UserConfigurable(
        default_factory=list,
        from_env=lambda: _safe_split(
            os.getenv("SHELL_ALLOWLIST", os.getenv("ALLOW_COMMANDS"))
        ),
    )

    # 文本转图片
    # 图片提供者，默认为 None，可以从环境变量中获取
    image_provider: Optional[str] = UserConfigurable(from_env="IMAGE_PROVIDER")
    # Huggingface 图像模型，默认为 "CompVis/stable-diffusion-v1-4"，可以从环境变量中获取
    huggingface_image_model: str = UserConfigurable(
        default="CompVis/stable-diffusion-v1-4", from_env="HUGGINGFACE_IMAGE_MODEL"
    )
    # SD WebUI URL，默认为 "http://localhost:7860"，可以从环境变量中获取
    sd_webui_url: Optional[str] = UserConfigurable(
        default="http://localhost:7860", from_env="SD_WEBUI_URL"
    )
    # 设置图像大小，默认为256
    image_size: int = UserConfigurable(
        default=256,
        from_env=lambda: int(v) if (v := os.getenv("IMAGE_SIZE")) else None,
    )

    # 音频转文本提供者，默认为"huggingface"
    audio_to_text_provider: str = UserConfigurable(
        default="huggingface", from_env="AUDIO_TO_TEXT_PROVIDER"
    )
    # Huggingface 音频转文本模型，可选
    huggingface_audio_to_text_model: Optional[str] = UserConfigurable(
        from_env="HUGGINGFACE_AUDIO_TO_TEXT_MODEL"
    )

    # 网页浏览设置
    # Selenium 浏览器，默认为"chrome"
    selenium_web_browser: str = UserConfigurable("chrome", from_env="USE_WEB_BROWSER")
    # 是否使用无头浏览器，默认为True
    selenium_headless: bool = UserConfigurable(
        default=True, from_env=lambda: os.getenv("HEADLESS_BROWSER", "True") == "True"
    )
    # 用户代理设置，默认为指定的用户代理字符串
    user_agent: str = UserConfigurable(
        default="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36",  # noqa: E501
        from_env="USER_AGENT",
    )

    ###################
    # 插件设置 #
    ###################
    # 插件目录，默认为"plugins"
    plugins_dir: str = UserConfigurable("plugins", from_env="PLUGINS_DIR")
    # 插件配置文件路径，默认为预定义的配置文件路径
    plugins_config_file: Path = UserConfigurable(
        default=PLUGINS_CONFIG_FILE,
        from_env=lambda: Path(f) if (f := os.getenv("PLUGINS_CONFIG_FILE")) else None,
    )
    # 插件配置对象，默认为空配置
    plugins_config: PluginsConfig = Field(
        default_factory=lambda: PluginsConfig(plugins={})
    )
    # 插件列表，默认为空列表，不包含在输出中
    plugins: list[AutoGPTPluginTemplate] = Field(default_factory=list, exclude=True)
    # 插件允许列表，默认为空列表，从环境变量中获取
    plugins_allowlist: list[str] = UserConfigurable(
        default_factory=list,
        from_env=lambda: _safe_split(os.getenv("ALLOWLISTED_PLUGINS")),
    )
    # 插件拒绝列表，默认为空列表，从环境变量中获取
    plugins_denylist: list[str] = UserConfigurable(
        default_factory=list,
        from_env=lambda: _safe_split(os.getenv("DENYLISTED_PLUGINS")),
    )
    # OpenAI 插件列表，默认为空列表，从环境变量中获取
    plugins_openai: list[str] = UserConfigurable(
        default_factory=list, from_env=lambda: _safe_split(os.getenv("OPENAI_PLUGINS"))
    )

    ###############
    # 凭证 #
    ###############
    # OpenAI
    # 定义一个可选的 OpenAICredentials 类型的变量 openai_credentials，默认为 None
    openai_credentials: Optional[OpenAICredentials] = None
    # 定义一个可选的 Path 类型的变量 azure_config_file，可由用户配置，默认为 AZURE_CONFIG_FILE
    azure_config_file: Optional[Path] = UserConfigurable(
        default=AZURE_CONFIG_FILE,
        from_env=lambda: Path(f) if (f := os.getenv("AZURE_CONFIG_FILE")) else None,
    )

    # Github
    # 定义一个可选的字符串类型的变量 github_api_key，可由用户配置，从环境变量中获取
    github_api_key: Optional[str] = UserConfigurable(from_env="GITHUB_API_KEY")
    # 定义一个可选的字符串类型的变量 github_username，可由用户配置，从环境变量中获取

    github_username: Optional[str] = UserConfigurable(from_env="GITHUB_USERNAME")

    # Google
    # 定义一个可选的字符串类型的变量 google_api_key，可由用户配置，从环境变量中获取
    google_api_key: Optional[str] = UserConfigurable(from_env="GOOGLE_API_KEY")
    # 定义一个可选的字符串类型的变量 google_custom_search_engine_id，可由用户配置，从环境变量中获取

    google_custom_search_engine_id: Optional[str] = UserConfigurable(
        from_env=lambda: os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID"),
    )

    # Huggingface
    # 定义一个可选的字符串类型的变量 huggingface_api_token，可由用户配置，从环境变量中获取
    huggingface_api_token: Optional[str] = UserConfigurable(
        from_env="HUGGINGFACE_API_TOKEN"
    )

    # Stable Diffusion
    # 定义一个可选的字符串类型的变量 sd_webui_auth，可由用户配置，从环境变量中获取
    sd_webui_auth: Optional[str] = UserConfigurable(from_env="SD_WEBUI_AUTH")

    # 验证器：验证 plugins 列表中的每个元素是否为 AutoGPTPluginTemplate 类型
    @validator("plugins", each_item=True)
    def validate_plugins(cls, p: AutoGPTPluginTemplate | Any):
        # 断言 p 是 AutoGPTPluginTemplate 类型的子类
        assert issubclass(
            p.__class__, AutoGPTPluginTemplate
        ), f"{p} does not subclass AutoGPTPluginTemplate"
        # 断言 p 的类名不是 "AutoGPTPluginTemplate"
        assert (
            p.__class__.__name__ != "AutoGPTPluginTemplate"
        ), f"Plugins must subclass AutoGPTPluginTemplate; {p} is a template instance"
        return p

    # 验证器：验证是否开启 openai_functions，并检查 smart_llm 是否支持 OpenAI Functions
    @validator("openai_functions")
    def validate_openai_functions(cls, v: bool, values: dict[str, Any]):
        if v:
            smart_llm = values["smart_llm"]
            # 断言 smart_llm 对应的模型支持函数调用 API
            assert OPEN_AI_CHAT_MODELS[smart_llm].has_function_call_api, (
                f"Model {smart_llm} does not support OpenAI Functions. "
                "Please disable OPENAI_FUNCTIONS or choose a suitable model."
            )
        return v
# 定义一个 ConfigBuilder 类，继承自 Configurable[Config] 泛型类
class ConfigBuilder(Configurable[Config]):
    # 默认设置为 Config 类的实例
    default_settings = Config()

    # 从环境变量构建配置的类方法，参数为项目根目录路径，默认为 PROJECT_ROOT
    @classmethod
    def build_config_from_env(cls, project_root: Path = PROJECT_ROOT) -> Config:
        """Initialize the Config class"""
        
        # 构建代理配置
        config = cls.build_agent_configuration()
        # 设置项目根目录路径
        config.project_root = project_root

        # 将相对路径转换为绝对路径
        for k in {
            "ai_settings_file",  # TODO: deprecate or repurpose
            "prompt_settings_file",  # TODO: deprecate or repurpose
            "plugins_config_file",  # TODO: move from project root
            "azure_config_file",  # TODO: move from project root
        }:
            setattr(config, k, project_root / getattr(config, k))

        # 如果 OpenAI 凭证存在且为 Azure 类型，且 Azure 配置文件存在，则加载 Azure 配置
        if (
            config.openai_credentials
            and config.openai_credentials.api_type == "azure"
            and (config_file := config.azure_config_file)
        ):
            config.openai_credentials.load_azure_config(config_file)

        # 加载插件配置
        config.plugins_config = PluginsConfig.load_config(
            config.plugins_config_file,
            config.plugins_denylist,
            config.plugins_allowlist,
        )

        # 返回配置
        return config


# 断言配置中是否设置了 OpenAI API 密钥，参数为 Config 类的实例
def assert_config_has_openai_api_key(config: Config) -> None:
    """Check if the OpenAI API key is set in config.py or as an environment variable."""
    # 如果没有配置 OpenAI 凭据
    if not config.openai_credentials:
        # 提示用户设置 OpenAI API 密钥
        print(
            Fore.RED
            + "Please set your OpenAI API key in .env or as an environment variable."
            + Fore.RESET
        )
        print("You can get your key from https://platform.openai.com/account/api-keys")
        # 用户输入 OpenAI API 密钥
        openai_api_key = input(
            "If you do have the key, please enter your OpenAI API key now:\n"
        )
        # 定义 OpenAI API 密钥的格式
        key_pattern = r"^sk-\w{48}"
        openai_api_key = openai_api_key.strip()
        # 检查用户输入的密钥是否符合格式
        if re.search(key_pattern, openai_api_key):
            # 将密钥设置为环境变量
            os.environ["OPENAI_API_KEY"] = openai_api_key
            # 创建 OpenAI 凭据对象
            config.openai_credentials = OpenAICredentials(
                api_key=SecretStr(openai_api_key)
            )
            # 提示用户成功设置 API 密钥
            print(
                Fore.GREEN
                + "OpenAI API key successfully set!\n"
                + Fore.YELLOW
                + "NOTE: The API key you've set is only temporary.\n"
                + "For longer sessions, please set it in .env file"
                + Fore.RESET
            )
        else:
            # 提示用户输入的 OpenAI API 密钥无效
            print("Invalid OpenAI API key!")
            # 退出程序
            exit(1)
# 定义一个函数，用于根据指定分隔符分割字符串。如果字符串为 None，则返回空列表。
def _safe_split(s: Union[str, None], sep: str = ",") -> list[str]:
    # 如果字符串为 None，则返回空列表
    if s is None:
        return []
    # 使用指定的分隔符对字符串进行分割，并返回结果列表
    return s.split(sep)
```