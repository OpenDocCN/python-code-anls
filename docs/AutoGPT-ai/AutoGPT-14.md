# AutoGPT源码解析 14

# `autogpts/autogpt/autogpt/models/context_item.py`

这段代码定义了一个名为“ContextItem”的抽象类，属于“ABC”中的一个子类。这个类包含三个抽象方法：“description”、“source”和“content”。

具体来说，这个类的“description”方法提供了一个字符串描述，这个描述将根据类的实现进行定义。同样，“source”方法提供了一个字符串，表示该“ContextItem”的来源，也可以根据类的实现进行定义。最后，“content”方法提供了一个字符串，表示该“ContextItem”的内容。

另外，这个类的“fmt”方法定义了一个格式化字符串的方法，这个方法根据“description”、“source”和“content”方法得到的信息进行格式化，并将它们拼接在一起，最后输出一个“fmt”字符串。

总之，这段代码定义了一个“ContextItem”类，用于表示在自动生成文本时需要考虑的一些信息，包括描述、来源和内容。这个类可以被用于自定义的自动化工具或者在自动生成文本时进行一些预处理。


```py
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from autogpt.commands.file_operations_utils import read_textual_file

logger = logging.getLogger(__name__)


class ContextItem(ABC):
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of the context item"""
        ...

    @property
    @abstractmethod
    def source(self) -> Optional[str]:
        """A string indicating the source location of the context item"""
        ...

    @property
    @abstractmethod
    def content(self) -> str:
        """The content represented by the context item"""
        ...

    def fmt(self) -> str:
        return (
            f"{self.description} (source: {self.source})\n"
            "```\n"
            f"{self.content}\n"
            "```py"
        )


```py

这段代码定义了一个名为 FileContextItem 的类，继承自 BaseModel 和 ContextItem 两个模型。

FileContextItem 类有两个方法，一个是 file\_path，它是通过 workspace\_path 和 file\_path\_in\_workspace 属性计算得出的路径。另一个是 description，它是通过 file\_path 和 description 属性计算得出的字符串。第三个方法是 source，它是通过 file\_path 和 source 属性计算得出的字符串。最后一个方法是 content，它是通过 read\_textual\_file 函数读取文件内容并返回的。

file\_path 的实现是通过 walkspace\_path 和 file\_path\_in\_workspace 属性计算得出的路径，这方便在需要时获取文件在 workspace 中的位置。description 和 source 的实现基本上是将 file\_path 属性中的内容进行了字符串处理，并输出了一个字符串。content 的实现则是通过调用 read\_textual\_file 函数读取文件内容，并在需要时将其返回。


```py
class FileContextItem(BaseModel, ContextItem):
    file_path_in_workspace: Path
    workspace_path: Path

    @property
    def file_path(self) -> Path:
        return self.workspace_path / self.file_path_in_workspace

    @property
    def description(self) -> str:
        return f"The current content of the file '{self.file_path_in_workspace}'"

    @property
    def source(self) -> str:
        return str(self.file_path_in_workspace)

    @property
    def content(self) -> str:
        return read_textual_file(self.file_path, logger)


```py

这段代码定义了一个名为FolderContextItem的类，继承自BaseModel和ContextItem类。FolderContextItem包含两个属性：path_in_workspace和workspace_path，分别表示 workspace 中包含此 folder 的路径和 workspace 的路径。

FolderContextItem还有一个名为path的属性，它是一个Path类型，使用了self.workspace_path 和 self.path_in_workspace 属性计算出来。

FolderContextItem还有一个名为description的属性，它返回了 folder 中所有文件名的列表，并按照一定的格式对它们进行了排序。

FolderContextItem还有一个名为source的属性，它返回了 folder 的完整路径，其中包含了文件名和文件夹路径。

FolderContextItem还有一个名为content的属性，它使用了workspace 和 path 的属性，对 folder 中的所有文件进行了遍历，并计算了每个文件的路径，然后将它们按照一定的格式对它们进行了排序，并返回了排序后的内容。


```py
class FolderContextItem(BaseModel, ContextItem):
    path_in_workspace: Path
    workspace_path: Path

    @property
    def path(self) -> Path:
        return self.workspace_path / self.path_in_workspace

    def __post_init__(self) -> None:
        assert self.path.exists(), "Selected path does not exist"
        assert self.path.is_dir(), "Selected path is not a directory"

    @property
    def description(self) -> str:
        return f"The contents of the folder '{self.path_in_workspace}' in the workspace"

    @property
    def source(self) -> str:
        return str(self.path_in_workspace)

    @property
    def content(self) -> str:
        items = [f"{p.name}{'/' if p.is_dir() else ''}" for p in self.path.iterdir()]
        items.sort()
        return "\n".join(items)


```py

这段代码定义了一个名为 `StaticContextItem` 的类，继承自 `BaseModel` 和 `ContextItem` 类。这个类的实例包含了以下字段：

- `item_description:` 字段，类型为 `str`，使用了 `alias` 参数，它的作用域是类的内部，并且不能被外面的代码访问或者修改。
- `item_source:` 字段，类型为 `str`，使用了 `alias` 参数，它的作用域是类的内部，并且不能被外面的代码访问或者修改。
- `item_content:` 字段，类型为 `str`，使用了 `alias` 参数，它的作用域是类的内部，并且不能被外面的代码访问或者修改。

这个类的目的是提供一个类来表示一个静态上下文中的项，例如一个界面组件、一个配置文件中的选项等等。这个类可以让用户在定义上下文的时候，只需要关注业务逻辑，而不需要关心具体的字段值是如何被决定的。


```py
class StaticContextItem(BaseModel, ContextItem):
    item_description: str = Field(alias="description")
    item_source: Optional[str] = Field(alias="source")
    item_content: str = Field(alias="content")

```py

# `autogpts/autogpt/autogpt/models/__init__.py`

很抱歉，我无法解释没有源代码的代码，因为缺乏上下文和解释，我无法理解代码的意图和具体实现。请提供代码以供我为您提供更好的帮助。


```py

```py

# `autogpts/autogpt/autogpt/plugins/plugins_config.py`

This is a Python class that manages the creation and manipulation of plugins configuration files.

It takes in four arguments:

* `plugins_config_file`: A file path to the configuration file, usually filled with yaml data, that contains the base configuration for the plugins.
* `plugins_denylist`: A list of strings that contain the disallowed plugin names that should be ignored by the plugin system.
* `plugins_allowlist`: A list of strings that contain the allowed plugin names that should be allowed to be used by the plugin system.

It has the following methods:

* `__init__`: Initializes the empty plugins configuration file and returns it.
* `get_plugins_config`: Retrieves the configuration for the given plugins and returns it.
* `set_plugins_denylist`: Sets the denylist for the plugins.
* `set_plugins_allowlist`: Sets the allowlist for the plugins.
* `create_empty_plugins_config`: Creates an empty plugins configuration file and fills it with values from the environment variables. This method is intended for backward compatibility and should be used in a development environment.
* `write_plugins_config`: Writes the plugins configuration file to the given file.
* `read_plugins_config`: Reads the plugins configuration file from the given file and returns it.
* `get_plugins_denylist_strings`: Returns a list of strings for the denylist.
* `get_plugins_allowlist_strings`: Returns a list of strings for the allowlist.

Note: The method `get_plugins_config` and the method `write_plugins_config` should be overridden in the base class, `PluginConfig` class, to provide a more robust interface for accessing the configuration file.


```py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import yaml
from pydantic import BaseModel

from autogpt.plugins.plugin_config import PluginConfig

logger = logging.getLogger(__name__)


class PluginsConfig(BaseModel):
    """Class for holding configuration of all plugins"""

    plugins: dict[str, PluginConfig]

    def __repr__(self):
        return f"PluginsConfig({self.plugins})"

    def get(self, name: str) -> Union[PluginConfig, None]:
        return self.plugins.get(name)

    def is_enabled(self, name) -> bool:
        plugin_config = self.plugins.get(name)
        return plugin_config is not None and plugin_config.enabled

    @classmethod
    def load_config(
        cls,
        plugins_config_file: Path,
        plugins_denylist: list[str],
        plugins_allowlist: list[str],
    ) -> "PluginsConfig":
        empty_config = cls(plugins={})

        try:
            config_data = cls.deserialize_config_file(
                plugins_config_file,
                plugins_denylist,
                plugins_allowlist,
            )
            if type(config_data) != dict:
                logger.error(
                    f"Expected plugins config to be a dict, got {type(config_data)}, continuing without plugins"
                )
                return empty_config
            return cls(plugins=config_data)

        except BaseException as e:
            logger.error(
                f"Plugin config is invalid, continuing without plugins. Error: {e}"
            )
            return empty_config

    @classmethod
    def deserialize_config_file(
        cls,
        plugins_config_file: Path,
        plugins_denylist: list[str],
        plugins_allowlist: list[str],
    ) -> dict[str, PluginConfig]:
        if not plugins_config_file.is_file():
            logger.warn("plugins_config.yaml does not exist, creating base config.")
            cls.create_empty_plugins_config(
                plugins_config_file,
                plugins_denylist,
                plugins_allowlist,
            )

        with open(plugins_config_file, "r") as f:
            plugins_config = yaml.load(f, Loader=yaml.FullLoader)

        plugins = {}
        for name, plugin in plugins_config.items():
            if type(plugin) == dict:
                plugins[name] = PluginConfig(
                    name=name,
                    enabled=plugin.get("enabled", False),
                    config=plugin.get("config", {}),
                )
            elif type(plugin) == PluginConfig:
                plugins[name] = plugin
            else:
                raise ValueError(f"Invalid plugin config data type: {type(plugin)}")
        return plugins

    @staticmethod
    def create_empty_plugins_config(
        plugins_config_file: Path,
        plugins_denylist: list[str],
        plugins_allowlist: list[str],
    ):
        """Create an empty plugins_config.yaml file. Fill it with values from old env variables."""
        base_config = {}

        logger.debug(f"Legacy plugin denylist: {plugins_denylist}")
        logger.debug(f"Legacy plugin allowlist: {plugins_allowlist}")

        # Backwards-compatibility shim
        for plugin_name in plugins_denylist:
            base_config[plugin_name] = {"enabled": False, "config": {}}

        for plugin_name in plugins_allowlist:
            base_config[plugin_name] = {"enabled": True, "config": {}}

        logger.debug(f"Constructed base plugins config: {base_config}")

        logger.debug(f"Creating plugin config file {plugins_config_file}")
        with open(plugins_config_file, "w+") as f:
            f.write(yaml.dump(base_config))
            return base_config

```py

# `autogpts/autogpt/autogpt/plugins/plugin_config.py`

这段代码定义了一个名为 `PluginConfig` 的类，旨在创建一个配置类，其中包含一个程序的单一插件的配置。

具体来说，这个类接受一个 `name` 字段，它是插件的唯一标识符。它还包含一个 `enabled` 字段，表示插件是否启用，它是根据 `pydantic.BaseModel` 的 `__post__` 方法定义的。最后，它包含一个 `config` 字段，它是根据 `pydantic.BaseModel` 的 `__post__` 方法定义的，可以包含任何类型的数据，以适应插件的配置需求。

`PluginConfig` 类是使用 `typing.Any` 类型声明的，这意味着它可以接受任何类型的数据，包括字符串、数字和数字。此外，它还继承自 `BaseModel` 类，继承了它的 `__post__` 方法定义的 `__BaseModel__` 方法。

这个 `PluginConfig` 类被用于定义一个程序中的单个插件的配置。通过将所需的配置数据传递给 `PluginConfig` 类的 `__post__` 方法，可以创建一个 `PluginConfig` 对象，从而使程序具有更好的可维护性和可扩展性。


```py
from typing import Any

from pydantic import BaseModel


class PluginConfig(BaseModel):
    """Class for holding configuration of a single plugin"""

    name: str
    enabled: bool = False
    config: dict[str, Any] = None

```py

# `autogpts/autogpt/autogpt/plugins/__init__.py`

这段代码是一个插件加载器，可以处理不同插件的加载。它通过以下几个步骤来实现：

1. 导入需要用到的库：from importlib.util import import modules
import inspect
import json
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, List
from urllib.parse import urlparse
from zipimport zipimporter

2. 定义了一个名为`HandlesLoadingOfPlugins`的类，这个类使用了`__future__`注解，表示在未来可能会用到的新特性。
3. 从`importlib.util`库中导入`modules`模块，用于加载和使用模块。
4. 从`inspect`库中导入`getsource`函数，用于获取对象的源代码。
5. 从`json`库中导入`json.dumps`函数，用于将对象的字符串转义序列化为JSON格式。
6. 从`logging`库中导入`logging.getLogger`函数，用于获取当前对象的日志输出。
7. 从`os`库中导入`os.path.abspath`函数，用于获取文件或目录的绝对路径。
8. 从`sys`库中导入`sys.path`目录，用于获取当前系统的库搜索路径。
9. 从`zipfile`库中导入`ZipFile`类，用于处理ZIP文件。
10. 从`typing`库中导入`TYPE_CHECKING`类型，用于检查输入的参数是否为`typing.T`类型。
11. 从`urllib.parse`库中导入`urlparse`函数，用于解析URL。
12. 从`zipimport`库中导入`ZipImporter`类，用于将ZIP文件中的内容导入到指定目录。

13. 定义了一个`load_plugins`方法，用于加载插件。这个方法需要一个参数`path`，表示插件的安装目录。
14. 在`load_plugins`方法中，首先使用`os.path.abspath`函数获取插件安装目录的绝对路径，然后使用`sys.path`目录中当前系统的库搜索路径，加载插件的源代码。
15. 最后，将加载的插件序列化为JSON格式，并输出到`logging.getLogger`对象的日志中。


```py
"""Handles loading of plugins."""
from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, List
from urllib.parse import urlparse
from zipimport import zipimporter

```py

这段代码使用了多个第三方库，包括 openapi_python_client、requests 和 auto_gpt_plugin_template。其中，openapi_python_client 是用于与 OpenAPI 文档交互的库，requests 是用于发送 HTTP 请求的库，auto_gpt_plugin_template 是用于自动生成 OpenAPI 文档的库。

这段代码的作用是定义了一个函数 inspect_zip_for_modules，该函数接收一个 zip 文件的路径作为参数，并返回 zip 文件中模块的名称。函数内部使用了两个小括号，分别表示开包函数内部要 inspect 的模块和外部要返回的模块。

inspect_zip_for_modules 函数的具体实现包括以下步骤：

1. 通过 requests 库发送 HTTP GET 请求，获取指定 zip 文件的内容。
2. 通过 zipfile 库读取 zip 文件内容。
3. 通过 zfile 对象，遍历 zip 文件中的所有文件名。
4. 对于每个文件名，通过 openapi_python_client 库检查其是否为Python定义的__init__.py文件，如果不是，则输出调试信息。
5. 如果 zip 文件中不存在定义的__init__.py文件，则输出调试信息。
6. 返回 zip 文件中定义的模块名称。

这段代码的作用是帮助开发者更方便地检查和排除 zip 文件中的问题，使得开发者在使用 auto_gpt_plugin_template 库时，能够正确地定位和排除问题。


```py
import openapi_python_client
import requests
from auto_gpt_plugin_template import AutoGPTPluginTemplate
from openapi_python_client.config import Config as OpenAPIConfig

if TYPE_CHECKING:
    from autogpt.config import Config

from autogpt.models.base_open_ai_plugin import BaseOpenAIPlugin

logger = logging.getLogger(__name__)


def inspect_zip_for_modules(zip_path: str, debug: bool = False) -> list[str]:
    """
    Inspect a zipfile for a modules.

    Args:
        zip_path (str): Path to the zipfile.
        debug (bool, optional): Enable debug logging. Defaults to False.

    Returns:
        list[str]: The list of module names found or empty list if none were found.
    """
    result = []
    with zipfile.ZipFile(zip_path, "r") as zfile:
        for name in zfile.namelist():
            if name.endswith("__init__.py") and not name.startswith("__MACOSX"):
                logger.debug(f"Found module '{name}' in the zipfile at: {name}")
                result.append(name)
    if len(result) == 0:
        logger.debug(f"Module '__init__.py' not found in the zipfile @ {zip_path}.")
    return result


```py

This is a function that retrieves an AI plugin from an OpenAPI specification. The function takes a URL as an input and returns a dictionary containing the plugin's manifest and OpenAPI specification.

Here's how the function works:

1. The function sends a GET request to the specified URL in a high-level HTTP client (e.g., `requests`).
2. If the request is successful (i.e., HTTP status code 200), the function parses the JSON response.
3. If the JSON response does not have a schema version of "v1", the function prints a warning message.
4. The function extracts the API type (i.e., "openapi") from the JSON response.
5. If the API type is "openapi", the function prints a warning message.
6. The function writes the AI plugin's manifest to a file.
7. The function reads the AI plugin's manifest from the file.
8. The function updates a dictionary (`manifests`) with the current AI plugin's manifest and OpenAPI specification.
9. The function returns the `manifests`.

This function uses the `requests` high-level HTTP client and the `json` library for parsing JSON.


```py
def write_dict_to_json_file(data: dict, file_path: str) -> None:
    """
    Write a dictionary to a JSON file.
    Args:
        data (dict): Dictionary to write.
        file_path (str): Path to the file.
    """
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def fetch_openai_plugins_manifest_and_spec(config: Config) -> dict:
    """
    Fetch the manifest for a list of OpenAI plugins.
        Args:
        urls (List): List of URLs to fetch.
    Returns:
        dict: per url dictionary of manifest and spec.
    """
    # TODO add directory scan
    manifests = {}
    for url in config.plugins_openai:
        openai_plugin_client_dir = f"{config.plugins_dir}/openai/{urlparse(url).netloc}"
        create_directory_if_not_exists(openai_plugin_client_dir)
        if not os.path.exists(f"{openai_plugin_client_dir}/ai-plugin.json"):
            try:
                response = requests.get(f"{url}/.well-known/ai-plugin.json")
                if response.status_code == 200:
                    manifest = response.json()
                    if manifest["schema_version"] != "v1":
                        logger.warn(
                            f"Unsupported manifest version: {manifest['schem_version']} for {url}"
                        )
                        continue
                    if manifest["api"]["type"] != "openapi":
                        logger.warn(
                            f"Unsupported API type: {manifest['api']['type']} for {url}"
                        )
                        continue
                    write_dict_to_json_file(
                        manifest, f"{openai_plugin_client_dir}/ai-plugin.json"
                    )
                else:
                    logger.warn(
                        f"Failed to fetch manifest for {url}: {response.status_code}"
                    )
            except requests.exceptions.RequestException as e:
                logger.warn(f"Error while requesting manifest from {url}: {e}")
        else:
            logger.info(f"Manifest for {url} already exists")
            manifest = json.load(open(f"{openai_plugin_client_dir}/ai-plugin.json"))
        if not os.path.exists(f"{openai_plugin_client_dir}/openapi.json"):
            openapi_spec = openapi_python_client._get_document(
                url=manifest["api"]["url"], path=None, timeout=5
            )
            write_dict_to_json_file(
                openapi_spec, f"{openai_plugin_client_dir}/openapi.json"
            )
        else:
            logger.info(f"OpenAPI spec for {url} already exists")
            openapi_spec = json.load(open(f"{openai_plugin_client_dir}/openapi.json"))
        manifests[url] = {"manifest": manifest, "openapi_spec": openapi_spec}
    return manifests


```py

这段代码定义了一个名为 `create_directory_if_not_exists` 的函数，它接受一个名为 `directory_path` 的字符串参数。函数的作用是在给定的目录路径是否存在时，创建该目录（如果目录不存在）。

函数首先检查给定的目录路径是否存在，如果目录不存在，它将尝试创建目录并输出一条调试消息。如果目录已存在，函数将输出一条通知消息。

函数的实现使用了 Python 的 `os` 和 `logging` 模块。`os.path.exists()` 函数用于检查给定的目录路径是否存在，如果存在，函数将返回 `True`，否则返回 `False`。`os.makedirs()` 函数用于创建目录，如果目录不存在，将自动创建目录并输出一条调试消息。`logging.debug()` 和 `logging.warn()` 函数用于在出现错误或警告时记录日志信息。


```py
def create_directory_if_not_exists(directory_path: str) -> bool:
    """
    Create a directory if it does not exist.
    Args:
        directory_path (str): Path to the directory.
    Returns:
        bool: True if the directory was created, else False.
    """
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            logger.debug(f"Created directory: {directory_path}")
            return True
        except OSError as e:
            logger.warn(f"Error creating directory {directory_path}: {e}")
            return False
    else:
        logger.info(f"Directory {directory_path} already exists")
        return True


```py

This is a function that creates an OpenAPI client for an application based on the specified manifest and specification. It uses the `openai` package to create the directory for the client if it does not already exist.

The function takes in a `Config` object and an optional `debug` parameter. The `Config` object is used to specify the configuration settings for the client, such as the project name and package name overrides.

The function returns a dictionary of data for each URL, including the manifest, specification, and client for that URL.


```py
def initialize_openai_plugins(
    manifests_specs: dict, config: Config, debug: bool = False
) -> dict:
    """
    Initialize OpenAI plugins.
    Args:
        manifests_specs (dict): per url dictionary of manifest and spec.
        config (Config): Config instance including plugins config
        debug (bool, optional): Enable debug logging. Defaults to False.
    Returns:
        dict: per url dictionary of manifest, spec and client.
    """
    openai_plugins_dir = f"{config.plugins_dir}/openai"
    if create_directory_if_not_exists(openai_plugins_dir):
        for url, manifest_spec in manifests_specs.items():
            openai_plugin_client_dir = f"{openai_plugins_dir}/{urlparse(url).hostname}"
            _meta_option = (openapi_python_client.MetaType.SETUP,)
            _config = OpenAPIConfig(
                **{
                    "project_name_override": "client",
                    "package_name_override": "client",
                }
            )
            prev_cwd = Path.cwd()
            os.chdir(openai_plugin_client_dir)

            if not os.path.exists("client"):
                client_results = openapi_python_client.create_new_client(
                    url=manifest_spec["manifest"]["api"]["url"],
                    path=None,
                    meta=_meta_option,
                    config=_config,
                )
                if client_results:
                    logger.warn(
                        f"Error creating OpenAPI client: {client_results[0].header} \n"
                        f" details: {client_results[0].detail}"
                    )
                    continue
            spec = importlib.util.spec_from_file_location(
                "client", "client/client/client.py"
            )
            module = importlib.util.module_from_spec(spec)

            try:
                spec.loader.exec_module(module)
            finally:
                os.chdir(prev_cwd)

            client = module.Client(base_url=url)
            manifest_spec["client"] = client
    return manifests_specs


```py

这段代码定义了一个名为`instantiate_openai_plugin_clients`的函数，它接受三个参数：`manifests_specs_clients`、`config`和`debug`。函数内部使用这两个参数来创建每个OpenAI插件的实例，并将它们存储在`plugins`字典中。

具体来说，函数内部首先定义了一个`plugins_specs_clients`字典，它包含每个URL对应的 manifest、spec 和 client。然后，函数内部定义了一个`config`变量，它是一个Config实例，其中包含了插件的配置。最后，函数内部定义了一个`debug`变量，它是一个布尔值，表示是否启用调试日志输出。

函数内部使用两个for循环来遍历`manifests_specs_clients`字典中的每个URL对应的manifest_spec_client。对于每个URL，函数内部创建一个`BaseOpenAIPlugin`实例，并将它添加到`plugins_specs_clients`字典中对应的URL。

最终，函数返回一个以URL为键的`plugins`字典，其中包含每个URL对应的所有OpenAI插件实例。


```py
def instantiate_openai_plugin_clients(
    manifests_specs_clients: dict, config: Config, debug: bool = False
) -> dict:
    """
    Instantiates BaseOpenAIPlugin instances for each OpenAI plugin.
    Args:
        manifests_specs_clients (dict): per url dictionary of manifest, spec and client.
        config (Config): Config instance including plugins config
        debug (bool, optional): Enable debug logging. Defaults to False.
    Returns:
          plugins (dict): per url dictionary of BaseOpenAIPlugin instances.

    """
    plugins = {}
    for url, manifest_spec_client in manifests_specs_clients.items():
        plugins[url] = BaseOpenAIPlugin(manifest_spec_client)
    return plugins


```py

This is a Python implementation of the `auto_api.plugins.PluginRegistry` class from the AutoGPT and OpenAI提醒大家插件系统的自动补全编程指南。

这个函数接受两个参数。第一个参数`config`是插件配置文件，第二个参数`debug`是调试模式，如果开启，则会输出调试信息。

函数内部首先定义了一系列方便用户查看插件配置的函数，如`is_enabled()`，`Enabled()`，`PluginModuleName()`等。

接着，函数使用`getattr()`函数获取了`__name__()`方法，并将其作为参数传递给`BaseOpenAIPlugin`类的构造函数。

接下来，函数使用`fetch_openai_plugins_manifest_and_spec()`函数获取OpenAI插件清单文件中的元数据，并检查配置文件中是否定义了相应的插件。如果是，函数会使用`initialize_openai_plugins()`函数加载插件，并检查插件是否启用。

最后，函数检查插件是否启用，并在启用的情况下输出插件信息。如果插件不启用，函数将输出调试信息。

插件加载器只加载了经过验证的插件，并在插件启用时加载它们。


```py
def scan_plugins(config: Config, debug: bool = False) -> List[AutoGPTPluginTemplate]:
    """Scan the plugins directory for plugins and loads them.

    Args:
        config (Config): Config instance including plugins config
        debug (bool, optional): Enable debug logging. Defaults to False.

    Returns:
        List[Tuple[str, Path]]: List of plugins.
    """
    loaded_plugins = []
    # Generic plugins
    plugins_path = Path(config.plugins_dir)

    plugins_config = config.plugins_config
    # Directory-based plugins
    for plugin_path in [f.path for f in os.scandir(config.plugins_dir) if f.is_dir()]:
        # Avoid going into __pycache__ or other hidden directories
        if plugin_path.startswith("__"):
            continue

        plugin_module_path = plugin_path.split(os.path.sep)
        plugin_module_name = plugin_module_path[-1]
        qualified_module_name = ".".join(plugin_module_path)

        try:
            __import__(qualified_module_name)
        except:
            logger.error(f"Failed to load {qualified_module_name}")
            continue
        plugin = sys.modules[qualified_module_name]

        if not plugins_config.is_enabled(plugin_module_name):
            logger.warn(
                f"Plugin folder {plugin_module_name} found but not configured. If this is a legitimate plugin, please add it to plugins_config.yaml (key: {plugin_module_name})."
            )
            continue

        for _, class_obj in inspect.getmembers(plugin):
            if (
                hasattr(class_obj, "_abc_impl")
                and AutoGPTPluginTemplate in class_obj.__bases__
            ):
                loaded_plugins.append(class_obj())

    # Zip-based plugins
    for plugin in plugins_path.glob("*.zip"):
        if moduleList := inspect_zip_for_modules(str(plugin), debug):
            for module in moduleList:
                plugin = Path(plugin)
                module = Path(module)
                logger.debug(f"Zipped Plugin: {plugin}, Module: {module}")
                zipped_package = zipimporter(str(plugin))
                try:
                    zipped_module = zipped_package.load_module(str(module.parent))
                except:
                    logger.error(f"Failed to load {str(module.parent)}")

                for key in dir(zipped_module):
                    if key.startswith("__"):
                        continue

                    a_module = getattr(zipped_module, key)
                    if not inspect.isclass(a_module):
                        continue

                    if (
                        issubclass(a_module, AutoGPTPluginTemplate)
                        and a_module.__name__ != "AutoGPTPluginTemplate"
                    ):
                        plugin_name = a_module.__name__
                        plugin_configured = plugins_config.get(plugin_name) is not None
                        plugin_enabled = plugins_config.is_enabled(plugin_name)

                        if plugin_configured and plugin_enabled:
                            logger.debug(
                                f"Loading plugin {plugin_name}. Enabled in plugins_config.yaml."
                            )
                            loaded_plugins.append(a_module())
                        elif plugin_configured and not plugin_enabled:
                            logger.debug(
                                f"Not loading plugin {plugin_name}. Disabled in plugins_config.yaml."
                            )
                        elif not plugin_configured:
                            logger.warn(
                                f"Not loading plugin {plugin_name}. Key '{plugin_name}' was not found in plugins_config.yaml. "
                                f"Zipped plugins should use the class name ({plugin_name}) as the key."
                            )
                    else:
                        if (
                            module_name := getattr(a_module, "__name__", str(a_module))
                        ) != "AutoGPTPluginTemplate":
                            logger.debug(
                                f"Skipping '{module_name}' because it doesn't subclass AutoGPTPluginTemplate."
                            )

    # OpenAI plugins
    if config.plugins_openai:
        manifests_specs = fetch_openai_plugins_manifest_and_spec(config)
        if manifests_specs.keys():
            manifests_specs_clients = initialize_openai_plugins(
                manifests_specs, config, debug
            )
            for url, openai_plugin_meta in manifests_specs_clients.items():
                if not plugins_config.is_enabled(url):
                    logger.warn(
                        f"OpenAI Plugin {plugin_module_name} found but not configured"
                    )
                    continue

                plugin = BaseOpenAIPlugin(openai_plugin_meta)
                loaded_plugins.append(plugin)

    if loaded_plugins:
        logger.info(f"\nPlugins found: {len(loaded_plugins)}\n" "--------------------")
    for plugin in loaded_plugins:
        logger.info(f"{plugin._name}: {plugin._version} - {plugin._description}")
    return loaded_plugins

```py

# `autogpts/autogpt/autogpt/processing/html.py`

这段代码定义了一个名为 `extract_hyperlinks` 的函数，它接受一个 BeautifulSoup 对象 `soup` 和一个基础 URL `base_url`。这个函数的作用是提取出 soup 中所有的链接，并将链接的文本和链接指向的 URL 组合成一个元组返回。

具体来说，这个函数首先使用 `find_all` 方法来找到 soup 中所有的链接，然后遍历链接，提取链接的文本和链接指向的 URL，最后将它们组合成一个元组并返回。它还使用了 `urljoin` 方法来将链接的 URL 连接到基础 URL。


```py
"""HTML processing functions"""
from __future__ import annotations

from bs4 import BeautifulSoup
from requests.compat import urljoin


def extract_hyperlinks(soup: BeautifulSoup, base_url: str) -> list[tuple[str, str]]:
    """Extract hyperlinks from a BeautifulSoup object

    Args:
        soup (BeautifulSoup): The BeautifulSoup object
        base_url (str): The base URL

    Returns:
        List[Tuple[str, str]]: The extracted hyperlinks
    """
    return [
        (link.text, urljoin(base_url, link["href"]))
        for link in soup.find_all("a", href=True)
    ]


```py

这段代码定义了一个名为 `format_hyperlinks` 的函数，它接受一个名为 `hyperlinks` 的列表参数。函数的作用是将传入的 `hyperlinks` 列表中的每个链接字符串格式化为字符串形式，其中链接字符串和链接 URL 分别存储在 `link_text` 和 `link_url` 变量中。最终返回一个新的列表，其中包含所有格式化后的链接字符串。


```py
def format_hyperlinks(hyperlinks: list[tuple[str, str]]) -> list[str]:
    """Format hyperlinks to be displayed to the user

    Args:
        hyperlinks (List[Tuple[str, str]]): The hyperlinks to format

    Returns:
        List[str]: The formatted hyperlinks
    """
    return [f"{link_text.strip()} ({link_url})" for link_text, link_url in hyperlinks]

```py

# `autogpts/autogpt/autogpt/processing/text.py`

这段代码是一个Python程序，定义了一些文本处理函数，包括日志、数学和类型定义。

1. `import logging`: 引入logging库，用于记录对话框中的信息和警告。
2. `import math`: 引入math库，用于数学计算。
3. `from typing import Iterator, Optional, TypeVar`: 引入typing库，用于定义输入和输出迭代器中的类型变量。
4. `import spacy`: 导入spacy库，以便从SpaCy模型中提取文本。
5. `from autogpt.config import Config`: 从autogpt库中导入Config库，以便初始化自动生成的配置。
6. `from autogpt.core.prompting import ChatPrompt`: 从autogpt库中导入ChatPrompt库，以便在对话框中获取用户输入。
7. `from autogpt.core.resource.model_providers import (` 导入autogpt库中的ModelProvider类，以便获取预训练模型。
8. `ChatMessage`: 定义一个接口，用于表示对话框中的消息。
9. `ChatModelProvider`: 定义一个接口，用于表示预训练模型的提供者。
10. `ModelTokenizer`: 定义一个接口，用于表示用于生成文本的tokenizer。

该程序的主要作用是定义文本处理函数，包括从用户输入中获取对话框中的消息，并使用预训练的模型来生成文本。


```py
"""Text processing functions"""
import logging
import math
from typing import Iterator, Optional, TypeVar

import spacy

from autogpt.config import Config
from autogpt.core.prompting import ChatPrompt
from autogpt.core.resource.model_providers import (
    ChatMessage,
    ChatModelProvider,
    ModelTokenizer,
)

```py

这段代码定义了一个名为 "logger" 的日志输出类和一个名为 "T" 的抽象类型变量。接下来定义了一个名为 "batch" 的函数，该函数接受一个包含多个 "T" 类型元素的列表 "sequence"，以及一个最大批处理长度 "max_batch_length" 和一个默认为 0 的 overlap 设置。

该函数的作用是将传入的 "sequence" 中的元素 batched，使得每个批次的长度都为 "max_batch_length"，可能会有最后一个批次长度不足 "max_batch_length"，这种情况下会引发出一个 "ValueError"。函数的实现使用了Python标准库中的 "yield" 语句，用于生成一个生成器函数，将 batched后的 "sequence" 中的每一段 "T" 元素作为生成器的 output 值，并随着生成器函数的遍历进度输出。


```py
logger = logging.getLogger(__name__)

T = TypeVar("T")


def batch(
    sequence: list[T], max_batch_length: int, overlap: int = 0
) -> Iterator[list[T]]:
    """Batch data from iterable into slices of length N. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if max_batch_length < 1:
        raise ValueError("n must be at least one")
    for i in range(0, len(sequence), max_batch_length - overlap):
        yield sequence[i : i + max_batch_length]


```py

这段代码定义了一个名为 `chunk_content` 的函数，用于将给定文本内容拆分成等长度的片段。

函数接收四个参数：

- `content`: 要分割的文本内容。
- `max_chunk_length`: 每个片段的最大长度，以字节为单位。
- `tokenizer`: 用于对文本进行分词的模型。
- `with_overlap`: 一个布尔值，表示是否允许片段的覆盖。如果为 `True`，则允许覆盖，否则不允许。

函数内部首先通过 `tokenizer.encode` 方法将文本转换为模型可读取的格式，并计算出文本的总长度。然后，函数根据 `max_chunk_length` 和 `with_overlap` 参数计算出每个片段的最大长度，并将 `MAX_OVERLAP` 设置为 `200`。接下来，函数根据计算出的最大片段长度和 `n_chunks` 计算出每个片段的实际最大长度，并设置 `chunk_length` 为 `max_chunk_length - chunk_length`。然后，函数使用 `batch` 函数将文本分成等长度的片段，并将片段传递给 `tokenizer.decode` 函数以获取每个片段的编码。

函数的作用是将给定文本内容拆分成等长度的片段，可以用于对文本进行分词、进行编号等操作。


```py
def chunk_content(
    content: str,
    max_chunk_length: int,
    tokenizer: ModelTokenizer,
    with_overlap: bool = True,
) -> Iterator[tuple[str, int]]:
    """Split content into chunks of approximately equal token length."""

    MAX_OVERLAP = 200  # limit overlap to save tokens

    tokenized_text = tokenizer.encode(content)
    total_length = len(tokenized_text)
    n_chunks = math.ceil(total_length / max_chunk_length)

    chunk_length = math.ceil(total_length / n_chunks)
    overlap = min(max_chunk_length - chunk_length, MAX_OVERLAP) if with_overlap else 0

    for token_batch in batch(tokenized_text, chunk_length + overlap, overlap):
        yield tokenizer.decode(token_batch), len(token_batch)


```py

This is a Python implementation of a `llm_provider` that can provide summaries of text messages using the language model provided by the `llm_provider.rollib` package. The `summarization_prompt` is a message that the user must provide before the text can be summarized. The `text` is the text message that the user wants to summarize.

The `create_chat_completion` method of the `llm_provider.rollib` package is used to summarize the text. The method takes several parameters such as `model_prompt` which is a list of messages, `model` which is the language model to use for summarization, `temperature` which is the temperature of the summary, and `max_tokens` which limits the number of tokens (such as words) that can be used in the summary.

The `summarize_text` function is then used to summarize the text. This function takes the text and the prompt as input and returns a tuple of the summary and a flag indicating if the summary is complete.

The `logger` function is used to log information about the summary process.

The `llm_provider` is initialized with the path to the language model and the tokenizer for the language model.

The `summarize_text` function summarizes the text by breaking it up into chunks and summarizing each chunk. The chunks are then compared to the summary to determine if a summary should be added to the list of summaries or not.


```py
async def summarize_text(
    text: str,
    llm_provider: ChatModelProvider,
    config: Config,
    instruction: Optional[str] = None,
    question: Optional[str] = None,
) -> tuple[str, None | list[tuple[str, str]]]:
    """Summarize text using the OpenAI API

    Args:
        text (str): The text to summarize
        config (Config): The config object
        instruction (str): Additional instruction for summarization, e.g. "focus on information related to polar bears", "omit personal information contained in the text"
        question (str): Question to answer in the summary

    Returns:
        str: The summary of the text
        list[(summary, chunk)]: Text chunks and their summary, if the text was chunked.
            None otherwise.
    """
    if not text:
        raise ValueError("No text to summarize")

    if instruction and question:
        raise ValueError("Parameters 'question' and 'instructions' cannot both be set")

    model = config.fast_llm

    if question:
        instruction = (
            f'include any information that can be used to answer the question "{question}". '
            "Do not directly answer the question itself"
        )

    summarization_prompt = ChatPrompt(messages=[])

    text_tlength = llm_provider.count_tokens(text, model)
    logger.info(f"Text length: {text_tlength} tokens")

    # reserve 50 tokens for summary prompt, 500 for the response
    max_chunk_length = llm_provider.get_token_limit(model) - 550
    logger.info(f"Max chunk length: {max_chunk_length} tokens")

    if text_tlength < max_chunk_length:
        # summarization_prompt.add("user", text)
        summarization_prompt.messages.append(
            ChatMessage.user(
                "Write a concise summary of the following text"
                f"{f'; {instruction}' if instruction is not None else ''}:"
                "\n\n\n"
                f'LITERAL TEXT: """{text}"""'
                "\n\n\n"
                "CONCISE SUMMARY: The text is best summarized as"
                # "Only respond with a concise summary or description of the user message."
            )
        )

        summary = (
            await llm_provider.create_chat_completion(
                model_prompt=summarization_prompt.messages,
                model_name=model,
                temperature=0,
                max_tokens=500,
            )
        ).response["content"]

        logger.debug(f"\n{'-'*16} SUMMARY {'-'*17}\n{summary}\n{'-'*42}\n")
        return summary.strip(), None

    summaries: list[str] = []
    chunks = list(
        split_text(
            text,
            config=config,
            max_chunk_length=max_chunk_length,
            tokenizer=llm_provider.get_tokenizer(model),
        )
    )

    for i, (chunk, chunk_length) in enumerate(chunks):
        logger.info(
            f"Summarizing chunk {i + 1} / {len(chunks)} of length {chunk_length} tokens"
        )
        summary, _ = await summarize_text(
            text=chunk,
            instruction=instruction,
            llm_provider=llm_provider,
            config=config,
        )
        summaries.append(summary)

    logger.info(f"Summarized {len(chunks)} chunks")

    summary, _ = await summarize_text(
        "\n\n".join(summaries),
        llm_provider=llm_provider,
        config=config,
    )
    return summary.strip(), [
        (summaries[i], chunks[i][0]) for i in range(0, len(chunks))
    ]


```py

This code appears to be a Python implementation of a text chunking algorithm. It is specifically designed to handle long sentences that contain multiple long words, with the goal of being able to efficiently splits such sentences into smaller chunks while preserving the meaning of the sentences.

The code starts by defining some constants, such as `max_chunk_length`, `expected_chunk_length`, `sentence_length`, `target_chunk_length`, and `chunk_content()`. It then enters a loop that splits the long sentence into chunks, appending each chunk (a sequence of words) to a list called `current_chunk`, and updating the length of each chunk.

The loop continues until the end of the sentence is reached, at which point the code checks whether the remaining chunk is longer than the maximum allowed chunk length, and splits it up if necessary. It also checks whether the last sentence of the chunk is shorter than half the chunk length, and splits it up if it is.

The code also includes a block of commented out code indicating that there may be a bug in the code where sentences are being splayed together instead of being split, and another block of code where the last sentence is being split up instead of being added to the current chunk.


```py
def split_text(
    text: str,
    config: Config,
    max_chunk_length: int,
    tokenizer: ModelTokenizer,
    with_overlap: bool = True,
) -> Iterator[tuple[str, int]]:
    """Split text into chunks of sentences, with each chunk not exceeding the maximum length

    Args:
        text (str): The text to split
        for_model (str): The model to chunk for; determines tokenizer and constraints
        config (Config): The config object
        with_overlap (bool, optional): Whether to allow overlap between chunks
        max_chunk_length (int, optional): The maximum length of a chunk

    Yields:
        str: The next chunk of text

    Raises:
        ValueError: when a sentence is longer than the maximum length
    """
    text_length = len(tokenizer.encode(text))

    if text_length < max_chunk_length:
        yield text, text_length
        return

    n_chunks = math.ceil(text_length / max_chunk_length)
    target_chunk_length = math.ceil(text_length / n_chunks)

    nlp: spacy.language.Language = spacy.load(config.browse_spacy_language_model)
    nlp.add_pipe("sentencizer")
    doc = nlp(text)
    sentences = [sentence.text.strip() for sentence in doc.sents]

    current_chunk: list[str] = []
    current_chunk_length = 0
    last_sentence = None
    last_sentence_length = 0

    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_length = len(tokenizer.encode(sentence))
        expected_chunk_length = current_chunk_length + 1 + sentence_length

        if (
            expected_chunk_length < max_chunk_length
            # try to create chunks of approximately equal size
            and expected_chunk_length - (sentence_length / 2) < target_chunk_length
        ):
            current_chunk.append(sentence)
            current_chunk_length = expected_chunk_length

        elif sentence_length < max_chunk_length:
            if last_sentence:
                yield " ".join(current_chunk), current_chunk_length
                current_chunk = []
                current_chunk_length = 0

                if with_overlap:
                    overlap_max_length = max_chunk_length - sentence_length - 1
                    if last_sentence_length < overlap_max_length:
                        current_chunk += [last_sentence]
                        current_chunk_length += last_sentence_length + 1
                    elif overlap_max_length > 5:
                        # add as much from the end of the last sentence as fits
                        current_chunk += [
                            list(
                                chunk_content(
                                    content=last_sentence,
                                    max_chunk_length=overlap_max_length,
                                    tokenizer=tokenizer,
                                )
                            ).pop()[0],
                        ]
                        current_chunk_length += overlap_max_length + 1

            current_chunk += [sentence]
            current_chunk_length += sentence_length

        else:  # sentence longer than maximum length -> chop up and try again
            sentences[i : i + 1] = [
                chunk
                for chunk, _ in chunk_content(sentence, target_chunk_length, tokenizer)
            ]
            continue

        i += 1
        last_sentence = sentence
        last_sentence_length = sentence_length

    if current_chunk:
        yield " ".join(current_chunk), current_chunk_length

```py

# `autogpts/autogpt/autogpt/processing/__init__.py`

很抱歉，我没有看到您提供代码。如果您可以提供代码，我将非常乐意为您解释其作用。


```py

```py

# `autogpts/autogpt/autogpt/prompts/default_prompts.py`

这段代码是一个Python脚本，它提供了一个帮助用户生成旨在最大化其自动售货机业务市场营销的5个目标和一个适当的基于角色的名称的建议。目标包括：

1. 问题解决，优先级排序，计划和执行以解决您的市场营销需求，作为您的虚拟首席市场营销官。
2. 利用人工智能技术，使业务能够发展壮大。
3. 提高客户意识，提高转化率，增加销售。
4. 通过优化社交媒体渠道和广告策略，提高品牌知名度和互动性，增加网站流量。
5. 利用数据分析和预测，提供有关市场营销活动的实时反馈，帮助您做出明智的决策。

脚本中还包含一个DEFAULT_SYSTEM_PROMPT_AICONFIG_AUTOMATIC变量，它是一个包含系统提示自动配置AI系统命令的模板，用于在用户指定任务时自动生成任务描述。


```py
#########################Setup.py#################################

DEFAULT_SYSTEM_PROMPT_AICONFIG_AUTOMATIC = """
Your task is to devise up to 5 highly effective goals and an appropriate role-based name (_GPT) for an autonomous agent, ensuring that the goals are optimally aligned with the successful completion of its assigned task.

The user will provide the task, you will provide only the output in the exact format specified below with no explanation or conversation.

Example input:
Help me with marketing my business

Example output:
Name: CMOGPT
Description: a professional digital marketer AI that assists Solopreneurs in growing their businesses by providing world-class expertise in solving marketing problems for SaaS, content products, agencies, and more.
Goals:
- Engage in effective problem-solving, prioritization, planning, and supporting execution to address your marketing needs as your virtual Chief Marketing Officer.

```py

This code appears to be a Python script that responds to user prompts or tasks by providing自动化的、具体的、简洁的提示和建议，以帮助用户做出明智的决策，而无需使用套话或过于冗长的解释。它的目的是提供一个清晰的、易于理解的输出，以回答用户的问题或指导他们完成任务。

该脚本包含了两个主要的功能：1）根据用户提供的任务或问题，自动生成具体的、行动able的建议；2）在用户面临不明确的信息或不确定性时，主动提出建议，确保营销策略保持方向。

具体来说，该脚本可以执行以下操作：

1. 如果用户提供了一个任务或问题，它将根据所提供的信息生成一个包含明确、简洁建议的输出。这些建议将以系统提示的格式出现，用户只需按照要求输出相应的信息即可。

2. 如果用户面临不明确的信息或不确定性，该脚本将主动提出建议，以帮助用户更好地理解问题或解决问题。这些建议将基于用户提供的信息进行生成，以确保用户可以按照建议采取相应的行动。

3. 脚本会自动生成一个关于某个主题的 Wikipedia 风格的文章。用户可以通过输入一个主题，来生成关于该主题的系统文章。

总结起来，该脚本是一个自动化的、多功能的工具，旨在帮助用户在需要时快速获得针对所面临的问题或任务的解答和建议。


```py
- Provide specific, actionable, and concise advice to help you make informed decisions without the use of platitudes or overly wordy explanations.

- Identify and prioritize quick wins and cost-effective campaigns that maximize results with minimal time and budget investment.

- Proactively take the lead in guiding you and offering suggestions when faced with unclear information or uncertainty to ensure your marketing strategy remains on track.
"""

DEFAULT_TASK_PROMPT_AICONFIG_AUTOMATIC = (
    "Task: '{{user_prompt}}'\n"
    "Respond only with the output in the exact format specified in the system prompt, with no explanation or conversation.\n"
)

DEFAULT_USER_DESIRE_PROMPT = "Write a wikipedia style article about the project: https://github.com/significant-gravitas/AutoGPT"  # Default prompt

```py

# `autogpts/autogpt/autogpt/prompts/prompt.py`

这段代码定义了一个名为DEFAULT_TRIGGERING_PROMPT的常量，它是一个字符串。

DEFAULT_TRIGGERING_PROMPT的值使用了多个参数字符串，分别是：

"Determine exactly one command to use next based on the given goals "

"and respond using the JSON schema specified previously:"

这个字符串是一个触发器，它用于定义TriggeringPrompt函数的输入参数。在这个函数中，可能会有一些用户定义的目标和进度，以及一个指定的JSONschema。通过对这些参数的值进行比较，DEFAULT_TRIGGERING_PROMPT函数会返回一个字符串，用于指定下一个命令应该使用哪个命令，同时也是基于用户定义的目标和进度以及指定的JSONschema来确定的。


```py
DEFAULT_TRIGGERING_PROMPT = (
    "Determine exactly one command to use next based on the given goals "
    "and the progress you have made so far, "
    "and respond using the JSON schema specified previously:"
)

```py

# `autogpts/autogpt/autogpt/prompts/utils.py`

这段代码定义了两个函数，分别是 `format_numbered_list` 和 `indent`。它们的作用如下：

1. `format_numbered_list` 函数接收一个列表 `items`，以及一个起始索引 `start_at`。这个函数的作用是 format 列表 `items` 中每个元素的数值，并将它们连接成一个字符串，使得每个元素都带有一个 `_` 占位符，然后按照它们在列表中的索引号插入这个字符串。最终返回一个字符串，它将所有元素连接成一个大的字符串，然后在字符串中插入 `start_at` 个 `_` 占位符，使得字符串中的每个元素看起来像是一个数字。
2. `indent` 函数接收一个字符串 `content`，以及一个 indentation 值 `indentation`。这个函数的作用是将 `content` 中的所有换行符替换成 `indentation` 中的字符，然后返回 `indentation`。

这两个函数的具体实现是为了让一个简单的数字列表更加易于阅读和维护。通过将数字列表转换成字符串，可以使得数字列表更容易与文档、代码中使用 ' numbers' 这个词更加平易，更加符合代码规范。通过通过字符串格式化，可以让数字列表更加规范，数字更加具有描述性，而且更加易于阅读，特别是在多行字符串中。通过通过 indent，可以增加代码的缩进层次，更加符合文档，也更加易于阅读和理解。


```py
from typing import Any


def format_numbered_list(items: list[Any], start_at: int = 1) -> str:
    return "\n".join(f"{i}. {str(item)}" for i, item in enumerate(items, start_at))


def indent(content: str, indentation: int | str = 4) -> str:
    if type(indentation) == int:
        indentation = " " * indentation
    return indentation + content.replace("\n", f"\n{indentation}")  # type: ignore

```py

# `autogpts/autogpt/autogpt/prompts/__init__.py`

很抱歉，我不能直接解释代码的作用，因为大多数代码都是使用特定的编程语言编写的，缺乏上下文和环境，我无法确定它会发生什么。如果您能提供更多信息，例如代码是在哪个程序中运行的，以及它达到了什么结果，我会尽力帮助您理解它的作用。


```py

```py

# `autogpts/autogpt/autogpt/speech/base.py`

这段代码定义了一个名为`VoiceBase`的类，作为所有语音类的基类。这个基类有一个`__init__`方法来初始化语音类的实例，包括API密钥、语音数据等。有一个`say`方法，用于在特定情境下输出文本并播放声音。还有一个名为`_setup`的私有方法，用于设置语音类的实例。

从这段代码中，我们可以看出这个基类创建了一个包含多个方法的类，这些方法可以用来定义和实现一个完整的语音应用程序。例如，要使用这个基类来实现一个智能语音助手，可以创建一个`VoiceBase`实例，并在其中添加不同的语音数据和功能。


```py
"""Base class for all voice classes."""
from __future__ import annotations

import abc
import re
from threading import Lock


class VoiceBase:
    """
    Base class for all voice classes.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the voice class.
        """
        self._url = None
        self._headers = None
        self._api_key = None
        self._voices = []
        self._mutex = Lock()
        self._setup(*args, **kwargs)

    def say(self, text: str, voice_index: int = 0) -> bool:
        """
        Say the given text.

        Args:
            text (str): The text to say.
            voice_index (int): The index of the voice to use.
        """
        text = re.sub(
            r"\b(?:https?://[-\w_.]+/?\w[-\w_.]*\.(?:[-\w_.]+/?\w[-\w_.]*\.)?[a-z]+(?:/[-\w_.%]+)*\b(?!\.))",
            "",
            text,
        )
        with self._mutex:
            return self._speech(text, voice_index)

    @abc.abstractmethod
    def _setup(self, *args, **kwargs) -> None:
        """
        Setup the voices, API key, etc.
        """

    @abc.abstractmethod
    def _speech(self, text: str, voice_index: int = 0) -> bool:
        """
        Play the given text.

        Args:
            text (str): The text to play.
        """

```py

# `autogpts/autogpt/autogpt/speech/eleven_labs.py`

这段代码是一个Python代码，它继承自`VoiceBase`类，用于实现一个语音助手。以下是代码的主要部分，以及其作用：

1. 导入`logging`、`os`、`requests`、`playsound`模块以及`autogpt.core.configuration`模块的`SystemConfiguration`和`UserConfigurable`类型。

2. 导入了`logging`模块的`getLogger`函数，用于在日志中记录信息。

3. 导入了`os`模块，用于在命令行中执行操作。

4. 导入了`requests`模块，用于向互联网请求数据。

5. 导入了`playsound`模块，用于播放声音文件。

6. 继承自`VoiceBase`类的`__init__`方法，用于初始化语音助手，其中加载了用户配置文件中的配置信息。

7. 实现了`SystemConfiguration`和`UserConfigurable`的`__call__`方法，用于设置语音助手的参数。

8. 加载了用户配置文件中的配置信息，并将其存储在`self.config`变量中。

9. 在`__call__`方法中，初始化日志记录器，并设置为当前对象的`logger`实例。

10. 在`__call__`方法中，发送请求获取当前用户的配置，并将其存储在`self.config.get`方法中。

11. 在`__call__`方法中，加载用户指定的配置文件，并将其存储在`self.config`变量中。

12. 在`__call__`方法中，创建日志记录器实例，并设置为当前对象的`logger`实例。

13. 在`__call__`方法中，设置语音助手为当前用户的配置，并启动播放声音文件的功能。


```py
"""ElevenLabs speech module"""
from __future__ import annotations

import logging
import os

import requests
from playsound import playsound

from autogpt.core.configuration import SystemConfiguration, UserConfigurable

from .base import VoiceBase

logger = logging.getLogger(__name__)

```py

This is a class that uses Eleven Labs.io's API to convert text to speech. It has a number of methods for controlling the output, including the ability to specify a custom voice by passing a voice ID to the `_use_custom_voice` method. The class also has a `speech` method that uses Eleven Labs.io's API to speak the text passed to it.


```py
PLACEHOLDERS = {"your-voice-id"}


class ElevenLabsConfig(SystemConfiguration):
    api_key: str = UserConfigurable()
    voice_id: str = UserConfigurable()


class ElevenLabsSpeech(VoiceBase):
    """ElevenLabs speech class"""

    def _setup(self, config: ElevenLabsConfig) -> None:
        """Set up the voices, API key, etc.

        Returns:
            None: None
        """

        default_voices = ["ErXwobaYiN019PkySvjV", "EXAVITQu4vr4xnSDxMaL"]
        voice_options = {
            "Rachel": "21m00Tcm4TlvDq8ikWAM",
            "Domi": "AZnzlk1XvdvUeBnXmlld",
            "Bella": "EXAVITQu4vr4xnSDxMaL",
            "Antoni": "ErXwobaYiN019PkySvjV",
            "Elli": "MF3mGyEYCl7XYWbV9V6O",
            "Josh": "TxGEqnHWrfWFTfGW9XjX",
            "Arnold": "VR6AewLTigWG4xSOukaG",
            "Adam": "pNInz6obpgDQGcFmaJgB",
            "Sam": "yoZ06aMxZJJ28mfd3POQ",
        }
        self._headers = {
            "Content-Type": "application/json",
            "xi-api-key": config.api_key,
        }
        self._voices = default_voices.copy()
        if config.voice_id in voice_options:
            config.voice_id = voice_options[config.voice_id]
        self._use_custom_voice(config.voice_id, 0)

    def _use_custom_voice(self, voice, voice_index) -> None:
        """Use a custom voice if provided and not a placeholder

        Args:
            voice (str): The voice ID
            voice_index (int): The voice index

        Returns:
            None: None
        """
        # Placeholder values that should be treated as empty
        if voice and voice not in PLACEHOLDERS:
            self._voices[voice_index] = voice

    def _speech(self, text: str, voice_index: int = 0) -> bool:
        """Speak text using elevenlabs.io's API

        Args:
            text (str): The text to speak
            voice_index (int, optional): The voice to use. Defaults to 0.

        Returns:
            bool: True if the request was successful, False otherwise
        """
        tts_url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/{self._voices[voice_index]}"
        )
        response = requests.post(tts_url, headers=self._headers, json={"text": text})

        if response.status_code == 200:
            with open("speech.mpeg", "wb") as f:
                f.write(response.content)
            playsound("speech.mpeg", True)
            os.remove("speech.mpeg")
            return True
        else:
            logger.warn("Request failed with status code:", response.status_code)
            logger.info("Response content:", response.content)
            return False

```py

# `autogpts/autogpt/autogpt/speech/gtts.py`

这段代码是一个自定义的 GTTS Voice，其中包括一个声音类和一个 `_speech` 方法。

具体来说，这个自定义的 GTTS Voice 类继承自 `VoiceBase` 类，因此继承了该类的多个方法，例如 `__init__` 和 `__speech` 方法。在这个自定义的 GTTS Voice 中，`__speech` 方法会实现播放文本的功能，具体实现是使用 `gtts` 库播放文本，并使用 `playsound` 库播放声音。

另外，在 `__init__` 方法中，调用了 `gtts.gTTS` 函数来生成语音，并使用 `gtts.iTTS` 函数将生成的音频保存为文件。最后，使用 `playsound` 库播放声音，并使用 `os.remove` 函数移除生成的音频文件，以便下次运行时可以重新生成。

总结起来，这段代码的作用是实现了一个自定义的 GTTS Voice，可以用来播放文本，但需要注意，由于使用了未经验证的第三方库 `gtts`，因此需要自行承担风险并确保库的安全性和可靠性。


```py
""" GTTS Voice. """
from __future__ import annotations

import os

import gtts
from playsound import playsound

from autogpt.speech.base import VoiceBase


class GTTSVoice(VoiceBase):
    """GTTS Voice."""

    def _setup(self) -> None:
        pass

    def _speech(self, text: str, _: int = 0) -> bool:
        """Play the given text."""
        tts = gtts.gTTS(text)
        tts.save("speech.mp3")
        playsound("speech.mp3", True)
        os.remove("speech.mp3")
        return True

```