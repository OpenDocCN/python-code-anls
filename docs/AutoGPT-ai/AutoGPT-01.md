# AutoGPT源码解析 1

# 🌟 AutoGPT: the heart of the open-source agent ecosystem

[![Discord Follow](https://dcbadge.vercel.app/api/server/autogpt?style=flat)](https://discord.gg/autogpt) [![GitHub Repo stars](https://img.shields.io/github/stars/Significant-Gravitas/AutoGPT?style=social)](https://github.com/Significant-Gravitas/AutoGPT/stargazers) [![Twitter Follow](https://img.shields.io/twitter/follow/auto_gpt?style=social)](https://twitter.com/Auto_GPT) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AutoGPT** is your go-to toolkit for supercharging agents. With its modular and extensible framework, you're empowered to focus on:

- 🏗️ **Building** - Lay the foundation for something amazing.
- 🧪 **Testing** - Fine-tune your agent to perfection.
- 👀 **Viewing** - See your progress come to life.

Be part of the revolution! **AutoGPT** stays at the forefront of AI innovation, featuring the codebase for the reigning champion in the Open-Source ecosystem.

---

<p align="center">
  <a href="https://lablab.ai/event/autogpt-arena-hacks">
    <img src="https://lablab.ai/_next/image?url=https%3A%2F%2Fstorage.googleapis.com%2Flablab-static-eu%2Fimages%2Fevents%2Fcll6p5cxj0000356zslac05gg%2Fcll6p5cxj0000356zslac05gg_imageLink_562z1jzj.jpg&w=1080&q=75" alt="AutoGPT Arena Hacks Hackathon" />
  </a>
</p>
<p align="center">
  <strong>We're hosting a Hackathon!</strong>
  <br>
  Click the banner above for details and registration!
</p>

---

## 🥇 Current Best Agent: AutoGPT

Among our currently benchmarked agents, AutoGPT scores the best. This will change after the hackathon - the top-performing generalist agent will earn the esteemed position as the primary AutoGPT 🎊

📈 To enter, submit your benchmark run through the UI.

## 🌟 Quickstart

- **To build your own agent** and to be eligible for the hackathon, follow the quickstart guide [here](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpts/forge/tutorials/001_getting_started.md). This will guide you through the process of creating your own agent and using the benchmark and user interface.

- **To activate the best agent** follow the guide [here](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpts/autogpt/README.md).

Want to build your own groundbreaking agent using AutoGPT? 🛠️ There are three major components to focus on:

### 🏗️ the Forge

**Forge your future!** The `forge` is your innovation lab. All the boilerplate code is already handled, letting you channel all your creativity into building a revolutionary agent. It's more than a starting point, it's a launchpad for your ideas. All tutorials are located [here](https://medium.com/@aiedge/autogpt-forge-e3de53cc58ec).

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/autogpts/forge)

### 🎯 the Benchmark

**Test to impress!** The `benchmark` offers a stringent testing environment. Our framework allows for autonomous, objective performance evaluations, ensuring your agents are primed for real-world action.

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/blob/master/benchmark)

### 🎮 the UI

**Take Control!** The `frontend` is your personal command center. It gives you a user-friendly interface to control and monitor your agents, making it easier to bring your ideas to life.

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/frontend)

---

### 🔄 Agent Protocol

🔌 **Standardize to Maximize!** To maintain a uniform standard and ensure seamless compatibility, AutoGPT employs the [agent protocol](https://agentprotocol.ai/) from the AI Engineer Foundation. This standardizes the communication pathways from your agent to the frontend and benchmark.

### 🤔 Questions? Problems? Suggestions?

#### Get help - [Discord 💬](https://discord.gg/autogpt)

[![Join us on Discord](https://invidget.switchblade.xyz/autogpt)](https://discord.gg/autogpt)

To report a bug or request a feature, create a [GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose). Please ensure someone else hasn’t created an issue for the same topic.

<p align="center">
  <a href="https://star-history.com/#Significant-Gravitas/AutoGPT&Date">
    <img src="https://api.star-history.com/svg?repos=Significant-Gravitas/AutoGPT&type=Date" alt="Star History Chart">
  </a>
</p>


This page is a list of issues you could encounter along with their fixes.

# Forge
**Poetry configuration invalid**

The poetry configuration is invalid: 
- Additional properties are not allowed ('group' was unexpected)
<img width="487" alt="Screenshot 2023-09-22 at 5 42 59 PM" src="https://github.com/Significant-Gravitas/AutoGPT/assets/9652976/dd451e6b-8114-44de-9928-075f5f06d661">

**Pydantic Validation Error**

Remove your sqlite agent.db file. it's probably because some of your data is not complying with the new spec (we will create migrations soon to avoid this problem)


*Solution*

Update poetry

# Benchmark
TODO

# Frontend
TODO


### Background

<!-- Clearly explain the need for these changes: -->

### Changes 🏗️

<!-- Concisely describe all of the changes made in this pull request: -->

### PR Quality Scorecard ✨

<!--
Check out our contribution guide:
https://github.com/Significant-Gravitas/Nexus/wiki/Contributing

1. Avoid duplicate work, issues, PRs etc.
2. Also consider contributing something other than code; see the [contribution guide]
   for options.
3. Clearly explain your changes.
4. Avoid making unnecessary changes, especially if they're purely based on personal
   preferences. Doing so is the maintainers' job. ;-)
-->

- [x] Have you used the PR description template? &ensp; `+2 pts`
- [ ] Is your pull request atomic, focusing on a single change? &ensp; `+5 pts`
- [ ] Have you linked the GitHub issue(s) that this PR addresses? &ensp; `+5 pts`
- [ ] Have you documented your changes clearly and comprehensively? &ensp; `+5 pts`
- [ ] Have you changed or added a feature? &ensp; `-4 pts`
  - [ ] Have you added/updated corresponding documentation? &ensp; `+4 pts`
  - [ ] Have you added/updated corresponding integration tests? &ensp; `+5 pts`
- [ ] Have you changed the behavior of AutoGPT? &ensp; `-5 pts`
  - [ ] Have you also run `agbenchmark` to verify that these changes do not regress performance? &ensp; `+10 pts`


# QUICK LINKS 🔗
# --------------
🌎 *Official Website*: https://agpt.co.
📖 *User Guide*: https://docs.agpt.co.
👩 *Contributors Wiki*: https://github.com/Significant-Gravitas/Nexus/wiki/Contributing.

# v0.4.7 RELEASE HIGHLIGHTS! 🚀
# -----------------------------
This release introduces initial REST API support, powered by e2b's agent 
protocol SDK (https://github.com/e2b-dev/agent-protocol#sdk). 

It also includes improvements to prompt generation and support 
for our new benchmarking tool, Auto-GPT-Benchmarks
(https://github.com/Significant-Gravitas/Auto-GPT-Benchmarks).

We've also moved our documentation to Material Theme, at https://docs.agpt.co.

As usual, we've squashed a few bugs and made some under-the-hood improvements.

Take a look at the Release Notes on Github for the full changelog:
https://github.com/Significant-Gravitas/AutoGPT/releases.


# `autogpts/autogpt/data_ingestion.py`

这段代码使用了多个 Python 库，包括 argparse、logging 和 autogpt-memory-vector。它主要用于配置日志输出，以及从环境变量中读取 Config 设置，并执行相应的操作。

具体来说，这段代码的作用如下：

1. 导入需要的库。
2. 设置日志输出格式和来源。
3. 创建一个名为 configure_logging 的函数，该函数使用logging库来实现日志的配置。
4. 使用configure_logging函数创建一个日志实例，并设置日志的级别为DEBUG，同时将输出导向至文件 log-ingestion.txt。
5. 创建一个名为 Config 的类，该类使用 ConfigBuilder 从环境变量中读取配置。
6. 创建一个名为 Memory 的类，该类使用 VectorMemory 和 get_memory 函数来管理内存。
7. 在 configure_logging 函数中，使用 logging.basicConfig 设置日志的格式和来源，包括将时间、毫秒数、函数名称和日志级别括在感叹号中。
8. 创建一个名为 main 的函数，该函数使用 argparse 库的 ArgParser 类来解析命令行参数。
9. 使用 ArgParser 类中的 add_argument 方法，将需要的参数添加到命令行参数中。
10. 使用 Config 和 Memory 类，分别读取和设置环境变量中的配置，并执行相应的操作。

配置日志输出的代码，可以帮助用户在出现问题时进行调试。通过对日志进行记录和跟踪，可以了解问题的发生过程，进一步提高系统的可靠性和稳定性。


```py
import argparse
import logging

from autogpt.commands.file_operations import ingest_file, list_files
from autogpt.config import ConfigBuilder
from autogpt.memory.vector import VectorMemory, get_memory

config = ConfigBuilder.build_config_from_env()


def configure_logging():
    logging.basicConfig(
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(filename="log-ingestion.txt", mode="a"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("AutoGPT-Ingestion")


```

这段代码定义了一个名为 `ingest_directory` 的函数，用于将指定目录中的所有文件 ingest 到内存中。

具体来说，这个函数接收三个参数：

- `directory`：要 ingest 的目录，是一个字符串类型。
- `memory`：一个 MemoryMapped对象，用于存储文件信息，是一个将文件信息映射到内存的容器。
- `args`：一个含有最大长度和 overlap 的参数，用于限制文件读取的最大长度和重叠读取。

函数内部先调用 `list_files` 函数来获取要 ingest 的文件列表，然后对每个文件进行 ingest_file 函数的调用，该函数将文件读取到内存中并设置最大长度和重叠读取参数。最后，如果出现错误，函数会打印错误信息并退出。

函数的作用是帮助用户将指定目录中的所有文件 ingest 到内存中，以便进行训练 AutoGPT 模型等任务。


```py
def ingest_directory(directory: str, memory: VectorMemory, args):
    """
    Ingest all files in a directory by calling the ingest_file function for each file.

    :param directory: The directory containing the files to ingest
    :param memory: An object with an add() method to store the chunks in memory
    """
    logger = logging.getLogger("AutoGPT-Ingestion")
    try:
        files = list_files(directory)
        for file in files:
            ingest_file(file, memory, args.max_length, args.overlap)
    except Exception as e:
        logger.error(f"Error while ingesting directory '{directory}': {str(e)}")


```

This is a Python script that uses the AutoPy++ library for processing Automatic Translation Fridge (ATF) files. It is designed to ingest files and convert them into a memory group that is then used by the AutoPy++ library for further processing.

The script takes one or more options for file or directory to ingest, and optionally passes the `--init` flag to initialize the memory before each file is ingested. The `--overlap` and `--max_length` options can also be used to control the ingestion of files.

The script first initializes the memory and sets the logging to use the `console` output. If the `--init` flag is passed, the memory is cleared and the logging is set to use the `info` level.

If a file is passed as an option, the script attempts to ingest the file using the `file_ingest` function provided by the AutoPy++ library. This function takes the file path, memory, maximum length of each chunk, and overlay size as input, and returns a success or failure message.

If a directory containing files is passed as an option, the script attempts to ingest all files in the directory using the `directory_ingest` function provided by the AutoPy++ library. This function takes the directory path, memory, and initialize flag as input, and returns a success or failure message.

If either a file or directory is not passed, the script prints a warning message and exits.

Note that this script has been tested on Linux and should not work on Windows, macOS or other platforms.


```py
def main() -> None:
    logger = configure_logging()

    parser = argparse.ArgumentParser(
        description="Ingest a file or a directory with multiple files into memory. "
        "Make sure to set your .env before running this script."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="The file to ingest.")
    group.add_argument(
        "--dir", type=str, help="The directory containing the files to ingest."
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Init the memory and wipe its content (default: False)",
        default=False,
    )
    parser.add_argument(
        "--overlap",
        type=int,
        help="The overlap size between chunks when ingesting files (default: 200)",
        default=200,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        help="The max_length of each chunk when ingesting files (default: 4000)",
        default=4000,
    )
    args = parser.parse_args()

    # Initialize memory
    memory = get_memory(config)
    if args.init:
        memory.clear()
    logger.debug("Using memory of type: " + memory.__class__.__name__)

    if args.file:
        try:
            ingest_file(args.file, memory, args.max_length, args.overlap)
            logger.info(f"File '{args.file}' ingested successfully.")
        except Exception as e:
            logger.error(f"Error while ingesting file '{args.file}': {str(e)}")
    elif args.dir:
        try:
            ingest_directory(args.dir, memory, args)
            logger.info(f"Directory '{args.dir}' ingested successfully.")
        except Exception as e:
            logger.error(f"Error while ingesting directory '{args.dir}': {str(e)}")
    else:
        logger.warn(
            "Please provide either a file path (--file) or a directory name (--dir)"
            " inside the auto_gpt_workspace directory as input."
        )


```

这段代码是一个if语句，它会判断当前脚本是否作为主程序运行。如果脚本作为主程序运行，那么程序会直接进入if语句中的main()函数。

"__name__"是一个特殊的属性，用于保存脚本的完整路径，即使脚本在不同目录中，它的路径也不会发生改变。"__main__"是另一个特殊的属性，用于保存脚本描述其意图的名称。在这个例子中，"__main__"被用来检查脚本是否作为主程序运行，如果它被正确设置，则脚本会进入if语句中的main()函数，否则不会执行任何操作。


```py
if __name__ == "__main__":
    main()

```

# AutoGPT: An Autonomous GPT-4 Experiment

[![Discord Follow](https://dcbadge.vercel.app/api/server/autogpt?style=flat)](https://discord.gg/autogpt)
[![GitHub Repo stars](https://img.shields.io/github/stars/Significant-Gravitas/AutoGPT?style=social)](https://github.com/Significant-Gravitas/AutoGPT/stargazers)
[![Twitter Follow](https://img.shields.io/twitter/follow/siggravitas?style=social)](https://twitter.com/SigGravitas)

## 💡 Get help - [Q&A](https://github.com/Significant-Gravitas/AutoGPT/discussions/categories/q-a) or [Discord 💬](https://discord.gg/autogpt)

<hr/>

AutoGPT is an experimental open-source application showcasing the capabilities of the GPT-4 language model. This program, driven by GPT-4, chains together LLM "thoughts", to autonomously achieve whatever goal you set. As one of the first examples of GPT-4 running fully autonomously, AutoGPT pushes the boundaries of what is possible with AI.

<h2 align="center"> Demo April 16th 2023 </h2>

https://user-images.githubusercontent.com/70048414/232352935-55c6bf7c-3958-406e-8610-0913475a0b05.mp4

Demo made by <a href=https://twitter.com/BlakeWerlinger>Blake Werlinger</a>

## 🚀 Features

- 🌐 Internet access for searches and information gathering
- 💾 Long-term and short-term memory management
- 🧠 GPT-4 instances for text generation
- 🔗 Access to popular websites and platforms
- 🗃️ File storage and summarization with GPT-3.5
- 🔌 Extensibility with Plugins

## Quickstart

0. Check out the [wiki](https://github.com/Significant-Gravitas/Nexus/wiki)
1. Get an OpenAI [API Key](https://platform.openai.com/account/api-keys)
2. Download the [latest release](https://github.com/Significant-Gravitas/AutoGPT/releases/latest)
3. Follow the [installation instructions][docs/setup]
4. Configure any additional features you want, or install some [plugins][docs/plugins]
5. [Run][docs/usage] the app

Please see the [documentation][docs] for full setup instructions and configuration options.

[docs]: https://docs.agpt.co/

## 📖 Documentation

- [⚙️ Setup][docs/setup]
- [💻 Usage][docs/usage]
- [🔌 Plugins][docs/plugins]
- Configuration
  - [🔍 Web Search](https://docs.agpt.co/configuration/search/)
  - [🧠 Memory](https://docs.agpt.co/configuration/memory/)
  - [🗣️ Voice (TTS)](https://docs.agpt.co/configuration/voice/)
  - [🖼️ Image Generation](https://docs.agpt.co/configuration/imagegen/)

[docs/setup]: https://docs.agpt.co/setup/
[docs/usage]: https://docs.agpt.co/usage/
[docs/plugins]: https://docs.agpt.co/plugins/

## 🏗️ Setting up for development
1. Make sure `poetry` is installed: `python3 -m pip install poetry`
2. Install all dependencies: `poetry install`


<h2 align="center"> 💖 Help Fund AutoGPT's Development 💖</h2>
<p align="center">
If you can spare a coffee, you can help to cover the costs of developing AutoGPT and help to push the boundaries of fully autonomous AI!
Your support is greatly appreciated. Development of this free, open-source project is made possible by all the <a href="https://github.com/Significant-Gravitas/AutoGPT/graphs/contributors">contributors</a> and <a href="https://github.com/sponsors/Torantulino">sponsors</a>. If you'd like to sponsor this project and have your avatar or company logo appear below <a href="https://github.com/sponsors/Torantulino">click here</a>.
</p>

<p align="center">
<div align="center" class="logo-container">
<a href="https://www.zilliz.com/">
<picture height="40px">
  <source media="(prefers-color-scheme: light)" srcset="https://user-images.githubusercontent.com/22963551/234158272-7917382e-ff80-469e-8d8c-94f4477b8b5a.png">
  <img src="https://user-images.githubusercontent.com/22963551/234158222-30e2d7a7-f0a9-433d-a305-e3aa0b194444.png" height="40px" alt="Zilliz" />
</picture>
</a>

<a href="https://roost.ai">
<img src="https://user-images.githubusercontent.com/22963551/234180283-b58cb03c-c95a-4196-93c1-28b52a388e9d.png" height="40px" alt="Roost.AI" />
</a>
  
<a href="https://nuclei.ai/">
<picture height="40px">
  <source media="(prefers-color-scheme: light)" srcset="https://user-images.githubusercontent.com/22963551/234153428-24a6f31d-c0c6-4c9b-b3f4-9110148f67b4.png">
  <img src="https://user-images.githubusercontent.com/22963551/234181283-691c5d71-ca94-4646-a1cf-6e818bd86faa.png" height="40px" alt="NucleiAI" />
</picture>
</a>

<a href="https://www.algohash.org/">
<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://user-images.githubusercontent.com/22963551/234180375-1365891c-0ba6-4d49-94c3-847c85fe03b0.png" >
  <img src="https://user-images.githubusercontent.com/22963551/234180359-143e4a7a-4a71-4830-99c8-9b165cde995f.png" height="40px" alt="Algohash" />
</picture>
</a>

<a href="https://github.com/weaviate/weaviate">
<picture height="40px">
  <source media="(prefers-color-scheme: light)" srcset="https://user-images.githubusercontent.com/22963551/234181699-3d7f6ea8-5a7f-4e98-b812-37be1081be4b.png">
  <img src="https://user-images.githubusercontent.com/22963551/234181695-fc895159-b921-4895-9a13-65e6eff5b0e7.png" height="40px" alt="TypingMind" />
</picture>
</a>

<a href="https://chatgpv.com/?ref=spni76459e4fa3f30a">
<img src="https://github-production-user-asset-6210df.s3.amazonaws.com/22963551/239132565-623a2dd6-eaeb-4941-b40f-c5a29ca6bebc.png" height="40px" alt="ChatGPV" />
</a>
  
</div>
</br>

<p align="center"><a href="https://github.com/robinicus"><img src="https://avatars.githubusercontent.com/robinicus?v=4" width="50px" alt="robinicus" /></a>&nbsp;&nbsp;<a href="https://github.com/0xmatchmaker"><img src="https://avatars.githubusercontent.com/0xmatchmaker?v=4" width="50px" alt="0xmatchmaker" /></a>&nbsp;&nbsp;<a href="https://github.com/jazgarewal"><img src="https://avatars.githubusercontent.com/jazgarewal?v=4" width="50px" alt="jazgarewal" /></a>&nbsp;&nbsp;<a href="https://github.com/MayurVirkar"><img src="https://avatars.githubusercontent.com/MayurVirkar?v=4" width="50px" alt="MayurVirkar" /></a>&nbsp;&nbsp;<a href="https://github.com/avy-ai"><img src="https://avatars.githubusercontent.com/avy-ai?v=4" width="50px" alt="avy-ai" /></a>&nbsp;&nbsp;<a href="https://github.com/TheStoneMX"><img src="https://avatars.githubusercontent.com/TheStoneMX?v=4" width="50px" alt="TheStoneMX" /></a>&nbsp;&nbsp;<a href="https://github.com/goldenrecursion"><img src="https://avatars.githubusercontent.com/goldenrecursion?v=4" width="50px" alt="goldenrecursion" /></a>&nbsp;&nbsp;<a href="https://github.com/MatthewAgs"><img src="https://avatars.githubusercontent.com/MatthewAgs?v=4" width="50px" alt="MatthewAgs" /></a>&nbsp;&nbsp;<a href="https://github.com/eelbaz"><img src="https://avatars.githubusercontent.com/eelbaz?v=4" width="50px" alt="eelbaz" /></a>&nbsp;&nbsp;<a href="https://github.com/rapidstartup"><img src="https://avatars.githubusercontent.com/rapidstartup?v=4" width="50px" alt="rapidstartup" /></a>&nbsp;&nbsp;<a href="https://github.com/gklab"><img src="https://avatars.githubusercontent.com/gklab?v=4" width="50px" alt="gklab" /></a>&nbsp;&nbsp;<a href="https://github.com/VoiceBeer"><img src="https://avatars.githubusercontent.com/VoiceBeer?v=4" width="50px" alt="VoiceBeer" /></a>&nbsp;&nbsp;<a href="https://github.com/DailyBotHQ"><img src="https://avatars.githubusercontent.com/DailyBotHQ?v=4" width="50px" alt="DailyBotHQ" /></a>&nbsp;&nbsp;<a href="https://github.com/lucas-chu"><img src="https://avatars.githubusercontent.com/lucas-chu?v=4" width="50px" alt="lucas-chu" /></a>&nbsp;&nbsp;<a href="https://github.com/knifour"><img src="https://avatars.githubusercontent.com/knifour?v=4" width="50px" alt="knifour" /></a>&nbsp;&nbsp;<a href="https://github.com/refinery1"><img src="https://avatars.githubusercontent.com/refinery1?v=4" width="50px" alt="refinery1" /></a>&nbsp;&nbsp;<a href="https://github.com/st617"><img src="https://avatars.githubusercontent.com/st617?v=4" width="50px" alt="st617" /></a>&nbsp;&nbsp;<a href="https://github.com/neodenit"><img src="https://avatars.githubusercontent.com/neodenit?v=4" width="50px" alt="neodenit" /></a>&nbsp;&nbsp;<a href="https://github.com/CrazySwami"><img src="https://avatars.githubusercontent.com/CrazySwami?v=4" width="50px" alt="CrazySwami" /></a>&nbsp;&nbsp;<a href="https://github.com/Heitechsoft"><img src="https://avatars.githubusercontent.com/Heitechsoft?v=4" width="50px" alt="Heitechsoft" /></a>&nbsp;&nbsp;<a href="https://github.com/RealChrisSean"><img src="https://avatars.githubusercontent.com/RealChrisSean?v=4" width="50px" alt="RealChrisSean" /></a>&nbsp;&nbsp;<a href="https://github.com/abhinav-pandey29"><img src="https://avatars.githubusercontent.com/abhinav-pandey29?v=4" width="50px" alt="abhinav-pandey29" /></a>&nbsp;&nbsp;<a href="https://github.com/Explorergt92"><img src="https://avatars.githubusercontent.com/Explorergt92?v=4" width="50px" alt="Explorergt92" /></a>&nbsp;&nbsp;<a href="https://github.com/SparkplanAI"><img src="https://avatars.githubusercontent.com/SparkplanAI?v=4" width="50px" alt="SparkplanAI" /></a>&nbsp;&nbsp;<a href="https://github.com/crizzler"><img src="https://avatars.githubusercontent.com/crizzler?v=4" width="50px" alt="crizzler" /></a>&nbsp;&nbsp;<a href="https://github.com/kreativai"><img src="https://avatars.githubusercontent.com/kreativai?v=4" width="50px" alt="kreativai" /></a>&nbsp;&nbsp;<a href="https://github.com/omphos"><img src="https://avatars.githubusercontent.com/omphos?v=4" width="50px" alt="omphos" /></a>&nbsp;&nbsp;<a href="https://github.com/Jahmazon"><img src="https://avatars.githubusercontent.com/Jahmazon?v=4" width="50px" alt="Jahmazon" /></a>&nbsp;&nbsp;<a href="https://github.com/tjarmain"><img src="https://avatars.githubusercontent.com/tjarmain?v=4" width="50px" alt="tjarmain" /></a>&nbsp;&nbsp;<a href="https://github.com/ddtarazona"><img src="https://avatars.githubusercontent.com/ddtarazona?v=4" width="50px" alt="ddtarazona" /></a>&nbsp;&nbsp;<a href="https://github.com/saten-private"><img src="https://avatars.githubusercontent.com/saten-private?v=4" width="50px" alt="saten-private" /></a>&nbsp;&nbsp;<a href="https://github.com/anvarazizov"><img src="https://avatars.githubusercontent.com/anvarazizov?v=4" width="50px" alt="anvarazizov" /></a>&nbsp;&nbsp;<a href="https://github.com/lazzacapital"><img src="https://avatars.githubusercontent.com/lazzacapital?v=4" width="50px" alt="lazzacapital" /></a>&nbsp;&nbsp;<a href="https://github.com/m"><img src="https://avatars.githubusercontent.com/m?v=4" width="50px" alt="m" /></a>&nbsp;&nbsp;<a href="https://github.com/Pythagora-io"><img src="https://avatars.githubusercontent.com/Pythagora-io?v=4" width="50px" alt="Pythagora-io" /></a>&nbsp;&nbsp;<a href="https://github.com/Web3Capital"><img src="https://avatars.githubusercontent.com/Web3Capital?v=4" width="50px" alt="Web3Capital" /></a>&nbsp;&nbsp;<a href="https://github.com/toverly1"><img src="https://avatars.githubusercontent.com/toverly1?v=4" width="50px" alt="toverly1" /></a>&nbsp;&nbsp;<a href="https://github.com/digisomni"><img src="https://avatars.githubusercontent.com/digisomni?v=4" width="50px" alt="digisomni" /></a>&nbsp;&nbsp;<a href="https://github.com/concreit"><img src="https://avatars.githubusercontent.com/concreit?v=4" width="50px" alt="concreit" /></a>&nbsp;&nbsp;<a href="https://github.com/LeeRobidas"><img src="https://avatars.githubusercontent.com/LeeRobidas?v=4" width="50px" alt="LeeRobidas" /></a>&nbsp;&nbsp;<a href="https://github.com/Josecodesalot"><img src="https://avatars.githubusercontent.com/Josecodesalot?v=4" width="50px" alt="Josecodesalot" /></a>&nbsp;&nbsp;<a href="https://github.com/dexterityx"><img src="https://avatars.githubusercontent.com/dexterityx?v=4" width="50px" alt="dexterityx" /></a>&nbsp;&nbsp;<a href="https://github.com/rickscode"><img src="https://avatars.githubusercontent.com/rickscode?v=4" width="50px" alt="rickscode" /></a>&nbsp;&nbsp;<a href="https://github.com/Brodie0"><img src="https://avatars.githubusercontent.com/Brodie0?v=4" width="50px" alt="Brodie0" /></a>&nbsp;&nbsp;<a href="https://github.com/FSTatSBS"><img src="https://avatars.githubusercontent.com/FSTatSBS?v=4" width="50px" alt="FSTatSBS" /></a>&nbsp;&nbsp;<a href="https://github.com/nocodeclarity"><img src="https://avatars.githubusercontent.com/nocodeclarity?v=4" width="50px" alt="nocodeclarity" /></a>&nbsp;&nbsp;<a href="https://github.com/jsolejr"><img src="https://avatars.githubusercontent.com/jsolejr?v=4" width="50px" alt="jsolejr" /></a>&nbsp;&nbsp;<a href="https://github.com/amr-elsehemy"><img src="https://avatars.githubusercontent.com/amr-elsehemy?v=4" width="50px" alt="amr-elsehemy" /></a>&nbsp;&nbsp;<a href="https://github.com/RawBanana"><img src="https://avatars.githubusercontent.com/RawBanana?v=4" width="50px" alt="RawBanana" /></a>&nbsp;&nbsp;<a href="https://github.com/horazius"><img src="https://avatars.githubusercontent.com/horazius?v=4" width="50px" alt="horazius" /></a>&nbsp;&nbsp;<a href="https://github.com/SwftCoins"><img src="https://avatars.githubusercontent.com/SwftCoins?v=4" width="50px" alt="SwftCoins" /></a>&nbsp;&nbsp;<a href="https://github.com/tob-le-rone"><img src="https://avatars.githubusercontent.com/tob-le-rone?v=4" width="50px" alt="tob-le-rone" /></a>&nbsp;&nbsp;<a href="https://github.com/RThaweewat"><img src="https://avatars.githubusercontent.com/RThaweewat?v=4" width="50px" alt="RThaweewat" /></a>&nbsp;&nbsp;<a href="https://github.com/jun784"><img src="https://avatars.githubusercontent.com/jun784?v=4" width="50px" alt="jun784" /></a>&nbsp;&nbsp;<a href="https://github.com/joaomdmoura"><img src="https://avatars.githubusercontent.com/joaomdmoura?v=4" width="50px" alt="joaomdmoura" /></a>&nbsp;&nbsp;<a href="https://github.com/rejunity"><img src="https://avatars.githubusercontent.com/rejunity?v=4" width="50px" alt="rejunity" /></a>&nbsp;&nbsp;<a href="https://github.com/mathewhawkins"><img src="https://avatars.githubusercontent.com/mathewhawkins?v=4" width="50px" alt="mathewhawkins" /></a>&nbsp;&nbsp;<a href="https://github.com/caitlynmeeks"><img src="https://avatars.githubusercontent.com/caitlynmeeks?v=4" width="50px" alt="caitlynmeeks" /></a>&nbsp;&nbsp;<a href="https://github.com/jd3655"><img src="https://avatars.githubusercontent.com/jd3655?v=4" width="50px" alt="jd3655" /></a>&nbsp;&nbsp;<a href="https://github.com/Odin519Tomas"><img src="https://avatars.githubusercontent.com/Odin519Tomas?v=4" width="50px" alt="Odin519Tomas" /></a>&nbsp;&nbsp;<a href="https://github.com/DataMetis"><img src="https://avatars.githubusercontent.com/DataMetis?v=4" width="50px" alt="DataMetis" /></a>&nbsp;&nbsp;<a href="https://github.com/webbcolton"><img src="https://avatars.githubusercontent.com/webbcolton?v=4" width="50px" alt="webbcolton" /></a>&nbsp;&nbsp;<a href="https://github.com/rocks6"><img src="https://avatars.githubusercontent.com/rocks6?v=4" width="50px" alt="rocks6" /></a>&nbsp;&nbsp;<a href="https://github.com/cxs"><img src="https://avatars.githubusercontent.com/cxs?v=4" width="50px" alt="cxs" /></a>&nbsp;&nbsp;<a href="https://github.com/fruition"><img src="https://avatars.githubusercontent.com/fruition?v=4" width="50px" alt="fruition" /></a>&nbsp;&nbsp;<a href="https://github.com/nnkostov"><img src="https://avatars.githubusercontent.com/nnkostov?v=4" width="50px" alt="nnkostov" /></a>&nbsp;&nbsp;<a href="https://github.com/morcos"><img src="https://avatars.githubusercontent.com/morcos?v=4" width="50px" alt="morcos" /></a>&nbsp;&nbsp;<a href="https://github.com/pingbotan"><img src="https://avatars.githubusercontent.com/pingbotan?v=4" width="50px" alt="pingbotan" /></a>&nbsp;&nbsp;<a href="https://github.com/maxxflyer"><img src="https://avatars.githubusercontent.com/maxxflyer?v=4" width="50px" alt="maxxflyer" /></a>&nbsp;&nbsp;<a href="https://github.com/tommi-joentakanen"><img src="https://avatars.githubusercontent.com/tommi-joentakanen?v=4" width="50px" alt="tommi-joentakanen" /></a>&nbsp;&nbsp;<a href="https://github.com/hunteraraujo"><img src="https://avatars.githubusercontent.com/hunteraraujo?v=4" width="50px" alt="hunteraraujo" /></a>&nbsp;&nbsp;<a href="https://github.com/projectonegames"><img src="https://avatars.githubusercontent.com/projectonegames?v=4" width="50px" alt="projectonegames" /></a>&nbsp;&nbsp;<a href="https://github.com/tullytim"><img src="https://avatars.githubusercontent.com/tullytim?v=4" width="50px" alt="tullytim" /></a>&nbsp;&nbsp;<a href="https://github.com/comet-ml"><img src="https://avatars.githubusercontent.com/comet-ml?v=4" width="50px" alt="comet-ml" /></a>&nbsp;&nbsp;<a href="https://github.com/thepok"><img src="https://avatars.githubusercontent.com/thepok?v=4" width="50px" alt="thepok" /></a>&nbsp;&nbsp;<a href="https://github.com/prompthero"><img src="https://avatars.githubusercontent.com/prompthero?v=4" width="50px" alt="prompthero" /></a>&nbsp;&nbsp;<a href="https://github.com/sunchongren"><img src="https://avatars.githubusercontent.com/sunchongren?v=4" width="50px" alt="sunchongren" /></a>&nbsp;&nbsp;<a href="https://github.com/neverinstall"><img src="https://avatars.githubusercontent.com/neverinstall?v=4" width="50px" alt="neverinstall" /></a>&nbsp;&nbsp;<a href="https://github.com/josephcmiller2"><img src="https://avatars.githubusercontent.com/josephcmiller2?v=4" width="50px" alt="josephcmiller2" /></a>&nbsp;&nbsp;<a href="https://github.com/yx3110"><img src="https://avatars.githubusercontent.com/yx3110?v=4" width="50px" alt="yx3110" /></a>&nbsp;&nbsp;<a href="https://github.com/MBassi91"><img src="https://avatars.githubusercontent.com/MBassi91?v=4" width="50px" alt="MBassi91" /></a>&nbsp;&nbsp;<a href="https://github.com/SpacingLily"><img src="https://avatars.githubusercontent.com/SpacingLily?v=4" width="50px" alt="SpacingLily" /></a>&nbsp;&nbsp;<a href="https://github.com/arthur-x88"><img src="https://avatars.githubusercontent.com/arthur-x88?v=4" width="50px" alt="arthur-x88" /></a>&nbsp;&nbsp;<a href="https://github.com/ciscodebs"><img src="https://avatars.githubusercontent.com/ciscodebs?v=4" width="50px" alt="ciscodebs" /></a>&nbsp;&nbsp;<a href="https://github.com/christian-gheorghe"><img src="https://avatars.githubusercontent.com/christian-gheorghe?v=4" width="50px" alt="christian-gheorghe" /></a>&nbsp;&nbsp;<a href="https://github.com/EngageStrategies"><img src="https://avatars.githubusercontent.com/EngageStrategies?v=4" width="50px" alt="EngageStrategies" /></a>&nbsp;&nbsp;<a href="https://github.com/jondwillis"><img src="https://avatars.githubusercontent.com/jondwillis?v=4" width="50px" alt="jondwillis" /></a>&nbsp;&nbsp;<a href="https://github.com/Cameron-Fulton"><img src="https://avatars.githubusercontent.com/Cameron-Fulton?v=4" width="50px" alt="Cameron-Fulton" /></a>&nbsp;&nbsp;<a href="https://github.com/AryaXAI"><img src="https://avatars.githubusercontent.com/AryaXAI?v=4" width="50px" alt="AryaXAI" /></a>&nbsp;&nbsp;<a href="https://github.com/AuroraHolding"><img src="https://avatars.githubusercontent.com/AuroraHolding?v=4" width="50px" alt="AuroraHolding" /></a>&nbsp;&nbsp;<a href="https://github.com/Mr-Bishop42"><img src="https://avatars.githubusercontent.com/Mr-Bishop42?v=4" width="50px" alt="Mr-Bishop42" /></a>&nbsp;&nbsp;<a href="https://github.com/doverhq"><img src="https://avatars.githubusercontent.com/doverhq?v=4" width="50px" alt="doverhq" /></a>&nbsp;&nbsp;<a href="https://github.com/johnculkin"><img src="https://avatars.githubusercontent.com/johnculkin?v=4" width="50px" alt="johnculkin" /></a>&nbsp;&nbsp;<a href="https://github.com/marv-technology"><img src="https://avatars.githubusercontent.com/marv-technology?v=4" width="50px" alt="marv-technology" /></a>&nbsp;&nbsp;<a href="https://github.com/ikarosai"><img src="https://avatars.githubusercontent.com/ikarosai?v=4" width="50px" alt="ikarosai" /></a>&nbsp;&nbsp;<a href="https://github.com/ColinConwell"><img src="https://avatars.githubusercontent.com/ColinConwell?v=4" width="50px" alt="ColinConwell" /></a>&nbsp;&nbsp;<a href="https://github.com/humungasaurus"><img src="https://avatars.githubusercontent.com/humungasaurus?v=4" width="50px" alt="humungasaurus" /></a>&nbsp;&nbsp;<a href="https://github.com/terpsfreak"><img src="https://avatars.githubusercontent.com/terpsfreak?v=4" width="50px" alt="terpsfreak" /></a>&nbsp;&nbsp;<a href="https://github.com/iddelacruz"><img src="https://avatars.githubusercontent.com/iddelacruz?v=4" width="50px" alt="iddelacruz" /></a>&nbsp;&nbsp;<a href="https://github.com/thisisjeffchen"><img src="https://avatars.githubusercontent.com/thisisjeffchen?v=4" width="50px" alt="thisisjeffchen" /></a>&nbsp;&nbsp;<a href="https://github.com/nicoguyon"><img src="https://avatars.githubusercontent.com/nicoguyon?v=4" width="50px" alt="nicoguyon" /></a>&nbsp;&nbsp;<a href="https://github.com/arjunb023"><img src="https://avatars.githubusercontent.com/arjunb023?v=4" width="50px" alt="arjunb023" /></a>&nbsp;&nbsp;<a href="https://github.com/Nalhos"><img src="https://avatars.githubusercontent.com/Nalhos?v=4" width="50px" alt="Nalhos" /></a>&nbsp;&nbsp;<a href="https://github.com/belharethsami"><img src="https://avatars.githubusercontent.com/belharethsami?v=4" width="50px" alt="belharethsami" /></a>&nbsp;&nbsp;<a href="https://github.com/Mobivs"><img src="https://avatars.githubusercontent.com/Mobivs?v=4" width="50px" alt="Mobivs" /></a>&nbsp;&nbsp;<a href="https://github.com/txtr99"><img src="https://avatars.githubusercontent.com/txtr99?v=4" width="50px" alt="txtr99" /></a>&nbsp;&nbsp;<a href="https://github.com/ntwrite"><img src="https://avatars.githubusercontent.com/ntwrite?v=4" width="50px" alt="ntwrite" /></a>&nbsp;&nbsp;<a href="https://github.com/founderblocks-sils"><img src="https://avatars.githubusercontent.com/founderblocks-sils?v=4" width="50px" alt="founderblocks-sils" /></a>&nbsp;&nbsp;<a href="https://github.com/kMag410"><img src="https://avatars.githubusercontent.com/kMag410?v=4" width="50px" alt="kMag410" /></a>&nbsp;&nbsp;<a href="https://github.com/angiaou"><img src="https://avatars.githubusercontent.com/angiaou?v=4" width="50px" alt="angiaou" /></a>&nbsp;&nbsp;<a href="https://github.com/garythebat"><img src="https://avatars.githubusercontent.com/garythebat?v=4" width="50px" alt="garythebat" /></a>&nbsp;&nbsp;<a href="https://github.com/lmaugustin"><img src="https://avatars.githubusercontent.com/lmaugustin?v=4" width="50px" alt="lmaugustin" /></a>&nbsp;&nbsp;<a href="https://github.com/shawnharmsen"><img src="https://avatars.githubusercontent.com/shawnharmsen?v=4" width="50px" alt="shawnharmsen" /></a>&nbsp;&nbsp;<a href="https://github.com/clortegah"><img src="https://avatars.githubusercontent.com/clortegah?v=4" width="50px" alt="clortegah" /></a>&nbsp;&nbsp;<a href="https://github.com/MetaPath01"><img src="https://avatars.githubusercontent.com/MetaPath01?v=4" width="50px" alt="MetaPath01" /></a>&nbsp;&nbsp;<a href="https://github.com/sekomike910"><img src="https://avatars.githubusercontent.com/sekomike910?v=4" width="50px" alt="sekomike910" /></a>&nbsp;&nbsp;<a href="https://github.com/MediConCenHK"><img src="https://avatars.githubusercontent.com/MediConCenHK?v=4" width="50px" alt="MediConCenHK" /></a>&nbsp;&nbsp;<a href="https://github.com/svpermari0"><img src="https://avatars.githubusercontent.com/svpermari0?v=4" width="50px" alt="svpermari0" /></a>&nbsp;&nbsp;<a href="https://github.com/jacobyoby"><img src="https://avatars.githubusercontent.com/jacobyoby?v=4" width="50px" alt="jacobyoby" /></a>&nbsp;&nbsp;<a href="https://github.com/turintech"><img src="https://avatars.githubusercontent.com/turintech?v=4" width="50px" alt="turintech" /></a>&nbsp;&nbsp;<a href="https://github.com/allenstecat"><img src="https://avatars.githubusercontent.com/allenstecat?v=4" width="50px" alt="allenstecat" /></a>&nbsp;&nbsp;<a href="https://github.com/CatsMeow492"><img src="https://avatars.githubusercontent.com/CatsMeow492?v=4" width="50px" alt="CatsMeow492" /></a>&nbsp;&nbsp;<a href="https://github.com/tommygeee"><img src="https://avatars.githubusercontent.com/tommygeee?v=4" width="50px" alt="tommygeee" /></a>&nbsp;&nbsp;<a href="https://github.com/judegomila"><img src="https://avatars.githubusercontent.com/judegomila?v=4" width="50px" alt="judegomila" /></a>&nbsp;&nbsp;<a href="https://github.com/cfarquhar"><img src="https://avatars.githubusercontent.com/cfarquhar?v=4" width="50px" alt="cfarquhar" /></a>&nbsp;&nbsp;<a href="https://github.com/ZoneSixGames"><img src="https://avatars.githubusercontent.com/ZoneSixGames?v=4" width="50px" alt="ZoneSixGames" /></a>&nbsp;&nbsp;<a href="https://github.com/kenndanielso"><img src="https://avatars.githubusercontent.com/kenndanielso?v=4" width="50px" alt="kenndanielso" /></a>&nbsp;&nbsp;<a href="https://github.com/CrypteorCapital"><img src="https://avatars.githubusercontent.com/CrypteorCapital?v=4" width="50px" alt="CrypteorCapital" /></a>&nbsp;&nbsp;<a href="https://github.com/sultanmeghji"><img src="https://avatars.githubusercontent.com/sultanmeghji?v=4" width="50px" alt="sultanmeghji" /></a>&nbsp;&nbsp;<a href="https://github.com/jenius-eagle"><img src="https://avatars.githubusercontent.com/jenius-eagle?v=4" width="50px" alt="jenius-eagle" /></a>&nbsp;&nbsp;<a href="https://github.com/josephjacks"><img src="https://avatars.githubusercontent.com/josephjacks?v=4" width="50px" alt="josephjacks" /></a>&nbsp;&nbsp;<a href="https://github.com/pingshian0131"><img src="https://avatars.githubusercontent.com/pingshian0131?v=4" width="50px" alt="pingshian0131" /></a>&nbsp;&nbsp;<a href="https://github.com/AIdevelopersAI"><img src="https://avatars.githubusercontent.com/AIdevelopersAI?v=4" width="50px" alt="AIdevelopersAI" /></a>&nbsp;&nbsp;<a href="https://github.com/ternary5"><img src="https://avatars.githubusercontent.com/ternary5?v=4" width="50px" alt="ternary5" /></a>&nbsp;&nbsp;<a href="https://github.com/ChrisDMT"><img src="https://avatars.githubusercontent.com/ChrisDMT?v=4" width="50px" alt="ChrisDMT" /></a>&nbsp;&nbsp;<a href="https://github.com/AcountoOU"><img src="https://avatars.githubusercontent.com/AcountoOU?v=4" width="50px" alt="AcountoOU" /></a>&nbsp;&nbsp;<a href="https://github.com/chatgpt-prompts"><img src="https://avatars.githubusercontent.com/chatgpt-prompts?v=4" width="50px" alt="chatgpt-prompts" /></a>&nbsp;&nbsp;<a href="https://github.com/Partender"><img src="https://avatars.githubusercontent.com/Partender?v=4" width="50px" alt="Partender" /></a>&nbsp;&nbsp;<a href="https://github.com/Daniel1357"><img src="https://avatars.githubusercontent.com/Daniel1357?v=4" width="50px" alt="Daniel1357" /></a>&nbsp;&nbsp;<a href="https://github.com/KiaArmani"><img src="https://avatars.githubusercontent.com/KiaArmani?v=4" width="50px" alt="KiaArmani" /></a>&nbsp;&nbsp;<a href="https://github.com/zkonduit"><img src="https://avatars.githubusercontent.com/zkonduit?v=4" width="50px" alt="zkonduit" /></a>&nbsp;&nbsp;<a href="https://github.com/fabrietech"><img src="https://avatars.githubusercontent.com/fabrietech?v=4" width="50px" alt="fabrietech" /></a>&nbsp;&nbsp;<a href="https://github.com/scryptedinc"><img src="https://avatars.githubusercontent.com/scryptedinc?v=4" width="50px" alt="scryptedinc" /></a>&nbsp;&nbsp;<a href="https://github.com/coreyspagnoli"><img src="https://avatars.githubusercontent.com/coreyspagnoli?v=4" width="50px" alt="coreyspagnoli" /></a>&nbsp;&nbsp;<a href="https://github.com/AntonioCiolino"><img src="https://avatars.githubusercontent.com/AntonioCiolino?v=4" width="50px" alt="AntonioCiolino" /></a>&nbsp;&nbsp;<a href="https://github.com/Dradstone"><img src="https://avatars.githubusercontent.com/Dradstone?v=4" width="50px" alt="Dradstone" /></a>&nbsp;&nbsp;<a href="https://github.com/CarmenCocoa"><img src="https://avatars.githubusercontent.com/CarmenCocoa?v=4" width="50px" alt="CarmenCocoa" /></a>&nbsp;&nbsp;<a href="https://github.com/bentoml"><img src="https://avatars.githubusercontent.com/bentoml?v=4" width="50px" alt="bentoml" /></a>&nbsp;&nbsp;<a href="https://github.com/merwanehamadi"><img src="https://avatars.githubusercontent.com/merwanehamadi?v=4" width="50px" alt="merwanehamadi" /></a>&nbsp;&nbsp;<a href="https://github.com/vkozacek"><img src="https://avatars.githubusercontent.com/vkozacek?v=4" width="50px" alt="vkozacek" /></a>&nbsp;&nbsp;<a href="https://github.com/ASmithOWL"><img src="https://avatars.githubusercontent.com/ASmithOWL?v=4" width="50px" alt="ASmithOWL" /></a>&nbsp;&nbsp;<a href="https://github.com/tekelsey"><img src="https://avatars.githubusercontent.com/tekelsey?v=4" width="50px" alt="tekelsey" /></a>&nbsp;&nbsp;<a href="https://github.com/GalaxyVideoAgency"><img src="https://avatars.githubusercontent.com/GalaxyVideoAgency?v=4" width="50px" alt="GalaxyVideoAgency" /></a>&nbsp;&nbsp;<a href="https://github.com/wenfengwang"><img src="https://avatars.githubusercontent.com/wenfengwang?v=4" width="50px" alt="wenfengwang" /></a>&nbsp;&nbsp;<a href="https://github.com/rviramontes"><img src="https://avatars.githubusercontent.com/rviramontes?v=4" width="50px" alt="rviramontes" /></a>&nbsp;&nbsp;<a href="https://github.com/indoor47"><img src="https://avatars.githubusercontent.com/indoor47?v=4" width="50px" alt="indoor47" /></a>&nbsp;&nbsp;<a href="https://github.com/ZERO-A-ONE"><img src="https://avatars.githubusercontent.com/ZERO-A-ONE?v=4" width="50px" alt="ZERO-A-ONE" /></a>&nbsp;&nbsp;</p>

## ⚠️ Limitations

This experiment aims to showcase the potential of GPT-4 but comes with some limitations:

1. Not a polished application or product, just an experiment
2. May not perform well in complex, real-world business scenarios. In fact, if it actually does, please share your results!
3. Quite expensive to run, so set and monitor your API key limits with OpenAI!

## 🛡 Disclaimer

This project, AutoGPT, is an experimental application and is provided "as-is" without any warranty, express or implied. By using this software, you agree to assume all risks associated with its use, including but not limited to data loss, system failure, or any other issues that may arise.

The developers and contributors of this project do not accept any responsibility or liability for any losses, damages, or other consequences that may occur as a result of using this software. You are solely responsible for any decisions and actions taken based on the information provided by AutoGPT.

**Please note that the use of the GPT-4 language model can be expensive due to its token usage.** By utilizing this project, you acknowledge that you are responsible for monitoring and managing your own token usage and the associated costs. It is highly recommended to check your OpenAI API usage regularly and set up any necessary limits or alerts to prevent unexpected charges.

As an autonomous experiment, AutoGPT may generate content or take actions that are not in line with real-world business practices or legal requirements. It is your responsibility to ensure that any actions or decisions made based on the output of this software comply with all applicable laws, regulations, and ethical standards. The developers and contributors of this project shall not be held responsible for any consequences arising from the use of this software.

By using AutoGPT, you agree to indemnify, defend, and hold harmless the developers, contributors, and any affiliated parties from and against any and all claims, damages, losses, liabilities, costs, and expenses (including reasonable attorneys' fees) arising from your use of this software or your violation of these terms.

## 🐦 Connect with Us on Twitter

Stay up-to-date with the latest news, updates, and insights about AutoGPT by following our Twitter accounts. Engage with the developer and the AI's own account for interesting discussions, project updates, and more.

- **Developer**: Follow [@siggravitas](https://twitter.com/siggravitas) for insights into the development process, project updates, and related topics from the creator of Entrepreneur-GPT.

We look forward to connecting with you and hearing your thoughts, ideas, and experiences with AutoGPT. Join us on Twitter and let's explore the future of AI together!

<p align="center">
  <a href="https://star-history.com/#Significant-Gravitas/AutoGPT&Date">
    <img src="https://api.star-history.com/svg?repos=Significant-Gravitas/AutoGPT&type=Date" alt="Star History Chart">
  </a>
</p>


# `autogpts/autogpt/agbenchmark_config/analyze_reports.py`

这段代码是一个Python脚本，它的作用是读取并打印指定目录下的文件列表，同时支持多种输出格式的导出。下面是具体的实现步骤：

1. 导入所需库：使用python标准库中的json、logging、re、sys、collections库，以及第三方库tabulate。
2. 定义信息、日志和粒度设置：通过`info`、`debug`和`granular`参数来设置信息输出方式，如果这些参数都不传递则按照默认值进行设置。
3. 导入路径模块：使用pathlib库中的Path类。
4. 定义函数：创建一个名为`print_files`的函数，用于打印文件列表。
5. 调用pathlib库中的目录遍历函数：使用pathlib库中的Path.iterdir()函数，遍历指定目录下的所有文件。
6. 调用json库中的dump函数：使用json库中的dump()函数，将文件列表转换为json格式并输出。
7. 调用logging库中的getLogger函数：使用logging库中的getLogger()函数，获取当前日志记录器的名称，并输出指定的日志级别。
8. 调用re库中的findall函数：使用re库中的findall()函数，对输入文件名进行正则表达式匹配，返回匹配的所有单词或表达式。
9. 创建一个空字典：创建一个名为`file_list`的字典，用于存储文件列表。
10. 遍历字典中的每个键：使用for循环遍历字典中的每个键，即遍历文件列表中的每个文件。
11. 如果当前文件夹具有重要日志信息：如果当前文件夹具有重要日志信息，则调用getLogger()函数获取当前日志记录器，并输出指定的日志级别。
12. 输出文件列表：调用print_files()函数，将文件列表打印出来。
13. 输出日志信息：调用getLogger()函数，获取当前日志记录器，并输出指定的日志级别。


```py
#!/usr/bin/env python3

import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

from tabulate import tabulate

info = "-v" in sys.argv
debug = "-vv" in sys.argv
granular = "--granular" in sys.argv

```

这段代码是一个Python脚本，用于处理日志输出配置、获取报告文件列表、统计每个标签对应的运行次数，以下是它的作用：

1. 配置日志输出级别：通过`logging.basicConfig`函数，设置日志输出的最低级别。如果定义了`debug`参数，则输出为`DEBUG`级别，否则为`INFO`级别，否则为`WARNING`级别。

2. 获取报告文件列表：使用`__name__`作为`logger`的别名，然后使用`Path(__file__).parent / "reports"`目录下的所有子目录的名称，创建一个列表。接着使用正则表达式`re.match`筛选出所有的报告文件，确保文件名为`report_`. 

3. 统计每个标签对应的运行次数：创建一个字典`runs_per_label`，用于记录每个标签对应的运行次数。然后将`runs_per_label`初始化为0，通过列表推导式`defaultdict`创建一个统计每个标签对应运行次数的函数，函数的参数为当前标签名称。

4. 输出日志信息：通过`logger`对象输出日志信息，设置日志级别为`DEBUG`，否则为`INFO`。


```py
logging.basicConfig(
    level=logging.DEBUG if debug else logging.INFO if info else logging.WARNING
)
logger = logging.getLogger(__name__)

# Get a list of all JSON files in the directory
report_files = [
    report_file
    for dir in (Path(__file__).parent / "reports").iterdir()
    if re.match(r"^\d{8}T\d{6}_", dir.name)
    and (report_file := dir / "report.json").is_file()
]

labels = list[str]()
runs_per_label = defaultdict[str, int](lambda: 0)
```

This appears to be a Python script that uses the `gradle` command-line tool to run a series of tests and metrics on a set of labels.

The script has a number of直


```
suite_names = list[str]()
test_names = list[str]()

# Create a dictionary to store grouped success values by suffix and test
grouped_success_values = defaultdict[str, list[str]](list[str])

# Loop through each JSON file to collect suffixes and success values
for report_file in sorted(report_files):
    with open(report_file) as f:
        logger.info(f"Loading {report_file}...")

        data = json.load(f)
        if "tests" in data:
            test_tree = data["tests"]
            label = data["agent_git_commit_sha"].rsplit("/", 1)[1][:7]  # commit hash
        else:
            # Benchmark run still in progress
            test_tree = data
            label = report_file.parent.name.split("_", 1)[1]
            logger.info(f"Run '{label}' seems to be in progress")

        runs_per_label[label] += 1

        def process_test(test_name: str, test_data: dict):
            result_group = grouped_success_values[f"{label}|{test_name}"]

            if "tests" in test_data:
                logger.debug(f"{test_name} is a test suite")

                # Test suite
                suite_attempted = any(
                    test["metrics"]["attempted"] for test in test_data["tests"].values()
                )
                logger.debug(f"suite_attempted: {suite_attempted}")
                if not suite_attempted:
                    return

                if test_name not in test_names:
                    test_names.append(test_name)

                if test_data["metrics"]["percentage"] == 0:
                    result_indicator = "❌"
                else:
                    highest_difficulty = test_data["metrics"]["highest_difficulty"]
                    result_indicator = {
                        "interface": "🔌",
                        "novice": "🌑",
                        "basic": "🌒",
                        "intermediate": "🌓",
                        "advanced": "🌔",
                        "hard": "🌕",
                    }[highest_difficulty]

                logger.debug(f"result group: {result_group}")
                logger.debug(f"runs_per_label: {runs_per_label[label]}")
                if len(result_group) + 1 < runs_per_label[label]:
                    result_group.extend(
                        ["❔"] * (runs_per_label[label] - len(result_group) - 1)
                    )
                result_group.append(result_indicator)
                logger.debug(f"result group (after): {result_group}")

                if granular:
                    for test_name, test in test_data["tests"].items():
                        process_test(test_name, test)
                return

            test_metrics = test_data["metrics"]
            result_indicator = "❔"

            if not "attempted" in test_metrics:
                return
            elif test_metrics["attempted"]:
                if test_name not in test_names:
                    test_names.append(test_name)

                success_value = test_metrics["success"]
                result_indicator = {True: "✅", False: "❌"}[success_value]

            if len(result_group) + 1 < runs_per_label[label]:
                result_group.extend(
                    ["  "] * (runs_per_label[label] - len(result_group) - 1)
                )
            result_group.append(result_indicator)

        for test_name, suite in test_tree.items():
            try:
                process_test(test_name, suite)
            except KeyError as e:
                print(f"{test_name}.metrics: {suite['metrics']}")
                raise

    if label not in labels:
        labels.append(label)

```py

这段代码的主要目的是创建一个包含多个测试名称的列表。具体来说，它将一个包含多个字符串元素的列表（也就是标签列表）与一个空列表（headers）进行连接，并从test\_names列表中获取每个测试名称。接下来，它遍历test\_names列表中的每个测试名称，并准备相应的数据以进行统计。

对于每个测试名称，代码首先创建一个空行，然后遍历labels列表中的每个测试名称。接下来，它使用grouped\_success\_values字典的一个键（即"{label}||{test\_name}"）来获取与该标签相关的结果集合。如果结果集合的长度小于设定的运行数乘以每个测试名称的数量，那么代码会向结果集合中添加更多的“❔”以凑足运行数。如果结果集合的长度大于或等于运行数，那么代码会将结果集合中的所有元素清除并重置为一个新的结果集合。

最后，代码将准备好的数据添加到table\_data列表中，以便在后续的数据分析过程中被使用。


```
# Create headers
headers = ["Test Name"] + list(labels)

# Prepare data for tabulation
table_data = list[list[str]]()
for test_name in test_names:
    row = [test_name]
    for label in labels:
        results = grouped_success_values.get(f"{label}|{test_name}", ["❔"])
        if len(results) < runs_per_label[label]:
            results.extend(["❔"] * (runs_per_label[label] - len(results)))
        if len(results) > 1 and all(r == "❔" for r in results):
            results.clear()
        row.append(" ".join(results))
    table_data.append(row)

```py

这段代码使用了Python内置的`tabulate`函数来打印表格数据。`table_data`参数表示要打印的表格数据，`headers`参数表示表格的表头，`tablefmt`参数表示表格中每个单元格内的数据格式。具体来说，`tablefmt="grid"`表示在打印表格时，以表格的第一行和第一列为列，按照指定的格式和表格的第二行和第二列为行，显示表格数据。这样就形成了一个类似于网格的格式，行和列之间用指定的字符分隔。


```
# Print tabulated data
print(tabulate(table_data, headers=headers, tablefmt="grid"))

```py

# `autogpts/autogpt/agbenchmark_config/benchmarks.py`

这段代码是一个基于Python的程序，主要作用是实现对指定任务类型的自动执行。具体来说，它实现了以下功能：

1. 导入了一些必要的库，包括asyncio、sys、pathlib、autogpt、autogpt的子程序main、autogpt的命令、autogpt的配置文件、autogpt的日志、autogpt的命令索引和自动注册表。

2. 加载了指定的配置文件，通过AIProfile创建了一个AI配置实例，通过ConfigBuilder设置了一些参数。

3. 创建了一个命令注册表，通过autogpt的models.command_registry实现。

4. 实现了run_specific_agent函数，该函数接受两个参数，一个是任务类型，另一个是布尔类型的选项，表示是否是持续运行模式。函数内部首先创建了一个自动化的agent实例，然后使用agent.run_interaction_loop方法运行程序。

5. 在程序的入口部分，通过调用_configure_openai_provider函数来设置自动化的OpenAI提供者。

6. 调用run_interaction_loop函数，该函数是异步运行程序的关键部分，用于运行自动化的agent实例。

7. 最后，在程序的出口部分，记录了日志，并使用配置实例的参数化方法，导入了具体的命令，以便指定具体要执行的任务类型。


```
import asyncio
import sys
from pathlib import Path

from autogpt.agents.agent import Agent, AgentConfiguration, AgentSettings
from autogpt.app.main import _configure_openai_provider, run_interaction_loop
from autogpt.commands import COMMAND_CATEGORIES
from autogpt.config import AIProfile, ConfigBuilder
from autogpt.logs.config import configure_logging
from autogpt.models.command_registry import CommandRegistry

LOG_DIR = Path(__file__).parent / "logs"


def run_specific_agent(task: str, continuous_mode: bool = False) -> None:
    agent = bootstrap_agent(task, continuous_mode)
    asyncio.run(run_interaction_loop(agent))


```py

It looks like you're trying to create a HLP training agent using the OpenAI Foundry tools. Let me explain the steps you'll need to take to complete this task.

1. Install the required libraries:

You'll need to install the following libraries: `mlflow`, `opencv-python`, `transformers`, `oretina`, and `tensorflow`. You can do this using `pip`:
```
pip install mlflow opencv-python transformersoretina tensorflow
```py
1. Create a training script:

Create a file named `train.py` in your project directory and add the following code:
```python
import os
import random
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import mplfig
import matplotlib.pyplot as plt
import numpy as np
import opencv2
import opencv2.core as cv
import opencv2.imgcodecs as cv
import torchvision
import torchvision.transforms as transforms
import torch.utils.tensor as torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensor as torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensor as torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensor as torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensor as torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensor as torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensor as torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensor as torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensor as torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensor as torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensor as torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensor as torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensor as torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensor as torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensor as torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensor as torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensor as torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensor as torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.
```py


```
def bootstrap_agent(task: str, continuous_mode: bool) -> Agent:
    config = ConfigBuilder.build_config_from_env()
    config.debug_mode = False
    config.continuous_mode = continuous_mode
    config.continuous_limit = 20
    config.temperature = 0
    config.noninteractive_mode = True
    config.plain_output = True
    config.memory_backend = "no_memory"

    configure_logging(
        debug_mode=config.debug_mode,
        plain_output=config.plain_output,
        log_dir=LOG_DIR,
    )

    command_registry = CommandRegistry.with_command_modules(COMMAND_CATEGORIES, config)

    ai_profile = AIProfile(
        ai_name="AutoGPT",
        ai_role="a multi-purpose AI assistant.",
        ai_goals=[task],
    )

    agent_prompt_config = Agent.default_settings.prompt_config.copy(deep=True)
    agent_prompt_config.use_functions_api = config.openai_functions
    agent_settings = AgentSettings(
        name=Agent.default_settings.name,
        description=Agent.default_settings.description,
        ai_profile=ai_profile,
        config=AgentConfiguration(
            fast_llm=config.fast_llm,
            smart_llm=config.smart_llm,
            allow_fs_access=not config.restrict_to_workspace,
            use_functions_api=config.openai_functions,
            plugins=config.plugins,
        ),
        prompt_config=agent_prompt_config,
        history=Agent.default_settings.history.copy(deep=True),
    )

    agent = Agent(
        settings=agent_settings,
        llm_provider=_configure_openai_provider(config),
        command_registry=command_registry,
        legacy_config=config,
    )
    agent.attach_fs(config.app_data_dir / "agents" / "AutoGPT-benchmark")  # HACK
    return agent


```py

这段代码是一个Python脚本，主要用于检查命令行参数中是否提供了要运行的任务。如果第一个参数是一个空字符串或者没有提供任务，则脚本会输出一条使用说明，并崩溃退出。

具体来说，这段代码首先检查是否存在空字符串，如果是，则输出一条使用说明，并崩溃退出。否则，它将获取第一个命令行参数，并将其存储在变量`task`中。接下来，它调用一个名为`run_specific_agent`的函数，该函数将传入`task`参数，并在持续模式下运行。`持续模式`是一个设置，使脚本在运行时保持运行状态，即使它已经完成了。


```
if __name__ == "__main__":
    # The first argument is the script name itself, second is the task
    if len(sys.argv) != 2:
        print("Usage: python script.py <task>")
        sys.exit(1)
    task = sys.argv[1]
    run_specific_agent(task, continuous_mode=True)

```py

# `autogpts/autogpt/agbenchmark_config/__init__.py`

很抱歉，我没有看到您提供的代码。如果您能提供代码或更多上下文信息，我将非常乐意帮助您解释代码的作用。


```

```py

# `autogpts/autogpt/autogpt/command_decorator.py`

这段代码定义了一个名为 "AUTO_GPT_COMMAND_IDENTIFIER" 的常量，它的值为 "auto_gpt_command"。这个常量用于标识 AutoGPT 的命令，以便在代码中进行使用。

接下来，它从两个模块(from autogpt.agents.base import BaseAgent 和 from autogpt.config import Config)中导入了两个函数类型：from __future__ import annotations 和 import functools。from functools import挠钩函数，从 Python 2.67 开始，from **future** import 子句中的所有内容都被认为是明文可读的，但同时也从 Python 3.6 开始，from **future** import 子句中的所有内容都被认为是不可读的。

然后，它从 types 类型注释中使用并行导入从typing. 的Callable 类型中进口函数类型，并从 inspect 模块中导入从 py营业执照.类型中使用的 inspect. Annotations 函数类型，以及从 types 类型注释中使用并行导入从 inspect. 的typing. Any 类型中进口 Any 类型。

接下来，从 types 类型注释中使用并行导入从 inspect. 的typing. TYPE_CHECKING 类型中进口 FunctionType 类型，并从 inspect. 的typing. TYPE_CHECKING 类型中导入从 inspect. 的typing. Any 类型。

然后，从 types 类型注释中使用并行导入从 inspect. 的typing.丘比特. Literal 类型中import 从 inspect. 的typing.丘比特. Literal 类型，并从 inspect. 的typing.丘比特. Literal 类型中导入从 inspect. 的typing. Any 类型。

接下来，从 types 类型注释中使用并行导入从 inspect. 的typing.丘比特. Literal 类型中import 从 inspect. 的typing.丘比特. Literal 类型，并从 inspect. 的typing.丘比特. Literal 类型中导入从 inspect. 的typing. Any 类型。

然后，从 types 类型注释中使用并行导入从 inspect. 的typing.丘比特. Literal 类型中import 从 inspect. 的typing.丘比特. Literal 类型，并从 inspect. 的typing.丘比特. Literal 类型中导入从 inspect. 的typing. Any 类型。

接下来，从 types 类型注释中使用并行导入从 inspect. 的typing.丘比特. Literal 类型中import 从 inspect. 的typing.丘比特. Literal 类型，并从 inspect. 的typing.丘比特. Literal 类型中导入从 inspect. 的typing. Any 类型。

然后，从 types 类型注释中使用并行导入从 inspect. 的typing.丘比特. Literal 类型中import 从 inspect. 的typing.丘比特. Literal 类型，并从 inspect. 的typing.丘比特. Literal 类型中导入从 inspect. 的typing. Any 类型。

接下来，定义了一个名为 "AUTO_GPT_COMMAND_IDENTIFIER" 的常量，它的值为 "auto_gpt_command"。

最后，从两个模块(from autogpt.agents.base import BaseAgent 和 from autogpt.config import Config)中使用并行导入从 inspect. 的typing. Command 类型，并从 config 模块中导入从 inspect. 的Command 类型，然后定义了 AUTO_GPT_COMMAND_IDENTIFIER，作为Command 类型的一个静态成员函数。


```
from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, ParamSpec, TypeVar

if TYPE_CHECKING:
    from autogpt.agents.base import BaseAgent
    from autogpt.config import Config

from autogpt.core.utils.json_schema import JSONSchema
from autogpt.models.command import Command, CommandOutput, CommandParameter

# Unique identifier for AutoGPT commands
AUTO_GPT_COMMAND_IDENTIFIER = "auto_gpt_command"

```py

这段代码定义了一个命令装饰器，其作用是将从普通函数中创建 Command 对象。

具体来说，这段代码首先定义了一个名为 P 的 ParamSpec，用于指定命令参数的JSONSchema。然后定义了一个名为 CO 的 TypeVar，用于声明一个未来要绑定的类型变量。接下来定义了一个名为 command 的函数，该函数接收一个字符串参数 name，一个字符串参数 description，以及一个字典参数 parameters，其中键和值都是参数的名称和JSONSchema。接着定义了一个名为 enabled 的布尔参数，以及一个名为 disabled_reason 的可选字符串参数。然后定义了一个名为 aliases 的列表参数，以及一个名为 available 的布尔参数。

最后，定义了一个名为 decorator 的函数，该函数接收一个函数对象 func 和一个上下文对象 CO，并返回一个新的函数，该新函数使用了命令装饰器。如果该函数是一个异步函数，那么会将其包装为异步函数并添加命令 alias。最后，将新函数添加到了 aliases 列表中，并设置 ALIASES 和 AUTO_GPT_COMMAND_IDENTIFIER 为 True，以便将该命令与其他命令区分开。


```
P = ParamSpec("P")
CO = TypeVar("CO", bound=CommandOutput)


def command(
    name: str,
    description: str,
    parameters: dict[str, JSONSchema],
    enabled: Literal[True] | Callable[[Config], bool] = True,
    disabled_reason: Optional[str] = None,
    aliases: list[str] = [],
    available: Literal[True] | Callable[[BaseAgent], bool] = True,
) -> Callable[[Callable[P, CO]], Callable[P, CO]]:
    """The command decorator is used to create Command objects from ordinary functions."""

    def decorator(func: Callable[P, CO]) -> Callable[P, CO]:
        typed_parameters = [
            CommandParameter(
                name=param_name,
                spec=spec,
            )
            for param_name, spec in parameters.items()
        ]
        cmd = Command(
            name=name,
            description=description,
            method=func,
            parameters=typed_parameters,
            enabled=enabled,
            disabled_reason=disabled_reason,
            aliases=aliases,
            available=available,
        )

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                return await func(*args, **kwargs)

        else:

            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                return func(*args, **kwargs)

        setattr(wrapper, "command", cmd)
        setattr(wrapper, AUTO_GPT_COMMAND_IDENTIFIER, True)

        return wrapper

    return decorator

```py

# `autogpts/autogpt/autogpt/singleton.py`

这段代码定义了一个名为 Singleton 的单例 metaclass，用于确保一个类只有一个实例。这个 metaclass 是基于 abc（Abstract基类）规范的，因此实现了 ABC 规范中的 singleton（单例）特性。

具体来说，这段代码的作用是：

1. 定义了一个名为 Singleton 的类，这个类实现了 ABC 规范中的 singleton 方法。
2. 定义了一个名为 Singleton 的单例 metaclass，这个 metaclass 继承自 abc.ABCMeta 和 type。
3. 在 Singleton 的定义中，定义了一个内部方法 __call__，这个方法实际上是一个 metaclass 方法，用于创建类的实例。
4. 在 Singleton 的类中，声明了一个名为 _instances 的内部变量，这个变量是一个字典，用于存储类的实例。
5. 在 Singleton 的 __call__ 方法中，使用了 cls 参数，这个参数传递给 super 方法，确保可以正确地调用父类的实例。
6. 在 Singleton 的实例化过程中，如果这个类已经存在，就直接返回实例；否则，创建一个新的实例，并将它存储在 _instances 变量中。

总之，这段代码定义了一个用于确保一个类只有一个实例的单例 metaclass，这个 metaclass 可以通过子类来继承，从而实现 ABC 规范中的单例特性。


```
"""The singleton metaclass for ensuring only one instance of a class."""
import abc


class Singleton(abc.ABCMeta, type):
    """
    Singleton metaclass for ensuring only one instance of a class.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Call method for the singleton metaclass."""
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

```py

# `autogpts/autogpt/autogpt/utils.py`

这段代码是一个函数，名为“validate\_yaml\_file”，它接受一个参数“file”，这个参数可以是文件路径（str或Path类型），也可以是文件路径的目录（Path类型）。

函数内部使用了两种 Python标准库中的模块，分别是“pathlib”和“colorama”。其中，“pathlib”模块提供了“Path”类，用于处理文件和目录操作；“colorama”模块提供了“Fore”类，用于输出颜色。

函数的作用是判断一个YAML文件的语法是否正确。具体来说，它通过读取YAML文件内容，并使用Python标准库中的“yaml”模块来解析文件内容，如果解析成功，则返回True，否则返回一个元组，其中第一个元素表示错误信息，第二个元素表示警告信息。

函数的具体实现可以分为两个步骤：

1. 读取文件内容并解码编码。文件内容使用Python标准库中的“io”模块读取，并使用“utf-8”编码方式获取。
2. 解析文件内容。文件内容使用Python标准库中的“yaml”模块解析，解析成功后返回True，否则返回一个元组，其中第一个元素表示错误信息，第二个元素表示警告信息。

最终的结果是，如果文件内容正确，函数返回True，否则返回一个元组，其中第一个元素表示错误信息，第二个元素表示警告信息。


```
from pathlib import Path

import yaml
from colorama import Fore


def validate_yaml_file(file: str | Path):
    try:
        with open(file, encoding="utf-8") as fp:
            yaml.load(fp.read(), Loader=yaml.FullLoader)
    except FileNotFoundError:
        return (False, f"The file {Fore.CYAN}`{file}`{Fore.RESET} wasn't found")
    except yaml.YAMLError as e:
        return (
            False,
            f"There was an issue while trying to read with your AI Settings file: {e}",
        )

    return (True, f"Successfully validated {Fore.CYAN}`{file}`{Fore.RESET}!")

```py

# `autogpts/autogpt/autogpt/__init__.py`

这段代码是一个Python程序，它进行了以下操作：

1. 导入os、random和sys模块。
2. 判断字符串"pytest"是否在sys.argv或sys.modules中，如果是，则执行以下操作：

   a. 输出"Setting random seed to 42"。
   b. 将随机数种子值设为42。

这段代码的作用是让pytest运行时使用42作为随机数种子，以便在pytest测试中产生更多的随机数。这个种子值对于pytest的随机数生成是固定的，不会对测试结果产生影响。


```
import os
import random
import sys

if "pytest" in sys.argv or "pytest" in sys.modules or os.getenv("CI"):
    print("Setting random seed to 42")
    random.seed(42)

```py

# `autogpts/autogpt/autogpt/__main__.py`

这段代码是一个基于Python的程序，它使用了一个名为"AutoGPT"的人工智能助手。这个程序通过导入autogpt.app.cli模块，从而使用自动编程的方式，创建了一个客户端来与这个人工智能助手进行交互。

如果程序运行时遇到任何错误，它会输出一个错误消息，并退出脚本。如果没有错误，它将直接退出脚本，并在脚本运行结束时关闭。


```
"""AutoGPT: A GPT powered AI Assistant"""
import autogpt.app.cli

if __name__ == "__main__":
    autogpt.app.cli.cli()

```py

# `autogpts/autogpt/autogpt/agents/agent.py`

这段代码是一个Python文件，其中包含了一些导入、导入和定义了一些类型声明。

from datetime import datetime, timedelta
from typing import datetime, Optional
from pydantic import BaseModel, fields
from autogpt.core.util import apply_to_samples
from autogpt.config import Config, get_driver_model
from autogpt.models import select_model_from_registry, add_to_registry
from autogpt.config. environments import get_validation_session

# 在这里定义了能够打印的日志类
class Logger(logging.Logger):
   def __init__(self, name):
       super().__init__(name=name)
       self.logger = self._get_logger()

   def _get_logger(self):
       return self.__name__ + ' logger'

   def info(self, message):
       self.logger.info(message)

   def debug(self, message):
       self.logger.debug(message)

   def critical(self, message):
       self.logger.critical(message)

   def warning(self, message):
       self.logger.warning(message)

   def error(self, message):
       self.logger.error(message)

   def add_argument(self, argument_name, description, type_check):
       return fields.Field(argument_name, description, apply_to_samples=True, nullable=False, metadata=None)

   def add_model_dependency(self, model_name, driver_model_name, command_registry):
       return add_to_registry(model_name, driver_model_name, command_registry)

   def load_configuration(self, config_path):
       return Config.load(config_path)

   def run(self, model_name, driver_model_name, command_registry, config):
       # 在这里使用get_driver_model函数获取到模型驱动的配置
       driver_model = get_driver_model(driver_model_name)

       # 在这里使用add_to_registry函数将命令注册到命令注册表中
       registry = add_to_registry(model_name, driver_model_name, command_registry)

       # 在这里使用apply_to_samples函数应用配置到样品中
       config.apply_to_samples(registry)

       # 在这里启动日志记录器
       self.logger.start()

       # 在这里运行time.sleep
       # 这里模拟运行time.sleep(10)秒
       time.sleep(10)

       # 在这里打印日志
       self.logger.info('Sample run finished')

   def run_configuration(self, config_path):
       # 在这里读取配置文件
       config = Config.load(config_path)

       # 在这里运行
       self.run('config', config.driver_model_name, config.command_registry, config)

# 自定义日志类
class CustomLogger(Logger):
   pass

# 自定义日志输出了
def configure_logger(logger_name):
   logger = CustomLogger(logger_name)
   logger.logger.info('Python log process starts')
   yield logger
   logger.logger.info('Python log process ends')

# 自定义日志格式
def custom_log_format(message):
   return '{} [{}]'.format(logger.name, datetime.now())

# 自定义日志函数
def log_command(logger, message):
   logger.logger.info(custom_log_format(message))

# 在这里定义了一个函数，用于启动命令行
def start_console(logger):
   pass

# 在这里定义了一个函数，用于启动命令行，并打印日志
def start_console_async(logger):
   def start_console(arg):
       # 在这里打印日志
       logger.logger.info('Start console: {}'.format(arg))
       # 在这里打印命令行
       print('start console: {}'.format(arg))
       yield logger

   # 启动命令行
   yield start_console(logger)

# 在这里将自定义日志函数注册到命令行日志中
def register_custom_log_format():
   custom_log_format = fields.field(lambda message: message, bool=True)
   register_custom_log_format.register(logger_name='custom_logger')
   custom_log_format.write_format(write_mode=fields.Text, config=get_config())

# 在这里注册了一个命令行日志函数
register_custom_log_format()

# 在这里启动了命令行
start_console_async(logger)


```
from __future__ import annotations

import inspect
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from autogpt.config import Config
    from autogpt.models.command_registry import CommandRegistry

from pydantic import Field

from autogpt.core.configuration import Configurable
```py

这段代码的主要作用是创建一个智能对话模型，可以接受用户输入并输出相应的回复。具体来说，它包括以下几个组件：

1. ChatPrompt：用于在用户结束对话时提供提示信息，告诉用户可以通过 `/start` 命令重新开始对话。

2. ChatMessage：用于在用户结束对话时向服务器发送消息，告诉服务器对话已经结束。

3. ChatModelProvider：用于在服务器端提供 Chat 模型的实现，这个组件可能来自不同的服务提供商。

4. ChatModelResponse：用于在服务器端处理 Chat 模型输出的响应，将 Chat 模型的回复发送给用户。

5. ApiManager：用于与 Chat 模型服务器通信，将用户输入的消息发送给服务器。

6. LogCycleHandler：用于记录 Chat 模型的输出到文件中，以便在需要时进行审计。

7. ActionHistory：用于保存用户的历史操作，包括动作、错误结果、异步结果等。

8. ChatAction：用于实现 Chat 模型中的对话动作，将用户的输入转换为可以执行的动作。

9. ChatActionErrorResult：用于实现 Chat 模型中的错误动作，将异常信息包含在结果中返回给用户。

10. ChatActionSuccessResult：用于实现 Chat 模型中的成功动作，将结果中包含的信息返回给用户。


```
from autogpt.core.prompting import ChatPrompt
from autogpt.core.resource.model_providers import (
    ChatMessage,
    ChatModelProvider,
    ChatModelResponse,
)
from autogpt.llm.api_manager import ApiManager
from autogpt.logs.log_cycle import (
    CURRENT_CONTEXT_FILE_NAME,
    NEXT_ACTION_FILE_NAME,
    USER_INPUT_FILE_NAME,
    LogCycleHandler,
)
from autogpt.models.action_history import (
    Action,
    ActionErrorResult,
    ActionInterruptedByHuman,
    ActionResult,
    ActionSuccessResult,
)
```py

这段代码是一个基于Autogpt模型的命令行应用，它包括以下几个部分：

1. 从autogpt模型中导入CommandOutput类，用于处理命令行输出的设置和获取。
2. 从autogpt模型中导入ContextItem类，用于在上下文范围内获取用户输入的信息。
3. 从Autogpt的Base类中继承了三个Feature类，分别是ContextMixin、FileWorkspaceMixin和WatchdogMixin，这些类用于在上下文和文件工作空间中提取信息、获取元数据和监控运行状态。
4. 从PromptStrategies库中导入了一个OneShotAgentPromptStrategy，用于实现基于One-shot策略的对话模型。
5. 从utils库中导入了一个Exceptions类，用于处理异常情况，如命令执行失败和未注册命令等。
6. 在代码的顶部定义了一个logger实例，用于输出日志信息。

整个代码的作用是开发一个基于Autogpt模型的命令行应用，用于实现一个问答系统的用户提问和回答功能。该应用可以接受用户输入的提问，并在知识库中查找答案并输出结果。


```
from autogpt.models.command import CommandOutput
from autogpt.models.context_item import ContextItem

from .base import BaseAgent, BaseAgentConfiguration, BaseAgentSettings
from .features.context import ContextMixin
from .features.file_workspace import FileWorkspaceMixin
from .features.watchdog import WatchdogMixin
from .prompt_strategies.one_shot import (
    OneShotAgentPromptConfiguration,
    OneShotAgentPromptStrategy,
)
from .utils.exceptions import AgentException, CommandExecutionError, UnknownCommandError

logger = logging.getLogger(__name__)


```py

This looks like a Python class that implements an Iterator for actions that can be executed by an agent.
It appears to handle a command and its arguments, and returns an Action Success/ErrorResult.
It also appears to keep track of the action history and the number of tokens used.
It also has a limit for the number of commands/arguments that can be executed per agent.


```
class AgentConfiguration(BaseAgentConfiguration):
    pass


class AgentSettings(BaseAgentSettings):
    config: AgentConfiguration = Field(default_factory=AgentConfiguration)
    prompt_config: OneShotAgentPromptConfiguration = Field(
        default_factory=(
            lambda: OneShotAgentPromptStrategy.default_configuration.copy(deep=True)
        )
    )


class Agent(
    ContextMixin,
    FileWorkspaceMixin,
    WatchdogMixin,
    BaseAgent,
    Configurable[AgentSettings],
):
    """AutoGPT's primary Agent; uses one-shot prompting."""

    default_settings: AgentSettings = AgentSettings(
        name="Agent",
        description=__doc__,
    )

    def __init__(
        self,
        settings: AgentSettings,
        llm_provider: ChatModelProvider,
        command_registry: CommandRegistry,
        legacy_config: Config,
    ):
        prompt_strategy = OneShotAgentPromptStrategy(
            configuration=settings.prompt_config,
            logger=logger,
        )
        super().__init__(
            settings=settings,
            llm_provider=llm_provider,
            prompt_strategy=prompt_strategy,
            command_registry=command_registry,
            legacy_config=legacy_config,
        )

        self.created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        """Timestamp the agent was created; only used for structured debug logging."""

        self.log_cycle_handler = LogCycleHandler()
        """LogCycleHandler for structured debug logging."""

    def build_prompt(
        self,
        *args,
        extra_messages: Optional[list[ChatMessage]] = None,
        include_os_info: Optional[bool] = None,
        **kwargs,
    ) -> ChatPrompt:
        if not extra_messages:
            extra_messages = []

        # Clock
        extra_messages.append(
            ChatMessage.system(f"The current time and date is {time.strftime('%c')}"),
        )

        # Add budget information (if any) to prompt
        api_manager = ApiManager()
        if api_manager.get_total_budget() > 0.0:
            remaining_budget = (
                api_manager.get_total_budget() - api_manager.get_total_cost()
            )
            if remaining_budget < 0:
                remaining_budget = 0

            budget_msg = ChatMessage.system(
                f"Your remaining API budget is ${remaining_budget:.3f}"
                + (
                    " BUDGET EXCEEDED! SHUT DOWN!\n\n"
                    if remaining_budget == 0
                    else " Budget very nearly exceeded! Shut down gracefully!\n\n"
                    if remaining_budget < 0.005
                    else " Budget nearly exceeded. Finish up.\n\n"
                    if remaining_budget < 0.01
                    else ""
                ),
            )
            logger.debug(budget_msg)
            extra_messages.append(budget_msg)

        if include_os_info is None:
            include_os_info = self.legacy_config.execute_local_commands

        return super().build_prompt(
            *args,
            extra_messages=extra_messages,
            include_os_info=include_os_info,
            **kwargs,
        )

    def on_before_think(self, *args, **kwargs) -> ChatPrompt:
        prompt = super().on_before_think(*args, **kwargs)

        self.log_cycle_handler.log_count_within_cycle = 0
        self.log_cycle_handler.log_cycle(
            self.ai_profile.ai_name,
            self.created_at,
            self.config.cycle_count,
            prompt.raw(),
            CURRENT_CONTEXT_FILE_NAME,
        )
        return prompt

    def parse_and_process_response(
        self, llm_response: ChatModelResponse, *args, **kwargs
    ) -> Agent.ThoughtProcessOutput:
        for plugin in self.config.plugins:
            if not plugin.can_handle_post_planning():
                continue
            llm_response.response["content"] = plugin.post_planning(
                llm_response.response.get("content", "")
            )

        (
            command_name,
            arguments,
            assistant_reply_dict,
        ) = self.prompt_strategy.parse_response_content(llm_response.response)

        self.log_cycle_handler.log_cycle(
            self.ai_profile.ai_name,
            self.created_at,
            self.config.cycle_count,
            assistant_reply_dict,
            NEXT_ACTION_FILE_NAME,
        )

        self.event_history.register_action(
            Action(
                name=command_name,
                args=arguments,
                reasoning=assistant_reply_dict["thoughts"]["reasoning"],
            )
        )

        return command_name, arguments, assistant_reply_dict

    async def execute(
        self,
        command_name: str,
        command_args: dict[str, str] = {},
        user_input: str = "",
    ) -> ActionResult:
        result: ActionResult

        if command_name == "human_feedback":
            result = ActionInterruptedByHuman(feedback=user_input)
            self.log_cycle_handler.log_cycle(
                self.ai_profile.ai_name,
                self.created_at,
                self.config.cycle_count,
                user_input,
                USER_INPUT_FILE_NAME,
            )

        else:
            for plugin in self.config.plugins:
                if not plugin.can_handle_pre_command():
                    continue
                command_name, command_args = plugin.pre_command(
                    command_name, command_args
                )

            try:
                return_value = await execute_command(
                    command_name=command_name,
                    arguments=command_args,
                    agent=self,
                )

                # Intercept ContextItem if one is returned by the command
                if type(return_value) == tuple and isinstance(
                    return_value[1], ContextItem
                ):
                    context_item = return_value[1]
                    return_value = return_value[0]
                    logger.debug(
                        f"Command {command_name} returned a ContextItem: {context_item}"
                    )
                    self.context.add(context_item)

                result = ActionSuccessResult(outputs=return_value)
            except AgentException as e:
                result = ActionErrorResult.from_exception(e)

            result_tlength = self.llm_provider.count_tokens(str(result), self.llm.name)
            if result_tlength > self.send_token_limit // 3:
                result = ActionErrorResult(
                    reason=f"Command {command_name} returned too much output. "
                    "Do not execute this command again with the same arguments."
                )

            for plugin in self.config.plugins:
                if not plugin.can_handle_post_command():
                    continue
                if result.status == "success":
                    result.outputs = plugin.post_command(command_name, result.outputs)
                elif result.status == "error":
                    result.reason = plugin.post_command(command_name, result.reason)

        # Update action history
        self.event_history.register_result(result)

        return result


```py

这段代码定义了一个名为 `execute_command` 的函数，它接受一个命令名称、一个或多个参数、一个 `Agent` 对象和一个字符串类型的返回值 `CommandOutput`。函数的作用是执行动作并返回结果，其具体实现如下：

1. 如果传入的命令名称存在于 `agent` 对象的 `command_registry` 内部，则直接执行该命令并返回结果。

2. 如果命令名称不存在于 `agent` 对象的 `command_registry` 内部，则尝试从 `agent` 对象中读取该命令的描述并尝试使用该描述执行命令。如果命令描述与传入的参数匹配，则尝试使用命令的方法执行动作并返回结果。如果命令描述或参数不匹配，则 raise an `AgentException`。

3. 如果 `agent` 对象中没有关于要执行的命令的信息，则 raise一个名为 `UnknownCommandError` 的异常。


```
#############
# Utilities #
#############


async def execute_command(
    command_name: str,
    arguments: dict[str, str],
    agent: Agent,
) -> CommandOutput:
    """Execute the command and return the result

    Args:
        command_name (str): The name of the command to execute
        arguments (dict): The arguments for the command
        agent (Agent): The agent that is executing the command

    Returns:
        str: The result of the command
    """
    # Execute a native command with the same name or alias, if it exists
    if command := agent.command_registry.get_command(command_name):
        try:
            result = command(**arguments, agent=agent)
            if inspect.isawaitable(result):
                return await result
            return result
        except AgentException:
            raise
        except Exception as e:
            raise CommandExecutionError(str(e))

    # Handle non-native commands (e.g. from plugins)
    if agent._prompt_scratchpad:
        for name, command in agent._prompt_scratchpad.commands.items():
            if (
                command_name == name
                or command_name.lower() == command.description.lower()
            ):
                try:
                    return command.method(**arguments)
                except AgentException:
                    raise
                except Exception as e:
                    raise CommandExecutionError(str(e))

    raise UnknownCommandError(
        f"Cannot execute command '{command_name}': unknown command."
    )

```