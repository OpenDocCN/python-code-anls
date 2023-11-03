# AutoGPT源码解析 0

## CLI Documentation

This document describes how to interact with the project's CLI (Command Line Interface). It includes the types of outputs you can expect from each command. Note that the `agents stop` command will terminate any process running on port 8000.

### 1. Entry Point for the CLI

Running the `./run` command without any parameters will display the help message, which provides a list of available commands and options. Additionally, you can append `--help` to any command to view help information specific to that command.

```pysh
./run
```

**Output**:

```py
Usage: cli.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  agent     Commands to create, start and stop agents
  benchmark  Commands to start the benchmark and list tests and categories
  setup      Installs dependencies needed for your system.
```

If you need assistance with any command, simply add the `--help` parameter to the end of your command, like so:

```pysh
./run COMMAND --help
```

This will display a detailed help message regarding that specific command, including a list of any additional options and arguments it accepts.

### 2. Setup Command

```pysh
./run setup
```

**Output**:

```py
Setup initiated
Installation has been completed.
```

This command initializes the setup of the project.

### 3. Agents Commands

**a. List All Agents**

```pysh
./run agent list
```

**Output**:

```
Available agents: 🤖
        🐙 forge
        🐙 autogpt
```py

Lists all the available agents.

**b. Create a New Agent**

```sh
./run agent create my_agent
```py

**Output**:

```
🎉 New agent 'my_agent' created and switched to the new directory in autogpts folder.
```py

Creates a new agent named 'my_agent'.

**c. Start an Agent**

```sh
./run agent start my_agent
```py

**Output**:

```
... (ASCII Art representing the agent startup)
[Date and Time] [forge.sdk.db] [DEBUG] 🐛  Initializing AgentDB with database_string: sqlite:///agent.db
[Date and Time] [forge.sdk.agent] [INFO] 📝  Agent server starting on http://0.0.0.0:8000
```py

Starts the 'my_agent' and displays startup ASCII art and logs.

**d. Stop an Agent**

```sh
./run agent stop
```py

**Output**:

```
Agent stopped
```py

Stops the running agent.

### 4. Benchmark Commands

**a. List Benchmark Categories**

```sh
./run benchmark categories list
```py

**Output**:

```
Available categories: 📚
        📖 code
        📖 safety
        📖 memory
        ... (and so on)
```py

Lists all available benchmark categories.

**b. List Benchmark Tests**

```sh
./run benchmark tests list
```py

**Output**:

```
Available tests: 📚
        📖 interface
                🔬 Search - TestSearch
                🔬 Write File - TestWriteFile
        ... (and so on)
```py

Lists all available benchmark tests.

**c. Show Details of a Benchmark Test**

```sh
./run benchmark tests details TestWriteFile
```py

**Output**:

```
TestWriteFile
-------------

        Category:  interface
        Task:  Write the word 'Washington' to a .txt file
        ... (and other details)
```py

Displays the details of the 'TestWriteFile' benchmark test.

**d. Start Benchmark for the Agent**

```sh
./run benchmark start my_agent
```py

**Output**:

```
(more details about the testing process shown whilst the test are running)
============= 13 failed, 1 passed in 0.97s ============...
```py

Displays the results of the benchmark tests on 'my_agent'.


# `cli.py`

这段代码是一个用于管理自动生成文章项目的简单文件。它包含以下几行。

1. 欢迎使用此文件，以帮助用户管理自动生成项目。
2. 如果想贡献，请仅使用Python自带的库。
3. 为了提高效率，将所需的库导入函数中，仅导入所需的库。


```
"""
This is a minimal file intended to be run by users to help them manage the autogpt projects.

If you want to contribute, please use only libraries that come as part of Python.
To ensure efficiency, add the imports to the functions so only what is needed is imported.
"""
try:
    import click
    import github
except ImportError:
    import os

    os.system("pip3 install click")
    os.system("pip3 install PyGithub")
    import click


```py

这段代码是一个Python类中的方法，是一个命令行脚本。该方法的作用是在终端窗口中显示一个名为“setup”的命令行选项。

该方法的参数是一个参数列表，其中包含三个参数：

1. 该方法在一个名为“setup”的命令行选项上使用了@click.group()的语法。这个语法告诉Python程序，该选项属于名为“setup”的组，并且该选项可以用于命令行脚本。

2. 该方法使用了一个名为“pass”的函数作为方法体。这个函数在这里并没有做任何实际的工作，它只是一个空函数，没有定义任何变量或执行任何代码。

3. 该方法使用了两次下面的@click.command()语法来定义命令行选项的功能。这些语法告诉Python程序，下面的命令行选项是一个命令，并且它属于名为“setup”的组。

4. 该方法在命令行选项的功能字符串中使用了以下格式字符：

```
d8888          888             .d8888b.  888888888 
     d8888          888            d88P  Y88b 888   Y88b    888     
    d88P888          888            888    888    888     
   d88P 888 888  888 888   d88b.  888        888   d88P    888     
  d88P  888 888  888 888  888   d88```py

这些格式字符告诉命令行程序，该选项将在终端窗口中显示一个带有“setup”字样的命令行选项，并且该选项将会在三个平台上生成一个带有星号*的感叹号。


```
@click.group()
def cli():
    pass


@cli.command()
def setup():
    """Installs dependencies needed for your system. Works with Linux, MacOS and Windows WSL."""
    import os
    import subprocess

    click.echo(
        click.style(
            """
       d8888          888             .d8888b.  8888888b. 88888888888 
      d88888          888            d88P  Y88b 888   Y88b    888     
     d88P888          888            888    888 888    888    888     
    d88P 888 888  888 888888 .d88b.  888        888   d88P    888     
   d88P  888 888  888 888   d88""88b 888  88888 8888888P"     888     
  d88P   888 888  888 888   888  888 888    888 888           888     
 d8888888888 Y88b 888 Y88b. Y88..88P Y88b  d88P 888           888     
```py

The script appears to be setting up the instructions for generating a new GitHub access token. It is using red text to indicate that certain steps are important and bolding the text to make it more prominent.

It looks like the script is checking if a `.github_access_token` file exists. If it does not exist, it will be created with the file extension and a message telling the user to copy the text into the file and run the `setup` command again.

If the `.github_access_token` file exists, the script will print instructions on how to configure the access token, including copying the file to the specified location and running the `setup` command again.

It is not clear from the script what the `setup` command is. It is likely a command to install the necessary packages or configuration files for the access token to work properly.


```
d88P     888  "Y88888  "Y888 "Y88P"   "Y8888P88 888           888     
                                                                                                                                       
""",
            fg="green",
        )
    )

    script_dir = os.path.dirname(os.path.realpath(__file__))
    setup_script = os.path.join(script_dir, "setup.sh")
    install_error = False
    if os.path.exists(setup_script):
        click.echo(click.style("🚀 Setup initiated...\n", fg="green"))
        try:
            subprocess.check_call([setup_script], cwd=script_dir)
        except subprocess.CalledProcessError:
            click.echo(
                click.style("❌ There was an issue with the installation.", fg="red")
            )
            install_error = True
    else:
        click.echo(
            click.style(
                "❌ Error: setup.sh does not exist in the current directory.", fg="red"
            )
        )
        install_error = True

    try:
        # Check if GitHub user name is configured
        user_name = (
            subprocess.check_output(["git", "config", "user.name"])
            .decode("utf-8")
            .strip()
        )
        user_email = (
            subprocess.check_output(["git", "config", "user.email"])
            .decode("utf-8")
            .strip()
        )

        if user_name and user_email:
            click.echo(
                click.style(
                    f"✅ GitHub account is configured with username: {user_name} and email: {user_email}",
                    fg="green",
                )
            )
        else:
            raise subprocess.CalledProcessError(
                returncode=1, cmd="git config user.name or user.email"
            )

    except subprocess.CalledProcessError:
        # If the GitHub account is not configured, print instructions on how to set it up
        click.echo(click.style("❌ GitHub account is not configured.", fg="red"))
        click.echo(
            click.style(
                "To configure your GitHub account, use the following commands:",
                fg="red",
            )
        )
        click.echo(
            click.style(
                '  git config --global user.name "Your GitHub Username"', fg="red"
            )
        )
        click.echo(
            click.style(
                '  git config --global user.email "Your GitHub Email"', fg="red"
            )
        )
        install_error = True
    print_access_token_instructions = False
    # Check for the existence of the .github_access_token file
    if os.path.exists(".github_access_token"):
        with open(".github_access_token", "r") as file:
            github_access_token = file.read().strip()
            if github_access_token:
                click.echo(
                    click.style(
                        "✅ GitHub access token loaded successfully.", fg="green"
                    )
                )
                # Check if the token has the required permissions
                import requests

                headers = {"Authorization": f"token {github_access_token}"}
                response = requests.get("https://api.github.com/user", headers=headers)
                if response.status_code == 200:
                    scopes = response.headers.get("X-OAuth-Scopes")
                    if "public_repo" in scopes or "repo" in scopes:
                        click.echo(
                            click.style(
                                "✅ GitHub access token has the required permissions.",
                                fg="green",
                            )
                        )
                    else:
                        install_error = True
                        click.echo(
                            click.style(
                                "❌ GitHub access token does not have the required permissions. Please ensure it has 'public_repo' or 'repo' scope.",
                                fg="red",
                            )
                        )
                else:
                    install_error = True
                    click.echo(
                        click.style(
                            "❌ Failed to validate GitHub access token. Please ensure it is correct.",
                            fg="red",
                        )
                    )
            else:
                install_error = True
                click.echo(
                    click.style(
                        "❌ GitHub access token file is empty. Please follow the instructions below to set up your GitHub access token.",
                        fg="red",
                    )
                )
                print_access_token_instructions = True
    else:
        # Create the .github_access_token file if it doesn't exist
        with open(".github_access_token", "w") as file:
            file.write("")
        install_error = True
        print_access_token_instructions = True

    if print_access_token_instructions:
        # Instructions to set up GitHub access token
        click.echo(
            click.style(
                "❌ To configure your GitHub access token, follow these steps:", fg="red"
            )
        )
        click.echo(
            click.style("\t1. Ensure you are logged into your GitHub account", fg="red")
        )
        click.echo(
            click.style("\t2. Navigate to https://github.com/settings/tokens", fg="red")
        )
        click.echo(click.style("\t3. Click on 'Generate new token'.", fg="red"))
        click.echo(click.style("\t4. Click on 'Generate new token (classic)'.", fg="red"))
        click.echo(
            click.style(
                "\t5. Fill out the form to generate a new token. Ensure you select the 'repo' scope.",
                fg="red",
            )
        )
        click.echo(
            click.style(
                "\t6. Open the '.github_access_token' file in the same directory as this script and paste the token into this file.",
                fg="red",
            )
        )
        click.echo(
            click.style("\t7. Save the file and run the setup command again.", fg="red")
        )
    if install_error:
        click.echo(
            click.style(
                "\n\n🔴 If you need help, please raise a ticket on GitHub at https://github.com/Significant-Gravitas/AutoGPT/issues\n\n",
                fg="magenta",
                bold=True,
            )
        )


```py

This is a command-line script that creates a new agent with a given name. The agent's code is stored in the file "autogpts/{agent_name}". If the agent name contains invalid characters, a failure message will be displayed. If the agent name is already in use, the script will notify the user and ask them to enter a new name. The user can then run the "run arena enter {agent_name}" command to enter the arena and play using the new agent.


```
@cli.group()
def agent():
    """Commands to create, start and stop agents"""
    pass


@agent.command()
@click.argument("agent_name")
def create(agent_name):
    """Create's a new agent with the agent name provided"""
    import os
    import re
    import shutil

    if not re.match("^[a-zA-Z0-9_-]*$", agent_name):
        click.echo(
            click.style(
                f"😞 Agent name '{agent_name}' is not valid. It should not contain spaces or special characters other than -_",
                fg="red",
            )
        )
        return
    try:
        new_agent_dir = f"./autogpts/{agent_name}"
        new_agent_name = f"{agent_name.lower()}.json"

        existing_arena_files = [name.lower() for name in os.listdir("./arena/")]

        if not os.path.exists(new_agent_dir) and not new_agent_name in existing_arena_files:
            shutil.copytree("./autogpts/forge", new_agent_dir)
            click.echo(
                click.style(
                    f"🎉 New agent '{agent_name}' created. The code for your new agent is in: autogpts/{agent_name}",
                    fg="green",
                )
            )
            click.echo(
                click.style(
                    f"🚀 If you would like to enter the arena, run './run arena enter {agent_name}'",
                    fg="yellow",
                )
            )
        else:
            click.echo(
                click.style(
                    f"😞 Agent '{agent_name}' already exists. Enter a different name for your agent, the name needs to be unique regardless of case",
                    fg="red",
                )
            )
    except Exception as e:
        click.echo(click.style(f"😢 An error occurred: {e}", fg="red"))


```py

这段代码是一个命令行工具，是一个@agent.command装饰，它接收两个参数，一个是agent_name，另一个是--no-setup选项，它是用来判断是否要重新构建环境。

工具的内部实现主要通过以下步骤：

1. 获取agent_name的定义，这个定义来自於autogpts库，在script_dir目录下有一个子目录，叫作{agent_name}，包含run和run_benchmark两个可执行文件，以及一个名为"agent_name.py"的文件。

2. 如果已经存在agent_name目录，并且run和run_benchmark文件都存在，则执行以下操作：

   1. 切换到agent_name目录
   2. 如果no_setup选项为真，则执行以下操作：
       2.1. 调用subprocess.Popen["./setup"]，将当前目录切换到script_dir，并等待其完成
       2.2. 调用subprocess.Popen["./run_benchmark"] serve，运行基准测试，并等待其完成
       2.3. 切换回当前目录
   3. 如果no_setup选项为假，则执行以下操作：
       2.3. 调用subprocess.Popen["./run"]，运行agent，并等待其完成

3. 如果agent_name目录不存在，则输出错误信息。


```
@agent.command()
@click.argument("agent_name")
@click.option("--no-setup", is_flag=True, help="Rebuilds your poetry env")
def start(agent_name, no_setup):
    """Start agent command"""
    import os
    import subprocess

    script_dir = os.path.dirname(os.path.realpath(__file__))
    agent_dir = os.path.join(script_dir, f"autogpts/{agent_name}")
    run_command = os.path.join(agent_dir, "run")
    run_bench_command = os.path.join(agent_dir, "run_benchmark")
    if os.path.exists(agent_dir) and os.path.isfile(run_command) and os.path.isfile(run_bench_command):
        os.chdir(agent_dir)
        if not no_setup:
            setup_process = subprocess.Popen(["./setup"], cwd=agent_dir)
            setup_process.wait()
        subprocess.Popen(["./run_benchmark", "serve"], cwd=agent_dir)
        click.echo(f"Benchmark Server starting please wait...")
        subprocess.Popen(["./run"], cwd=agent_dir)
        click.echo(f"Agent '{agent_name}' starting please wait...")
    elif not os.path.exists(agent_dir):
        click.echo(
            click.style(
                f"😞 Agent '{agent_name}' does not exist. Please create the agent first.",
                fg="red",
            )
        )
    else:
        click.echo(
            click.style(
                f"😞 Run command does not exist in the agent '{agent_name}' directory.",
                fg="red",
            )
        )


```py

这段代码是一个命令行工具中的命令函数，用于停止指定 agent 的进程。具体来说，该函数会尝试使用 lsof 命令来列出所有正在运行的 agent 进程并输出它们的 PID(进程 ID)，然后使用 os.kill 函数来停止指定 PID 的 agent 进程。如果某个进程无法被停止，函数会输出一条消息并退出。

函数中包含两个参数，第一个参数是一个字符串 "stop"，用于标识该命令函数。第二个参数 stop 参数是一个命令行参数，用于指定要停止的 agent 进程的 PID。函数内部还包含一个自定义函数 try_stop，该函数使用 lsof 命令来查找所有正在运行的 agent 进程并停止指定 PID 的进程。如果该函数内部出现的任何一个命令行工具无法正常退出，它会向用户输出一条消息并退出。


```
@agent.command()
def stop():
    """Stop agent command"""
    import os
    import signal
    import subprocess

    try:
        pids = subprocess.check_output(["lsof", "-t", "-i", ":8000"]).split()
        if isinstance(pids, int):
            os.kill(int(pids), signal.SIGTERM)
        else:
            for pid in pids:
                os.kill(int(pid), signal.SIGTERM)
    except subprocess.CalledProcessError:
        click.echo("No process is running on port 8000")

    try:
        pids = int(subprocess.check_output(["lsof", "-t", "-i", ":8080"]))
        if isinstance(pids, int):
            os.kill(int(pids), signal.SIGTERM)
        else:
            for pid in pids:
                os.kill(int(pid), signal.SIGTERM)
    except subprocess.CalledProcessError:
        click.echo("No process is running on port 8080")

```py

这段代码是一个Python类，名为"list"。该类定义了一个命令方法，用于列出系统中所有可用的AI代理程序。

具体来说，该方法通过调用os.listdir()函数获取所有与agent.py(或类似的名称)相关联的目录，并使用os.path.isdir()函数检查每个目录是否包含一个名为agents的文件。如果是，该方法将提取该文件夹中的所有代理程序列表并打印出来。如果列表中代理程序数量少于1个，该方法将打印一个错误消息。如果列表中代理程序数量不存在，该方法将打印一个错误消息。如果agents.py目录不存在，该方法将打印一个错误消息。如果该方法内部发生任何错误，该方法将打印一个错误消息并返回。

该方法使用了agent.py文件中的import语句，该语句在代码中导入os模块以访问操作系统中的文件系统操作。


```
@agent.command()
def list():
    """List agents command"""
    import os

    try:
        agents_dir = "./autogpts"
        agents_list = [
            d
            for d in os.listdir(agents_dir)
            if os.path.isdir(os.path.join(agents_dir, d))
        ]
        if agents_list:
            click.echo(click.style("Available agents: 🤖", fg="green"))
            for agent in agents_list:
                click.echo(click.style(f"\t🐙 {agent}", fg="blue"))
        else:
            click.echo(click.style("No agents found 😞", fg="red"))
    except FileNotFoundError:
        click.echo(click.style("The autogpts directory does not exist 😢", fg="red"))
    except Exception as e:
        click.echo(click.style(f"An error occurred: {e} 😢", fg="red"))


```py

这段代码是一个命令行工具，名为“benchmark”。该工具的作用是启动一个基准测试，并为测试指定类和测试类型，同时忽略可能出现在命令行输入中的任何未知选项。

具体来说，该工具包含一个名为“start”的命令和一个名为“benchmark”的函数。在“start”命令中，使用了一个名为“agent_name”的参数和一个名为“subprocess_args”的参数，其中“subprocess_args”参数允许用户传递一个或多个参数。函数中，首先创建了一个名为“agent_dir”的目录，用于存储基准测试脚本，然后使用创建的目录作为工作目录。接着，使用Python的“subprocess”模块中的“Popen”函数来运行基准测试，其中包含传递给“subprocess_args”的参数。最后，显示一条消息，表明基准测试是否成功运行，并使用绿色字体强调了运行成功的消息，如果没有成功运行，则使用红色字体显示一条消息。


```
@cli.group()
def benchmark():
    """Commands to start the benchmark and list tests and categories"""
    pass


@benchmark.command(
    context_settings=dict(
        ignore_unknown_options=True,
    )
)
@click.argument("agent_name")
@click.argument("subprocess_args", nargs=-1, type=click.UNPROCESSED)
def start(agent_name, subprocess_args):
    """Starts the benchmark command"""
    import os
    import subprocess

    script_dir = os.path.dirname(os.path.realpath(__file__))
    agent_dir = os.path.join(script_dir, f"autogpts/{agent_name}")
    benchmark_script = os.path.join(agent_dir, "run_benchmark")
    if os.path.exists(agent_dir) and os.path.isfile(benchmark_script):
        os.chdir(agent_dir)
        subprocess.Popen([benchmark_script, *subprocess_args], cwd=agent_dir)
        click.echo(
            click.style(
                f"🚀 Running benchmark for '{agent_name}' with subprocess arguments: {' '.join(subprocess_args)}",
                fg="green",
            )
        )
    else:
        click.echo(
            click.style(
                f"😞 Agent '{agent_name}' does not exist. Please create the agent first.",
                fg="red",
            )
        )


```py

这段代码是一个Python脚本，定义了一个名为"benchmark_categories"的函数。该函数的作用是在给定目录下查找所有与"benchmark"和"agbenchmark"相关的挑战相关的JSON文件，并将这些文件中的"category"键的值添加到"categories"集合中。

具体来说，函数在以下步骤中实现了：

1. 使用os.path.dirname()函数获取当前文件目录。
2. 使用os.path.join()函数构建目标目录的路径，即包含JSON文件的目录。
3. 使用glob.glob()函数获取目录中的所有JSON文件。
4. 对于每个JSON文件，函数使用os.path.join()函数构建一个 exclude 目录的路径，其中包含一个名为"deprecated"的目录。然后，函数使用with open()函数读取JSON文件内容，并使用json.load()函数将其转换为Python可用的数据结构。
5. 函数尝试打开每个JSON文件，如果文件无法打开，则输出错误消息并跳过该文件。
6. 如果所有JSON文件都打开并读取成功，则函数输出绿色字体，每个文件夹夹标题，并在其后面添加感叹号。

最后，函数将打印红色字体，表示没有找到任何类。


```
@benchmark.group(name="categories")
def benchmark_categories():
    """Benchmark categories group command"""
    pass


@benchmark_categories.command(name="list")
def benchmark_categories_list():
    """List benchmark categories command"""
    import glob
    import json
    import os

    categories = set()

    # Get the directory of this file
    this_dir = os.path.dirname(os.path.abspath(__file__))

    glob_path = os.path.join(
        this_dir, "./benchmark/agbenchmark/challenges/**/[!deprecated]*/data.json"
    )
    # Use it as the base for the glob pattern, excluding 'deprecated' directory
    for data_file in glob.glob(glob_path, recursive=True):
        if 'deprecated' not in data_file:
            with open(data_file, "r") as f:
                try:
                    data = json.load(f)
                    categories.update(data.get("category", []))
                except json.JSONDecodeError:
                    print(f"Error: {data_file} is not a valid JSON file.")
                    continue
                except IOError:
                    print(f"IOError: file could not be read: {data_file}")
                    continue

    if categories:
        click.echo(click.style("Available categories: 📚", fg="green"))
        for category in categories:
            click.echo(click.style(f"\t📖 {category}", fg="blue"))
    else:
        click.echo(click.style("No categories found 😞", fg="red"))


```py

This is a Python script that retrieves benchmark data from a JSON file and benchmarks against it. The JSON file contains information about categories, tests, and their names.

The script first checks the base directory and the JSON file, and then loops through the tests directory. If a test file is found, it reads the JSON data and benchmarks against it. The tests are compared to the benchmarks, and if a test is found to be missing, it is highlighted.

If no tests are found, the script prints a message. If a test is found to be invalid, such as a missing or improperly formatted JSON file, it also prints a message.

The script then outputs a table with the available tests, with the first column being the categories and the second column being the tests. The tests are output with a colored font for better readability.


```
@benchmark.group(name="tests")
def benchmark_tests():
    """Benchmark tests group command"""
    pass


@benchmark_tests.command(name="list")
def benchmark_tests_list():
    """List benchmark tests command"""
    import glob
    import json
    import os
    import re

    tests = {}

    # Get the directory of this file
    this_dir = os.path.dirname(os.path.abspath(__file__))

    glob_path = os.path.join(
        this_dir, "./benchmark/agbenchmark/challenges/**/[!deprecated]*/data.json"
    )
    # Use it as the base for the glob pattern, excluding 'deprecated' directory
    for data_file in glob.glob(glob_path, recursive=True):
        if 'deprecated' not in data_file:
            with open(data_file, "r") as f:
                try:
                    data = json.load(f)
                    category = data.get("category", [])
                    test_name = data.get("name", "")
                    if category and test_name:
                        if category[0] not in tests:
                            tests[category[0]] = []
                        tests[category[0]].append(test_name)
                except json.JSONDecodeError:
                    print(f"Error: {data_file} is not a valid JSON file.")
                    continue
                except IOError:
                    print(f"IOError: file could not be read: {data_file}")
                    continue

    if tests:
        click.echo(click.style("Available tests: 📚", fg="green"))
        for category, test_list in tests.items():
            click.echo(click.style(f"\t📖 {category}", fg="blue"))
            for test in sorted(test_list):
                test_name = (
                    " ".join(word for word in re.split("([A-Z][a-z]*)", test) if word)
                    .replace("_", "")
                    .replace("C L I", "CLI")
                    .replace("  ", " ")
                )
                test_name_padded = f"{test_name:<40}"
                click.echo(click.style(f"\t\t🔬 {test_name_padded} - {test}", fg="cyan"))
    else:
        click.echo(click.style("No tests found 😞", fg="red"))


```py

This is a Python script that uses the Click library to handle command-line interactions with the tool. It reads a file called "data.json" that contains information about files on the computer, including their names, descriptions, and side effects, and then outputs some information about each file.

The script reads the file and outputs the following information about each file:

* Name
* Description
* Difficulty
* Side Effects

For files with a score higher than 4.0, the script also outputs additional information:

* Difficulty type
* Additional side effects

The script uses a simple format for the output, with different information for each type of file.


```
@benchmark_tests.command(name="details")
@click.argument("test_name")
def benchmark_tests_details(test_name):
    """Benchmark test details command"""
    import glob
    import json
    import os

    # Get the directory of this file
    this_dir = os.path.dirname(os.path.abspath(__file__))

    glob_path = os.path.join(
        this_dir, "./benchmark/agbenchmark/challenges/**/[!deprecated]*/data.json"
    )
    # Use it as the base for the glob pattern, excluding 'deprecated' directory
    for data_file in glob.glob(glob_path, recursive=True):
        with open(data_file, "r") as f:
            try:
                data = json.load(f)
                if data.get("name") == test_name:
                    click.echo(
                        click.style(
                            f"\n{data.get('name')}\n{'-'*len(data.get('name'))}\n",
                            fg="blue",
                        )
                    )
                    click.echo(
                        click.style(
                            f"\tCategory:  {', '.join(data.get('category'))}",
                            fg="green",
                        )
                    )
                    click.echo(click.style(f"\tTask:  {data.get('task')}", fg="green"))
                    click.echo(
                        click.style(
                            f"\tDependencies:  {', '.join(data.get('dependencies')) if data.get('dependencies') else 'None'}",
                            fg="green",
                        )
                    )
                    click.echo(
                        click.style(f"\tCutoff:  {data.get('cutoff')}\n", fg="green")
                    )
                    click.echo(
                        click.style("\tTest Conditions\n\t-------", fg="magenta")
                    )
                    click.echo(
                        click.style(
                            f"\t\tAnswer: {data.get('ground').get('answer')}",
                            fg="magenta",
                        )
                    )
                    click.echo(
                        click.style(
                            f"\t\tShould Contain: {', '.join(data.get('ground').get('should_contain'))}",
                            fg="magenta",
                        )
                    )
                    click.echo(
                        click.style(
                            f"\t\tShould Not Contain: {', '.join(data.get('ground').get('should_not_contain'))}",
                            fg="magenta",
                        )
                    )
                    click.echo(
                        click.style(
                            f"\t\tFiles: {', '.join(data.get('ground').get('files'))}",
                            fg="magenta",
                        )
                    )
                    click.echo(
                        click.style(
                            f"\t\tEval: {data.get('ground').get('eval').get('type')}\n",
                            fg="magenta",
                        )
                    )
                    click.echo(click.style("\tInfo\n\t-------", fg="yellow"))
                    click.echo(
                        click.style(
                            f"\t\tDifficulty: {data.get('info').get('difficulty')}",
                            fg="yellow",
                        )
                    )
                    click.echo(
                        click.style(
                            f"\t\tDescription: {data.get('info').get('description')}",
                            fg="yellow",
                        )
                    )
                    click.echo(
                        click.style(
                            f"\t\tSide Effects: {', '.join(data.get('info').get('side_effects'))}",
                            fg="yellow",
                        )
                    )
                    break

            except json.JSONDecodeError:
                print(f"Error: {data_file} is not a valid JSON file.")
                continue
            except IOError:
                print(f"IOError: file could not be read: {data_file}")
                continue

```py

This script appears to be a simple script for submitting an agent to an arena. The script first pulls the latest changes from the specified Git repository, and then creates a new branch called `arena_submission_{agent_name}`. It then creates a dictionary with the necessary information for a submission, and if the `--branch` option was passed, creates a new branch called `arena_submission_{agent_name}`.

The script then submits the agent to the arena by creating a new branch called `arena_submission_{agent_name}`, and then making a Git push to the parent repository. Finally, it creates a pull request to the parent repository.

Note that the script assumes that the arena is set up to accept submissions, and that the agent has been configured to participate in the arena. Additionally, the script creates a new directory called `arena` to hold the JSON file for the submission, and a new branch called `arena_submission_{agent_name}` in the directory.


```
@cli.group()
def arena():
    """Commands to enter the arena"""
    pass


@arena.command()
@click.argument("agent_name")
@click.option("--branch", default="master", help="Branch to use instead of master")
def enter(agent_name, branch):
    import json
    import os
    import subprocess
    from datetime import datetime

    from github import Github

    # Check if the agent_name directory exists in the autogpts directory
    agent_dir = f"./autogpts/{agent_name}"
    if not os.path.exists(agent_dir):
        click.echo(
            click.style(
                f"❌ The directory for agent '{agent_name}' does not exist in the autogpts directory.",
                fg="red",
            )
        )
        click.echo(
            click.style(
                f"🚀 Run './run agent create {agent_name}' to create the agent.",
                fg="yellow",
            )
        )

        return
    else:
        # Check if the agent has already entered the arena
        try:
            subprocess.check_output(
                [
                    "git",
                    "rev-parse",
                    "--verify",
                    "--quiet",
                    f"arena_submission_{agent_name}",
                ]
            )
        except subprocess.CalledProcessError:
            pass
        else:
            click.echo(
                click.style(
                    f"⚠️  The agent '{agent_name}' has already entered the arena. To update your submission, follow these steps:",
                    fg="yellow",
                )
            )
            click.echo(
                click.style(
                    f"1. Get the git hash of your submission by running 'git rev-parse HEAD' on the branch you want to submit to the arena.",
                    fg="yellow",
                )
            )
            click.echo(
                click.style(
                    f"2. Change the branch to 'arena_submission_{agent_name}' by running 'git checkout arena_submission_{agent_name}'.",
                    fg="yellow",
                )
            )
            click.echo(
                click.style(
                    f"3. Modify the 'arena/{agent_name}.json' to include the new commit hash of your submission (the hash you got from step 1) and an up-to-date timestamp by running './run arena update {agent_name} hash --branch x'.",
                    fg="yellow",
                )
            )
            click.echo(
                click.style(
                    f"Note: The '--branch' option is only needed if you want to change the branch that will be used.",
                    fg="yellow",
                )
            )
            return

    # Check if there are staged changes
    staged_changes = [
        line
        for line in subprocess.check_output(["git", "status", "--porcelain"])
        .decode("utf-8")
        .split("\n")
        if line and line[0] in ("A", "M", "D", "R", "C")
    ]
    if staged_changes:
        click.echo(
            click.style(
                f"❌ There are staged changes. Please commit or stash them and run the command again.",
                fg="red",
            )
        )
        return

    try:
        # Load GitHub access token from file
        with open(".github_access_token", "r") as file:
            github_access_token = file.read().strip()

        # Get GitHub repository URL
        github_repo_url = (
            subprocess.check_output(["git", "config", "--get", "remote.origin.url"])
            .decode("utf-8")
            .strip()
        )

        if github_repo_url.startswith("git@"):
            github_repo_url = (
                github_repo_url.replace(":", "/")
                .replace("git@", "https://")
                .replace(".git", "")
            )

        # If --branch is passed, use it instead of master
        if branch:
            branch_to_use = branch
        else:
            branch_to_use = "master"

        # Get the commit hash of HEAD of the branch_to_use
        commit_hash_to_benchmark = (
            subprocess.check_output(["git", "rev-parse", branch_to_use])
            .decode("utf-8")
            .strip()
        )

        arena_submission_branch = f"arena_submission_{agent_name}"
        # Create a new branch called arena_submission_{agent_name}
        subprocess.check_call(["git", "checkout", "-b", arena_submission_branch])
        # Create a dictionary with the necessary fields
        data = {
            "github_repo_url": github_repo_url,
            "timestamp": datetime.utcnow().isoformat(),
            "commit_hash_to_benchmark": commit_hash_to_benchmark,
        }

        # If --branch was passed, add branch_to_benchmark to the JSON file
        if branch:
            data["branch_to_benchmark"] = branch

        # Create agent directory if it does not exist
        subprocess.check_call(["mkdir", "-p", "arena"])

        # Create a JSON file with the data
        with open(f"arena/{agent_name}.json", "w") as json_file:
            json.dump(data, json_file, indent=4)

        # Create a commit with the specified message
        subprocess.check_call(["git", "add", f"arena/{agent_name}.json"])
        subprocess.check_call(
            ["git", "commit", "-m", f"{agent_name} entering the arena"]
        )

        # Push the commit
        subprocess.check_call(["git", "push", "origin", arena_submission_branch])

        # Create a PR into the parent repository
        g = Github(github_access_token)
        repo_name = github_repo_url.replace("https://github.com/", '')
        repo = g.get_repo(repo_name)
        parent_repo = repo.parent
        if parent_repo:
            pr_message = f"""
```py

这段代码是一个欢迎参加AutoGPT Arena Hacks Hackathon的界面，它向参赛者介绍了他们的项目，并询问了一些关于他们的团队和项目的信息。这些信息将用于评分和比较参赛者之间的差异。

其中，`AutoGPT`可能是一个基于预训练的语言模型的人工智能，`Arena`是一个比赛或竞赛场地的名称，`Hacks`可能是一个编程术语，表示这个比赛旨在发现和分享编程技巧和工具。

在这段注释中，还有一些欢迎参赛者的说明，包括一个关于如何构建项目的指导，以及一个关于项目想法的要求。


```
### 🌟 Welcome to the AutoGPT Arena Hacks Hackathon! 🌟

Hey there amazing builders! We're thrilled to have you join this exciting journey. Before you dive deep into building, we'd love to know more about you and the awesome project you are envisioning. Fill out the template below to kickstart your hackathon journey. May the best agent win! 🏆

#### 🤖 Team Introduction

- **Agent Name:** {agent_name}
- **Team Members:** (Who are the amazing minds behind this team? Do list everyone along with their roles!)
- **Repository Link:** [{github_repo_url.replace('https://github.com/', '')}]({github_repo_url})

#### 🌟 Project Vision

- **Starting Point:** (Are you building from scratch or starting with an existing agent? Do tell!)
- **Preliminary Ideas:** (Share your initial ideas and what kind of project you are aiming to build. We are all ears!)
  
```py

这段代码是一个 prize category 的示例，它会向参加竞赛的团队介绍并提供一些说明。它包括以下内容：

目标奖品：这个奖品抓住了您的注意力，您希望在竞赛中获得此奖品。

为什么这个奖品：我们希望了解为什么您觉得这个奖品是您的团队的一个合适的适合。

介绍视频：提供一个介绍您团队的视频，让您有机会分享您如何为竞赛做好准备。

Additional Notes：如果您有其他要分享的事情，请在这里放在一起。

Compliance with Hackathon Rules：我们想提醒您遵守竞赛规则。


```
#### 🏆 Prize Category

- **Target Prize:** (Which prize caught your eye? Which one are you aiming for?)
- **Why this Prize:** (We'd love to know why this prize feels like the right fit for your team!)

#### 🎬 Introduction Video

- **Video Link:** (If you'd like, share a short video where you introduce your team and talk about your project. We'd love to see your enthusiastic faces!)

#### 📝 Notes and Compliance

- **Additional Notes:** (Any other things you want to share? We're here to listen!)
- **Compliance with Hackathon Rules:** (Just a gentle reminder to stick to the rules outlined for the hackathon)

#### ✅ Checklist

```py

这段代码是一个Python脚本，它执行了一系列的操作来进入特定的React Native实验室。它包括以下内容：

1. 读取并遵循Hackathon Rules。
2. 确认项目为开源并采用MIT License。
3. 检查注册电子邮件是否与OpenAI帐户匹配，如果有，将在PR上获得bonus credits（如果适用）。
4. 将仓库切换到主分支。
5. 在主分支上运行git checkout命令，以确保代码已准备好进行测试。
6. 如果遇到任何错误，将输出错误消息并返回。
7. 在完成实验室操作后，将仓库切换回主分支。


```
- [ ] We have read and are aligned with the [Hackathon Rules](https://lablab.ai/event/autogpt-arena-hacks).
- [ ] We confirm that our project will be open-source and adhere to the MIT License.
- [ ] Our lablab.ai registration email matches our OpenAI account to claim the bonus credits (if applicable).
"""
            head = f"{repo.owner.login}:{arena_submission_branch}"
            pr = parent_repo.create_pull(
                title=f"{agent_name} entering the arena",
                body=pr_message,
                head=head,
                base=branch_to_use,
            )
            click.echo(
                click.style(
                    f"🚀 {agent_name} has entered the arena! Please edit your PR description at the following URL: {pr.html_url}",
                    fg="green",
                )
            )
        else:
            click.echo(
                click.style(
                    "❌ This repository does not have a parent repository to sync with.",
                    fg="red",
                )
            )
            return

        # Switch back to the master branch
        subprocess.check_call(["git", "checkout", branch_to_use])

    except Exception as e:
        click.echo(click.style(f"❌ An error occurred: {e}", fg="red"))
        # Switch back to the master branch
        subprocess.check_call(["git", "checkout", branch_to_use])


```py

This is a simple script that performs the following tasks:

1. Updates the agent JSON file with the current branch and the agent name.
2. Checks if the file already exists for the given agent name.
3. Loads the existing data from the JSON file.
4. Updates the commit hash and timestamp for the current branch.
5. Updates the branch\_to\_benchmark key in the JSON file if the --branch option is passed.
6. Writes the updated data back to the JSON file.
7. Supports both agent name and --branch options.

To run this script, you can use the following command:
```php
./run arena agent <agent_name> <--branch>
```py
This will update the agent JSON file with the specified agent name and the current branch, or the --branch option if specified. If the --branch option is used, it will update the branch\_to\_benchmark key in the JSON file.

Note: This script assumes that you have the arena command and the click command installed and running on your system.


```
@arena.command()
@click.argument("agent_name")
@click.argument("hash")
@click.option("--branch", default=None, help="Branch to use instead of current branch")
def update(agent_name, hash, branch):
    import json
    import os
    from datetime import datetime
    import subprocess

    # Check if the agent_name.json file exists in the arena directory
    agent_json_file = f"./arena/{agent_name}.json"
    # Check if they are on the correct branch
    current_branch = (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode("utf-8")
        .strip()
    )
    correct_branch = f"arena_submission_{agent_name}"
    if current_branch != correct_branch:
        click.echo(
            click.style(
                f"❌ You are not on the correct branch. Please switch to the '{correct_branch}' branch.",
                fg="red",
            )
        )
        return

    if not os.path.exists(agent_json_file):
        click.echo(
            click.style(
                f"❌ The file for agent '{agent_name}' does not exist in the arena directory.",
                fg="red",
            )
        )
        click.echo(
            click.style(
                f"⚠️ You need to enter the arena first. Run './run arena enter {agent_name}'",
                fg="yellow",
            )
        )
        return
    else:
        # Load the existing data
        with open(agent_json_file, "r") as json_file:
            data = json.load(json_file)

        # Update the commit hash and timestamp
        data["commit_hash_to_benchmark"] = hash
        data["timestamp"] = datetime.utcnow().isoformat()

        # If --branch was passed, update the branch_to_benchmark in the JSON file
        if branch:
            data["branch_to_benchmark"] = branch

        # Write the updated data back to the JSON file
        with open(agent_json_file, "w") as json_file:
            json.dump(data, json_file, indent=4)

        click.echo(
            click.style(
                f"🚀 The file for agent '{agent_name}' has been updated in the arena directory.",
                fg="green",
            )
        )

```py

这段代码是一个Python脚本，其中包含一个if语句。

if语句是一种条件语句，用于检查一个操作是否为真。在这个例子中，if语句的判断条件是`__name__ == "__main__"`。

if __name__ == "__main__":`是一个特殊的字符串，用于表示Python脚本是否作为主程序运行。如果这个条件为真，那么if语句中的代码将被执行。

cli()是一个函数，用于执行客户端命令行界面(CLI)交互。因此，如果if语句的判断条件为真，那么这段代码将执行cli()函数，导致该函数成为交互式命令行程序。


```
if __name__ == "__main__":
    cli()

```

# Code of Conduct for AutoGPT

## 1. Purpose

The purpose of this Code of Conduct is to provide guidelines for contributors to the AutoGPT projects on GitHub. We aim to create a positive and inclusive environment where all participants can contribute and collaborate effectively. By participating in this project, you agree to abide by this Code of Conduct.

## 2. Scope

This Code of Conduct applies to all contributors, maintainers, and users of the AutoGPT project. It extends to all project spaces, including but not limited to issues, pull requests, code reviews, comments, and other forms of communication within the project.

## 3. Our Standards

We encourage the following behavior:

* Being respectful and considerate to others
* Actively seeking diverse perspectives
* Providing constructive feedback and assistance
* Demonstrating empathy and understanding

We discourage the following behavior:

* Harassment or discrimination of any kind
* Disrespectful, offensive, or inappropriate language or content
* Personal attacks or insults
* Unwarranted criticism or negativity

## 4. Reporting and Enforcement

If you witness or experience any violations of this Code of Conduct, please report them to the project maintainers by email or other appropriate means. The maintainers will investigate and take appropriate action, which may include warnings, temporary or permanent bans, or other measures as necessary.

Maintainers are responsible for ensuring compliance with this Code of Conduct and may take action to address any violations.

## 5. Acknowledgements

This Code of Conduct is adapted from the [Contributor Covenant](https://www.contributor-covenant.org/version/2/0/code_of_conduct.html).

## 6. Contact

If you have any questions or concerns, please contact the project maintainers on Discord:
https://discord.gg/autogpt


# AutoGPT Contribution Guide
If you are reading this, you are probably looking for our **[contribution guide]**,
which is part of our [knowledge base].

[contribution guide]: https://github.com/Significant-Gravitas/Nexus/wiki/Contributing
[knowledge base]: https://github.com/Significant-Gravitas/Nexus/wiki

## In short
1. Avoid duplicate work, issues, PRs etc.
2. Also consider contributing something other than code; see the [contribution guide]
   for options.
3. Create a draft PR before starting work on non-small changes. Also post your proposal
   in the [dev channel].
4. Clearly explain your changes when submitting a PR.
5. Don't submit stuff that's broken.
6. Avoid making unnecessary changes, especially if they're purely based on your personal
   preferences. Doing so is the maintainers' job.

[dev channel]: https://discord.com/channels/1092243196446249134/1095817829405704305

## Why instructions like these are necessary
We would like to say "We value all contributions". After all, we are an open-source
project, so we should welcome any input that people are willing to give, right?

Well, the reality is that some contributions are SUPER-valuable, while others create
more trouble than they are worth and actually _create_ work for the core team. So to
ensure maximum chances of a smooth ride, please stick to the guidelines.

If you wish to involve with the project (beyond just contributing PRs), please read the
wiki [catalyzing](https://github.com/Significant-Gravitas/Nexus/wiki/Catalyzing) page.

In fact, why not just look through the whole wiki (it's only a few pages) and
hop on our Discord. See you there! :-)

❤️ & 🔆
The team @ AutoGPT
https://discord.gg/autogpt


# Quickstart Guide

> For the complete getting started [tutorial series](https://aiedge.medium.com/autogpt-forge-e3de53cc58ec) <- click here

Welcome to the Quickstart Guide! This guide will walk you through the process of setting up and running your own AutoGPT agent. Whether you're a seasoned AI developer or just starting out, this guide will provide you with the necessary steps to jumpstart your journey in the world of AI development with AutoGPT.

## System Requirements

This project supports Linux (Debian based), Mac, and Windows Subsystem for Linux (WSL). If you are using a Windows system, you will need to install WSL. You can find the installation instructions for WSL [here](https://learn.microsoft.com/en-us/windows/wsl/).


## Getting Setup
1. **Fork the Repository**
   To fork the repository, follow these steps:
   - Navigate to the main page of the repository.

   ![Repository](docs/content/imgs/quickstart/001_repo.png)
   - In the top-right corner of the page, click Fork.

   ![Create Fork UI](docs/content/imgs/quickstart/002_fork.png)
   - On the next page, select your GitHub account to create the fork under.
   - Wait for the forking process to complete. You now have a copy of the repository in your GitHub account.

2. **Clone the Repository**
   To clone the repository, you need to have Git installed on your system. If you don't have Git installed, you can download it from [here](https://git-scm.com/downloads). Once you have Git installed, follow these steps:
   - Open your terminal.
   - Navigate to the directory where you want to clone the repository.
   - Run the git clone command for the fork you just created

   ![Clone the Repository](docs/content/imgs/quickstart/003_clone.png)

   - Then open your project in your ide

   ![Open the Project in your IDE](docs/content/imgs/quickstart/004_ide.png)

4. **Setup the Project**
    Next we need to setup the required dependencies. We have a tool for helping you do all the tasks you need to on the repo.
    It can be accessed by running the `run` command by typing `./run` in the terminal.

    The first command you need to use is `./run setup` This will guide you through the process of setting up your system.
    Initially you will get instructions for installing flutter, chrome and setting up your github access token like the following image:

    > Note: for advanced users. The github access token is only needed for the ./run arena enter command so the system can automatically create a PR

    
    ![Setup the Project](docs/content/imgs/quickstart/005_setup.png)

### For Windows Users

If you're a Windows user and experience issues after installing WSL, follow the steps below to resolve them. 

#### Update WSL 
Run the following command in Powershell or Command Prompt to:
1. Enable the optional WSL and Virtual Machine Platform components.
2. Download and install the latest Linux kernel.
3. Set WSL 2 as the default.
4. Download and install the Ubuntu Linux distribution (a reboot may be required).

```pyshell
wsl --install
```

For more detailed information and additional steps, refer to [Microsoft's WSL Setup Environment Documentation](https://learn.microsoft.com/en-us/windows/wsl/setup/environment).

#### Resolve FileNotFoundError or "No such file or directory" Errors
When you run `./run setup`, if you encounter errors like `No such file or directory` or `FileNotFoundError`, it might be because Windows-style line endings (CRLF - Carriage Return Line Feed) are not compatible with Unix/Linux style line endings (LF - Line Feed).

To resolve this, you can use the `dos2unix` utility to convert the line endings in your script from CRLF to LF. Here’s how to install and run `dos2unix` on the script:

```pyshell
sudo apt update
sudo apt install dos2unix
dos2unix ./run
```

After executing the above commands, running `./run setup` should work successfully. 

#### Store Project Files within the WSL File System
If you continue to experience issues, consider storing your project files within the WSL file system instead of the Windows file system. This method avoids issues related to path translations and permissions and provides a more consistent development environment.
    
    You can keep running the command to get feedback on where you are up to with your setup. 
    When setup has been completed, the command will return an output like this:

   ![Setup Complete](docs/content/imgs/quickstart/006_setup_complete.png)

## Creating Your Agent

    Now setup has been completed its time to create your agent template. 
    Do so by running the `./run agent create YOUR_AGENT_NAME` replacing YOUR_AGENT_NAME with a name of your choice. Examples of valid names: swiftyosgpt or SwiftyosAgent or swiftyos_agent

   ![Create an Agent](docs/content/imgs/quickstart/007_create_agent.png)

    Upon creating your agent its time to offically enter the Arena!
    Do so by running `./run arena enter YOUR_AGENT_NAME`

   ![Enter the Arena](docs/content/imgs/quickstart/008_enter_arena.png)

   > Note: for adavanced yours, create a new branch and create a file called YOUR_AGENT_NAME.json in the arena directory. Then commit this and create a PR to merge into the main repo. Only single file entries will be permitted. The json file needs the following format. 
   ```pyjson
   {
    "github_repo_url": "https://github.com/Swiftyos/YourAgentName",
    "timestamp": "2023-09-18T10:03:38.051498",
    "commit_hash_to_benchmark": "ac36f7bfc7f23ad8800339fa55943c1405d80d5e",
    "branch_to_benchmark": "master"
   }
   ```
   - github_repo_url: the url to your fork
   - timestamp: timestamp of the last update of this file
   - commit_hash_to_benchmark: the commit hash of your entry. You update each time you have an something ready to be officially entered into the hackathon
   - branch_to_benchmark: the branch you are using to develop your agent on, default is master.


## Running your Agent

Your agent can started using the `./run agent start YOUR_AGENT_NAME`

This start the agent on `http://localhost:8000/`

![Start the Agent](docs/content/imgs/quickstart/009_start_agent.png)

The frontend can be accessed from `http://localhost:8000/`, you will first need to login using either a google account or your github account.

![Login](docs/content/imgs/quickstart/010_login.png)

Upon logging in you will get a page that looks something like this. With your task history down the left hand side of the page and the 'chat' window to send tasks to your agent.

![Login](docs/content/imgs/quickstart/011_home.png)

When you have finished with your agent, or if you just need to restart it, use Ctl-C to end the session then you can re-run the start command.

If you are having issues and want to ensure the agent has been stopped there is a `./run agent stop` command which will kill the process using port 8000, which should be the agent. 

## Benchmarking your Agent

The benchmarking system can also be accessed using the cli too:

```pybash
agpt % ./run benchmark
Usage: cli.py benchmark [OPTIONS] COMMAND [ARGS]...

  Commands to start the benchmark and list tests and categories

Options:
  --help  Show this message and exit.

Commands:
  categories  Benchmark categories group command
  start       Starts the benchmark command
  tests       Benchmark tests group command
agpt % ./run benchmark categories     
Usage: cli.py benchmark categories [OPTIONS] COMMAND [ARGS]...

  Benchmark categories group command

Options:
  --help  Show this message and exit.

Commands:
  list  List benchmark categories command
agpt % ./run benchmark tests      
Usage: cli.py benchmark tests [OPTIONS] COMMAND [ARGS]...

  Benchmark tests group command

Options:
  --help  Show this message and exit.

Commands:
  details  Benchmark test details command
  list     List benchmark tests command
```

The benchmark has been split into different categories of skills you can test your agent on. You can see what categories are available with
```pybash
./run benchmark categories list
# And what tests are available with
./run benchmark tests list
```

![Login](docs/content/imgs/quickstart/012_tests.png)


Finally you can run the benchmark with

```pybash
./run benchmark start YOUR_AGENT_NAME

```

>
