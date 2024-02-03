# `.\AutoGPT\cli.py`

```
"""
This is a minimal file intended to be run by users to help them manage the autogpt projects.

If you want to contribute, please use only libraries that come as part of Python.
To ensure efficiency, add the imports to the functions so only what is needed is imported.
"""
# 尝试导入 click 和 github 库，如果导入失败则导入 os 库
try:
    import click
    import github
except ImportError:
    import os
    # 使用系统命令安装 click 和 PyGithub 库
    os.system("pip3 install click")
    os.system("pip3 install PyGithub")
    import click

# 创建命令组
@click.group()
def cli():
    pass

# 创建命令函数
@cli.command()
def setup():
    """Installs dependencies needed for your system. Works with Linux, MacOS and Windows WSL."""
    import os
    import subprocess

    # 输出彩色文本
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
d88P     888  "Y88888  "Y888 "Y88P"   "Y8888P88 888           888     
""",
            fg="green",
        )
    )

    # 获取当前脚本所在目录路径
    script_dir = os.path.dirname(os.path.realpath(__file__))
    setup_script = os.path.join(script_dir, "setup.sh")
    install_error = False
    # 如果 setup.sh 脚本存在
    if os.path.exists(setup_script):
        click.echo(click.style("🚀 Setup initiated...\n", fg="green"))
        try:
            # 在当前目录下执行 setup.sh 脚本
            subprocess.check_call([setup_script], cwd=script_dir)
        except subprocess.CalledProcessError:
            click.echo(
                click.style("❌ There was an issue with the installation.", fg="red")
            )
            install_error = True
    else:
        # 如果 setup.sh 文件在当前目录中不存在，则输出错误信息
        click.echo(
            click.style(
                "❌ Error: setup.sh does not exist in the current directory.", fg="red"
            )
        )
        # 设置安装错误标志为 True
        install_error = True

    try:
        # 检查是否配置了 git 用户信息
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
            # 如果配置了 git 用户信息，则输出成功信息
            click.echo(
                click.style(
                    f"✅ Git is configured with name '{user_name}' and email '{user_email}'",
                    fg="green",
                )
            )
        else:
            # 如果未配置 git 用户信息，则引发 CalledProcessError
            raise subprocess.CalledProcessError(
                returncode=1, cmd="git config user.name or user.email"
            )

    except subprocess.CalledProcessError:
        # 如果未配置 GitHub 账户，则输出设置指令
        click.echo(click.style("⚠️ Git user is not configured.", fg="red"))
        click.echo(
            click.style(
                "To configure Git with your user info, use the following commands:",
                fg="red",
            )
        )
        click.echo(
            click.style(
                '  git config --global user.name "Your (user)name"', fg="red"
            )
        )
        click.echo(
            click.style(
                '  git config --global user.email "Your email"', fg="red"
            )
        )
        # 设置安装错误标志为 True
        install_error = True

    print_access_token_instructions = False

    # 检查是否存在 .github_access_token 文件
    else:
        # 如果不存在 .github_access_token 文件，则创建该文件
        with open(".github_access_token", "w") as file:
            file.write("")
        # 设置安装错误标志为 True，设置打印访问令牌指令标志为 True
        install_error = True
        print_access_token_instructions = True
    # 如果需要打印访问令牌的设置说明
    if print_access_token_instructions:
        # 打印设置 GitHub 访问令牌的说明
        click.echo(
            click.style(
                "💡 To configure your GitHub access token, follow these steps:", fg="red"
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

    # 如果存在安装错误
    if install_error:
        # 打印错误信息，并提供 GitHub 上的问题反馈链接
        click.echo(
            click.style(
                "\n\n🔴 If you need help, please raise a ticket on GitHub at https://github.com/Significant-Gravitas/AutoGPT/issues\n\n",
                fg="magenta",
                bold=True,
            )
        )
# 创建一个名为 agent 的命令组
@cli.group()
def agent():
    """Commands to create, start and stop agents"""
    pass


# 创建一个名为 create 的命令，用于创建新的 agent
@agent.command()
@click.argument("agent_name")
def create(agent_name):
    """Create's a new agent with the agent name provided"""
    import os
    import re
    import shutil

    # 检查 agent 名称是否合法，不包含空格或特殊字符
    if not re.match(r"\w*$", agent_name):
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

        # 如果 agent 目录不存在且 agent 名称不在现有文件中，则复制模板文件创建新的 agent
        if not os.path.exists(new_agent_dir) and not new_agent_name in existing_arena_files:
            shutil.copytree("./autogpts/forge", new_agent_dir)
            click.echo(
                click.style(
                    f"🎉 New agent '{agent_name}' created. The code for your new agent is in: autogpts/{agent_name}",
                    fg="green",
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


# 创建一个名为 start 的命令，用于启动 agent
@agent.command()
@click.argument("agent_name")
@click.option(
    "--no-setup",
    is_flag=True,
    help="Disables running the setup script before starting the agent",
)
def start(agent_name, no_setup):
    """Start agent command"""
    import os
    import subprocess

    # 获取当前脚本的目录路径和 agent 目录路径
    script_dir = os.path.dirname(os.path.realpath(__file__))
    agent_dir = os.path.join(script_dir, f"autogpts/{agent_name}")
    # 设置运行命令的路径为 agent 目录下的 run 文件
    run_command = os.path.join(agent_dir, "run")
    # 拼接运行基准测试命令的路径
    run_bench_command = os.path.join(agent_dir, "run_benchmark")
    # 检查代理目录是否存在，并且运行命令文件和运行基准测试命令文件是否存在
    if os.path.exists(agent_dir) and os.path.isfile(run_command) and os.path.isfile(run_bench_command):
        # 切换当前工作目录到代理目录
        os.chdir(agent_dir)
        # 如果不是无需设置，则执行 setup 脚本
        if not no_setup:
            setup_process = subprocess.Popen(["./setup"], cwd=agent_dir)
            setup_process.wait()
        # 在代理目录下执行运行基准测试命令
        subprocess.Popen(["./run_benchmark", "serve"], cwd=agent_dir)
        # 输出提示信息，表示基准测试服务器正在启动
        click.echo(f"Benchmark Server starting please wait...")
        # 在代理目录下执行运行命令
        subprocess.Popen(["./run"], cwd=agent_dir)
        # 输出提示信息，表示代理正在启动
        click.echo(f"Agent '{agent_name}' starting please wait...")
    # 如果代理目录不存在
    elif not os.path.exists(agent_dir):
        # 输出错误信息，表示代理不存在，需要先创建代理
        click.echo(
            click.style(
                f"😞 Agent '{agent_name}' does not exist. Please create the agent first.",
                fg="red",
            )
        )
    # 如果运行命令不存在于代理目录中
    else:
        # 输出错误信息，表示代理目录中不存在运行命令
        click.echo(
            click.style(
                f"😞 Run command does not exist in the agent '{agent_name}' directory.",
                fg="red",
            )
        )
# 定义一个停止代理的命令函数
@agent.command()
def stop():
    """Stop agent command"""
    # 导入必要的模块
    import os
    import signal
    import subprocess

    # 尝试获取运行在8000端口上的进程的PID
    try:
        pids = subprocess.check_output(["lsof", "-t", "-i", ":8000"]).split()
        # 如果PID是整数类型
        if isinstance(pids, int):
            # 终止该进程
            os.kill(int(pids), signal.SIGTERM)
        else:
            # 遍历所有PID并终止相应进程
            for pid in pids:
                os.kill(int(pid), signal.SIGTERM)
    except subprocess.CalledProcessError:
        click.echo("No process is running on port 8000")

    # 尝试获取运行在8080端口上的进程的PID
    try:
        pids = int(subprocess.check_output(["lsof", "-t", "-i", ":8080"]))
        # 如果PID是整数类型
        if isinstance(pids, int):
            # 终止该进程
            os.kill(int(pids), signal.SIGTERM)
        else:
            # 遍历所有PID并终止相应进程
            for pid in pids:
                os.kill(int(pid), signal.SIGTERM)
    except subprocess.CalledProcessError:
        click.echo("No process is running on port 8080")

# 定义一个列出代理的命令函数
@agent.command()
def list():
    """List agents command"""
    # 导入必要的模块
    import os

    try:
        # 指定代理目录
        agents_dir = "./autogpts"
        # 获取目录下所有文件夹的列表
        agents_list = [
            d
            for d in os.listdir(agents_dir)
            if os.path.isdir(os.path.join(agents_dir, d))
        ]
        # 如果代理列表不为空
        if agents_list:
            # 输出可用代理信息
            click.echo(click.style("Available agents: 🤖", fg="green"))
            for agent in agents_list:
                click.echo(click.style(f"\t🐙 {agent}", fg="blue"))
        else:
            # 输出未找到代理信息
            click.echo(click.style("No agents found 😞", fg="red"))
    except FileNotFoundError:
        # 输出代理目录不存在的信息
        click.echo(click.style("The autogpts directory does not exist 😢", fg="red"))
    except Exception as e:
        # 输出其他错误信息
        click.echo(click.style(f"An error occurred: {e} 😢", fg="red"))


# 定义一个基准测试命令组
@cli.group()
def benchmark():
    """Commands to start the benchmark and list tests and categories"""
    pass


# 定义一个启动基准测试命令函数
@benchmark.command(
    context_settings=dict(
        ignore_unknown_options=True,
    )
)
@click.argument("agent_name")
@click.argument("subprocess_args", nargs=-1, type=click.UNPROCESSED)
def start(agent_name, subprocess_args):
    """Starts the benchmark command"""
    # 导入必要的模块
    import os
    # 导入 subprocess 模块，用于执行外部命令
    import subprocess
    
    # 获取当前脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # 根据 agent_name 拼接 agent 目录的路径
    agent_dir = os.path.join(script_dir, f"autogpts/{agent_name}")
    # 拼接 benchmark 脚本的路径
    benchmark_script = os.path.join(agent_dir, "run_benchmark")
    
    # 如果 agent_dir 存在且 benchmark_script 是文件
    if os.path.exists(agent_dir) and os.path.isfile(benchmark_script):
        # 切换当前工作目录到 agent_dir
        os.chdir(agent_dir)
        # 使用 subprocess 模块执行 benchmark_script 脚本，传入 subprocess_args 参数
        subprocess.Popen([benchmark_script, *subprocess_args], cwd=agent_dir)
        # 输出运行 benchmark 的信息
        click.echo(
            click.style(
                f"🚀 Running benchmark for '{agent_name}' with subprocess arguments: {' '.join(subprocess_args)}",
                fg="green",
            )
        )
    else:
        # 输出 agent 不存在的信息
        click.echo(
            click.style(
                f"😞 Agent '{agent_name}' does not exist. Please create the agent first.",
                fg="red",
            )
        )
# 创建一个名为 benchmark_categories 的命令组
@benchmark.group(name="categories")
def benchmark_categories():
    """Benchmark categories group command"""
    pass

# 在 benchmark_categories 命令组中创建一个名为 list 的命令
@benchmark_categories.command(name="list")
def benchmark_categories_list():
    """List benchmark categories command"""
    import glob
    import json
    import os

    # 创建一个空集合用于存储类别
    categories = set()

    # 获取当前文件所在目录的绝对路径
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建 glob 模式，用于匹配数据文件路径，排除 'deprecated' 目录
    glob_path = os.path.join(
        this_dir, "./benchmark/agbenchmark/challenges/**/[!deprecated]*/data.json"
    )

    # 遍历匹配到的数据文件
    for data_file in glob.glob(glob_path, recursive=True):
        # 如果文件路径中不包含 'deprecated'，则打开文件进行处理
        if 'deprecated' not in data_file:
            with open(data_file, "r") as f:
                try:
                    # 尝试解析 JSON 文件内容
                    data = json.load(f)
                    # 将类别添加到集合中
                    categories.update(data.get("category", []))
                except json.JSONDecodeError:
                    # 捕获 JSON 解析错误
                    print(f"Error: {data_file} is not a valid JSON file.")
                    continue
                except IOError:
                    # 捕获文件读取错误
                    print(f"IOError: file could not be read: {data_file}")
                    continue

    # 如果存在类别，则输出可用类别信息
    if categories:
        click.echo(click.style("Available categories: 📚", fg="green"))
        for category in categories:
            click.echo(click.style(f"\t📖 {category}", fg="blue"))
    else:
        # 如果没有找到类别，则输出提示信息
        click.echo(click.style("No categories found 😞", fg="red"))

# 创建一个名为 benchmark_tests 的命令组
@benchmark.group(name="tests")
def benchmark_tests():
    """Benchmark tests group command"""
    pass

# 在 benchmark_tests 命令组中创建一个名为 list 的命令
@benchmark_tests.command(name="list")
def benchmark_tests_list():
    """List benchmark tests command"""
    import glob
    import json
    import os
    import re

    # 创建一个空字典用于存储测试信息
    tests = {}

    # 获取当前文件所在目录的绝对路径
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建 glob 模式，用于匹配数据文件路径，排除 'deprecated' 目录
    glob_path = os.path.join(
        this_dir, "./benchmark/agbenchmark/challenges/**/[!deprecated]*/data.json"
    )
    # 使用 glob 模块的 glob 函数匹配指定路径下的文件，设置 recursive=True 可以递归查找子目录
    for data_file in glob.glob(glob_path, recursive=True):
        # 排除包含 'deprecated' 的文件路径
        if 'deprecated' not in data_file:
            # 打开文件，读取文件内容
            with open(data_file, "r") as f:
                try:
                    # 尝试解析 JSON 文件内容
                    data = json.load(f)
                    # 获取数据中的 category 和 name 字段
                    category = data.get("category", [])
                    test_name = data.get("name", "")
                    # 如果 category 和 test_name 都存在
                    if category and test_name:
                        # 如果 category[0] 不在 tests 字典中，则添加
                        if category[0] not in tests:
                            tests[category[0]] = []
                        # 将 test_name 添加到对应 category 的列表中
                        tests[category[0]].append(test_name)
                except json.JSONDecodeError:
                    # 捕获 JSON 解析错误
                    print(f"Error: {data_file} is not a valid JSON file.")
                    continue
                except IOError:
                    # 捕获文件读取错误
                    print(f"IOError: file could not be read: {data_file}")
                    continue

    # 如果 tests 字典不为空
    if tests:
        # 输出可用测试的提示信息
        click.echo(click.style("Available tests: 📚", fg="green"))
        # 遍历 tests 字典，输出每个 category 下的测试列表
        for category, test_list in tests.items():
            click.echo(click.style(f"\t📖 {category}", fg="blue"))
            # 对测试列表进行排序，并输出每个测试的信息
            for test in sorted(test_list):
                # 对测试名称进行格式化处理，去除下划线和空格
                test_name = (
                    " ".join(word for word in re.split("([A-Z][a-z]*)", test) if word)
                    .replace("_", "")
                    .replace("C L I", "CLI")
                    .replace("  ", " ")
                )
                # 对测试名称进行左对齐填充
                test_name_padded = f"{test_name:<40}"
                # 输出测试信息
                click.echo(click.style(f"\t\t🔬 {test_name_padded} - {test}", fg="cyan"))
    else:
        # 如果 tests 字典为空，输出未找到测试的提示信息
        click.echo(click.style("No tests found 😞", fg="red"))
# 定义一个名为 benchmark_tests_details 的命令，接受一个名为 test_name 的参数
@benchmark_tests.command(name="details")
def benchmark_tests_details(test_name):
    """Benchmark test details command"""
    # 导入必要的模块
    import glob
    import json
    import os

    # 获取当前文件所在目录的绝对路径
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建 glob 模式的路径，用于匹配指定目录下的文件
    glob_path = os.path.join(
        this_dir, "./benchmark/agbenchmark/challenges/**/[!deprecated]*/data.json"
    )
    # 将该路径作为基础路径，排除 'deprecated' 目录

# 定义一个名为 arena 的命令组
@cli.group()
def arena():
    """Commands to enter the arena"""
    pass

# 定义一个名为 enter 的命令，接受一个名为 agent_name 的参数和一个名为 branch 的选项参数
@arena.command()
def enter(agent_name, branch):
    # 导入必要的模块
    import json
    import os
    import subprocess
    from datetime import datetime

    from github import Github

    # 检查 autogpts 目录中是否存在 agent_name 目录
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
        # 检查代理是否已经进入竞技场
        try:
            # 检查是否存在名为'arena_submission_{agent_name}'的提交记录
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
            # 如果没有提交记录，则继续执行
            pass
        else:
            # 如果存在提交记录，则提示代理已经进入竞技场，提供更新提交的步骤
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

    # 检查是否有暂存的更改
    # 通过 subprocess 模块执行 git status 命令，获取工作区和暂存区的状态信息
    # 通过 decode 方法将字节流解码成字符串，使用 utf-8 编码
    # 通过 split 方法将字符串按行分割，形成列表
    # 只保留状态为 A、M、D、R、C 的文件
    staged_changes = [
        line
        for line in subprocess.check_output(["git", "status", "--porcelain"])
        .decode("utf-8")
        .split("\n")
        if line and line[0] in ("A", "M", "D", "R", "C")
    ]
    # 如果有暂存的更改
    if staged_changes:
        # 输出提示信息，提醒用户提交或存储暂存的更改后再运行命令
        click.echo(
            click.style(
                f"❌ There are staged changes. Please commit or stash them and run the command again.",
                fg="red",
            )
        )
        # 返回，不继续执行后续代码
        return
### 🌟 Welcome to the AutoGPT Arena Hacks Hackathon! 🌟

Hey there amazing builders! We're thrilled to have you join this exciting journey. Before you dive deep into building, we'd love to know more about you and the awesome project you are envisioning. Fill out the template below to kickstart your hackathon journey. May the best agent win! 🏆

#### 🤖 Team Introduction

- **Agent Name:** {agent_name}
- **Team Members:** (Who are the amazing minds behind this team? Do list everyone along with their roles!)
- **Repository Link:** [{github_repo_url.replace('https://github.com/', '')}]({github_repo_url})

#### 🌟 Project Vision

- **Starting Point:** (Are you building from scratch or starting with an existing agent? Do tell!)
- **Preliminary Ideas:** (Share your initial ideas and what kind of project you are aiming to build. We are all ears!)
  
#### 🏆 Prize Category

- **Target Prize:** (Which prize caught your eye? Which one are you aiming for?)
- **Why this Prize:** (We'd love to know why this prize feels like the right fit for your team!)

#### 🎬 Introduction Video

- **Video Link:** (If you'd like, share a short video where you introduce your team and talk about your project. We'd love to see your enthusiastic faces!)

#### 📝 Notes and Compliance

- **Additional Notes:** (Any other things you want to share? We're here to listen!)
- **Compliance with Hackathon Rules:** (Just a gentle reminder to stick to the rules outlined for the hackathon)

#### ✅ Checklist

- [ ] We have read and are aligned with the [Hackathon Rules](https://lablab.ai/event/autogpt-arena-hacks).
- [ ] We confirm that our project will be open-source and adhere to the MIT License.
- [ ] Our lablab.ai registration email matches our OpenAI account to claim the bonus credits (if applicable).
# 创建一个由 owner.login 和 arena_submission_branch 组成的字符串作为 pull request 的 head
head = f"{repo.owner.login}:{arena_submission_branch}"
# 在 parent_repo 上创建一个 pull request，标题为 agent_name entering the arena，内容为 pr_message，head 为上面创建的 head，base 为 branch_to_use
pr = parent_repo.create_pull(
    title=f"{agent_name} entering the arena",
    body=pr_message,
    head=head,
    base=branch_to_use,
)
# 输出提示信息，显示 agent_name 已经进入竞技场，提供 PR 描述的 URL
click.echo(
    click.style(
        f"🚀 {agent_name} has entered the arena! Please edit your PR description at the following URL: {pr.html_url}",
        fg="green",
    )
)
# 如果没有 parent repository，输出错误信息并返回
else:
    click.echo(
        click.style(
            "❌ This repository does not have a parent repository to sync with.",
            fg="red",
        )
    )
    return

# 切换回 master 分支
subprocess.check_call(["git", "checkout", branch_to_use])

except Exception as e:
    # 输出错误信息
    click.echo(click.style(f"❌ An error occurred: {e}", fg="red"))
    # 切换回 master 分支
    subprocess.check_call(["git", "checkout", branch_to_use])


@arena.command()
@click.argument("agent_name")
@click.argument("hash")
@click.option("--branch", default=None, help="Branch to use instead of current branch")
def update(agent_name, hash, branch):
    import json
    import os
    from datetime import datetime
    import subprocess

    # 检查 arena 目录中是否存在 agent_name.json 文件
    agent_json_file = f"./arena/{agent_name}.json"
    # 检查当前所在分支
    current_branch = (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode("utf-8")
        .strip()
    )
    # 设置正确的分支名为 arena_submission_agent_name
    correct_branch = f"arena_submission_{agent_name}"
    # 如果当前分支不是正确的分支，输出错误信息并返回
    if current_branch != correct_branch:
        click.echo(
            click.style(
                f"❌ You are not on the correct branch. Please switch to the '{correct_branch}' branch.",
                fg="red",
            )
        )
        return
    # 检查代理 JSON 文件是否存在
    if not os.path.exists(agent_json_file):
        # 如果文件不存在，输出错误信息
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
        # 返回空值
        return
    else:
        # 如果文件存在，加载现有数据
        with open(agent_json_file, "r") as json_file:
            data = json.load(json_file)

        # 更新提交哈希和时间戳
        data["commit_hash_to_benchmark"] = hash
        data["timestamp"] = datetime.utcnow().isoformat()

        # 如果传递了 --branch 参数，更新 JSON 文件中的 branch_to_benchmark
        if branch:
            data["branch_to_benchmark"] = branch

        # 将更新后的数据写回 JSON 文件
        with open(agent_json_file, "w") as json_file:
            json.dump(data, json_file, indent=4)

        # 输出成功更新信息
        click.echo(
            click.style(
                f"🚀 The file for agent '{agent_name}' has been updated in the arena directory.",
                fg="green",
            )
        )
# 如果当前脚本被直接执行，则调用 cli() 函数
if __name__ == "__main__":
    cli()
```