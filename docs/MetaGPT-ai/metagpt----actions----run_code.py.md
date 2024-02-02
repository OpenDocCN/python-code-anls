# `MetaGPT\metagpt\actions\run_code.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:46
@Author  : alexanderwu
@File    : run_code.py
@Modified By: mashenquan, 2023/11/27.
            1. Mark the location of Console logs in the PROMPT_TEMPLATE with markdown code-block formatting to enhance
            the understanding for the LLM.
            2. Fix bug: Add the "install dependency" operation.
            3. Encapsulate the input of RunCode into RunCodeContext and encapsulate the output of RunCode into
            RunCodeResult to standardize and unify parameter passing between WriteCode, RunCode, and DebugError.
            4. According to section 2.2.3.5.7 of RFC 135, change the method of transferring file content
            (code files, unit test files, log files) from using the message to using the file name.
            5. Merged the `Config` class of send18:dev branch to take over the set/get operations of the Environment
            class.
"""
import subprocess
from typing import Tuple

from pydantic import Field

from metagpt.actions.action import Action
from metagpt.config import CONFIG
from metagpt.logs import logger
from metagpt.schema import RunCodeContext, RunCodeResult
from metagpt.utils.exceptions import handle_exception

# 定义一个模板，用于生成运行代码的提示信息
PROMPT_TEMPLATE = """
Role: You are a senior development and qa engineer, your role is summarize the code running result.
If the running result does not include an error, you should explicitly approve the result.
On the other hand, if the running result indicates some error, you should point out which part, the development code or the test code, produces the error,
and give specific instructions on fixing the errors. Here is the code info:
{context}
Now you should begin your analysis
---
## instruction:
Please summarize the cause of the errors and give correction instruction
## File To Rewrite:
Determine the ONE file to rewrite in order to fix the error, for example, xyz.py, or test_xyz.py
## Status:
Determine if all of the code works fine, if so write PASS, else FAIL,
WRITE ONLY ONE WORD, PASS OR FAIL, IN THIS SECTION
## Send To:
Please write Engineer if the errors are due to problematic development codes, and QaEngineer to problematic test codes, and NoOne if there are no errors,
WRITE ONLY ONE WORD, Engineer OR QaEngineer OR NoOne, IN THIS SECTION.
---
You should fill in necessary instruction, status, send to, and finally return all content between the --- segment line.
"""

# 定义一个模板，用于生成运行代码的上下文信息
CONTEXT = """
## Development Code File Name
{code_file_name}
## Development Code

{code}

## Test File Name
{test_file_name}
## Test Code

{test_code}

## Running Command
{command}
## Running Output
standard output: 

{outs}

standard errors: 

{errs}

"""

# 定义一个运行代码的类
class RunCode(Action):
    name: str = "RunCode"
    context: RunCodeContext = Field(default_factory=RunCodeContext)

    # 运行文本代码的方法
    @classmethod
    async def run_text(cls, code) -> Tuple[str, str]:
        try:
            # 创建一个命名空间，执行代码
            namespace = {}
            exec(code, namespace)
        except Exception as e:
            return "", str(e)
        return namespace.get("result", ""), ""

    # 运行脚本代码的方法
    @classmethod
    async def run_script(cls, working_directory, additional_python_paths=[], command=[]) -> Tuple[str, str]:
        working_directory = str(working_directory)
        additional_python_paths = [str(path) for path in additional_python_paths]

        # 复制当前的环境变量
        env = CONFIG.new_environ()

        # 修改PYTHONPATH环境变量
        additional_python_paths = [working_directory] + additional_python_paths
        additional_python_paths = ":".join(additional_python_paths)
        env["PYTHONPATH"] = additional_python_paths + ":" + env.get("PYTHONPATH", "")
        RunCode._install_dependencies(working_directory=working_directory, env=env)

        # 启动子进程
        process = subprocess.Popen(
            command, cwd=working_directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )
        logger.info(" ".join(command))

        try:
            # 等待进程完成，设置超时时间
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            logger.info("The command did not complete within the given timeout.")
            process.kill()  # 如果超时则终止进程
            stdout, stderr = process.communicate()
        return stdout.decode("utf-8"), stderr.decode("utf-8")

    # 运行方法
    async def run(self, *args, **kwargs) -> RunCodeResult:
        logger.info(f"Running {' '.join(self.context.command)}")
        if self.context.mode == "script":
            outs, errs = await self.run_script(
                command=self.context.command,
                working_directory=self.context.working_directory,
                additional_python_paths=self.context.additional_python_paths,
            )
        elif self.context.mode == "text":
            outs, errs = await self.run_text(code=self.context.code)

        logger.info(f"{outs=}")
        logger.info(f"{errs=}")

        context = CONTEXT.format(
            code=self.context.code,
            code_file_name=self.context.code_filename,
            test_code=self.context.test_code,
            test_file_name=self.context.test_filename,
            command=" ".join(self.context.command),
            outs=outs[:500],  # 截断输出以避免标记溢出
            errs=errs[:10000],  # 截断错误以避免标记溢出
        )

        prompt = PROMPT_TEMPLATE.format(context=context)
        rsp = await self._aask(prompt)
        return RunCodeResult(summary=rsp, stdout=outs, stderr=errs)

    # 安装依赖的方法
    @staticmethod
    @handle_exception(exception_type=subprocess.CalledProcessError)
    def _install_via_subprocess(cmd, check, cwd, env):
        return subprocess.run(cmd, check=check, cwd=cwd, env=env)

    # 安装依赖的方法
    @staticmethod
    def _install_dependencies(working_directory, env):
        install_command = ["python", "-m", "pip", "install", "-r", "requirements.txt"]
        logger.info(" ".join(install_command))
        RunCode._install_via_subprocess(install_command, check=True, cwd=working_directory, env=env)

        install_pytest_command = ["python", "-m", "pip", "install", "pytest"]
        logger.info(" ".join(install_pytest_command))
        RunCode._install_via_subprocess(install_pytest_command, check=True, cwd=working_directory, env=env)

```