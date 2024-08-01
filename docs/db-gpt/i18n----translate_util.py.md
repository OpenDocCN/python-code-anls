# `.\DB-GPT-src\i18n\translate_util.py`

```py
# 导入所需的模块和库
from typing import List, Dict, Any
import asyncio
import os
import argparse
from dbgpt.core import (
    SystemPromptTemplate,
    HumanPromptTemplate,
    ChatPromptTemplate,
    ModelOutput,
    LLMClient,
)
from dbgpt.core.operators import PromptBuilderOperator, RequestBuilderOperator
from dbgpt.core.awel import (
    DAG,
    MapOperator,
    InputOperator,
    InputSource,
    JoinOperator,
    IteratorTrigger,
)
from dbgpt.model.operators import LLMOperator
from dbgpt.model.proxy import OpenAILLMClient
from dbgpt.model.proxy.base import TiktokenProxyTokenizer

# 定义用于翻译的模板文本
PROMPT_ZH = """
你是一位精通{language}的专业翻译，尤其擅长将 Python 国际化（i18n）工具 gettext 的 po(Portable Object) 
内容中的空缺的{language}的部分补充完整。

...（部分内容省略）

策略：保持原有格式，不要遗漏任何信息，遵守原意的前提下让内容更通俗易懂、符合{language}表达习惯，但要保留原有格式不变。

返回格式如下：
{response}

样例1：
{example_1_input}

输出：
{example_1_output}

样例2:
{example_2_input}

输出：
{example_2_output}


请一步步思考，翻译以下内容为{language}：
"""

# TODO: translate examples to target language

# 定义翻译结果的模板
response = """
{意译结果}
"""

# 定义示例1的输入
example_1_input = """
#: ../dbgpt/storage/vector_store/chroma_store.py:21
msgid "Chroma Vector Store"
msgstr ""
"""

# 定义示例1的期望输出
example_1_output_1 = """
#: ../dbgpt/storage/vector_store/chroma_store.py:21
msgid "Chroma Vector Store"
msgstr "Chroma 向量化存储"
"""

# 定义示例2的输入
example_2_input = """
#: ../dbgpt/model/operators/llm_operator.py:66
msgid "LLM Operator"
msgstr ""

#: ../dbgpt/model/operators/llm_operator.py:69
msgid "The LLM operator."
msgstr ""

#: ../dbgpt/model/operators/llm_operator.py:72
#: ../dbgpt/model/operators/llm_operator.py:120
msgid "LLM Client"
msgstr ""
"""

# 定义示例2的期望输出
example_2_output = """
#: ../dbgpt/model/operators/llm_operator.py:66
msgid "LLM Operator"
msgstr "LLM 算子"

#: ../dbgpt/model/operators/llm_operator.py:69
msgid "The LLM operator."
msgstr "LLM 算子。"

#: ../dbgpt/model/operators/llm_operator.py:72
#: ../dbgpt/model/operators/llm_operator.py:120
msgid "LLM Client"
msgstr "LLM 客户端"
"""
vocabulary_map = {
    "zh_CN": {
        "Transformer": "Transformer",  # 中文简称到英文原文的映射
        "Token": "Token",  # 中文简称到英文原文的映射
        "LLM/Large Language Model": "大语言模型",  # 中文简称到中文全称的映射
        "Generative AI": "生成式 AI",  # 中文简称到中文全称的映射
        "Operator": "算子",  # 中文简称到中文原文的映射
        "DAG": "工作流",  # 中文简称到中文原文的映射
        "AWEL": "AWEL",  # 中文简称到中文原文的映射
        "RAG": "RAG",  # 中文简称到中文原文的映射
        "DB-GPT": "DB-GPT",  # 中文简称到中文原文的映射
        "AWEL flow": "AWEL 工作流",  # 中文简称到中文原文的映射
    },
    "default": {
        "Transformer": "Transformer",  # 默认语言环境下的映射，与 "zh_CN" 相同
        "Token": "Token",  # 默认语言环境下的映射，与 "zh_CN" 相同
        "LLM/Large Language Model": "Large Language Model",  # 默认语言环境下的映射，与 "zh_CN" 相同
        "Generative AI": "Generative AI",  # 默认语言环境下的映射，与 "zh_CN" 相同
        "Operator": "Operator",  # 默认语言环境下的映射，与 "zh_CN" 相同
        "DAG": "DAG",  # 默认语言环境下的映射，与 "zh_CN" 相同
        "AWEL": "AWEL",  # 默认语言环境下的映射，与 "zh_CN" 相同
        "RAG": "RAG",  # 默认语言环境下的映射，与 "zh_CN" 相同
        "DB-GPT": "DB-GPT",  # 默认语言环境下的映射，与 "zh_CN" 相同
        "AWEL flow": "AWEL flow",  # 默认语言环境下的映射，与 "zh_CN" 相同
    },
}


class ReadPoFileOperator(MapOperator[str, List[str]]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def map(self, file_path: str) -> List[str]:
        return await self.blocking_func_to_async(self.read_file, file_path)

    def read_file(self, file_path: str) -> List[str]:
        with open(file_path, "r") as f:
            return f.readlines()  # 读取文件的所有行并返回为列表


class ParsePoFileOperator(MapOperator[List[str], List[str]]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def map(self, content_lines: List[str]) -> List[str]:
        block_lines = extract_messages_with_comments(content_lines)
        return block_lines


def extract_messages_with_comments(lines: List[str]):
    messages = []  # 存储提取出的消息内容的列表
    current_msg = []  # 当前消息块的列表
    has_start = False  # 是否已开始处理消息块的标志
    has_msgid = False  # 是否已找到msgid的标志
    sep = "#: .."  # 分隔符
    for line in lines:
        if line.startswith(sep):  # 如果行以分隔符开头
            has_start = True
            if current_msg and has_msgid:
                # 开始一个新的消息块
                messages.append("".join(current_msg))
                current_msg = []
                has_msgid = False
                current_msg.append(line)
            else:
                current_msg.append(line)
        elif has_start and line.startswith("msgid"):  # 如果已开始且行以"msgid"开头
            has_msgid = True
            current_msg.append(line)
        elif has_start:
            current_msg.append(line)
        else:
            print("Skip line:", line)  # 跳过的行，打印该行内容
    if current_msg:
        messages.append("".join(current_msg))

    return messages  # 返回提取出的消息列表


class BatchOperator(JoinOperator[str]):
    def __init__(
        self,
        llm_client: LLMClient,
        model_name: str = "gpt-3.5-turbo",  # 使用的语言模型的名称，默认为 "gpt-3.5-turbo"
        max_new_token: int = 4096,  # 最大新标记数，默认为 4096
        **kwargs,
    ):
        self._tokenizer = TiktokenProxyTokenizer()  # 初始化一个分词器
        self._llm_client = llm_client  # 语言模型客户端对象
        self._model_name = model_name  # 使用的语言模型的名称
        self._max_new_token = max_new_token  # 最大新标记数
        super().__init__(combine_function=self.batch_run, **kwargs)  # 调用父类构造函数，设置合并函数为 batch_run
    async def batch_run(self, blocks: List[str], ext_dict: Dict[str, Any]) -> str:
        # 从扩展字典或默认值中获取最大新令牌数量
        max_new_token = ext_dict.get("max_new_token", self._max_new_token)
        # 从扩展字典或默认值中获取并行处理的数量
        parallel_num = ext_dict.get("parallel_num", 5)
        # 从扩展字典或默认值中获取模型名称
        model_name = ext_dict.get("model_name", self._model_name)
        
        # 将原始文本分割成批量块，每个块的长度不超过max_new_token
        batch_blocks = await self.split_blocks(blocks, model_name, max_new_token)
        
        new_blocks = []
        # 将每个分割后的块组装成包含用户输入和扩展字典的字典列表
        for block in batch_blocks:
            new_blocks.append({"user_input": "".join(block), **ext_dict})
        
        # 创建一个有向无环图（DAG）用于执行分割块的任务流程
        with DAG("split_blocks_dag"):
            # 创建一个迭代器触发器，以新块的迭代器作为数据源
            trigger = IteratorTrigger(data=InputSource.from_iterable(new_blocks))
            # 创建一个提示生成器操作符，配置不同的提示模板
            prompt_task = PromptBuilderOperator(
                ChatPromptTemplate(
                    messages=[
                        SystemPromptTemplate.from_template(PROMPT_ZH),
                        HumanPromptTemplate.from_template("{user_input}"),
                    ],
                )
            )
            # 创建一个请求生成器操作符，配置模型名称、温度和最大新令牌数量
            model_pre_handle_task = RequestBuilderOperator(
                model=model_name, temperature=0.1, max_new_tokens=4096
            )
            # 创建一个语言模型操作符，使用开放的AI语言模型客户端
            llm_task = LLMOperator(OpenAILLMClient())
            # 创建一个输出解析器任务
            out_parse_task = OutputParser()

            # 定义任务流程，依次连接触发器、提示生成器、请求生成器、语言模型和输出解析器
            (
                trigger
                >> prompt_task
                >> model_pre_handle_task
                >> llm_task
                >> out_parse_task
            )
        
        # 触发并行执行任务流程，并获取结果
        results = await trigger.trigger(parallel_num=parallel_num)
        outs = []
        # 提取每个结果中的输出数据
        for _, out_data in results:
            outs.append(out_data)
        
        # 将输出数据以双换行符分隔并返回
        return "\n\n".join(outs)

    async def split_blocks(
        self, blocks: List[str], model_name: str, max_new_token: int
    ) -> List[List[str]]:
        batch_blocks = []
        last_block_end = 0
        # 循环直到所有块都被分割
        while last_block_end < len(blocks):
            start = last_block_end
            # 使用二分搜索确定每个块的分割点
            split_point = await self.bin_search(
                blocks[start:], model_name, max_new_token
            )
            new_end = start + split_point + 1
            # 将每个分割后的块添加到批处理块列表中
            batch_blocks.append(blocks[start:new_end])
            last_block_end = new_end
        
        # 检查所有块的总长度是否与原始块长度相同
        if sum(len(block) for block in batch_blocks) != len(blocks):
            raise ValueError("Split blocks error.")
        
        # 检查每个块的令牌数量是否在限制范围内
        for block in batch_blocks:
            block_tokens = await self._llm_client.count_token(model_name, "".join(block))
            if block_tokens > max_new_token:
                raise ValueError(
                    f"Block size {block_tokens} exceeds the max token limit "
                    f"{max_new_token}, your bin_search function is wrong."
                )
        
        # 返回批处理块列表
        return batch_blocks

    async def bin_search(
        self, blocks: List[str], model_name: str, max_new_token: int
    ) -> int:
    ) -> int:
        """Binary search to find the split point."""
        # 初始化左右边界
        l, r = 0, len(blocks) - 1
        # 二分查找算法，直到左边界超过右边界为止
        while l < r:
            # 计算中间点
            mid = l + r + 1 >> 1
            # 调用异步方法，计算当前分割点及之前所有块的总令牌数
            current_tokens = await self._llm_client.count_token(
                model_nam, "".join(blocks[: mid + 1])
            )
            # 如果当前令牌数小于等于最大新增令牌数，则将左边界移到中间点
            if current_tokens <= max_new_token:
                l = mid
            # 否则将右边界移到中间点的左侧
            else:
                r = mid - 1
        # 返回右边界作为最终的分割点
        return r
# 定义一个输出解析器类，继承自 MapOperator 类，处理 ModelOutput 类型映射到字符串类型
class OutputParser(MapOperator[ModelOutput, str]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # 异步方法，将模型输出的文本内容去除首尾空白字符后返回
    async def map(self, model_output: ModelOutput) -> str:
        content = model_output.text
        return content.strip()


# 定义一个保存翻译后的 PO 文件的操作类，继承自 JoinOperator 类
class SaveTranslatedPoFileOperator(JoinOperator[str]):
    def __init__(self, **kwargs):
        # 使用 combine_function 参数初始化父类 JoinOperator，指定保存文件的具体操作函数为 self.save_file
        super().__init__(combine_function=self.save_file, **kwargs)

    # 异步方法，调用阻塞函数转换为异步操作，保存翻译后的内容到指定文件路径
    async def save_file(self, translated_content: str, file_path: str) -> str:
        return await self.blocking_func_to_async(
            self._save_file, translated_content, file_path
        )

    # 阻塞函数，实际执行文件保存操作，将翻译后的内容写入文件中
    def _save_file(self, translated_content: str, file_path: str) -> str:
        output_file = file_path.replace(".po", "_ai_translated.po")
        with open(output_file, "w") as f:
            f.write(translated_content)
        return translated_content


# 主程序入口，创建一个 Directed Acyclic Graph (DAG) 对象，并定义任务流程
with DAG("translate_po_dag") as dag:
    # 定义节点
    llm_client = OpenAILLMClient()
    input_task = InputOperator(input_source=InputSource.from_callable())
    read_po_file_task = ReadPoFileOperator()
    parse_po_file_task = ParsePoFileOperator()

    # 使用 ChatGPT 处理的批处理任务，设置最大新词数为 1024
    batch_task = BatchOperator(llm_client, max_new_token=1024)
    save_translated_po_file_task = SaveTranslatedPoFileOperator()

    # 定义任务流程，将输入任务通过映射操作提取文件路径，依次执行读取、解析、批处理操作
    (input_task
     >> MapOperator(lambda x: x["file_path"])  # 映射操作：提取文件路径
     >> read_po_file_task  # 读取 PO 文件
     >> parse_po_file_task  # 解析 PO 文件
     >> batch_task)  # 批处理任务

    # 对输入任务再次映射操作，提取扩展字典，并传递给批处理任务处理
    input_task >> MapOperator(lambda x: x["ext_dict"]) >> batch_task

    # 将批处理任务的输出传递给保存翻译后的 PO 文件任务
    batch_task >> save_translated_po_file_task

    # 同样将输入任务的文件路径再次映射操作，传递给保存翻译后的 PO 文件任务
    input_task >> MapOperator(lambda x: x["file_path"]) >> save_translated_po_file_task


# 异步函数，运行翻译 PO 文件的 DAG，接收任务、语言描述、模块名称等参数
async def run_translate_po_dag(
    task,
    language: str,
    language_desc: str,
    module_name: str,
    max_new_token: int = 1024,
    parallel_num=10,
    model_name: str = "gpt-3.5-turbo",
):
    # 构建完整路径，指向语言目录下的特定文件
    full_path = os.path.join(
        "./locales", language, "LC_MESSAGES", f"dbgpt_{module_name}.po"
    )

    # 获取对应语言的词汇表，若无则使用默认词汇表
    vocabulary = vocabulary_map.get(language, vocabulary_map["default"])
    # 构建词汇表字符串，格式为键值对列表
    vocabulary_str = "\n".join([f"  * {k} -> {v}" for k, v in vocabulary.items()])

    # 构建扩展字典，包含语言描述、词汇表、响应等信息
    ext_dict = {
        "language_desc": language_desc,
        "vocabulary": vocabulary_str,
        "response": response,
        "language": language_desc,
        "example_1_input": example_1_input,
        "example_1_output": example_1_output_1,
        "example_2_input": example_2_input,
        "example_2_output": example_2_output,
        "max_new_token": max_new_token,
        "parallel_num": parallel_num,
        "model_name": model_name,
    }

    try:
        # 调用任务对象的 call 方法，传入文件路径和扩展字典，执行任务流程
        result = await task.call({"file_path": full_path, "ext_dict": ext_dict})
        return result
    except Exception as e:
        # 捕获异常，打印错误信息
        print(f"Error in {module_name}: {e}")


# 主程序入口，判断是否为直接运行脚本的入口
if __name__ == "__main__":
    # 在这里添加额外的主程序逻辑或初始化步骤
    # 定义所有要翻译的模块列表
    all_modules = [
        "agent",
        "app",
        "cli",
        "client",
        "configs",
        "core",
        "datasource",
        "model",
        "rag",
        "serve",
        "storage",
        "train",
        "util",
        "vis",
    ]
    # 定义语言到语言描述的映射字典
    lang_map = {
        "zh_CN": "简体中文",
        "ja": "日本語",
        "fr": "Français",
        "ko": "한국어",
        "ru": "русский",
    }

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加 --modules 参数，用于指定要翻译的模块，默认为所有模块
    parser.add_argument(
        "--modules",
        type=str,
        default=",".join(all_modules),
        help="Modules to translate, 'all' for all modules, split by ','.",
    )
    # 添加 --lang 参数，用于指定翻译的目标语言，默认为简体中文
    parser.add_argument(
        "--lang",
        type=str,
        default="zh_CN",
        help="Language to translate, 'all' for all languages, split by ','.",
    )
    # 添加 --max_new_token 参数，用于指定最大新增标记数，默认为 1024
    parser.add_argument("--max_new_token", type=int, default=1024)
    # 添加 --parallel_num 参数，用于指定并行翻译的数量，默认为 10
    parser.add_argument("--parallel_num", type=int, default=10)
    # 添加 --model_name 参数，用于指定使用的模型名称，默认为 gpt-3.5-turbo
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo")

    # 解析命令行参数
    args = parser.parse_args()
    # 打印解析后的参数信息
    print(f"args: {args}")

    # 根据命令行参数设置模型名称
    model_name = args.model_name
    # 根据命令行参数设置要翻译的模块列表
    modules = all_modules if args.modules == "all" else args.modules.strip().split(",")
    # 根据命令行参数设置最大新增标记数
    max_new_token = args.max_new_token
    # 根据命令行参数设置并行翻译的数量
    parallel_num = args.parallel_num
    # 根据命令行参数设置要翻译的语言列表
    langs = lang_map.keys() if args.lang == "all" else args.lang.strip().split(",")

    # 检查所有指定的语言是否在语言映射字典中，如果不在则抛出 ValueError
    for lang in langs:
        if lang not in lang_map:
            raise ValueError(
                f"Language {lang} not supported, now only support {','.join(lang_map.keys())}."
            )

    # 遍历每种语言和每个模块，调用翻译函数进行翻译
    for lang in langs:
        lang_desc = lang_map[lang]
        for module in modules:
            asyncio.run(
                run_translate_po_dag(
                    save_translated_po_file_task,
                    lang,
                    lang_desc,
                    module,
                    max_new_token,
                    parallel_num,
                    model_name,
                )
            )
```