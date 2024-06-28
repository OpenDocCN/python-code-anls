# `.\commands\run.py`

```py
# 导入必要的模块和函数
from argparse import ArgumentParser

from ..pipelines import Pipeline, PipelineDataFormat, get_supported_tasks, pipeline
from ..utils import logging
from . import BaseTransformersCLICommand

# 获取当前模块的日志记录器对象
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 根据文件路径推断输入文件格式的函数
def try_infer_format_from_ext(path: str):
    # 如果路径为空，则默认返回"pipe"
    if not path:
        return "pipe"

    # 遍历已支持的数据格式列表
    for ext in PipelineDataFormat.SUPPORTED_FORMATS:
        # 如果路径以当前格式结尾，则返回该格式
        if path.endswith(ext):
            return ext
    
    # 如果无法推断出格式，则抛出异常
    raise Exception(
        f"Unable to determine file format from file extension {path}. "
        f"Please provide the format through --format {PipelineDataFormat.SUPPORTED_FORMATS}"
    )

# 创建运行命令的工厂函数，根据参数配置创建相应的pipeline和数据格式对象
def run_command_factory(args):
    # 根据参数配置创建pipeline对象
    nlp = pipeline(
        task=args.task,
        model=args.model if args.model else None,
        config=args.config,
        tokenizer=args.tokenizer,
        device=args.device,
    )
    
    # 根据输入文件路径推断数据格式，或者直接使用给定的格式参数
    format = try_infer_format_from_ext(args.input) if args.format == "infer" else args.format
    
    # 根据参数创建数据格式对象
    reader = PipelineDataFormat.from_str(
        format=format,
        output_path=args.output,
        input_path=args.input,
        column=args.column if args.column else nlp.default_input_names,
        overwrite=args.overwrite,
    )
    
    # 返回运行命令对象，传入创建的pipeline和数据格式对象
    return RunCommand(nlp, reader)

# 定义运行命令的类，继承自BaseTransformersCLICommand
class RunCommand(BaseTransformersCLICommand):
    def __init__(self, nlp: Pipeline, reader: PipelineDataFormat):
        self._nlp = nlp  # 存储传入的pipeline对象
        self._reader = reader  # 存储传入的数据格式对象

    @staticmethod
    def run(self):
        # 初始化 NLP 模型和输出结果列表
        nlp, outputs = self._nlp, []

        # 遍历数据读取器中的每个条目
        for entry in self._reader:
            # 如果数据读取器支持多列输入，则调用 NLP 模型的处理方法
            if self._reader.is_multi_columns:
                output = nlp(**entry)
            else:
                output = nlp(entry)

            # 如果输出是字典，则将其添加到输出列表中
            if isinstance(output, dict):
                outputs.append(output)
            else:
                # 如果输出不是字典，假定它是一个列表，将其扩展到输出列表中
                outputs += output

        # 保存数据
        if self._nlp.binary_output:
            # 如果 NLP 模型要求二进制输出，将输出保存为二进制文件
            binary_path = self._reader.save_binary(outputs)
            # 记录警告，指出当前管道需要以二进制格式保存输出
            logger.warning(f"Current pipeline requires output to be in binary format, saving at {binary_path}")
        else:
            # 否则，将输出保存到适当的位置
            self._reader.save(outputs)
```