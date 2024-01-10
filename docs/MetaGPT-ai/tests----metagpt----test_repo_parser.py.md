# `MetaGPT\tests\metagpt\test_repo_parser.py`

```

# 从 pathlib 模块中导入 Path 类
from pathlib import Path
# 从 pprint 模块中导入 pformat 函数
from pprint import pformat
# 从 metagpt.const 模块中导入 METAGPT_ROOT 变量
from metagpt.const import METAGPT_ROOT
# 从 metagpt.logs 模块中导入 logger 对象
from metagpt.logs import logger
# 从 metagpt.repo_parser 模块中导入 RepoParser 类
from metagpt.repo_parser import RepoParser

# 定义测试函数 test_repo_parser
def test_repo_parser():
    # 创建 RepoParser 对象，指定基础目录为 METAGPT_ROOT / "metagpt" / "strategy"
    repo_parser = RepoParser(base_directory=METAGPT_ROOT / "metagpt" / "strategy")
    # 生成符号列表
    symbols = repo_parser.generate_symbols()
    # 记录符号列表
    logger.info(pformat(symbols))

    # 断言 "tot_schema.py" 是否在符号列表中
    assert "tot_schema.py" in str(symbols)

    # 生成结构，模式为 "json"，并断言输出路径存在
    output_path = repo_parser.generate_structure(mode="json")
    assert output_path.exists()
    # 生成结构，模式为 "csv"，并断言输出路径存在
    output_path = repo_parser.generate_structure(mode="csv")
    assert output_path.exists()

# 定义错误测试函数 test_error
def test_error():
    """_parse_file should return empty list when file not existed"""
    # 调用 _parse_file 方法，传入不存在的文件路径，断言返回空列表
    rsp = RepoParser._parse_file(Path("test_not_existed_file.py"))
    assert rsp == []

```