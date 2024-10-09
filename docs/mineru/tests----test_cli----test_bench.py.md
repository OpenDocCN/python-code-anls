# `.\MinerU\tests\test_cli\test_bench.py`

```
"""
bench
"""  # 声明模块名称为 bench
import os  # 导入 os 模块，用于与操作系统交互
import shutil  # 导入 shutil 模块，用于文件操作和高级文件处理
import json  # 导入 json 模块，用于处理 JSON 数据
from lib import calculate_score  # 从 lib 模块导入 calculate_score，用于分数计算
import pytest  # 导入 pytest 模块，用于单元测试
from conf import conf  # 从 conf 模块导入 conf，用于获取配置信息

code_path = os.environ.get('GITHUB_WORKSPACE')  # 获取环境变量 'GITHUB_WORKSPACE' 的值
pdf_dev_path = conf.conf["pdf_dev_path"]  # 从配置中获取 pdf_dev_path
pdf_res_path = conf.conf["pdf_res_path"]  # 从配置中获取 pdf_res_path

class TestBench():  # 定义测试类 TestBench
    """
    test bench
    """  # 测试基准的文档字符串说明
    def test_ci_ben(self):  # 定义测试方法 test_ci_ben
        """
        ci benchmark
        """  # 文档字符串，说明此方法用于 CI 基准测试
        fr = open(os.path.join(pdf_dev_path, "result.json"), "r", encoding="utf-8")  # 打开指定路径的 result.json 文件以读取
        lines = fr.readlines()  # 读取文件中的所有行
        last_line = lines[-1].strip()  # 获取文件的最后一行，并去掉首尾空白字符
        last_score = json.loads(last_line)  # 将最后一行 JSON 字符串解析为字典
        last_simscore = last_score["average_sim_score"]  # 提取最后得分的平均相似度
        last_editdistance = last_score["average_edit_distance"]  # 提取最后得分的平均编辑距离
        last_bleu = last_score["average_bleu_score"]  # 提取最后得分的 BLEU 分数
        os.system(f"python tests/test_cli/lib/pre_clean.py --tool_name mineru --download_dir {pdf_dev_path}")  # 执行命令行指令，运行清理脚本
        now_score = get_score()  # 调用 get_score 函数获取当前得分
        print ("now_score:", now_score)  # 打印当前得分
        if not os.path.exists(os.path.join(pdf_dev_path, "ci")):  # 检查 'ci' 目录是否存在
            os.makedirs(os.path.join(pdf_dev_path, "ci"), exist_ok=True)  # 如果不存在，则创建 'ci' 目录
        fw = open(os.path.join(pdf_dev_path, "ci", "result.json"), "w+", encoding="utf-8")  # 打开或创建 result.json 文件以写入
        fw.write(json.dumps(now_score) + "\n")  # 将当前得分写入文件，并添加换行符
        now_simscore = now_score["average_sim_score"]  # 提取当前得分的平均相似度
        now_editdistance = now_score["average_edit_distance"]  # 提取当前得分的平均编辑距离
        now_bleu = now_score["average_bleu_score"]  # 提取当前得分的 BLEU 分数
        assert last_simscore <= now_simscore  # 断言当前的平均相似度不低于最后的平均相似度
        assert last_editdistance <= now_editdistance  # 断言当前的编辑距离不高于最后的编辑距离
        assert last_bleu <= now_bleu  # 断言当前的 BLEU 分数不低于最后的 BLEU 分数


def get_score():  # 定义函数 get_score
    """
    get score
    """  # 文档字符串，说明此函数用于获取分数
    score = calculate_score.Scoring(os.path.join(pdf_dev_path, "result.json"))  # 创建 Scoring 对象，传入 result.json 文件路径
    score.calculate_similarity_total("mineru", pdf_dev_path)  # 计算与 'mineru' 相关的总相似度
    res = score.summary_scores()  # 获取分数汇总
    return res  # 返回分数汇总结果
```