# `.\MinerU\tests\test_cli\test_bench_gpu.py`

```
# 导入 pytest 测试框架
import pytest
# 导入操作系统相关模块
import os
# 从配置模块导入配置
from conf import conf
# 导入 JSON 处理模块
import json
# 从 magic_pdf.pipe 导入 UNIPipe 类
from magic_pdf.pipe.UNIPipe import UNIPipe
# 从 magic_pdf.rw 导入 DiskReaderWriter 类
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
# 从 lib 导入 calculate_score 模块
from lib import calculate_score
# 获取 PDF 结果路径配置
import shutil
pdf_res_path = conf.conf["pdf_res_path"]
# 获取代码路径配置
code_path = conf.conf["code_path"]
# 获取 PDF 开发路径配置
pdf_dev_path = conf.conf["pdf_dev_path"]

# 定义测试类 TestCliCuda
class TestCliCuda:
    """
    测试 CLI CUDA 功能
    """
    # 定义测试方法 test_pdf_sdk_cuda
    def test_pdf_sdk_cuda(self):
        """
        测试 PDF SDK CUDA 功能
        """
        # 清理 PDF 结果路径
        clean_magicpdf(pdf_res_path)
        # 将 PDF 转换为 Markdown
        pdf_to_markdown()
        # 打开生成的结果 JSON 文件，读取其内容
        fr = open(os.path.join(pdf_dev_path, "result.json"), "r", encoding="utf-8")
        lines = fr.readlines()  # 读取所有行
        last_line = lines[-1].strip()  # 获取最后一行并去除空白字符
        last_score = json.loads(last_line)  # 将最后一行 JSON 数据解析为字典
        last_simscore = last_score["average_sim_score"]  # 获取上一次的平均相似度分数
        last_editdistance = last_score["average_edit_distance"]  # 获取上一次的平均编辑距离
        last_bleu = last_score["average_bleu_score"]  # 获取上一次的平均 BLEU 分数
        # 执行预清理脚本，下载数据到指定目录
        os.system(f"python tests/test_cli/lib/pre_clean.py --tool_name mineru --download_dir {pdf_dev_path}")
        # 获取当前得分
        now_score = get_score()
        print ("now_score:", now_score)  # 输出当前得分
        # 检查 CI 目录是否存在，不存在则创建
        if not os.path.exists(os.path.join(pdf_dev_path, "ci")):
            os.makedirs(os.path.join(pdf_dev_path, "ci"), exist_ok=True)
        # 打开并创建 CI 结果 JSON 文件以写入当前得分
        fw = open(os.path.join(pdf_dev_path, "ci", "result.json"), "w+", encoding="utf-8")
        fw.write(json.dumps(now_score) + "\n")  # 将当前得分写入文件
        now_simscore = now_score["average_sim_score"]  # 获取当前的平均相似度分数
        now_editdistance = now_score["average_edit_distance"]  # 获取当前的平均编辑距离
        now_bleu = now_score["average_bleu_score"]  # 获取当前的平均 BLEU 分数
        # 断言当前分数大于等于上一个分数
        assert last_simscore <= now_simscore
        assert last_editdistance <= now_editdistance
        assert last_bleu <= now_bleu

# 定义 PDF 转 Markdown 的函数
def pdf_to_markdown():
    """
    将 PDF 转换为 Markdown 格式
    """
    demo_names = list()  # 创建示例名称列表
    pdf_path = os.path.join(pdf_dev_path, "pdf")  # 获取 PDF 文件路径
    # 遍历 PDF 目录，找到所有 PDF 文件
    for pdf_file in os.listdir(pdf_path):
        if pdf_file.endswith('.pdf'):  # 仅处理 PDF 文件
            demo_names.append(pdf_file.split('.')[0])  # 添加不带扩展名的文件名
    # 对每个示例名称进行处理
    for demo_name in demo_names:
        pdf_path = os.path.join(pdf_dev_path, "pdf", f"{demo_name}.pdf")  # 构建 PDF 文件路径
        cmd = "magic-pdf pdf-command --pdf %s --inside_model true" % (pdf_path)  # 构建命令行指令
        os.system(cmd)  # 执行命令
        dir_path = os.path.join(pdf_dev_path, "mineru")  # 获取结果目录路径
        # 如果目录不存在，则创建
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        res_path = os.path.join(dir_path, f"{demo_name}.md")  # 构建 Markdown 结果路径
        src_path = os.path.join(pdf_res_path, demo_name, "auto", f"{demo_name}.md")  # 源 Markdown 路径
        shutil.copy(src_path, res_path)  # 复制生成的 Markdown 文件到结果目录

# 定义获取分数的函数
def get_score():
    """
    获取评分结果
    """
    score = calculate_score.Scoring(os.path.join(pdf_dev_path, "result.json"))  # 创建评分对象
    score.calculate_similarity_total("mineru", pdf_dev_path)  # 计算相似度
    res = score.summary_scores()  # 获取评分摘要
    return res  # 返回评分结果

# 定义清理 PDF 结果的函数
def clean_magicpdf(pdf_res_path):
    """
    清理 PDF 结果目录
    """
    cmd = "rm -rf %s" % (pdf_res_path)  # 构建删除命令
    os.system(cmd)  # 执行删除命令
```