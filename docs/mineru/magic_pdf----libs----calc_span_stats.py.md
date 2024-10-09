# `.\MinerU\magic_pdf\libs\calc_span_stats.py`

```
# 导入所需的库和模块
import os  # 用于与操作系统交互的模块
import csv  # 用于读取和写入 CSV 文件的模块
import json  # 用于处理 JSON 数据的模块
import pandas as pd  # 导入 pandas 库用于数据处理
from pandas import DataFrame as df  # 将 DataFrame 简写为 df
from matplotlib import pyplot as plt  # 导入绘图模块
from termcolor import cprint  # 导入用于彩色打印的模块

"""
执行此脚本的方法：

1. 确保目录 code-clean/tmp/unittest/md/ 下存在 pdf_dic.json 文件，例如：

    code-clean/tmp/unittest/md/scihub/scihub_00500000/libgen.scimag00527000-00527999.zip_10.1002/app.25178/pdf_dic.json
    
2. 在 code-clean 目录下，执行以下命令：

    $ python -m libs.calc_span_stats
    
"""


def print_green_on_red(text):
    # 以绿色字体在红色背景上打印文本，并添加换行
    cprint(text, "green", "on_red", attrs=["bold"], end="\n\n")


def print_green(text):
    # 打印空行
    print()
    # 以绿色字体打印文本，并添加换行
    cprint(text, "green", attrs=["bold"], end="\n\n")


def print_red(text):
    # 打印空行
    print()
    # 以红色字体打印文本，并添加换行
    cprint(text, "red", attrs=["bold"], end="\n\n")


def safe_get(dict_obj, key, default):
    # 从字典中安全获取值，如果键不存在则返回默认值
    val = dict_obj.get(key)
    if val is None:
        return default  # 如果值为 None，返回默认值
    else:
        return val  # 否则返回找到的值


class SpanStatsCalc:
    """计算跨度的统计数据。"""

    def draw_charts(self, span_stats: pd.DataFrame, fig_num: int, save_path: str):
        """在一个图中绘制多个图形。"""
        # 创建一个画布
        fig = plt.figure(fig_num, figsize=(20, 20))

        pass  # 这里的具体绘制逻辑尚未实现


def __find_pdf_dic_files(
    jf_name="pdf_dic.json",  # 要查找的 JSON 文件名
    base_code_name="code-clean",  # 基础代码目录名
    tgt_base_dir_name="tmp",  # 目标基础目录名
    unittest_dir_name="unittest",  # 单元测试目录名
    md_dir_name="md",  # md 目录名
    book_names=[
        "scihub",  # 书籍名称列表，包含 "scihub"
    ],  # 其他可能的值包括 "zlib"、"arxiv" 等
):
    pdf_dict_files = []  # 初始化 PDF 字典文件列表

    curr_dir = os.path.dirname(__file__)  # 获取当前文件的目录

    for i in range(len(curr_dir)):  # 遍历当前目录的每个字符
        if curr_dir[i : i + len(base_code_name)] == base_code_name:  # 检查是否找到基础代码目录名
            base_code_dir_name = curr_dir[: i + len(base_code_name)]  # 设置基础代码目录路径
            for book_name in book_names:  # 遍历书籍名称列表
                # 创建要搜索的相对目录路径
                search_dir_relative_name = os.path.join(tgt_base_dir_name, unittest_dir_name, md_dir_name, book_name)
                if os.path.exists(base_code_dir_name):  # 检查基础代码目录是否存在
                    # 创建搜索目录的绝对路径
                    search_dir_name = os.path.join(base_code_dir_name, search_dir_relative_name)
                    for root, dirs, files in os.walk(search_dir_name):  # 遍历搜索目录
                        for file in files:  # 遍历文件
                            if file == jf_name:  # 检查文件名是否匹配
                                # 将符合条件的文件绝对路径添加到列表中
                                pdf_dict_files.append(os.path.join(root, file))
                break  # 一旦找到基础代码目录，终止循环

    return pdf_dict_files  # 返回找到的 PDF 字典文件列表


def combine_span_texts(group_df, span_stats):
    combined_span_texts = []  # 初始化组合跨度文本列表
    # 遍历 group_df 数据框中的每一行
    for _, row in group_df.iterrows():
        # 当前行的索引作为当前跨度的 ID
        curr_span_id = row.name
        # 获取当前跨度的文本
        curr_span_text = row["span_text"]

        # 计算前一个跨度的 ID
        pre_span_id = curr_span_id - 1
        # 如果前一个跨度的 ID 在 span_stats 中，获取其文本，否则为空字符串
        pre_span_text = span_stats.at[pre_span_id, "span_text"] if pre_span_id in span_stats.index else ""

        # 计算后一个跨度的 ID
        next_span_id = curr_span_id + 1
        # 如果后一个跨度的 ID 在 span_stats 中，获取其文本，否则为空字符串
        next_span_text = span_stats.at[next_span_id, "span_text"] if next_span_id in span_stats.index else ""

        # 如果当前跨度是上标，pointer_sign 为右箭头，否则为下箭头
        pointer_sign = "→ → → "
        # 将前一个、当前和后一个跨度的文本结合在一起，并加上指示符
        combined_text = "\n".join([pointer_sign + pre_span_text, pointer_sign + curr_span_text, pointer_sign + next_span_text])
        # 将组合的文本添加到 combined_span_texts 列表中
        combined_span_texts.append(combined_text)

    # 将所有组合的文本以双换行符连接成一个字符串并返回
    return "\n\n".join(combined_span_texts)
# 设置 Pandas 选项，以便显示完整的文本内容
# pd.set_option("display.max_colwidth", None)  # 设置为 None 来显示完整的文本
# 设置 Pandas 选项，以便显示更多行的数据
pd.set_option("display.max_rows", None)  # 设置为 None 来显示更多的行

# 主函数，负责处理 PDF 字典文件
def main():
    # 查找所有的 PDF 字典文件
    pdf_dict_files = __find_pdf_dic_files()
    # print(pdf_dict_files)  # 打印找到的文件列表（被注释掉）

    # 创建一个 SpanStatsCalc 实例，用于计算统计数据
    span_stats_calc = SpanStatsCalc()

    # 遍历每一个 PDF 字典文件
    for pdf_dict_file in pdf_dict_files:
        # 打印分隔线以便于输出区分
        print("-" * 100)
        # 打印当前正在处理的文件名
        print_green_on_red(f"Processing {pdf_dict_file}")

        # 以 UTF-8 编码打开 PDF 字典文件
        with open(pdf_dict_file, "r", encoding="utf-8") as f:
            # 读取 JSON 格式的数据到 pdf_dict 中
            pdf_dict = json.load(f)

            # 计算原始数据的统计信息
            raw_df = span_stats_calc.calc_stats_per_dict(pdf_dict)
            # 定义保存原始统计数据的路径
            save_path = pdf_dict_file.replace("pdf_dic.json", "span_stats_raw.csv")
            # 将原始数据保存为 CSV 文件
            raw_df.to_csv(save_path, index=False)

            # 过滤出 span_is_superscript 列为 1 的数据
            filtered_df = raw_df[raw_df["span_is_superscript"] == 1]
            # 如果没有找到超脚本数据，则打印提示并跳过
            if filtered_df.empty:
                print("No superscript span found!")
                continue

            # 按 span_font_name、span_font_size、span_font_color 进行分组
            filtered_grouped_df = filtered_df.groupby(["span_font_name", "span_font_size", "span_font_color"])

            # 合并每个分组的 span 文本
            combined_span_texts = filtered_grouped_df.apply(combine_span_texts, span_stats=raw_df)  # type: ignore

            # 计算每个分组的计数并重置索引
            final_df = filtered_grouped_df.size().reset_index(name="count")
            # 将合并后的 span 文本添加到 final_df 中
            final_df["span_texts"] = combined_span_texts.reset_index(level=[0, 1, 2], drop=True)

            # 打印最终的 DataFrame
            print(final_df)

            # 将 span_texts 中的换行符替换为回车换行符
            final_df["span_texts"] = final_df["span_texts"].apply(lambda x: x.replace("\n", "\r\n"))

            # 定义保存最终统计数据的路径
            save_path = pdf_dict_file.replace("pdf_dic.json", "span_stats_final.csv")
            # 使用 UTF-8 编码并添加 BOM，确保所有字段被双引号包围
            final_df.to_csv(save_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

            # 创建一个 2x2 的图表布局
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))

            # 按照 span_font_name 分类作图
            final_df.groupby("span_font_name")["count"].sum().plot(kind="bar", ax=axs[0, 0], title="By Font Name")

            # 按照 span_font_size 分类作图
            final_df.groupby("span_font_size")["count"].sum().plot(kind="bar", ax=axs[0, 1], title="By Font Size")

            # 按照 span_font_color 分类作图
            final_df.groupby("span_font_color")["count"].sum().plot(kind="bar", ax=axs[1, 0], title="By Font Color")

            # 按照 span_font_name、span_font_size 和 span_font_color 共同分类作图
            grouped = final_df.groupby(["span_font_name", "span_font_size", "span_font_color"])
            grouped["count"].sum().unstack().plot(kind="bar", ax=axs[1, 1], title="Combined Grouping")

            # 调整布局以避免重叠
            plt.tight_layout()

            # 显示图表
            # plt.show()  # 可以选择显示图表（被注释掉）

            # 保存图表到 PNG 文件
            save_path = pdf_dict_file.replace("pdf_dic.json", "span_stats_combined.png")
            plt.savefig(save_path)

            # 清除当前画布以准备下一次绘图
            plt.clf()

# 如果当前模块是主程序，则调用 main 函数
if __name__ == "__main__":
    main()
```