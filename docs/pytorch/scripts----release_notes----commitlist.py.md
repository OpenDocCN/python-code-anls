# `.\pytorch\scripts\release_notes\commitlist.py`

```py
import argparse
import csv
import dataclasses
import os
import pprint
import re
from collections import defaultdict
from pathlib import Path
from typing import List

import common
from common import (
    features_to_dict,
    frontend_categories,
    get_commit_data_cache,
    run,
    topics,
)

"""
Example Usages

Create a new commitlist for consumption by categorize.py.
Said commitlist contains commits between v1.5.0 and f5bc91f851.

    python commitlist.py --create-new tags/v1.5.0 f5bc91f851

Update the existing commitlist to commit bfcb687b9c.

    python commitlist.py --update-to bfcb687b9c

"""

@dataclasses.dataclass(frozen=False)
class Commit:
    commit_hash: str
    category: str
    topic: str
    title: str
    files_changed: str
    pr_link: str
    author: str
    accepter_1: str  # This is not a list so that it is easier to put in a spreadsheet
    accepter_2: str
    accepter_3: str
    merge_into: str = None

    def __repr__(self):
        return (
            f"Commit({self.commit_hash}, {self.category}, {self.topic}, {self.title})"
        )

# 获取所有Commit类中的字段名组成的元组
commit_fields = tuple(f.name for f in dataclasses.fields(Commit))


class CommitList:
    # NB: Private ctor. Use `from_existing` or `create_new`.
    def __init__(self, path: str, commits: List[Commit]):
        self.path = path  # 初始化CommitList对象的路径
        self.commits = commits  # 初始化CommitList对象的提交列表

    @staticmethod
    def from_existing(path):
        # 从指定路径读取已有的Commit列表
        commits = CommitList.read_from_disk(path)
        return CommitList(path, commits)

    @staticmethod
    def create_new(path, base_version, new_version):
        # 创建一个新的Commit列表文件，包含指定版本范围内的提交
        if os.path.exists(path):
            raise ValueError(
                "Attempted to create a new commitlist but one exists already!"
            )
        commits = CommitList.get_commits_between(base_version, new_version)
        return CommitList(path, commits)

    @staticmethod
    def read_from_disk(path) -> List[Commit]:
        # 从磁盘文件中读取Commit列表数据
        with open(path) as csvfile:
            reader = csv.DictReader(csvfile)
            rows = []
            for row in reader:
                if row.get("new_title", "") != "":
                    row["title"] = row["new_title"]
                filtered_rows = {k: row.get(k, "") for k in commit_fields}
                rows.append(Commit(**filtered_rows))
        return rows

    def write_result(self):
        # 将Commit列表写入到文件中
        self.write_to_disk_static(self.path, self.commits)

    @staticmethod
    def write_to_disk_static(path, commit_list):
        # 静态方法：将Commit列表写入到指定路径的文件中
        os.makedirs(Path(path).parent, exist_ok=True)
        with open(path, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(commit_fields)
            for commit in commit_list:
                writer.writerow(dataclasses.astuple(commit))

    @staticmethod
    def keywordInFile(file, keywords):
        # 检查文件中是否包含给定的关键词列表中的任意一个关键词
        for key in keywords:
            if key in file:
                return True
        return False

    @staticmethod
    # 生成提交对象的方法，根据提交哈希获取提交数据并生成 Commit 对象
    def gen_commit(commit_hash):
        # 从提交数据缓存中获取指定提交哈希的特征项
        feature_item = get_commit_data_cache().get(commit_hash)
        # 将特征项转换为字典形式
        features = features_to_dict(feature_item)
        # 根据特征项的内容进行分类，获取分类和主题
        category, topic = CommitList.categorize(features)
        # 获取接受者列表的前三项，若不足三项则用空字符串补齐
        a1, a2, a3 = (features["accepters"] + ("", "", ""))[:3]
        # 如果特征项包含 Pull Request 号，则生成对应的 GitHub 链接，否则为 None
        if features["pr_number"] is not None:
            pr_link = f"https://github.com/pytorch/pytorch/pull/{features['pr_number']}"
        else:
            pr_link = None
        # 将文件修改列表转换为字符串形式，用空格分隔各文件名
        files_changed_str = " ".join(features["files_changed"])
        # 返回生成的 Commit 对象
        return Commit(
            commit_hash,
            category,
            topic,
            features["title"],
            files_changed_str,
            pr_link,
            features["author"],
            a1,
            a2,
            a3,
        )

    @staticmethod
    # 静态方法：重新映射提交的类别名称
    def category_remapper(category: str) -> str:
        # 如果类别在前端类别列表中，则在类别名称后面加上 "_frontend" 后缀
        if category in frontend_categories:
            category = category + "_frontend"
            return category
        # 如果类别是 "Meta API"，则将其映射为 "composability"
        if category == "Meta API":
            category = "composability"
            return category
        # 如果类别在量化模块的类别列表中，则映射为量化模块的通用名称
        if category in common.quantization.categories:
            category = common.quantization.name
            return category
        # 如果类别在分布式模块的类别列表中，则映射为分布式模块的通用名称
        if category in common.distributed.categories:
            category = common.distributed.name
            return category
        # 若无需映射，则直接返回原始类别名称
        return category

    @staticmethod
    # 静态方法：根据标题中的括号类别信息，匹配并返回对应类别
    def bracket_category_matcher(title: str):
        """Categorize a commit based on the presence of a bracketed category in the title.

        Args:
            title (str): title to search

        Returns:
            optional[str]
        """
        # 定义括号类别与对应类别的映射关系列表
        pairs = [
            ("[dynamo]", "dynamo"),
            ("[torchdynamo]", "dynamo"),
            ("[torchinductor]", "inductor"),
            ("[inductor]", "inductor"),
            ("[codemod", "skip"),
            ("[profiler]", "profiler"),
            ("[functorch]", "functorch"),
            ("[autograd]", "autograd_frontend"),
            ("[quantization]", "quantization"),
            ("[nn]", "nn_frontend"),
            ("[complex]", "complex_frontend"),
            ("[mps]", "mps"),
            ("[optimizer]", "optimizer_frontend"),
            ("[xla]", "xla"),
        ]
        # 将标题转换为小写以进行比较
        title_lower = title.lower()
        # 遍历映射关系列表，查找标题中是否包含括号类别信息
        for bracket, category in pairs:
            if bracket in title_lower:
                return category
        # 若标题中无括号类别信息，则返回 None
        return None
    # 获取两个版本之间的提交记录
    def get_commits_between(base_version, new_version):
        # 构建 git 命令，找到两个版本的合并基点
        cmd = f"git merge-base {base_version} {new_version}"
        rc, merge_base, _ = run(cmd)
        # 确保命令成功执行
        assert rc == 0

        # 构建 git 命令，获取从合并基点到新版本的逆序提交记录列表
        cmd = f"git log --reverse --oneline {merge_base}..{new_version}"
        rc, commits, _ = run(cmd)
        # 确保命令成功执行
        assert rc == 0

        # 将提交记录拆分为行列表
        log_lines = commits.split("\n")
        # 解析每行提交记录，分离提交哈希和标题
        hashes, titles = zip(*[log_line.split(" ", 1) for log_line in log_lines])
        # 根据提交哈希列表生成提交对象列表
        return [CommitList.gen_commit(commit_hash) for commit_hash in hashes]

    # 根据类别和主题过滤提交记录
    def filter(self, *, category=None, topic=None):
        commits = self.commits
        # 如果指定了类别，则筛选具有指定类别的提交
        if category is not None:
            commits = [commit for commit in commits if commit.category == category]
        # 如果指定了主题，则筛选具有指定主题的提交
        if topic is not None:
            commits = [commit for commit in commits if commit.topic == topic]
        return commits

    # 更新提交记录至新版本
    def update_to(self, new_version):
        # 获取当前提交记录的最后一个提交哈希
        last_hash = self.commits[-1].commit_hash
        # 获取从最后一个提交哈希到新版本之间的所有提交记录
        new_commits = CommitList.get_commits_between(last_hash, new_version)
        # 将新获取的提交记录追加到当前提交记录列表中
        self.commits += new_commits

    # 统计提交记录的类别和主题数量
    def stat(self):
        # 使用 defaultdict 初始化 counts 字典，以统计类别和主题的提交数
        counts = defaultdict(lambda: defaultdict(int))
        # 遍历所有提交记录，更新类别和主题的计数
        for commit in self.commits:
            counts[commit.category][commit.topic] += 1
        return counts
# 创建新的版本提交列表
def create_new(path, base_version, new_version):
    # 使用 CommitList 类创建一个新的提交列表对象，基于给定的路径、基础版本和新版本
    commits = CommitList.create_new(path, base_version, new_version)
    # 将提交列表对象写入结果
    commits.write_result()


# 更新现有版本的提交列表
def update_existing(path, new_version):
    # 使用 CommitList 类从现有路径创建提交列表对象
    commits = CommitList.from_existing(path)
    # 更新提交列表对象到新版本
    commits.update_to(new_version)
    # 将更新后的提交列表对象写入结果
    commits.write_result()


# 使用新的过滤器重新运行
def rerun_with_new_filters(path):
    # 使用 CommitList 类从现有路径创建当前提交列表对象
    current_commits = CommitList.from_existing(path)
    # 遍历当前提交列表中的提交，进行分类和主题更新
    for i, commit in enumerate(current_commits.commits):
        current_category = commit.category
        # 如果当前分类为 "Uncategorized" 或者不在常见分类列表中，则更新分类和主题
        if (
            current_category == "Uncategorized"
            or current_category not in common.categories
        ):
            # 获取提交数据缓存中的特征项
            feature_item = get_commit_data_cache().get(commit.commit_hash)
            # 将特征项转换成字典形式的特征
            features = features_to_dict(feature_item)
            # 使用 CommitList 类的 categorize 方法对特征进行分类
            category, topic = CommitList.categorize(features)
            # 更新当前提交对象的分类和主题
            current_commits.commits[i] = dataclasses.replace(
                commit, category=category, topic=topic
            )
    # 将更新后的提交列表对象写入结果
    current_commits.write_result()


# 获取提交哈希值或 PR 链接
def get_hash_or_pr_url(commit: Commit):
    # 获取提交对象中的 PR 链接
    pr_link = commit.pr_link
    # 如果 PR 链接为空，则返回提交的哈希值
    if pr_link is None:
        return commit.commit_hash
    else:
        # 使用正则表达式匹配 GitHub PR 链接
        regex = r"https://github.com/pytorch/pytorch/pull/([0-9]+)"
        matches = re.findall(regex, pr_link)
        # 如果匹配结果为空，则返回提交的哈希值
        if len(matches) == 0:
            return commit.commit_hash
        # 否则返回包含 PR 编号的 Markdown 链接
        return f"[#{matches[0]}]({pr_link})"


# 将提交列表对象转换为 Markdown 格式
def to_markdown(commit_list: CommitList, category):
    # 清理提交标题的格式，去除标题中的 PR 编号
    def cleanup_title(commit):
        match = re.match(r"(.*) \(#\d+\)", commit.title)
        if match is None:
            return commit.title
        return match.group(1)

    # 创建合并映射的默认字典
    merge_mapping = defaultdict(list)
    # 遍历提交列表中的提交对象，构建合并映射
    for commit in commit_list.commits:
        if commit.merge_into:
            merge_mapping[commit.merge_into].append(commit)

    # 获取提交数据缓存
    cdc = get_commit_data_cache()
    # 初始化 Markdown 行列表
    lines = [f"\n## {category}\n"]
    # 遍历主题列表
    for topic in topics:
        lines.append(f"### {topic}\n")
        # 根据分类和主题筛选提交列表中的提交对象
        commits = commit_list.filter(category=category, topic=topic)
        # 处理包含下划线的主题
        if "_" in topic:
            commits.extend(
                commit_list.filter(category=category, topic=topic.replace("_", " "))
            )
        # 处理包含空格的主题
        if " " in topic:
            commits.extend(
                commit_list.filter(category=category, topic=topic.replace(" ", "_"))
            )
        # 遍历筛选后的提交对象列表
        for commit in commits:
            # 如果提交被合并，则跳过
            if commit.merge_into:
                continue
            # 获取与当前提交相关的所有提交
            all_related_commits = merge_mapping[commit.commit_hash] + [commit]
            # 构建提交列表的 Markdown 格式
            commit_list_md = ", ".join(
                get_hash_or_pr_url(c) for c in all_related_commits
            )
            result = f"- {cleanup_title(commit)} ({commit_list_md})\n"
            lines.append(result)
    # 返回 Markdown 行列表
    return lines


# 获取 Markdown 头部信息
def get_markdown_header(category):
    # 构建 Markdown 头部信息
    header = f"""
# Release Notes worksheet {category}

The main goal of this process is to rephrase all the commit messages below to make them clear and easy to read by the end user. You should follow the following instructions to do so:
"""
    return header
# 定义主函数，用于创建提交列表工具
def main():
    # 创建参数解析器，描述工具用途
    parser = argparse.ArgumentParser(description="Tool to create a commit list")

    # 创建互斥参数组，只能选择其中一个参数
    group = parser.add_mutually_exclusive_group(required=True)
    # 添加选项参数"--create-new"或"--create_new"，需要两个参数值
    group.add_argument("--create-new", "--create_new", nargs=2)
    # 添加选项参数"--update-to"或"--update_to"，需要一个参数值
    group.add_argument("--update-to", "--update_to")
    
    # 我在添加新的自动分类过滤器时发现这个标志很有用。
    # 第一次运行commitlist.py后，如果在此文件中添加了任何新的过滤器，
    # 使用"rerun_with_new_filters"重新运行将更新现有的commitlist.csv文件，
    # 添加一个参数组到参数解析器，用于重新运行带有新过滤器的操作
    group.add_argument(
        "--rerun-with-new-filters", "--rerun_with_new_filters", action="store_true"
    )
    # 添加一个参数到参数组，用于执行统计操作
    group.add_argument("--stat", action="store_true")
    # 添加一个参数到参数组，用于导出 Markdown 格式的数据
    group.add_argument("--export-markdown", "--export_markdown", action="store_true")
    # 添加一个参数到参数组，用于导出 CSV 格式的数据
    group.add_argument(
        "--export-csv-categories", "--export_csv_categories", action="store_true"
    )
    # 添加一个路径参数到参数解析器，默认为 "results/commitlist.csv"
    parser.add_argument("--path", default="results/commitlist.csv")
    # 解析命令行参数
    args = parser.parse_args()

    # 如果参数 args.create_new 存在，则创建新的提交列表，并保存到指定路径
    if args.create_new:
        create_new(args.path, args.create_new[0], args.create_new[1])
        print(
            "Finished creating new commit list. Results have been saved to results/commitlist.csv"
        )
        return
    # 如果参数 args.update_to 存在，则更新现有的提交列表到指定版本
    if args.update_to:
        update_existing(args.path, args.update_to)
        return
    # 如果参数 args.rerun_with_new_filters 存在，则使用新的过滤器重新运行操作
    if args.rerun_with_new_filters:
        rerun_with_new_filters(args.path)
        return
    # 如果参数 args.stat 存在，则从现有提交列表中获取提交信息，并执行统计操作
    if args.stat:
        commits = CommitList.from_existing(args.path)
        stats = commits.stat()
        pprint.pprint(stats)
        return

    # 如果参数 args.export_csv_categories 存在，则从现有提交列表中获取统计数据的类别，并导出为 CSV 文件
    if args.export_csv_categories:
        commits = CommitList.from_existing(args.path)
        categories = list(commits.stat().keys())
        for category in categories:
            print(f"Exporting {category}...")
            filename = f"results/export/result_{category}.csv"
            CommitList.write_to_disk_static(filename, commits.filter(category=category))
        return

    # 如果参数 args.export_markdown 存在，则从现有提交列表中获取统计数据的类别，并导出为 Markdown 文件
    if args.export_markdown:
        commits = CommitList.from_existing(args.path)
        categories = list(commits.stat().keys())
        for category in categories:
            print(f"Exporting {category}...")
            lines = get_markdown_header(category)
            lines += to_markdown(commits, category)
            filename = f"results/export/result_{category}.md"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                f.writelines(lines)
        return

    # 如果没有任何参数匹配，则抛出断言错误
    raise AssertionError
# 如果这个模块是作为主程序执行（而不是被导入到其他模块中执行），则执行 main() 函数
if __name__ == "__main__":
    main()
```