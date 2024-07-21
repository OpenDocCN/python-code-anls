# `.\pytorch\scripts\release_notes\categorize.py`

```py
# 导入必要的模块
import argparse  # 用于解析命令行参数
import os  # 提供与操作系统交互的功能
import textwrap  # 提供用于文本包装的实用工具
from pathlib import Path  # 提供处理文件和目录路径的类

import common  # 导入自定义的 common 模块

# 与分类器相关的导入
from classifier import (  # 从 classifier 模块导入以下内容：
    CategoryConfig,  # 分类配置类
    CommitClassifier,  # 提交分类器类
    CommitClassifierInputs,  # 提交分类器输入类
    get_author_map,  # 获取作者映射函数
    get_file_map,  # 获取文件映射函数
    XLMR_BASE,  # XLM-R 模型基本路径
)
from commitlist import CommitList  # 导入 CommitList 类，用于处理提交列表
from common import get_commit_data_cache, topics  # 从 common 模块导入数据缓存函数和主题列表

import torch  # 导入 PyTorch 深度学习框架

# 代码定义了一个 Categorizer 类，用于对提交进行分类
class Categorizer:
    def __init__(self, path, category="Uncategorized", use_classifier: bool = False):
        # 获取提交数据缓存
        self.cache = get_commit_data_cache()
        # 从现有路径创建 CommitList 对象，用于管理提交列表
        self.commits = CommitList.from_existing(path)
        
        # 如果需要使用分类器
        if use_classifier:
            print("Using a classifier to aid with categorization.")
            # 判断是否可以使用 GPU 加速，否则使用 CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # 创建分类器配置
            classifier_config = CategoryConfig(common.categories)
            # 获取作者映射
            author_map = get_author_map(
                Path("results/classifier"), regen_data=False, assert_stored=True
            )
            # 获取文件映射
            file_map = get_file_map(
                Path("results/classifier"), regen_data=False, assert_stored=True
            )
            # 初始化提交分类器对象，并将其移动到适当的设备上
            self.classifier = CommitClassifier(
                XLMR_BASE, author_map, file_map, classifier_config
            ).to(device)
            # 加载预训练好的分类器模型权重
            self.classifier.load_state_dict(
                torch.load(Path("results/classifier/commit_classifier.pt"))
            )
            # 设置分类器为评估模式
            self.classifier.eval()
        else:
            self.classifier = None
        
        # 设置分类器的特定类别，默认为 'Uncategorized'
        # 所有其他类别必须是真实存在的
        self.category = category

    def categorize(self):
        # 根据指定类别过滤提交列表
        commits = self.commits.filter(category=self.category)
        total_commits = len(self.commits.commits)
        already_done = total_commits - len(commits)
        i = 0
        # 遍历符合条件的提交
        while i < len(commits):
            cur_commit = commits[i]
            next_commit = commits[i + 1] if i + 1 < len(commits) else None
            # 处理当前提交，返回可能的跳转位置
            jump_to = self.handle_commit(
                cur_commit, already_done + i + 1, total_commits, commits
            )
            
            # 根据返回值更新计数器
            if jump_to is not None:
                i = jump_to
            elif next_commit is None:
                i = len(commits)
            else:
                i = commits.index(next_commit)

    def features(self, commit):
        # 返回指定提交的特征数据，从缓存中获取
        return self.cache.get(commit.commit_hash)
        def potential_reverts_of(self, commit, commits):
            # 可能的回滚原因关键词列表
            submodule_update_str = [
                "Update TensorPipe submodule",
                "Updating submodules",
                "Automated submodule update",
            ]
            # 如果提交标题包含任一回滚关键词，则返回空列表，表示没有回滚
            if any(a in commit.title for a in submodule_update_str):
                return []

            # 获取当前提交的特征
            features = self.features(commit)
            # 如果特征中标签包含"Reverted"，则设置回滚原因为 GithubBot: Reverted
            if "Reverted" in features.labels:
                reasons = {"GithubBot": "Reverted"}
            else:
                reasons = {}

            # 获取当前提交在提交列表中的索引
            index = commits.index(commit)
            # 去除标题末尾的 " (#35011)" 标记，得到清理后的标题
            cleaned_title = commit.title[:-10]
            # NB: index + 2 是暂时的估计
            # 根据清理后的标题查找可能的回滚提交，更新到原因字典中
            reasons.update(
                {
                    (index + 2 + delta): cand
                    for delta, cand in enumerate(commits[index + 1 :])
                    if cleaned_title in cand.title
                    and commit.commit_hash != cand.commit_hash
                }
            )
            # 返回包含回滚原因的字典
            return reasons

        def handle_commit(self, commit, i, total, commits):
            # 获取可能的回滚原因字典
            potential_reverts = self.potential_reverts_of(commit, commits)
            # 如果存在可能的回滚原因，添加警告信息
            if potential_reverts:
                potential_reverts = f"!!!POTENTIAL REVERTS!!!: {potential_reverts}"
            else:
                potential_reverts = ""

            # 获取当前提交的特征
            features = self.features(commit)
            # 如果分类器存在，则获取提交的作者、修改的文件，并生成分类器输入对象
            if self.classifier is not None:
                # 有些提交可能没有作者信息，则设为"Unknown"
                author = features.author if features.author else "Unknown"
                files = " ".join(features.files_changed)
                # 创建分类器输入对象
                classifier_input = CommitClassifierInputs(
                    title=[features.title], files=[files], author=[author]
                )
                # 使用分类器预测提交的分类
                classifier_category = self.classifier.get_most_likely_category_name(
                    classifier_input
                )[0]

            else:
                # 如果分类器不存在，则使用提交自带的分类信息
                classifier_category = commit.category

            breaking_alarm = ""
            # 如果提交标签包含"module: bc-breaking"，添加断崖式变更警告信息
            if "module: bc-breaking" in features.labels:
                breaking_alarm += "\n!!!!!! BC BREAKING !!!!!!"

            # 如果提交标签包含"module: deprecation"，添加弃用警告信息
            if "module: deprecation" in features.labels:
                breaking_alarm += "\n!!!!!! DEPRECATION !!!!!!"

            # 清空终端窗口内容
            os.system("clear")
            # 生成并返回视图内容，使用了文本缩进处理
            view = textwrap.dedent(
                f"""\
# 定义一个类 Categorizer，用于处理提交分类的工具
class Categorizer:
    # 初始化方法，接受文件路径、默认分类和是否使用分类器作为参数
    def __init__(self, csv_file, default_category, use_classifier):
        # 将文件路径存储到对象属性中
        self.csv_file = csv_file
        # 将默认分类存储到对象属性中
        self.default_category = default_category
        # 将是否使用分类器的标志存储到对象属性中
        self.use_classifier = use_classifier
        # 创建一个 Commits 对象，用于处理提交
        self.commits = Commits(self.csv_file)

    # 分类方法，用于调用分类器进行提交分类
    def categorize(self):
        # 获取所有未分类的提交
        unclassified_commits = self.commits.get_unclassified()
        # 如果没有未分类的提交，直接返回
        if not unclassified_commits:
            print("No unclassified commits found.")
            return
        
        # 输出分类器的视图，显示当前的分类和主题
        print(f"[{i}/{total}]")
        print("=" * 80)
        print(features.title)
        print(f"{potential_reverts} {breaking_alarm}")
        print(features.body)
        print(f"Files changed: {features.files_changed}")
        print(f"Labels: {features.labels}")
        print(f"Current category: {commit.category}")
        print(f"Select from: {', '.join(common.categories)}")
        
        # 进入分类器交互循环，直到选择有效的分类
        cat_choice = None
        while cat_choice is None:
            print("Enter category: ")
            value = input(f"{classifier_category} ").strip()
            if len(value) == 0:
                # 用户按下回车并接受默认值
                cat_choice = classifier_category
                continue
            # 查找以输入值开头的所有分类选项
            choices = [cat for cat in common.categories if cat.startswith(value)]
            if len(choices) != 1:
                # 如果找到多个匹配项，则提示用户重新输入
                print(f"Possible matches: {choices}, try again")
                continue
            # 确认唯一的匹配项作为选择的分类
            cat_choice = choices[0]
        
        print(f"\nSelected: {cat_choice}")
        print(f"\nCurrent topic: {commit.topic}")
        print(f"Select from: {', '.join(topics)}")
        
        # 进入主题选择循环，直到选择有效的主题
        topic_choice = None
        while topic_choice is None:
            value = input("topic> ").strip()
            if len(value) == 0:
                topic_choice = commit.topic
                continue
            # 查找以输入值开头的所有主题选项
            choices = [cat for cat in topics if cat.startswith(value)]
            if len(choices) != 1:
                # 如果找到多个匹配项，则提示用户重新输入
                print(f"Possible matches: {choices}, try again")
                continue
            # 确认唯一的匹配项作为选择的主题
            topic_choice = choices[0]
        
        # 更新提交对象的分类和主题属性
        self.update_commit(commit, cat_choice, topic_choice)
        return None

    # 更新提交方法，用于将分类和主题应用到提交对象
    def update_commit(self, commit, category, topic):
        # 断言所选分类在通用分类列表中
        assert category in common.categories
        # 断言所选主题在主题列表中
        assert topic in topics
        # 更新提交对象的分类属性
        commit.category = category
        # 更新提交对象的主题属性
        commit.topic = topic
        # 将更新后的结果写入提交对象
        self.commits.write_result()


# 主函数，用于解析命令行参数并执行分类工具
def main():
    parser = argparse.ArgumentParser(description="Tool to help categorize commits")
    # 添加命令行参数 --category，用于指定默认分类
    parser.add_argument(
        "--category",
        type=str,
        default="Uncategorized",
        help='Which category to filter by. "Uncategorized", None, or a category name',
    )
    # 添加命令行参数 --file，用于指定提交记录 CSV 文件的位置
    parser.add_argument(
        "--file",
        help="The location of the commits CSV",
        default="results/commitlist.csv",
    )
    # 添加命令行参数 --use_classifier，用于启用或禁用分类器辅助分类
    parser.add_argument(
        "--use_classifier",
        action="store_true",
        help="Whether or not to use a classifier to aid in categorization.",
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 创建分类器对象，并传入命令行参数
    categorizer = Categorizer(args.file, args.category, args.use_classifier)
    # 调用分类方法开始执行分类流程
    categorizer.categorize()


# 如果当前脚本作为主程序运行，则调用主函数开始执行程序
if __name__ == "__main__":
    main()
```