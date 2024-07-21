# `.\pytorch\scripts\release_notes\test_release_notes.py`

```py
# 导入必要的模块tempfile和unittest

import tempfile
import unittest

# 从commitlist模块中导入CommitList类
from commitlist import CommitList

# 定义测试类TestCommitList，继承自unittest.TestCase
class TestCommitList(unittest.TestCase):

    # 定义测试方法test_create_new，测试创建新的CommitList对象
    def test_create_new(self):
        # 使用临时目录作为上下文管理器，确保测试环境的清理
        with tempfile.TemporaryDirectory() as tempdir:
            # 拼接得到commitlist.csv文件的路径
            commit_list_path = f"{tempdir}/commitlist.csv"
            # 创建新的CommitList对象，并指定版本和初始提交哈希
            commit_list = CommitList.create_new(
                commit_list_path, "v1.5.0", "6000dca5df"
            )
            # 断言CommitList对象中的提交数为33
            self.assertEqual(len(commit_list.commits), 33)
            # 断言第一个提交的提交哈希为"7335f079abb"
            self.assertEqual(commit_list.commits[0].commit_hash, "7335f079abb")
            # 断言第一个提交的标题以"[pt][quant] qmul and qadd"开头
            self.assertTrue(
                commit_list.commits[0].title.startswith("[pt][quant] qmul and qadd")
            )
            # 断言最后一个提交的提交哈希为"6000dca5df6"
            self.assertEqual(commit_list.commits[-1].commit_hash, "6000dca5df6")
            # 断言最后一个提交的标题以"[nomnigraph] Copy device option when customize "开头
            self.assertTrue(
                commit_list.commits[-1].title.startswith(
                    "[nomnigraph] Copy device option when customize "
                )
            )

    # 定义测试方法test_read_write，测试读写CommitList对象
    def test_read_write(self):
        # 使用临时目录作为上下文管理器，确保测试环境的清理
        with tempfile.TemporaryDirectory() as tempdir:
            # 拼接得到commitlist.csv文件的路径
            commit_list_path = f"{tempdir}/commitlist.csv"
            # 创建新的CommitList对象，并指定版本和初始提交哈希
            initial = CommitList.create_new(commit_list_path, "v1.5.0", "7543e7e558")
            # 将初始CommitList对象写入磁盘
            initial.write_to_disk()

            # 从已存在的文件中读取CommitList对象到expected
            expected = CommitList.from_existing(commit_list_path)
            # 修改expected中倒数第二个提交的类别为"foobar"
            expected.commits[-2].category = "foobar"
            # 将修改后的expected对象写入磁盘
            expected.write_to_disk()

            # 从文件中读取CommitList对象到commit_list
            commit_list = CommitList.from_existing(commit_list_path)
            # 逐一比较commit_list和expected中的提交对象是否相等
            for commit, expected_commit in zip(commit_list.commits, expected.commits):
                self.assertEqual(commit, expected_commit)

    # 定义测试方法test_update_to，测试更新CommitList对象至指定提交
    def test_update_to(self):
        # 使用临时目录作为上下文管理器，确保测试环境的清理
        with tempfile.TemporaryDirectory() as tempdir:
            # 拼接得到commitlist.csv文件的路径
            commit_list_path = f"{tempdir}/commitlist.csv"
            # 创建新的CommitList对象，并指定版本和初始提交哈希
            initial = CommitList.create_new(commit_list_path, "v1.5.0", "7543e7e558")
            # 修改initial中倒数第二个提交的类别为"foobar"
            initial.commits[-2].category = "foobar"
            # 断言initial中的提交数为2143
            self.assertEqual(len(initial.commits), 2143)
            # 将修改后的initial对象写入磁盘
            initial.write_to_disk()

            # 从文件中读取CommitList对象到commit_list
            commit_list = CommitList.from_existing(commit_list_path)
            # 更新commit_list至指定提交"5702a28b26"
            commit_list.update_to("5702a28b26")
            # 断言更新后commit_list中的提交数为2143 + 4
            self.assertEqual(len(commit_list.commits), 2143 + 4)
            # 断言commit_list倒数第五个提交与initial的最后一个提交相等
            self.assertEqual(commit_list.commits[-5], initial.commits[-1])

# 当脚本直接执行时，运行所有的测试用例
if __name__ == "__main__":
    unittest.main()
```