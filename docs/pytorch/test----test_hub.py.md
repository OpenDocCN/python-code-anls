# `.\pytorch\test\test_hub.py`

```py
# Owner(s): ["module: hub"]

# 导入所需的库和模块
import os
import tempfile
import unittest
import warnings
from unittest.mock import patch

# 导入PyTorch相关模块
import torch
import torch.hub as hub
from torch.testing._internal.common_utils import IS_SANDCASTLE, retry, TestCase

# 定义函数：计算给定state_dict中所有张量元素的总和
def sum_of_state_dict(state_dict):
    s = 0
    for v in state_dict.values():
        s += v.sum()
    return s

# 预定义常量：hub模型state_dict所有元素的总和
SUM_OF_HUB_EXAMPLE = 431080

# 预定义常量：torch hub示例模型的发布地址
TORCHHUB_EXAMPLE_RELEASE_URL = (
    "https://github.com/ailzhang/torchhub_example/releases/download/0.1/mnist_init_ones"
)

# 跳过测试条件：如果在Sandcastle环境下，无法访问外部资源
@unittest.skipIf(IS_SANDCASTLE, "Sandcastle cannot ping external")
class TestHub(TestCase):
    def setUp(self):
        super().setUp()
        # 设置测试环境：保存之前的hub目录，创建临时目录作为hub目录
        self.previous_hub_dir = torch.hub.get_dir()
        self.tmpdir = tempfile.TemporaryDirectory("hub_dir")
        torch.hub.set_dir(self.tmpdir.name)
        # 设置受信任列表的路径
        self.trusted_list_path = os.path.join(torch.hub.get_dir(), "trusted_list")

    def tearDown(self):
        super().tearDown()
        # 清理测试环境：恢复之前的hub目录设置，清理临时目录
        torch.hub.set_dir(self.previous_hub_dir)  # 可能不需要，但无妨清理
        self.tmpdir.cleanup()

    # 断言受信任列表为空的辅助方法
    def _assert_trusted_list_is_empty(self):
        with open(self.trusted_list_path) as f:
            assert not f.readlines()

    # 断言受信任列表中包含特定行的辅助方法
    def _assert_in_trusted_list(self, line):
        with open(self.trusted_list_path) as f:
            assert line in (l.strip() for l in f)

    # 带重试功能的测试方法：从GitHub加载模型
    @retry(Exception, tries=3)
    def test_load_from_github(self):
        hub_model = hub.load(
            "ailzhang/torchhub_example",
            "mnist",
            source="github",
            pretrained=True,
            verbose=False,
        )
        # 断言加载的模型state_dict的元素总和与预定义常量相等
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE)

    # 带重试功能的测试方法：从本地目录加载模型
    @retry(Exception, tries=3)
    def test_load_from_local_dir(self):
        local_dir = hub._get_cache_or_reload(
            "ailzhang/torchhub_example",
            force_reload=False,
            trust_repo=True,
            calling_fn=None,
        )
        hub_model = hub.load(
            local_dir, "mnist", source="local", pretrained=True, verbose=False
        )
        # 断言加载的模型state_dict的元素总和与预定义常量相等
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE)

    # 带重试功能的测试方法：从指定分支加载模型
    @retry(Exception, tries=3)
    def test_load_from_branch(self):
        hub_model = hub.load(
            "ailzhang/torchhub_example:ci/test_slash",
            "mnist",
            pretrained=True,
            verbose=False,
        )
        # 断言加载的模型state_dict的元素总和与预定义常量相等
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE)

    # 带重试功能的测试方法：测试加载模型的鲁棒性
    @retry(Exception, tries=3)
    # 定义测试用例，验证 torch.hub.get_dir() 的行为
    def test_get_set_dir(self):
        # 获取之前的 hub 目录
        previous_hub_dir = torch.hub.get_dir()
        # 创建临时目录作为 hub 目录
        with tempfile.TemporaryDirectory("hub_dir") as tmpdir:
            # 设置当前 hub 目录为临时目录
            torch.hub.set_dir(tmpdir)
            # 断言当前 hub 目录与设置的临时目录相同
            self.assertEqual(torch.hub.get_dir(), tmpdir)
            # 断言之前的 hub 目录与临时目录不相同
            self.assertNotEqual(previous_hub_dir, tmpdir)

            # 使用 torch.hub 加载模型
            hub_model = hub.load(
                "ailzhang/torchhub_example", "mnist", pretrained=True, verbose=False
            )
            # 断言加载的模型参数的总和与预期值相等
            self.assertEqual(
                sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE
            )
            # 断言临时目录中存在指定的文件夹
            assert os.path.exists(
                os.path.join(tmpdir, "ailzhang_torchhub_example_master")
            )

        # 测试 set_dir 方法是否正确调用了 expanduser()
        # 针对 https://github.com/pytorch/pytorch/issues/69761 的非回归测试
        new_dir = os.path.join("~", "hub")
        # 设置 hub 目录为新的目录
        torch.hub.set_dir(new_dir)
        # 断言当前 hub 目录已经被 expanduser 处理过
        self.assertEqual(torch.hub.get_dir(), os.path.expanduser(new_dir))

    # 带重试机制的测试函数，验证 hub.list 方法
    @retry(Exception, tries=3)
    def test_list_entrypoints(self):
        # 调用 hub.list 方法获取模型入口列表
        entry_lists = hub.list("ailzhang/torchhub_example", trust_repo=True)
        # 断言 "mnist" 在入口列表中
        self.assertObjectIn("mnist", entry_lists)

    # 带重试机制的测试函数，验证 hub.download_url_to_file 方法
    @retry(Exception, tries=3)
    def test_download_url_to_file(self):
        # 使用临时目录作为下载文件的存储位置
        with tempfile.TemporaryDirectory() as tmpdir:
            f = os.path.join(tmpdir, "temp")
            # 下载文件到指定路径 f，不显示进度条
            hub.download_url_to_file(TORCHHUB_EXAMPLE_RELEASE_URL, f, progress=False)
            # 加载下载的文件内容
            loaded_state = torch.load(f)
            # 断言加载的状态字典的总和与预期值相等
            self.assertEqual(sum_of_state_dict(loaded_state), SUM_OF_HUB_EXAMPLE)
            # 检查下载的文件是否具有默认的文件权限
            f_ref = os.path.join(tmpdir, "reference")
            open(f_ref, "w").close()
            expected_permissions = oct(os.stat(f_ref).st_mode & 0o777)
            actual_permissions = oct(os.stat(f).st_mode & 0o777)
            assert actual_permissions == expected_permissions

    # 带重试机制的测试函数，验证 hub.load_state_dict_from_url 方法
    @retry(Exception, tries=3)
    def test_load_state_dict_from_url(self):
        # 从 URL 加载状态字典
        loaded_state = hub.load_state_dict_from_url(TORCHHUB_EXAMPLE_RELEASE_URL)
        # 断言加载的状态字典的总和与预期值相等
        self.assertEqual(sum_of_state_dict(loaded_state), SUM_OF_HUB_EXAMPLE)

        # 指定文件名加载状态字典
        file_name = "the_file_name"
        loaded_state = hub.load_state_dict_from_url(
            TORCHHUB_EXAMPLE_RELEASE_URL, file_name=file_name
        )
        # 断言在指定的目录下存在预期的文件路径
        expected_file_path = os.path.join(torch.hub.get_dir(), "checkpoints", file_name)
        self.assertTrue(os.path.exists(expected_file_path))
        # 断言加载的状态字典的总和与预期值相等
        self.assertEqual(sum_of_state_dict(loaded_state), SUM_OF_HUB_EXAMPLE)

        # 仅加载权重部分的安全加载状态字典
        loaded_state = hub.load_state_dict_from_url(
            TORCHHUB_EXAMPLE_RELEASE_URL, weights_only=True
        )
        # 断言加载的状态字典的总和与预期值相等
        self.assertEqual(sum_of_state_dict(loaded_state), SUM_OF_HUB_EXAMPLE)
    # 定义测试函数，用于加载遗留的 ZIP 格式检查点
    def test_load_legacy_zip_checkpoint(self):
        # 捕获警告信息
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # 设置警告的过滤器为始终显示
            # 使用 torch hub 加载预训练模型 "ailzhang/torchhub_example" 的 "mnist_zip" 部分
            hub_model = hub.load(
                "ailzhang/torchhub_example", "mnist_zip", pretrained=True, verbose=False
            )
            # 断言模型状态字典的所有值的和是否等于预定义常量 SUM_OF_HUB_EXAMPLE
            self.assertEqual(
                sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE
            )
            # 断言是否有任何警告信息包含特定文本，提示将来会弃用默认的 zipfile 格式
            assert any(
                "will be deprecated in favor of default zipfile" in str(w) for w in ws
            )

    # 测试由 >=1.6 发行版生成的默认 zipfile 序列化格式
    @retry(Exception, tries=3)
    def test_load_zip_1_6_checkpoint(self):
        # 使用 torch hub 加载预训练模型 "ailzhang/torchhub_example" 的 "mnist_zip_1_6" 部分
        hub_model = hub.load(
            "ailzhang/torchhub_example",
            "mnist_zip_1_6",
            pretrained=True,
            verbose=False,
            trust_repo=True,
        )
        # 断言模型状态字典的所有值的和是否等于预定义常量 SUM_OF_HUB_EXAMPLE
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE)

    # 测试解析 torch hub 仓库信息的函数
    @retry(Exception, tries=3)
    def test_hub_parse_repo_info(self):
        # 如果指定了分支，则解析输入并返回相应的元组
        self.assertEqual(torch.hub._parse_repo_info("a/b:c"), ("a", "b", "c"))
        # 对于 torchvision，默认分支是 main
        self.assertEqual(
            torch.hub._parse_repo_info("pytorch/vision"), ("pytorch", "vision", "main")
        )
        # 对于 torchhub_example 仓库，默认分支仍然是 master
        self.assertEqual(
            torch.hub._parse_repo_info("ailzhang/torchhub_example"),
            ("ailzhang", "torchhub_example", "master"),
        )

    # 测试从 forked 仓库加载提交的情况
    @retry(Exception, tries=3)
    def test_load_commit_from_forked_repo(self):
        # 断言加载来自 forked 仓库的提交时是否引发 ValueError 异常
        with self.assertRaisesRegex(ValueError, "If it's a commit from a forked repo"):
            torch.hub.load("pytorch/vision:4e2c216", "resnet18")

    # 测试在 trust_repo 设置为 False 时的情况：输入为空字符串
    @retry(Exception, tries=3)
    @patch("builtins.input", return_value="")
    def test_trust_repo_false_emptystring(self, patched_input):
        # 断言加载来自 "ailzhang/torchhub_example" 的 "mnist_zip_1_6" 部分时是否引发异常
        with self.assertRaisesRegex(Exception, "Untrusted repository."):
            torch.hub.load(
                "ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=False
            )
        # 断言调用输入函数一次
        self._assert_trusted_list_is_empty()
        patched_input.assert_called_once()

        # 重置 mock 对象
        patched_input.reset_mock()
        # 再次断言加载来自 "ailzhang/torchhub_example" 的 "mnist_zip_1_6" 部分时是否引发异常
        with self.assertRaisesRegex(Exception, "Untrusted repository."):
            torch.hub.load(
                "ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=False
            )
        # 再次断言调用输入函数一次
        self._assert_trusted_list_is_empty()
        patched_input.assert_called_once()

    # 测试在 trust_repo 设置为 False 时的情况：输入为 "no"
    @retry(Exception, tries=3)
    @patch("builtins.input", return_value="no")
    def test_trust_repo_false_no(self, patched_input):
        # 使用断言检查是否抛出异常，并验证异常信息为 "Untrusted repository."
        with self.assertRaisesRegex(Exception, "Untrusted repository."):
            # 使用 torch.hub.load 加载名为 "mnist_zip_1_6" 的模型，设置 trust_repo=False 表示不信任仓库
            torch.hub.load(
                "ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=False
            )
        # 调用自定义方法验证 trusted_list 是否为空
        self._assert_trusted_list_is_empty()
        # 断言 patched_input 方法被调用了一次
        patched_input.assert_called_once()

        # 重置 patched_input 方法的 mock 对象
        patched_input.reset_mock()
        # 再次使用断言检查是否抛出异常，并验证异常信息为 "Untrusted repository."
        with self.assertRaisesRegex(Exception, "Untrusted repository."):
            # 再次尝试加载 "mnist_zip_1_6" 模型，设置 trust_repo=False 表示不信任仓库
            torch.hub.load(
                "ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=False
            )
        # 再次调用自定义方法验证 trusted_list 是否为空
        self._assert_trusted_list_is_empty()
        # 断言 patched_input 方法被调用了一次
        patched_input.assert_called_once()

    @retry(Exception, tries=3)
    @patch("builtins.input", return_value="y")
    def test_trusted_repo_false_yes(self, patched_input):
        # 加载 "mnist_zip_1_6" 模型，设置 trust_repo=False 表示不信任仓库
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=False)
        # 调用自定义方法验证 "ailzhang_torchhub_example" 是否在 trusted_list 中
        self._assert_in_trusted_list("ailzhang_torchhub_example")
        # 断言 patched_input 方法被调用了一次
        patched_input.assert_called_once()

        # 加载第二次时使用 "check"，不应该要求用户输入
        patched_input.reset_mock()
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo="check")
        # 断言 patched_input 方法未被调用
        patched_input.assert_not_called()

        # 再次使用 False 加载，仍然应该要求用户输入
        patched_input.reset_mock()
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=False)
        # 断言 patched_input 方法被调用了一次
        patched_input.assert_called_once()

    @retry(Exception, tries=3)
    @patch("builtins.input", return_value="no")
    def test_trust_repo_check_no(self, patched_input):
        # 使用断言检查是否抛出异常，并验证异常信息为 "Untrusted repository."
        with self.assertRaisesRegex(Exception, "Untrusted repository."):
            # 使用 torch.hub.load 加载 "mnist_zip_1_6" 模型，设置 trust_repo="check" 表示检查仓库
            torch.hub.load(
                "ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo="check"
            )
        # 调用自定义方法验证 trusted_list 是否为空
        self._assert_trusted_list_is_empty()
        # 断言 patched_input 方法被调用了一次
        patched_input.assert_called_once()

        # 重置 patched_input 方法的 mock 对象
        patched_input.reset_mock()
        # 再次使用断言检查是否抛出异常，并验证异常信息为 "Untrusted repository."
        with self.assertRaisesRegex(Exception, "Untrusted repository."):
            # 再次尝试加载 "mnist_zip_1_6" 模型，设置 trust_repo="check" 表示检查仓库
            torch.hub.load(
                "ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo="check"
            )
        # 断言 patched_input 方法被调用了一次
        patched_input.assert_called_once()

    @retry(Exception, tries=3)
    @patch("builtins.input", return_value="y")
    def test_trust_repo_check_yes(self, patched_input):
        # 加载 "mnist_zip_1_6" 模型，设置 trust_repo="check" 表示检查仓库
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo="check")
        # 调用自定义方法验证 "ailzhang_torchhub_example" 是否在 trusted_list 中
        self._assert_in_trusted_list("ailzhang_torchhub_example")
        # 断言 patched_input 方法被调用了一次
        patched_input.assert_called_once()

        # 加载第二次时使用 "check"，不应该要求用户输入
        patched_input.reset_mock()
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo="check")
        # 断言 patched_input 方法未被调用
        patched_input.assert_not_called()

    @retry(Exception, tries=3)
    def test_trust_repo_true(self):
        # 加载 "mnist_zip_1_6" 模型，设置 trust_repo=True 表示信任仓库
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=True)
        # 调用自定义方法验证 "ailzhang_torchhub_example" 是否在 trusted_list 中
        self._assert_in_trusted_list("ailzhang_torchhub_example")

    @retry(Exception, tries=3)
    # 定义一个测试方法，用于验证加载内置的受信任所有者仓库
    def test_trust_repo_builtin_trusted_owners(self):
        # 使用 torch.hub.load() 方法加载 pytorch/vision 仓库中的 resnet18 模型，
        # 并指定 trust_repo 参数为 "check"，表示检查仓库的受信任状态
        torch.hub.load("pytorch/vision", "resnet18", trust_repo="check")
        # 调用内部方法 _assert_trusted_list_is_empty()，验证受信任列表为空

    # 使用装饰器 retry(Exception, tries=3) 标记的测试方法，用于验证 trust_repo 参数为 None 的情况
    def test_trust_repo_none(self):
        # 使用 warnings.catch_warnings() 捕获警告信息
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # 设置警告过滤器为 always，表示始终记录警告
            # 使用 torch.hub.load() 方法加载 "ailzhang/torchhub_example" 仓库中的 mnist_zip_1_6 模型，
            # 并指定 trust_repo 参数为 None，表示不信任该仓库
            torch.hub.load(
                "ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=None
            )
            # 断言捕获的警告数量为 1
            assert len(w) == 1
            # 断言最后一条警告是 UserWarning 类型的子类
            assert issubclass(w[-1].category, UserWarning)
            # 断言警告消息中包含特定文本，提示从未信任的仓库下载和运行代码
            assert (
                "You are about to download and run code from an untrusted repository"
                in str(w[-1].message)
            )

        # 调用内部方法 _assert_trusted_list_is_empty()，验证受信任列表为空

    # 使用装饰器 retry(Exception, tries=3) 标记的测试方法，用于验证 trust_repo 参数为 True 的情况
    def test_trust_repo_legacy(self):
        # 首先加载 "ailzhang/torchhub_example" 仓库中的 mnist_zip_1_6 模型，
        # 并指定 trust_repo 参数为 True，表示信任该仓库
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=True)
        
        # 删除 self.trusted_list_path 文件，模拟删除允许列表文件
        os.remove(self.trusted_list_path)
        
        # 再次加载 "ailzhang/torchhub_example" 仓库中的 mnist_zip_1_6 模型，
        # 并指定 trust_repo 参数为 "check"，表示检查仓库的受信任状态
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo="check")
        
        # 调用内部方法 _assert_trusted_list_is_empty()，验证受信任列表为空
```