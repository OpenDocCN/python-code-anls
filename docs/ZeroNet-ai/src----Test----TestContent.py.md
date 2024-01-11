# `ZeroNet\src\Test\TestContent.py`

```
# 导入所需的模块
import json
import time
import io

import pytest

# 从 Crypt 模块中导入 CryptBitcoin 类
from Crypt import CryptBitcoin
# 从 Content.ContentManager 模块中导入 VerifyError 和 SignError 类
from Content.ContentManager import VerifyError, SignError
# 从 util.SafeRe 模块中导入 UnsafePatternError 类

# 使用 pytest 的 usefixtures 装饰器，重置设置
@pytest.mark.usefixtures("resetSettings")
class TestContent:
    # 设置私钥
    privatekey = "5KUh3PvNm5HUWoCfSUfcYvfQ2g3PrRNJWr6Q9eqdBGu23mtMntv"

    # 定义测试用例 testInclude，接受 site 参数
    def testInclude(self, site):
        # 从父级 content.json 中获取规则
        rules = site.content_manager.getRules("data/test_include/content.json")

        # 断言验证规则中的签名者
        assert rules["signers"] == ["15ik6LeBWnACWfaika1xqGapRZ1zh3JpCo"]  # 有效的签名者
        assert rules["user_name"] == "test"  # 额外数据
        assert rules["max_size"] == 20000  # 文件的最大大小
        assert not rules["includes_allowed"]  # 不允许更多的包含
        assert rules["files_allowed"] == "data.json"  # 允许的文件模式

        # 获取 "data/test_include/content.json" 的有效签名者
        valid_signers = site.content_manager.getValidSigners("data/test_include/content.json")
        assert "15ik6LeBWnACWfaika1xqGapRZ1zh3JpCo" in valid_signers  # 在父级 content.json 中定义的额外有效签名者
        assert "1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT" in valid_signers  # 站点本身
        assert len(valid_signers) == 2  # 没有更多的有效签名者

        # 获取 "data/users/content.json" 的有效签名者
        valid_signers = site.content_manager.getValidSigners("data/users/content.json")
        assert "1LSxsKfC9S9TVXGGNSM3vPHjyW82jgCX5f" in valid_signers  # 在父级 content.json 中定义的额外有效签名者
        assert "1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT" in valid_signers  # 站点本身
        assert len(valid_signers) == 2

        # 获取根 content.json 的有效签名者
        assert site.content_manager.getValidSigners("content.json") == ["1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT"]

    # 使用 pytest 的 parametrize 装饰器，传入参数列表 ["content.json", "data/test_include/content.json", "data/users/content.json"]
    @pytest.mark.parametrize("inner_path", ["content.json", "data/test_include/content.json", "data/users/content.json"])
    # 测试签名功能，验证私钥是否有效
    def testSign(self, site, inner_path):
        # 使用错误的私钥进行签名，预期会抛出 SignError 异常
        with pytest.raises(SignError) as err:
            site.content_manager.sign(inner_path, privatekey="5aaa3PvNm5HUWoCfSUfcYvfQ2g3PrRNJWr6Q9eqdBGu23mtMnaa", filewrite=False)
        assert "Private key invalid" in str(err.value)

        # 使用正确的私钥进行签名
        content = site.content_manager.sign(inner_path, privatekey=self.privatekey, filewrite=False)
        content_old = site.content_manager.contents[inner_path]  # 签名前的内容
        assert not content_old == content  # 时间戳已更改
        assert site.address in content["signs"]  # 使用站点的私钥进行签名
        if inner_path == "content.json":
            assert len(content["files"]) == 17
        elif inner_path == "data/test-include/content.json":
            assert len(content["files"]) == 1
        elif inner_path == "data/users/content.json":
            assert len(content["files"]) == 0

        # 除了修改时间戳和签名之外，其他内容应该与之前一样
        assert (
            {key: val for key, val in content_old.items() if key not in ["modified", "signs", "sign", "zeronet_version"]}
            ==
            {key: val for key, val in content.items() if key not in ["modified", "signs", "sign", "zeronet_version"]}
        )
    # 测试签名可选文件的方法，传入站点对象
    def testSignOptionalFiles(self, site):
        # 遍历站点内容管理器中的哈希字段列表，并移除所有哈希字段
        for hash in list(site.content_manager.hashfield):
            site.content_manager.hashfield.remove(hash)

        # 断言站点内容管理器中的哈希字段列表长度为0
        assert len(site.content_manager.hashfield) == 0

        # 将content.json文件的optional字段设置为"((data/img/zero.*))"，并进行签名
        content_optional = site.content_manager.sign(privatekey=self.privatekey, filewrite=False, remove_missing_optional=True)

        # 删除content.json文件的optional字段，并进行签名
        del site.content_manager.contents["content.json"]["optional"]
        content_nooptional = site.content_manager.sign(privatekey=self.privatekey, filewrite=False, remove_missing_optional=True)

        # 断言如果没有匹配模式，则没有可选文件
        assert len(content_nooptional.get("files_optional", {})) == 0
        # 断言有可选文件
        assert len(content_optional["files_optional"]) > 0
        # 断言站点内容管理器中的哈希字段列表长度等于content_optional中的可选文件数量
        assert len(site.content_manager.hashfield) == len(content_optional["files_optional"])
        # 断言没有可选文件时，content_nooptional中的文件数量大于content_optional中的文件数量
        assert len(content_nooptional["files"]) > len(content_optional["files"])
    # 测试文件信息的方法，接受一个站点对象作为参数
    def testFileInfo(self, site):
        # 断言 index.html 文件中包含 "sha512" 信息
        assert "sha512" in site.content_manager.getFileInfo("index.html")
        # 断言 data/img/domain.png 文件的 content_inner_path 为 "content.json"
        assert site.content_manager.getFileInfo("data/img/domain.png")["content_inner_path"] == "content.json"
        # 断言 data/users/hello.png 文件的 content_inner_path 为 "data/users/content.json"
        assert site.content_manager.getFileInfo("data/users/hello.png")["content_inner_path"] == "data/users/content.json"
        # 断言 data/users/content.json 文件的 content_inner_path 为 "data/users/content.json"
        assert site.content_manager.getFileInfo("data/users/content.json")["content_inner_path"] == "data/users/content.json"
        # 断言不存在的文件返回空值
        assert not site.content_manager.getFileInfo("notexist")

        # 可选文件
        file_info_optional = site.content_manager.getFileInfo("data/optional.txt")
        # 断言可选文件中包含 "sha512" 信息
        assert "sha512" in file_info_optional
        # 断言可选文件的 optional 属性为 True
        assert file_info_optional["optional"] is True

        # 不存在的用户 content.json 文件
        assert "cert_signers" in site.content_manager.getFileInfo("data/users/unknown/content.json")

        # 可选用户文件
        file_info_optional = site.content_manager.getFileInfo("data/users/1CjfbrbwtP8Y2QjPy12vpTATkUT7oSiPQ9/peanut-butter-jelly-time.gif")
        # 断言可选用户文件中包含 "sha512" 信息
        assert "sha512" in file_info_optional
        # 断言可选用户文件的 optional 属性为 True
        assert file_info_optional["optional"] is True
    # 定义测试函数，用于验证内部路径
    def testVerifyInnerPath(self, site, crypt_bitcoin_lib):
        # 定义内部路径
        inner_path = "content.json"
        # 从站点存储中加载 JSON 数据
        data_dict = site.storage.loadJson(inner_path)

        # 遍历好的相对路径列表
        for good_relative_path in ["data.json", "out/data.json", "Any File [by none] (1).jpg", "árvzítűrő/tükörfúrógép.txt"]:
            # 更新数据字典中的文件信息
            data_dict["files"] = {good_relative_path: {"sha512": "369d4e780cc80504285f13774ca327fe725eed2d813aad229e62356b07365906", "size": 505}}

            # 如果数据字典中存在 "sign" 键，则删除
            if "sign" in data_dict:
                del data_dict["sign"]
            # 删除数据字典中的 "signs" 键
            del data_dict["signs"]
            # 更新数据字典中的 "signs" 键
            data_dict["signs"] = {
                "1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT": CryptBitcoin.sign(json.dumps(data_dict, sort_keys=True), self.privatekey)
            }
            # 将数据字典转换为字节流
            data = io.BytesIO(json.dumps(data_dict).encode())
            # 断言站点内容管理器验证文件，不忽略相同文件
            assert site.content_manager.verifyFile(inner_path, data, ignore_same=False)

        # 遍历坏的相对路径列表
        for bad_relative_path in ["../data.json", "data/" * 100, "invalid|file.jpg", "con.txt", "any/con.txt"]:
            # 更新数据字典中的文件信息
            data_dict["files"] = {bad_relative_path: {"sha512": "369d4e780cc80504285f13774ca327fe725eed2d813aad229e62356b07365906", "size": 505}}

            # 如果数据字典中存在 "sign" 键，则删除
            if "sign" in data_dict:
                del data_dict["sign"]
            # 删除数据字典中的 "signs" 键
            del data_dict["signs"]
            # 更新数据字典中的 "signs" 键
            data_dict["signs"] = {
                "1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT": CryptBitcoin.sign(json.dumps(data_dict, sort_keys=True), self.privatekey)
            }
            # 将数据字典转换为字节流
            data = io.BytesIO(json.dumps(data_dict).encode())
            # 使用 pytest 断言捕获 VerifyError 异常
            with pytest.raises(VerifyError) as err:
                site.content_manager.verifyFile(inner_path, data, ignore_same=False)
            # 断言错误信息中包含 "Invalid relative path"
            assert "Invalid relative path" in str(err.value)

    # 使用 pytest.mark.parametrize 标记参数化测试
    @pytest.mark.parametrize("key", ["ignore", "optional"])
    # 测试不安全的模式签名，将指定的键值对应的内容设置为正则表达式模式
    def testSignUnsafePattern(self, site, key):
        site.content_manager.contents["content.json"][key] = "([a-zA-Z]+)*"
        # 使用 pytest 断言检查是否会引发 UnsafePatternError 异常
        with pytest.raises(UnsafePatternError) as err:
            # 对 content.json 进行签名，使用私钥进行签名，但不写入文件
            site.content_manager.sign("content.json", privatekey=self.privatekey, filewrite=False)
        # 使用 pytest 断言检查异常信息中是否包含 "Potentially unsafe"
        assert "Potentially unsafe" in str(err.value)
    
    # 测试不安全的模式验证，设置指定文件的允许文件名模式为正则表达式模式
    def testVerifyUnsafePattern(self, site, crypt_bitcoin_lib):
        site.content_manager.contents["content.json"]["includes"]["data/test_include/content.json"]["files_allowed"] = "([a-zA-Z]+)*"
        # 使用 pytest 断言检查是否会引发 UnsafePatternError 异常
        with pytest.raises(UnsafePatternError) as err:
            # 打开指定文件，验证文件内容，不忽略相同内容
            with site.storage.open("data/test_include/content.json") as data:
                site.content_manager.verifyFile("data/test_include/content.json", data, ignore_same=False)
        # 使用 pytest 断言检查异常信息中是否包含 "Potentially unsafe"
        assert "Potentially unsafe" in str(err.value)
    
        # 设置用户内容中的权限规则的文件名模式为正则表达式模式
        site.content_manager.contents["data/users/content.json"]["user_contents"]["permission_rules"]["([a-zA-Z]+)*"] = {"max_size": 0}
        # 使用 pytest 断言检查是否会引发 UnsafePatternError 异常
        with pytest.raises(UnsafePatternError) as err:
            # 打开指定文件，验证文件内容，不忽略相同内容
            with site.storage.open("data/users/1C5sgvWaSgfaTpV5kjBCnCiKtENNMYo69q/content.json") as data:
                site.content_manager.verifyFile("data/users/1C5sgvWaSgfaTpV5kjBCnCiKtENNMYo69q/content.json", data, ignore_same=False)
        # 使用 pytest 断言检查异常信息中是否包含 "Potentially unsafe"
        assert "Potentially unsafe" in str(err.value)
    # 测试路径验证函数，接受一个站点对象作为参数
    def testPathValidation(self, site):
        # 断言给定的路径是有效的相对路径
        assert site.content_manager.isValidRelativePath("test.txt")
        # 断言给定的路径是有效的相对路径
        assert site.content_manager.isValidRelativePath("test/!@#$%^&().txt")
        # 断言给定的路径是有效的相对路径
        assert site.content_manager.isValidRelativePath("ÜøßÂŒƂÆÇ.txt")
        # 断言给定的路径是有效的相对路径
        assert site.content_manager.isValidRelativePath("тест.текст")
        # 断言给定的路径是有效的相对路径
        assert site.content_manager.isValidRelativePath("𝐮𝐧𝐢𝐜𝐨𝐝𝐞𝑖𝑠𝒂𝒆𝒔𝒐𝒎𝒆")

        # 根据 https://stackoverflow.com/questions/1976007/what-characters-are-forbidden-in-windows-and-linux-directory-names 中的规则进行测试

        # 断言给定的路径是无效的相对路径
        assert not site.content_manager.isValidRelativePath("any\\hello.txt")  # \ not allowed
        # 断言给定的路径是无效的相对路径
        assert not site.content_manager.isValidRelativePath("/hello.txt")  # Cannot start with /
        # 断言给定的路径是无效的相对路径
        assert not site.content_manager.isValidRelativePath("\\hello.txt")  # Cannot start with \
        # 断言给定的路径是无效的相对路径
        assert not site.content_manager.isValidRelativePath("../hello.txt")  # Not allowed .. in path
        # 断言给定的路径是无效的相对路径
        assert not site.content_manager.isValidRelativePath("\0hello.txt")  # NULL character
        # 断言给定的路径是无效的相对路径
        assert not site.content_manager.isValidRelativePath("\31hello.txt")  # 0-31 (ASCII control characters)
        # 断言给定的路径是无效的相对路径
        assert not site.content_manager.isValidRelativePath("any/hello.txt ")  # Cannot end with space
        # 断言给定的路径是无效的相对路径
        assert not site.content_manager.isValidRelativePath("any/hello.txt.")  # Cannot end with dot
        # 断言给定的路径是有效的相对路径
        assert site.content_manager.isValidRelativePath(".hello.txt")  # Allow start with dot
        # 断言给定的路径是无效的相对路径
        assert not site.content_manager.isValidRelativePath("any/CON")  # Protected names on Windows
        # 断言给定的路径是无效的相对路径
        assert not site.content_manager.isValidRelativePath("CON/any.txt")
        # 断言给定的路径是无效的相对路径
        assert not site.content_manager.isValidRelativePath("any/lpt1.txt")
        # 断言给定的路径是有效的相对路径
        assert site.content_manager.isValidRelativePath("any/CONAN")
        # 断言给定的路径是无效的相对路径
        assert not site.content_manager.isValidRelativePath("any/CONOUT$")
        # 断言给定的路径是无效的相对路径
        assert not site.content_manager.isValidRelativePath("a" * 256)  # Max 255 characters allowed
```