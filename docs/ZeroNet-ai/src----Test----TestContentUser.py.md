# `ZeroNet\src\Test\TestContentUser.py`

```
# 导入 json 模块
import json
# 导入 io 模块
import io

# 导入 pytest 模块
import pytest

# 从 Crypt 模块中导入 CryptBitcoin 类
from Crypt import CryptBitcoin
# 从 Content.ContentManager 模块中导入 VerifyError, SignError 类
from Content.ContentManager import VerifyError, SignError

# 使用 pytest 的 usefixtures 装饰器，重置设置
@pytest.mark.usefixtures("resetSettings")
# 定义 TestContentUser 类
class TestContentUser:
    # 测试给定站点的签名者
    def testSigners(self, site):
        # 获取不存在用户文件的文件信息
        file_info = site.content_manager.getFileInfo("data/users/notexist/data.json")
        # 断言文件信息中的内容内部路径
        assert file_info["content_inner_path"] == "data/users/notexist/content.json"
        # 获取不存在用户文件夹中的子文件的文件信息
        file_info = site.content_manager.getFileInfo("data/users/notexist/a/b/data.json")
        # 断言文件信息中的内容内部路径
        assert file_info["content_inner_path"] == "data/users/notexist/content.json"
        # 获取不存在用户文件夹中的内容文件的有效签名者
        valid_signers = site.content_manager.getValidSigners("data/users/notexist/content.json")
        # 断言有效签名者列表
        assert valid_signers == ["14wgQ4VDDZNoRMFF4yCDuTrBSHmYhL3bet", "notexist", "1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT"]

        # 获取存在用户文件夹中的内容文件的有效签名者
        valid_signers = site.content_manager.getValidSigners("data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json")
        # 断言有效签名者列表中包含站点地址
        assert '1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT' in valid_signers
        # 断言有效签名者列表中包含在data/users/content.json中定义的管理员用户
        assert '14wgQ4VDDZNoRMFF4yCDuTrBSHmYhL3bet' in valid_signers
        # 断言有效签名者列表中包含用户本身
        assert '1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C' in valid_signers
        # 断言有效签名者列表长度为3
        assert len(valid_signers) == 3

        # 获取被禁止用户的内容文件的有效签名者
        user_content = site.storage.loadJson("data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json")
        user_content["cert_user_id"] = "bad@zeroid.bit"

        valid_signers = site.content_manager.getValidSigners("data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json", user_content)
        # 断言有效签名者列表中包含站点地址
        assert '1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT' in valid_signers
        # 断言有效签名者列表中包含在data/users/content.json中定义的管理员用户
        assert '14wgQ4VDDZNoRMFF4yCDuTrBSHmYhL3bet' in valid_signers
        # 断言有效签名者列表中不包含用户本身
        assert '1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C' not in valid_signers
    # 定义测试规则的方法，传入站点对象参数
    def testRules(self, site):
        # 从 data/users/content.json 中加载用户内容数据
        user_content = site.storage.loadJson("data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json")

        # 已知用户
        user_content["cert_auth_type"] = "web"
        user_content["cert_user_id"] = "nofish@zeroid.bit"
        # 获取用户规则
        rules = site.content_manager.getRules("data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json", user_content)
        # 断言最大大小规则为 100000
        assert rules["max_size"] == 100000
        # 断言用户 ID 在签名者列表中
        assert "1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C" in rules["signers"]

        # 未知用户
        user_content["cert_auth_type"] = "web"
        user_content["cert_user_id"] = "noone@zeroid.bit"
        # 获取用户规则
        rules = site.content_manager.getRules("data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json", user_content)
        # 断言最大大小规则为 10000
        assert rules["max_size"] == 10000
        # 断言用户 ID 在签名者列表中
        assert "1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C" in rules["signers"]

        # 根据认证类型提供更大的大小限制的用户
        user_content["cert_auth_type"] = "bitmsg"
        user_content["cert_user_id"] = "noone@zeroid.bit"
        # 获取用户规则
        rules = site.content_manager.getRules("data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json", user_content)
        # 断言最大大小规则为 15000
        assert rules["max_size"] == 15000
        # 断言用户 ID 在签名者列表中
        assert "1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C" in rules["signers"]

        # 被禁止的用户
        user_content["cert_auth_type"] = "web"
        user_content["cert_user_id"] = "bad@zeroid.bit"
        # 获取用户规则
        rules = site.content_manager.getRules("data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json", user_content)
        # 断言用户 ID 不在签名者列表中
        assert "1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C" not in rules["signers"]
    # 测试地址规则的函数，接受一个站点对象作为参数
    def testRulesAddress(self, site):
        # 定义用户内部路径
        user_inner_path = "data/users/1CjfbrbwtP8Y2QjPy12vpTATkUT7oSiPQ9/content.json"
        # 从站点存储中加载用户内容
        user_content = site.storage.loadJson(user_inner_path)

        # 获取用户规则
        rules = site.content_manager.getRules(user_inner_path, user_content)
        # 断言最大大小规则为10000
        assert rules["max_size"] == 10000
        # 断言用户地址在签名者列表中
        assert "1CjfbrbwtP8Y2QjPy12vpTATkUT7oSiPQ9" in rules["signers"]

        # 获取站点内容管理器中的用户内容
        users_content = site.content_manager.contents["data/users/content.json"]

        # 根据地址禁止用户
        users_content["user_contents"]["permissions"]["1CjfbrbwtP8Y2QjPy12vpTATkUT7oSiPQ9"] = False
        # 重新获取用户规则
        rules = site.content_manager.getRules(user_inner_path, user_content)
        # 断言用户地址不在签名者列表中
        assert "1CjfbrbwtP8Y2QjPy12vpTATkUT7oSiPQ9" not in rules["signers"]

        # 修改最大允许大小
        users_content["user_contents"]["permissions"]["1CjfbrbwtP8Y2QjPy12vpTATkUT7oSiPQ9"] = {"max_size": 20000}
        # 重新获取用户规则
        rules = site.content_manager.getRules(user_inner_path, user_content)
        # 断言最大大小规则为20000
        assert rules["max_size"] == 20000
    # 定义一个测试函数，用于验证地址
    def testVerifyAddress(self, site):
        # 设置私钥
        privatekey = "5KUh3PvNm5HUWoCfSUfcYvfQ2g3PrRNJWr6Q9eqdBGu23mtMntv"  # For 1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT
        # 设置用户内部路径
        user_inner_path = "data/users/1CjfbrbwtP8Y2QjPy12vpTATkUT7oSiPQ9/content.json"
        # 从站点存储中加载 JSON 数据
        data_dict = site.storage.loadJson(user_inner_path)
        # 获取站点内容管理器中的用户内容
        users_content = site.content_manager.contents["data/users/content.json"]

        # 将数据字典转换为字节流
        data = io.BytesIO(json.dumps(data_dict).encode())
        # 断言站点内容管理器验证文件的结果
        assert site.content_manager.verifyFile(user_inner_path, data, ignore_same=False)

        # 测试 15k data.json 文件时的错误
        data_dict["files"]["data.json"]["size"] = 1024 * 15
        # 在签名之前删除签名
        del data_dict["signs"]  # Remove signs before signing
        # 对数据字典进行签名
        data_dict["signs"] = {
            "1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT": CryptBitcoin.sign(json.dumps(data_dict, sort_keys=True), privatekey)
        }
        # 将数据字典转换为字节流
        data = io.BytesIO(json.dumps(data_dict).encode())
        # 使用 pytest 断言捕获 VerifyError 异常
        with pytest.raises(VerifyError) as err:
            site.content_manager.verifyFile(user_inner_path, data, ignore_same=False)
        # 断言异常信息中包含特定字符串
        assert "Include too large" in str(err.value)

        # 根据地址给予更多空间
        users_content["user_contents"]["permissions"]["1CjfbrbwtP8Y2QjPy12vpTATkUT7oSiPQ9"] = {"max_size": 20000}
        # 在签名之前删除签名
        del data_dict["signs"]  # Remove signs before signing
        # 对数据字典进行签名
        data_dict["signs"] = {
            "1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT": CryptBitcoin.sign(json.dumps(data_dict, sort_keys=True), privatekey)
        }
        # 将数据字典转换为字节流
        data = io.BytesIO(json.dumps(data_dict).encode())
        # 断言站点内容管理器验证文件的结果
        assert site.content_manager.verifyFile(user_inner_path, data, ignore_same=False)
    # 定义一个测试方法，用于测试在给定站点上创建新文件
    def testNewFile(self, site):
        # 设置私钥，用于对文件进行签名
        privatekey = "5KUh3PvNm5HUWoCfSUfcYvfQ2g3PrRNJWr6Q9eqdBGu23mtMntv"  # For 1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT
        # 设置内部路径，指定文件的存储位置
        inner_path = "data/users/1NEWrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json"

        # 在指定路径下写入 JSON 数据
        site.storage.writeJson(inner_path, {"test": "data"})
        # 对指定路径下的文件进行签名
        site.content_manager.sign(inner_path, privatekey)
        # 断言测试，确保文件中包含指定的数据
        assert "test" in site.storage.loadJson(inner_path)

        # 删除指定路径下的文件
        site.storage.delete(inner_path)
```