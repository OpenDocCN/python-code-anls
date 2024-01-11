# `ZeroNet\plugins\disabled-Bootstrapper\BootstrapperPlugin.py`

```
# 导入时间模块
import time

# 从util模块中导入helper函数
from util import helper

# 从Plugin模块中导入PluginManager类
from Plugin import PluginManager

# 从当前目录下的BootstrapperDb模块中导入BootstrapperDb类
from .BootstrapperDb import BootstrapperDb

# 从Crypt模块中导入CryptRsa类
from Crypt import CryptRsa

# 从Config模块中导入config变量
from Config import config

# 如果当前作用域中没有db变量，则创建BootstrapperDb对象并赋值给db变量
if "db" not in locals().keys():  # Share during reloads
    db = BootstrapperDb()

# 将FileRequestPlugin类注册到PluginManager的"FileRequest"插件中
@PluginManager.registerTo("FileRequest")
class FileRequestPlugin(object):
    # 检查Onion签名
    def checkOnionSigns(self, onions, onion_signs, onion_sign_this):
        # 如果没有Onion签名或者签名数量与Onion数量不一致，则返回False
        if not onion_signs or len(onion_signs) != len(set(onions)):
            return False

        # 如果当前时间减去Onion签名时间大于3分钟，则返回False
        if time.time() - float(onion_sign_this) > 3 * 60:
            return False  # Signed out of allowed 3 minutes

        onions_signed = []
        # 检查Onion签名
        for onion_publickey, onion_sign in onion_signs.items():
            # 如果Onion签名验证通过，则将对应的Onion地址添加到onions_signed列表中
            if CryptRsa.verify(onion_sign_this.encode(), onion_publickey, onion_sign):
                onions_signed.append(CryptRsa.publickeyToOnion(onion_publickey))
            else:
                break

        # 检查已签名的Onion地址是否与宣布的一致
        if sorted(onions_signed) == sorted(set(onions)):
            return True
        else:
            return False

# 将UiRequestPlugin类注册到PluginManager的"UiRequest"插件中
@PluginManager.registerTo("UiRequest")
class UiRequestPlugin(object):
    # 使用helper模块中的encodeResponse装饰器对方法进行编码响应处理
    @helper.encodeResponse
    # 定义一个名为 actionStatsBootstrapper 的方法，参数中包含 self
    def actionStatsBootstrapper(self):
        # 调用类中的 sendHeader 方法
        self.sendHeader()

        # 添加内联样式，设置字体和表格样式
        yield """
        <style>
         * { font-family: monospace; white-space: pre }
         table td, table th { text-align: right; padding: 0px 10px }
        </style>
        """

        # 从数据库中查询 hash 表中的所有数据
        hash_rows = db.execute("SELECT * FROM hash").fetchall()
        # 遍历查询结果中的每一行数据
        for hash_row in hash_rows:
            # 从数据库中查询 peer 表和 peer_to_hash 表的联合数据
            peer_rows = db.execute(
                "SELECT * FROM peer LEFT JOIN peer_to_hash USING (peer_id) WHERE hash_id = :hash_id",
                {"hash_id": hash_row["hash_id"]}
            ).fetchall()

            # 返回哈希值、添加日期和对应的对等节点数量
            yield "<br>%s (added: %s, peers: %s)<br>" % (
                str(hash_row["hash"]).encode().hex(), hash_row["date_added"], len(peer_rows)
            )
            # 遍历对等节点数据
            for peer_row in peer_rows:
                # 返回对等节点的类型、地址、端口、添加日期和公告日期
                yield " - {type} {address}:{port} added: {date_added}, announced: {date_announced}<br>".format(**dict(peer_row))
```