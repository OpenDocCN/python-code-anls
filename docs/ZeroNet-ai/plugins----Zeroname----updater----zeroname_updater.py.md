# `ZeroNet\plugins\Zeroname\updater\zeroname_updater.py`

```
# 导入必要的模块
from __future__ import print_function
import time
import json
import os
import sys
import re
import socket

from six import string_types
# 导入子进程调用模块
from subprocess import call
# 导入比特币 RPC 客户端模块
from bitcoinrpc.authproxy import AuthServiceProxy

# 定义发布函数
def publish():
    print("* Signing and Publishing...")
    # 调用命令行执行签名和发布操作
    call(" ".join(command_sign_publish), shell=True)

# 定义处理名称操作的函数
def processNameOp(domain, value, test=False):
    # 如果值不是以 "{" 开头，则返回 False
    if not value.strip().startswith("{"):
        return False
    try:
        # 尝试将值解析为 JSON 格式
        data = json.loads(value)
    except Exception as err:
        print("Json load error: %s" % err)
        return False
    # 如果数据中既没有 "zeronet" 也没有 "map"，则返回 False
    if "zeronet" not in data and "map" not in data:
        print("No zeronet and no map in ", data.keys())
        return False
    # 如果数据中包含 "map"，则将其转换为 Zeronet 格式并重新调用该函数
    if "map" in data:
        data_map = data["map"]
        new_value = {}
        for subdomain in data_map:
            if "zeronet" in data_map[subdomain]:
                new_value[subdomain] = data_map[subdomain]["zeronet"]
        if "zeronet" in data and isinstance(data["zeronet"], string_types):
            new_value[""] = data["zeronet"]
        if len(new_value) > 0:
            return processNameOp(domain, json.dumps({"zeronet": new_value}), test)
        else:
            return False
    # 如果数据中包含 "zeronet"，则将其转换为 Zeronet 格式并重新调用该函数
    if "zeronet" in data and isinstance(data["zeronet"], string_types):
        return processNameOp(domain, json.dumps({"zeronet": { "": data["zeronet"]}}), test)
    # 如果 "zeronet" 不是字符串类型，则返回 False
    if not isinstance(data["zeronet"], dict):
        print("Not dict: ", data["zeronet"])
        return False
    # 如果域名不符合规范，则返回 False
    if not re.match("^[a-z0-9]([a-z0-9-]{0,62}[a-z0-9])?$", domain):
        print("Invalid domain: ", domain)
        return False
    # 如果测试条件为真，则返回True
    if test:
        return True

    # 如果命令行参数中包含"slave"，则打印信息并等待30秒
    if "slave" in sys.argv:
        print("Waiting for master update arrive")
        time.sleep(30)  # Wait 30 sec to allow master updater

    # 注意：需要存在文件data/names.json并且包含"{}"才能正常工作
    # 读取names_path文件的二进制内容
    names_raw = open(names_path, "rb").read()
    # 将二进制内容解析为JSON格式
    names = json.loads(names_raw)
    # 遍历zeronet数据中的子域名和地址
    for subdomain, address in data["zeronet"].items():
        # 将子域名转换为小写
        subdomain = subdomain.lower()
        # 从地址中移除非字母和数字的字符
        address = re.sub("[^A-Za-z0-9]", "", address)
        # 打印子域名、域名和地址
        print(subdomain, domain, "->", address)
        # 如果子域名存在
        if subdomain:
            # 如果子域名符合域名规范，则将其添加到names字典中
            if re.match("^[a-z0-9]([a-z0-9-]{0,62}[a-z0-9])?$", subdomain):
                names["%s.%s.bit" % (subdomain, domain)] = address
            else:
                # 如果子域名不符合规范，则打印错误信息
                print("Invalid subdomain:", domain, subdomain)
        else:
            # 如果子域名不存在，则将域名和地址添加到names字典中
            names["%s.bit" % domain] = address

    # 将更新后的names字典转换为JSON格式，并进行格式化和排序
    new_names_raw = json.dumps(names, indent=2, sort_keys=True)
    # 如果更新后的JSON内容与原内容不同，则将更新后的内容写入names_path文件，并返回True
    if new_names_raw != names_raw:
        open(names_path, "wb").write(new_names_raw)
        print("-", domain, "Changed")
        return True
    else:
        # 如果更新后的JSON内容与原内容相同，则打印信息并返回False
        print("-", domain, "Not changed")
        return False
# 处理区块的函数，根据给定的区块ID和测试标志来处理区块
def processBlock(block_id, test=False):
    # 打印正在处理的区块编号
    print("Processing block #%s..." % block_id)
    # 记录开始处理的时间
    s = time.time()
    # 获取指定区块ID对应的区块哈希值
    block_hash = rpc.getblockhash(block_id)
    # 获取指定区块哈希值对应的区块数据
    block = rpc.getblock(block_hash)

    # 打印正在检查的交易数量
    print("Checking %s tx" % len(block["tx"]))
    # 记录更新的交易数量
    updated = 0
    # 遍历区块中的每一笔交易
    for tx in block["tx"]:
        try:
            # 获取指定交易ID对应的原始交易数据
            transaction = rpc.getrawtransaction(tx, 1)
            # 遍历交易中的每一个输出
            for vout in transaction.get("vout", []):
                # 检查输出中是否包含名称操作
                if "scriptPubKey" in vout and "nameOp" in vout["scriptPubKey"] and "name" in vout["scriptPubKey"]["nameOp"]:
                    # 获取名称操作的相关信息
                    name_op = vout["scriptPubKey"]["nameOp"]
                    # 调用处理名称操作的函数，并累加更新的数量
                    updated += processNameOp(name_op["name"].replace("d/", ""), name_op["value"], test)
        except Exception as err:
            # 打印处理交易时出现的错误
            print("Error processing tx #%s %s" % (tx, err))
    # 打印处理完成所花费的时间和更新的数量
    print("Done in %.3fs (updated %s)." % (time.time() - s, updated))
    # 返回更新的数量
    return updated

# 初始化与RPC的连接
def initRpc(config):
    """Initialize Namecoin RPC"""
    # 初始化RPC连接的相关信息
    rpc_data = {
        'connect': '127.0.0.1',
        'port': '8336',
        'user': 'PLACEHOLDER',
        'password': 'PLACEHOLDER',
        'clienttimeout': '900'
    }
    try:
        # 打开配置文件并读取内容
        fptr = open(config, 'r')
        lines = fptr.readlines()
        fptr.close()
    except:
        return None  # Or take some other appropriate action

    # 解析配置文件中的RPC连接信息
    for line in lines:
        if not line.startswith('rpc'):
            continue
        key_val = line.split(None, 1)[0]
        (key, val) = key_val.split('=', 1)
        if not key or not val:
            continue
        rpc_data[key[3:]] = val

    # 根据解析得到的信息构建RPC连接的URL
    url = 'http://%(user)s:%(password)s@%(connect)s:%(port)s' % rpc_data

    # 返回构建好的RPC连接URL和客户端超时时间
    return url, int(rpc_data['clienttimeout'])

# 加载配置文件...

# 检查平台是Windows还是Linux
# 在Linux上，Namecoin安装在~/.namecoin目录下，而在Windows上，它在%appdata%/Namecoin目录下
if sys.platform == "win32":
    # 设置Namecoin的安装位置为Windows下的路径
    namecoin_location = os.getenv('APPDATA') + "/Namecoin/"
else:
    # 设置Namecoin的安装位置为Linux下的路径
    namecoin_location = os.path.expanduser("~/.namecoin/")
# 拼接配置文件路径
config_path = namecoin_location + 'zeroname_config.json'
# 如果配置文件不存在，则创建示例配置文件
if not os.path.isfile(config_path):  # Create sample config
    # 写入示例配置文件内容
    open(config_path, "w").write(
        json.dumps({'site': 'site', 'zeronet_path': '/home/zeronet', 'privatekey': '', 'lastprocessed': 223910}, indent=2)
    )
    # 打印示例配置文件路径
    print("* Example config written to %s" % config_path)
    # 退出程序
    sys.exit(0)

# 读取配置文件内容
config = json.load(open(config_path))
# 拼接文件路径
names_path = "%s/data/%s/data/names.json" % (config["zeronet_path"], config["site"])
# 改变工作目录，告诉脚本 Zeronet 安装的位置
os.chdir(config["zeronet_path"])

# 设置签名和发布的参数
command_sign_publish = [sys.executable, "zeronet.py", "siteSign", config["site"], config["privatekey"], "--publish"]
# 如果是 Windows 平台，对参数进行处理
if sys.platform == 'win32':
    command_sign_publish = ['"%s"' % param for param in command_sign_publish]

# 初始化 RPC 连接
rpc_auth, rpc_timeout = initRpc(namecoin_location + "namecoin.conf")
rpc = AuthServiceProxy(rpc_auth, timeout=rpc_timeout)

# 获取节点版本信息
node_version = rpc.getnetworkinfo()['version']

# 循环直到连接成功
while 1:
    try:
        # 等待1秒
        time.sleep(1)
        # 如果节点版本小于160000，则获取最新区块高度
        if node_version < 160000 :
            last_block = int(rpc.getinfo()["blocks"])
        else:
            last_block = int(rpc.getblockchaininfo()["blocks"])
        # 连接成功，跳出循环
        break 
    # 如果超时，则打印"."并刷新输出，继续循环
    except socket.timeout:  # Timeout
        print(".", end=' ')
        sys.stdout.flush()
    # 如果出现异常，则打印异常信息，等待5秒后重新初始化 RPC 连接
    except Exception as err:
        print("Exception", err.__class__, err)
        time.sleep(5)
        rpc = AuthServiceProxy(rpc_auth, timeout=rpc_timeout)

# 如果配置文件中的最后处理块号为空，则将其设置为最新区块高度
if not config["lastprocessed"]:  # First startup: Start processing from last block
    config["lastprocessed"] = last_block

# 测试域名解析
print("- Testing domain parsing...")
# 断言处理区块函数的返回值，用于测试
assert processBlock(223911, test=True) # Testing zeronetwork.bit
assert processBlock(227052, test=True) # Testing brainwallets.bit
assert not processBlock(236824, test=True) # Utf8 domain name (invalid should skip)
assert not processBlock(236752, test=True) # Uppercase domain (invalid should skip)
# 对区块进行处理，测试模式下应该通过
assert processBlock(236870, test=True) 
# 对区块进行处理，测试 namecoin 标准 artifaxradio.bit 应该通过
assert processBlock(438317, test=True) 
# 打印跳过的区块信息
# sys.exit(0)

# 解析跳过的区块
print("- Parsing skipped blocks...")
# 是否需要发布标志位
should_publish = False
# 遍历区块范围
for block_id in range(config["lastprocessed"], last_block + 1):
    # 处理区块，如果成功则设置发布标志位
    if processBlock(block_id):
        should_publish = True
# 更新最后处理的区块
config["lastprocessed"] = last_block

# 如果需要发布，则执行发布操作
if should_publish:
    publish()

# 无限循环，等待新的区块
while 1:
    print("- Waiting for new block")
    sys.stdout.flush()
    # 再次无限循环，等待新的区块
    while 1:
        try:
            # 休眠1秒
            time.sleep(1)
            # 根据节点版本不同，调用不同的等待区块方法
            if node_version < 160000 :
                rpc.waitforblock()
            else:
                rpc.waitfornewblock()
            print("Found")
            break  # 区块找到
        except socket.timeout:  # 超时
            print(".", end=' ')
            sys.stdout.flush()
        except Exception as err:
            print("Exception", err.__class__, err)
            # 休眠5秒
            time.sleep(5)
            # 重新创建 RPC 连接
            rpc = AuthServiceProxy(rpc_auth, timeout=rpc_timeout)

    # 根据节点版本不同，获取最新区块高度
    if node_version < 160000 :
        last_block = int(rpc.getinfo()["blocks"])
    else:
        last_block = int(rpc.getblockchaininfo()["blocks"])
    should_publish = False
    # 遍历区块范围
    for block_id in range(config["lastprocessed"] + 1, last_block + 1):
        # 处理区块，如果成功则设置发布标志位
        if processBlock(block_id):
            should_publish = True

    # 更新最后处理的区块
    config["lastprocessed"] = last_block
    # 写入配置文件
    open(config_path, "w").write(json.dumps(config, indent=2))

    # 如果需要发布，则执行发布操作
    if should_publish:
        publish()
```