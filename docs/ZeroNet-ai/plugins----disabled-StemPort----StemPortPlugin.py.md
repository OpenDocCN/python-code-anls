# `ZeroNet\plugins\disabled-StemPort\StemPortPlugin.py`

```
# 导入日志模块
import logging
# 导入异常追踪模块
import traceback

# 导入 socket 模块
import socket
# 导入 stem 模块
import stem
# 从 stem 模块中导入 Signal 类
from stem import Signal
# 从 stem.control 模块中导入 Controller 类
from stem.control import Controller
# 从 stem.socket 模块中导入 ControlPort 类
from stem.socket import ControlPort

# 从 Plugin 模块中导入 PluginManager 类
from Plugin import PluginManager
# 从 Config 模块中导入 config 对象
from Config import config
# 从 Debug 模块中导入 Debug 类
from Debug import Debug

# 如果配置中的 tor 不等于 "disable"，则执行以下代码块
if config.tor != "disable":
    # 导入 gevent 中的 monkey 模块
    from gevent import monkey
    # 对 time 模块进行 monkey patch
    monkey.patch_time()
    # 对 socket 模块进行 monkey patch，禁用 DNS 查询
    monkey.patch_socket(dns=False)
    # 对 thread 模块进行 monkey patch
    monkey.patch_thread()
    # 打印提示信息
    print("Stem Port Plugin: modules are patched.")
# 如果配置中的 tor 等于 "disable"，则执行以下代码块
else:
    # 打印提示信息
    print("Stem Port Plugin: Tor mode disabled. Module patching skipped.")


# 定义一个继承自 ControlPort 类的 PatchedControlPort 类
class PatchedControlPort(ControlPort):
    # 重写 _make_socket 方法
    def _make_socket(self):
        try:
            # 如果 socket 模块中存在 socket_noproxy 方法，则使用非代理的 socket
            if "socket_noproxy" in dir(socket):
                control_socket = socket.socket_noproxy(socket.AF_INET, socket.SOCK_STREAM)
            # 否则使用普通的 socket
            else:
                control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            # 连接控制端口
            control_socket.connect((self._control_addr, self._control_port))
            return control_socket
        # 捕获 socket 错误并抛出 stem.SocketError 异常
        except socket.error as exc:
            raise stem.SocketError(exc)

# 定义一个函数，用于从指定地址和端口创建 Controller 对象
def from_port(address = '127.0.0.1', port = 'default'):
    # 导入 stem.connection 模块
    import stem.connection

    # 如果地址不是有效的 IPv4 地址，则抛出 ValueError 异常
    if not stem.util.connection.is_valid_ipv4_address(address):
        raise ValueError('Invalid IP address: %s' % address)
    # 如果端口不是默认值且不是有效的端口号，则抛出 ValueError 异常
    elif port != 'default' and not stem.util.connection.is_valid_port(port):
        raise ValueError('Invalid port: %s' % port)

    # 如果端口是默认值，则抛出 ValueError 异常
    if port == 'default':
        raise ValueError('Must specify a port')
    # 否则创建 PatchedControlPort 对象
    else:
        control_port = PatchedControlPort(address, port)

    # 返回 Controller 对象
    return Controller(control_port)

# 将 TorManagerPlugin 类注册到 PluginManager 的 "TorManager" 中
@PluginManager.registerTo("TorManager")
class TorManagerPlugin(object):
    # 连接到控制器，记录认证信息和连接状态
    def connectController(self):
        self.log.info("Authenticate using Stem... %s:%s" % (self.ip, self.port))

        try:
            # 使用线程锁确保多线程安全
            with self.lock:
                # 如果配置了 Tor 密码，则使用密码连接到控制器
                if config.tor_password:
                    controller = from_port(port=self.port, password=config.tor_password)
                else:
                    # 否则直接连接到控制器
                    controller = from_port(port=self.port)
                # 认证控制器
                controller.authenticate()
                # 将控制器对象赋值给实例变量
                self.controller = controller
                # 更新连接状态
                self.status = "Connected (via Stem)"
        except Exception as err:
            # 捕获异常并记录错误信息
            print("\n")
            traceback.print_exc()
            print("\n")

            # 清空控制器对象和更新连接状态为错误
            self.controller = None
            self.status = "Error (%s)" % err
            self.log.error("Tor stem connect error: %s" % Debug.formatException(err))

        # 返回控制器对象
        return self.controller


    # 断开连接
    def disconnect(self):
        # 关闭控制器连接并清空控制器对象
        self.controller.close()
        self.controller = None


    # 重置 Tor 路由
    def resetCircuits(self):
        try:
            # 发送信号请求重置 Tor 路由
            self.controller.signal(Signal.NEWNYM)
        except Exception as err:
            # 捕获异常并记录错误信息
            self.status = "Stem reset circuits error (%s)" % err
            self.log.error("Stem reset circuits error: %s" % err)


    # 创建 .onion 地址和密钥
    def makeOnionAndKey(self):
        try:
            # 创建临时隐藏服务，并设置不等待发布
            service = self.controller.create_ephemeral_hidden_service(
                {self.fileserver_port: self.fileserver_port},
                await_publication = False
            )
            # 检查私钥类型是否为 RSA1024
            if service.private_key_type != "RSA1024":
                raise Exception("ZeroNet doesn't support crypto " + service.private_key_type)

            # 记录创建的 .onion 地址，并返回 .onion 地址和私钥
            self.log.debug("Stem created %s.onion (async descriptor publication)" % service.service_id)
            return (service.service_id, service.private_key)

        except Exception as err:
            # 捕获异常并记录错误信息
            self.status = "AddOnion error (Stem: %s)" % err
            self.log.error("Failed to create hidden service with Stem: " + err)
            return False
    # 删除指定的 Tor 隐式服务
    def delOnion(self, address):
        try:
            # 尝试移除指定的隐式服务
            self.controller.remove_ephemeral_hidden_service(address)
            # 如果成功移除，则返回 True
            return True
        except Exception as err:
            # 如果移除失败，则记录错误信息
            self.status = "DelOnion error (Stem: %s)" % err
            self.log.error("Stem failed to delete %s.onion: %s" % (address, err))
            # 断开连接
            self.disconnect() # Why? - 为什么要断开连接？
            # 返回 False
            return False


    # 发送请求命令
    def request(self, cmd):
        # 使用锁保护代码块
        with self.lock:
            # 如果未启用，则返回 False
            if not self.enabled:
                return False
            else:
                # 记录警告信息
                self.log.error("[WARNING] StemPort self.request should not be called")
                # 返回空字符串
                return ""

    # 发送命令
    def send(self, cmd, conn=None):
        # 记录警告信息
        self.log.error("[WARNING] StemPort self.send should not be called")
        # 返回空字符串
        return ""
```