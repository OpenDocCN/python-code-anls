# `KubiScan\api\api_client.py`

```
# 从 kubernetes 模块中导入 client 和 config
from kubernetes import client, config
# 从 shutil 模块中导入 copyfile
from shutil import copyfile
# 从 os 模块中导入所有内容
import os
# 从 tempfile 模块中导入 mkstemp
from tempfile import mkstemp
# 从 shutil 模块中导入 move
from shutil import move
# 从 kubernetes.client.configuration 模块中导入 Configuration
from kubernetes.client.configuration import Configuration
# 从 kubernetes.client.api_client 模块中导入 ApiClient
from kubernetes.client.api_client import ApiClient

# TODO: 在 bug 解决后应该移除：
# https://github.com/kubernetes-client/python/issues/577
# 从 api.api_client_temp 模块中导入 ApiClientTemp
from api.api_client_temp import ApiClientTemp

# 以下变量已被注释掉，因为在运行 `kubiscan -h` 时会导致 bug
# 异常被忽略：<bound method ApiClient.__del__ of <kubernetes.client.api_client.ApiClient object ...
# 这与 https://github.com/kubernetes-client/python/issues/411 有关
#api_temp = ApiClientTemp()
#CoreV1Api = client.CoreV1Api()
#RbacAuthorizationV1Api = client.RbacAuthorizationV1Api()

# 将 api_temp 变量设置为 None
api_temp = None
# 定义全局变量，用于存储 API 对象和配置信息
CoreV1Api = None
RbacAuthorizationV1Api = None
configuration = None
api_version = None

# 判断当前是否在容器中运行
def running_in_container():
    running_in_a_container = os.getenv('RUNNING_IN_A_CONTAINER')
    if running_in_a_container is not None and running_in_a_container == 'true':
        return True
    return False

# 替换文件中的指定内容
def replace(file_path, pattern, subst):
    # 创建临时文件
    fh, abs_path = mkstemp()
    with os.fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                if pattern in line:
                   # 将匹配到的内容替换后写入新文件
                   new_file.write(line.replace(pattern, subst))
    # 如果条件不成立，执行以下操作
    else:
        # 将行写入新文件
        new_file.write(line)
    # 删除原始文件
    os.remove(file_path)
    # 移动新文件
    move(abs_path, file_path)

# 初始化 API
def api_init(kube_config_file=None, host=None, token_filename=None, cert_filename=None, context=None):
    # 声明全局变量
    global CoreV1Api
    global RbacAuthorizationV1Api
    global api_temp
    global configuration
    global api_version
    # 如果存在主机和令牌文件名
    if host and token_filename:
        # 打印使用的令牌和主机地址
        print("Using token from " + token_filename + " on ip address " + host)
        # 将令牌文件名转换为绝对路径
        token_filename = os.path.abspath(token_filename)
        # 如果存在证书文件名，将其转换为绝对路径
        if cert_filename:
            cert_filename = os.path.abspath(cert_filename)
        # 加载并设置 BearerTokenLoader 配置
        configuration = BearerTokenLoader(host=host, token_filename=token_filename, cert_filename=cert_filename).load_and_set()
# 创建 CoreV1Api 和 RbacAuthorizationV1Api 对象
CoreV1Api = client.CoreV1Api()
RbacAuthorizationV1Api = client.RbacAuthorizationV1Api()
# 创建 ApiClientTemp 对象
api_temp = ApiClientTemp(configuration=configuration)

# 如果存在 kube_config_file，则加载 kube 配置文件
elif kube_config_file:
    print("Using kube config file.")
    config.load_kube_config(os.path.abspath(kube_config_file))
    # 创建 CoreV1Api 和 RbacAuthorizationV1Api 对象
    CoreV1Api = client.CoreV1Api()
    RbacAuthorizationV1Api = client.RbacAuthorizationV1Api()
    # 从配置文件创建新的客户端
    api_from_config = config.new_client_from_config(kube_config_file)
    # 创建 ApiClientTemp 对象
    api_temp = ApiClientTemp(configuration=api_from_config.configuration)
else:
    print("Using kube config file.")
    # 创建空的 Configuration 对象
    configuration = Configuration()
    # 获取环境变量 KUBISCAN_CONFIG_PATH 的值
    kubeconfig_path = os.getenv('KUBISCAN_CONFIG_PATH')
    # 如果在容器中运行且未设置 kubeconfig_path，则输出提示信息
    if running_in_container() and kubeconfig_path is None:
        # TODO: 考虑使用来自 Kubernetes 创建的容器的 config.load_incluster_config()。需要具有特权权限的服务账户。必须挂载卷
# 获取容器卷路径，如果环境变量中没有设置，则默认为 /tmp
container_volume_prefix = os.getenv('KUBISCAN_VOLUME_PATH', '/tmp')
# 获取 kubeconfig 备份路径，如果环境变量中没有设置，则默认为 /opt/kubiscan/config_bak
kube_config_bak_path = os.getenv('KUBISCAN_CONFIG_BACKUP_PATH', '/opt/kubiscan/config_bak')
# 如果 kubeconfig 备份文件不存在，则将当前配置文件复制到备份路径，并替换其中的路径信息
if not os.path.isfile(kube_config_bak_path):
    copyfile(container_volume_prefix + os.path.expandvars('$CONF_PATH'), kube_config_bak_path)
    replace(kube_config_bak_path, ': /', f': {container_volume_prefix}/')

# 根据备份的 kubeconfig 文件加载配置，如果备份文件存在则使用备份文件，否则使用默认配置文件
config.load_kube_config(kube_config_bak_path, context=context, client_configuration=configuration)
# 创建 API 客户端
api_client = ApiClient(configuration=configuration)
# 获取 API 版本信息
api_version = client.VersionApi(api_client=api_client)
# 创建 CoreV1Api 对象
CoreV1Api = client.CoreV1Api(api_client=api_client)
# 创建 RbacAuthorizationV1Api 对象
RbacAuthorizationV1Api = client.RbacAuthorizationV1Api(api_client=api_client)
# 创建临时 API 客户端
api_temp = ApiClientTemp(configuration=configuration)

# 定义 BearerTokenLoader 类
class BearerTokenLoader(object):
    # 初始化方法，接收主机名、token 文件名和证书文件名作为参数
    def __init__(self, host, token_filename, cert_filename=None):
        self._token_filename = token_filename
        self._cert_filename = cert_filename
# 设置主机地址
self._host = host
# 默认开启 SSL 验证
self._verify_ssl = True

# 如果没有证书文件，则关闭 SSL 验证，并禁用不安全请求警告
if not self._cert_filename:
    self._verify_ssl = False
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 加载并设置配置
def load_and_set(self):
    # 加载配置
    self._load_config()
    # 设置配置
    configuration = self._set_config()
    return configuration

# 加载配置
def _load_config(self):
    # 设置主机地址为 https 协议
    self._host = "https://" + self._host

    # 如果服务令牌文件不存在，则抛出异常
    if not os.path.isfile(self._token_filename):
        raise Exception("Service token file does not exists.")
# 打开指定的 token 文件
with open(self._token_filename) as f:
    # 读取文件内容并去除末尾的换行符
    self.token = f.read().rstrip('\n')
    # 如果 token 为空，则抛出异常
    if not self.token:
        raise Exception("Token file exists but empty.")

# 如果存在服务证书文件
if self._cert_filename:
    # 如果服务证书文件不存在，则抛出异常
    if not os.path.isfile(self._cert_filename):
        raise Exception("Service certification file does not exists.")
    
    # 打开服务证书文件
    with open(self._cert_filename) as f:
        # 如果服务证书文件内容为空，则抛出异常
        if not f.read().rstrip('\n'):
            raise Exception("Cert file exists but empty.")

# 将服务证书文件名赋值给 ssl_ca_cert
self.ssl_ca_cert = self._cert_filename

# 设置配置信息
def _set_config(self):
    # 创建配置对象
    configuration = client.Configuration()
    # 设置主机地址
    configuration.host = self._host
    # 设置 SSL CA 证书
    configuration.ssl_ca_cert = self.ssl_ca_cert
# 设置配置对象的 SSL 验证选项
configuration.verify_ssl = self._verify_ssl
# 设置配置对象的 API 授权信息
configuration.api_key['authorization'] = "bearer " + self.token
# 将配置对象设置为默认配置
client.Configuration.set_default(configuration)
# 返回配置对象
return configuration
```