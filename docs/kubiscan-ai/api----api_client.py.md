# `KubiScan\api\api_client.py`

```py
# 导入所需的模块
from kubernetes import client, config
from shutil import copyfile
import os
from tempfile import mkstemp
from shutil import move
from kubernetes.client.configuration import Configuration
from kubernetes.client.api_client import ApiClient

# TODO: 应在解决了 bug 之后移除：
# https://github.com/kubernetes-client/python/issues/577
from api.api_client_temp import ApiClientTemp

# 以下变量已被注释，因为在运行 `kubiscan -h` 时会导致 bug
# 异常被忽略: <bound method ApiClient.__del__ of <kubernetes.client.api_client.ApiClient object ...
# 这与 https://github.com/kubernetes-client/python/issues/411 有关
#api_temp = ApiClientTemp()
#CoreV1Api = client.CoreV1Api()
#RbacAuthorizationV1Api = client.RbacAuthorizationV1Api()

api_temp = None
CoreV1Api = None
RbacAuthorizationV1Api = None
configuration = None
api_version = None

# 检查是否在容器中运行
def running_in_container():
    running_in_a_container = os.getenv('RUNNING_IN_A_CONTAINER')
    if running_in_a_container is not None and running_in_a_container == 'true':
        return True
    return False

# 替换文件中的内容
def replace(file_path, pattern, subst):
    # 创建临时文件
    fh, abs_path = mkstemp()
    with os.fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                if pattern in line:
                   new_file.write(line.replace(pattern, subst))
                else:
                   new_file.write(line)
    # 删除原始文件
    os.remove(file_path)
    # 移动新文件
    move(abs_path, file_path)

# 初始化 API
def api_init(kube_config_file=None, host=None, token_filename=None, cert_filename=None, context=None):
    global CoreV1Api
    global RbacAuthorizationV1Api
    global api_temp
    global configuration
    global api_version
    # 如果存在主机和令牌文件名
    if host and token_filename:
        # 打印使用令牌文件和主机地址
        print("Using token from " + token_filename + " on ip address " + host)
        # 将令牌文件名转换为绝对路径
        token_filename = os.path.abspath(token_filename)
        # 如果存在证书文件名，将证书文件名转换为绝对路径
        if cert_filename:
            cert_filename = os.path.abspath(cert_filename)
        # 使用BearerTokenLoader加载主机、令牌文件名和证书文件名的配置
        configuration = BearerTokenLoader(host=host, token_filename=token_filename, cert_filename=cert_filename).load_and_set()

        # 创建CoreV1Api对象
        CoreV1Api = client.CoreV1Api()
        # 创建RbacAuthorizationV1Api对象
        RbacAuthorizationV1Api = client.RbacAuthorizationV1Api()
        # 创建ApiClientTemp对象
        api_temp = ApiClientTemp(configuration=configuration)

    # 如果存在kube配置文件
    elif kube_config_file:
        # 打印使用kube配置文件
        print("Using kube config file.")
        # 加载kube配置文件并转换为绝对路径
        config.load_kube_config(os.path.abspath(kube_config_file))
        # 创建CoreV1Api对象
        CoreV1Api = client.CoreV1Api()
        # 创建RbacAuthorizationV1Api对象
        RbacAuthorizationV1Api = client.RbacAuthorizationV1Api()
        # 从配置文件创建新的客户端
        api_from_config = config.new_client_from_config(kube_config_file)
        # 创建ApiClientTemp对象
        api_temp = ApiClientTemp(configuration=api_from_config.configuration)
    else:
        # 打印提示信息，使用 kube 配置文件
        print("Using kube config file.")
        # 创建 Configuration 对象
        configuration = Configuration()

        # 获取环境变量 KUBISCAN_CONFIG_PATH 的值
        kubeconfig_path = os.getenv('KUBISCAN_CONFIG_PATH')
        # 如果在容器中运行且未设置 kubeconfig_path
        if running_in_container() and kubeconfig_path is None:
            # TODO: 考虑使用 Kubernetes 创建的容器中的 config.load_incluster_config()。需要具有特权权限的服务账户。必须挂载卷
            # 获取容器卷的前缀，默认为 /tmp
            container_volume_prefix = os.getenv('KUBISCAN_VOLUME_PATH', '/tmp')
            # 获取备份的 kubeconfig 文件路径，默认为 /opt/kubiscan/config_bak
            kube_config_bak_path = os.getenv('KUBISCAN_CONFIG_BACKUP_PATH', '/opt/kubiscan/config_bak')
            # 如果备份的 kubeconfig 文件不存在
            if not os.path.isfile(kube_config_bak_path):
                # 复制当前 kubeconfig 文件到备份路径
                copyfile(container_volume_prefix + os.path.expandvars('$CONF_PATH'), kube_config_bak_path)
                # 替换备份路径中的 ': /' 为 ': /tmp'，确保路径正确
                replace(kube_config_bak_path, ': /', f': {container_volume_prefix}/')

            # 加载备份的 kubeconfig 文件，设置上下文和客户端配置
            config.load_kube_config(kube_config_bak_path, context=context, client_configuration=configuration)
        else:
            # 加载指定路径的 kubeconfig 文件，设置上下文和客户端配置
            config.load_kube_config(config_file=kubeconfig_path, context=context, client_configuration=configuration)

        # 创建 ApiClient 对象
        api_client = ApiClient(configuration=configuration)
        # 创建 VersionApi 对象
        api_version = client.VersionApi(api_client=api_client)
        # 创建 CoreV1Api 对象
        CoreV1Api = client.CoreV1Api(api_client=api_client)
        # 创建 RbacAuthorizationV1Api 对象
        RbacAuthorizationV1Api = client.RbacAuthorizationV1Api(api_client=api_client)
        # 创建 ApiClientTemp 对象
        api_temp = ApiClientTemp(configuration=configuration)
# 定义一个名为 BearerTokenLoader 的类
class BearerTokenLoader(object):
    # 初始化方法，接受主机名、令牌文件名和证书文件名作为参数
    def __init__(self, host, token_filename, cert_filename=None):
        # 设置实例变量
        self._token_filename = token_filename
        self._cert_filename = cert_filename
        self._host = host
        self._verify_ssl = True

        # 如果没有证书文件名，则禁用 SSL 验证
        if not self._cert_filename:
            self._verify_ssl = False
            # 导入 urllib3 库并禁用警告
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # 加载配置并设置方法
    def load_and_set(self):
        # 调用加载配置方法
        self._load_config()
        # 调用设置配置方法
        configuration = self._set_config()
        # 返回配置对象
        return configuration

    # 加载配置方法
    def _load_config(self):
        # 将主机名添加到 https:// 前缀
        self._host = "https://" + self._host
        # 如果令牌文件不存在，则抛出异常
        if not os.path.isfile(self._token_filename):
            raise Exception("Service token file does not exists.")
        # 读取令牌文件内容并去除换行符
        with open(self._token_filename) as f:
            self.token = f.read().rstrip('\n')
            # 如果令牌文件存在但为空，则抛出异常
            if not self.token:
                raise Exception("Token file exists but empty.")
        # 如果有证书文件名
        if self._cert_filename:
            # 如果证书文件不存在，则抛出异常
            if not os.path.isfile(self._cert_filename):
                raise Exception("Service certification file does not exists.")
            # 读取证书文件内容并去除换行符
            with open(self._cert_filename) as f:
                if not f.read().rstrip('\n'):
                    raise Exception("Cert file exists but empty.")
        # 设置 SSL CA 证书
        self.ssl_ca_cert = self._cert_filename

    # 设置配置方法
    def _set_config(self):
        # 创建配置对象
        configuration = client.Configuration()
        # 设置主机名
        configuration.host = self._host
        # 设置 SSL CA 证书
        configuration.ssl_ca_cert = self.ssl_ca_cert
        # 设置是否验证 SSL
        configuration.verify_ssl = self._verify_ssl
        # 设置 API 密钥
        configuration.api_key['authorization'] = "bearer " + self.token
        # 设置默认配置
        client.Configuration.set_default(configuration)
        # 返回配置对象
        return configuration
```