# `KubiScan\api\api_client.py`

```

# 导入所需的模块
from kubernetes import client, config
from shutil import copyfile
import os
from tempfile import mkstemp
from shutil import move
from kubernetes.client.configuration import Configuration
from kubernetes.client.api_client import ApiClient

# 导入临时的 API 客户端，用于解决已知的 bug
from api.api_client_temp import ApiClientTemp

# 定义全局变量
api_temp = None
CoreV1Api = None
RbacAuthorizationV1Api = None
configuration = None
api_version = None

# 判断是否在容器中运行
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

# 初始化 API 客户端
def api_init(kube_config_file=None, host=None, token_filename=None, cert_filename=None, context=None):
    global CoreV1Api
    global RbacAuthorizationV1Api
    global api_temp
    global configuration
    global api_version
    if host and token_filename:
        # 使用远程主机和 token 进行初始化
        token_filename = os.path.abspath(token_filename)
        if cert_filename:
            cert_filename = os.path.abspath(cert_filename)
        configuration = BearerTokenLoader(host=host, token_filename=token_filename, cert_filename=cert_filename).load_and_set()

        CoreV1Api = client.CoreV1Api()
        RbacAuthorizationV1Api = client.RbacAuthorizationV1Api()
        api_temp = ApiClientTemp(configuration=configuration)

    elif kube_config_file:
        # 使用 kube 配置文件进行初始化
        config.load_kube_config(os.path.abspath(kube_config_file))
        CoreV1Api = client.CoreV1Api()
        RbacAuthorizationV1Api = client.RbacAuthorizationV1Api()
        api_from_config = config.new_client_from_config(kube_config_file)
        api_temp = ApiClientTemp(configuration=api_from_config.configuration)
    else:
        # 使用默认配置进行初始化
        configuration = Configuration()

        kubeconfig_path = os.getenv('KUBISCAN_CONFIG_PATH')
        if running_in_container() and kubeconfig_path is None:
            # 在容器中运行时，加载 kube 配置文件
            container_volume_prefix = os.getenv('KUBISCAN_VOLUME_PATH', '/tmp')
            kube_config_bak_path = os.getenv('KUBISCAN_CONFIG_BACKUP_PATH', '/opt/kubiscan/config_bak')
            if not os.path.isfile(kube_config_bak_path):
                copyfile(container_volume_prefix + os.path.expandvars('$CONF_PATH'), kube_config_bak_path)
                replace(kube_config_bak_path, ': /', f': {container_volume_prefix}/')

            config.load_kube_config(kube_config_bak_path, context=context, client_configuration=configuration)
        else:
            config.load_kube_config(config_file=kubeconfig_path, context=context, client_configuration=configuration)

        api_client = ApiClient(configuration=configuration)
        api_version = client.VersionApi(api_client=api_client)
        CoreV1Api = client.CoreV1Api(api_client=api_client)
        RbacAuthorizationV1Api = client.RbacAuthorizationV1Api(api_client=api_client)
        api_temp = ApiClientTemp(configuration=configuration)

# BearerTokenLoader 类，用于加载和设置配置
class BearerTokenLoader(object):
    def __init__(self, host, token_filename, cert_filename=None):
        self._token_filename = token_filename
        self._cert_filename = cert_filename
        self._host = host
        self._verify_ssl = True

        if not self._cert_filename:
            self._verify_ssl = False
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def load_and_set(self):
        self._load_config()
        configuration = self._set_config()
        return configuration

    def _load_config(self):
        self._host = "https://" + self._host

        if not os.path.isfile(self._token_filename):
            raise Exception("Service token file does not exists.")

        with open(self._token_filename) as f:
            self.token = f.read().rstrip('\n')
            if not self.token:
                raise Exception("Token file exists but empty.")

        if self._cert_filename:
            if not os.path.isfile(self._cert_filename):
                raise Exception("Service certification file does not exists.")

            with open(self._cert_filename) as f:
                if not f.read().rstrip('\n'):
                    raise Exception("Cert file exists but empty.")

        self.ssl_ca_cert = self._cert_filename

    def _set_config(self):
        configuration = client.Configuration()
        configuration.host = self._host
        configuration.ssl_ca_cert = self.ssl_ca_cert
        configuration.verify_ssl = self._verify_ssl
        configuration.api_key['authorization'] = "bearer " + self.token
        client.Configuration.set_default(configuration)
        return configuration

```