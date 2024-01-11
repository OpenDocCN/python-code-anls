# `KubiScan\api\api_client_temp.py`

```
# 临时的 API 客户端类，用于绕过 bug https://github.com/kubernetes-client/python/issues/577
# 当 bug 被解决后，可以删除这部分代码，并在 utils.py 中使用原始的 API 调用
# 它仅用于 list_cluster_role_binding()

# 编码声明，使用 utf-8 编码
# Kubernetes
# 由 Swagger Codegen https://github.com/swagger-api/swagger-codegen 生成，没有提供描述
# OpenAPI 规范版本：v1.10.1
# 由 https://github.com/swagger-api/swagger-codegen.git 生成

# 导入必要的模块
from __future__ import absolute_import
import os
import re
import json
import mimetypes
import tempfile
from multiprocessing.pool import ThreadPool
from datetime import date, datetime

# 导入兼容 Python 2 和 Python 3 的库
from six import PY3, integer_types, iteritems, text_type
from six.moves.urllib.parse import quote

# 导入 Kubernetes 客户端相关模块
from kubernetes.client import models, V1ObjectMeta, V1RoleRef, V1Subject, V1ClusterRoleBinding, V1ClusterRole, V1ClusterRoleList, V1ClusterRoleBindingList, V1PolicyRule
from kubernetes.client.configuration import Configuration
from kubernetes.client.rest import ApiException, RESTClientObject

# 定义 ApiClientTemp 类
class ApiClientTemp(object):
    """
    通用的 Swagger 客户端库构建 API 客户端。

    Swagger 通用 API 客户端。此客户端处理客户端-服务器通信，并且在所有实现中都是不变的。每个应用程序的方法和模型的具体信息都是从 Swagger 模板生成的。

    注意：此类是由 swagger 代码生成程序自动生成的。
    参考：https://github.com/swagger-api/swagger-codegen
    请勿手动编辑此类。

    :param host: 调用服务器的基本路径。
    :param header_name: 在调用 API 时传递的标头。
    :param header_value: 在调用 API 时传递的标头值。
    """

    # 定义原始类型
    PRIMITIVE_TYPES = (float, bool, bytes, text_type) + integer_types
    # 定义原生类型映射字典，将字符串类型映射为对应的内置类型
    NATIVE_TYPES_MAPPING = {
        'int': int,
        'long': int if PY3 else long,
        'float': float,
        'str': str,
        'bool': bool,
        'date': date,
        'datetime': datetime,
        'object': object,
    }
    
    # 初始化方法，接受配置、头部名称、头部值和 cookie 作为参数
    def __init__(self, configuration=None, header_name=None, header_value=None, cookie=None):
        # 如果配置为空，则创建一个新的配置对象
        if configuration is None:
            configuration = Configuration()
        self.configuration = configuration
    
        # 创建线程池对象
        self.pool = ThreadPool()
        # 创建 REST 客户端对象
        self.rest_client = RESTClientObject(configuration)
        # 初始化默认头部为空字典
        self.default_headers = {}
        # 如果头部名称不为空，则将头部名称和头部值添加到默认头部字典中
        if header_name is not None:
            self.default_headers[header_name] = header_value
        # 设置 cookie
        self.cookie = cookie
        # 设置默认的 User-Agent
        self.user_agent = 'Swagger-Codegen/6.0.0/python'
    
    # 析构方法，关闭线程池
    def __del__(self):
        self.pool.close()
        # 'self.pool.join()' causes a hang in some environments
        # self.pool.join()
    
    # 定义 user_agent 属性的 getter 方法
    @property
    def user_agent(self):
        """
        Gets user agent.
        """
        return self.default_headers['User-Agent']
    
    # 定义 user_agent 属性的 setter 方法
    @user_agent.setter
    def user_agent(self, value):
        """
        Sets user agent.
        """
        self.default_headers['User-Agent'] = value
    
    # 设置默认头部的方法，接受头部名称和头部值作为参数
    def set_default_header(self, header_name, header_value):
        self.default_headers[header_name] = header_value
    # 对象序列化，构建一个 JSON POST 对象
    def sanitize_for_serialization(self, obj):
        """
        Builds a JSON POST object.

        If obj is None, return None.
        If obj is str, int, long, float, bool, return directly.
        If obj is datetime.datetime, datetime.date
            convert to string in iso8601 format.
        If obj is list, sanitize each element in the list.
        If obj is dict, return the dict.
        If obj is swagger model, return the properties dict.

        :param obj: The data to serialize.
        :return: The serialized form of data.
        """
        # 如果 obj 为 None，则返回 None
        if obj is None:
            return None
        # 如果 obj 是基本类型，则直接返回
        elif isinstance(obj, self.PRIMITIVE_TYPES):
            return obj
        # 如果 obj 是列表，则对列表中的每个元素进行序列化
        elif isinstance(obj, list):
            return [self.sanitize_for_serialization(sub_obj)
                    for sub_obj in obj]
        # 如果 obj 是元组，则对元组中的每个元素进行序列化
        elif isinstance(obj, tuple):
            return tuple(self.sanitize_for_serialization(sub_obj)
                         for sub_obj in obj)
        # 如果 obj 是日期时间类型，则转换为 iso8601 格式的字符串
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()

        # 如果 obj 是字典，则直接返回
        if isinstance(obj, dict):
            obj_dict = obj
        else:
            # 将模型对象转换为字典，除了属性 `swagger_types`, `attribute_map`
            # 以及值不为 None 的属性。将属性名转换为模型定义中的 JSON 键
            obj_dict = {obj.attribute_map[attr]: getattr(obj, attr)
                        for attr, _ in iteritems(obj.swagger_types)
                        if getattr(obj, attr) is not None}

        # 对字典中的每个键值对进行序列化
        return {key: self.sanitize_for_serialization(val)
                for key, val in iteritems(obj_dict)}
    # 将响应反序列化为对象
    def deserialize(self, response, response_type):
        """
        Deserializes response into an object.

        :param response: RESTResponse object to be deserialized.
        :param response_type: class literal for
            deserialized object, or string of class name.

        :return: deserialized object.
        """
        # 处理文件下载
        # 将响应体保存到临时文件中，并返回实例
        if response_type == "file":
            return self.__deserialize_file(response)

        # 从响应对象中获取数据
        try:
            data = json.loads(response.data)
        except ValueError:
            data = response.data

        return data
        # 返回 self.__deserialize(data, response_type)
    # 将数据反序列化为对象
    def __deserialize(self, data, klass):
        """
        Deserializes dict, list, str into an object.

        :param data: dict, list or str.  # 数据可以是字典、列表或字符串
        :param klass: class literal, or string of class name.  # klass可以是类的字面量，也可以是类名的字符串

        :return: object.  # 返回一个对象
        """
        if data is None:  # 如果数据为空，则返回空
            return None

        if type(klass) == str:  # 如果klass是字符串类型
            if klass.startswith('list['):  # 如果klass以'list['开头
                sub_kls = re.match('list\[(.*)\]', klass).group(1)  # 获取列表元素的类名
                return [self.__deserialize(sub_data, sub_kls)  # 递归调用__deserialize函数，将列表元素反序列化
                        for sub_data in data]

            if klass.startswith('dict('):  # 如果klass以'dict('开头
                sub_kls = re.match('dict\(([^,]*), (.*)\)', klass).group(2)  # 获取字典值的类名
                return {k: self.__deserialize(v, sub_kls)  # 递归调用__deserialize函数，将字典值反序列化
                        for k, v in iteritems(data)}

            # 将字符串转换为类
            if klass in self.NATIVE_TYPES_MAPPING:  # 如果klass在NATIVE_TYPES_MAPPING中
                klass = self.NATIVE_TYPES_MAPPING[klass]  # 使用NATIVE_TYPES_MAPPING中的映射
            else:
                klass = getattr(models, klass)  # 否则，将klass转换为模块models中的类

        if klass in self.PRIMITIVE_TYPES:  # 如果klass是原始类型
            return self.__deserialize_primitive(data, klass)  # 调用__deserialize_primitive函数，将数据反序列化为原始类型
        elif klass == object:  # 如果klass是object类型
            return self.__deserialize_object(data)  # 调用__deserialize_object函数，将数据反序列化为对象
        elif klass == date:  # 如果klass是date类型
            return self.__deserialize_date(data)  # 调用__deserialize_date函数，将数据反序列化为日期类型
        elif klass == datetime:  # 如果klass是datetime类型
            return self.__deserialize_datatime(data)  # 调用__deserialize_datatime函数，将数据反序列化为日期时间类型
        else:
            return self.__deserialize_model(data, klass)  # 调用__deserialize_model函数，将数据反序列化为指定类的对象
    # 将参数转换为元组列表，格式化集合数据
    def parameters_to_tuples(self, params, collection_formats):
        """
        Get parameters as list of tuples, formatting collections.

        :param params: Parameters as dict or list of two-tuples
        :param dict collection_formats: Parameter collection formats
        :return: Parameters as list of tuples, collections formatted
        """
        # 初始化新的参数列表
        new_params = []
        # 如果集合格式为空，则初始化为空字典
        if collection_formats is None:
            collection_formats = {}
        # 遍历参数字典或参数元组列表
        for k, v in iteritems(params) if isinstance(params, dict) else params:
            # 如果参数名在集合格式中
            if k in collection_formats:
                # 获取集合格式
                collection_format = collection_formats[k]
                # 如果集合格式为'multi'
                if collection_format == 'multi':
                    # 将参数值拆分为元组，添加到新参数列表中
                    new_params.extend((k, value) for value in v)
                else:
                    # 根据不同的集合格式，使用不同的分隔符
                    if collection_format == 'ssv':
                        delimiter = ' '
                    elif collection_format == 'tsv':
                        delimiter = '\t'
                    elif collection_format == 'pipes':
                        delimiter = '|'
                    else:  # csv is the default
                        delimiter = ','
                    # 将参数值转换为字符串，并使用分隔符连接，添加到新参数列表中
                    new_params.append(
                        (k, delimiter.join(str(value) for value in v)))
            else:
                # 如果参数名不在集合格式中，直接添加到新参数列表中
                new_params.append((k, v))
        # 返回新的参数列表
        return new_params
    def prepare_post_parameters(self, post_params=None, files=None):
        """
        Builds form parameters.

        :param post_params: Normal form parameters.
        :param files: File parameters.
        :return: Form parameters with files.
        """
        # 初始化参数列表
        params = []

        # 如果存在普通表单参数，则将其添加到参数列表中
        if post_params:
            params = post_params

        # 如果存在文件参数
        if files:
            # 遍历文件参数
            for k, v in iteritems(files):
                # 如果文件参数为空，则跳过
                if not v:
                    continue
                # 如果文件参数是列表，则遍历列表
                file_names = v if type(v) is list else [v]
                for n in file_names:
                    # 以二进制只读模式打开文件
                    with open(n, 'rb') as f:
                        # 获取文件名和文件数据
                        filename = os.path.basename(f.name)
                        filedata = f.read()
                        # 获取文件的 MIME 类型
                        mimetype = mimetypes. \
                                       guess_type(filename)[0] or 'application/octet-stream'
                        # 将文件参数添加到参数列表中
                        params.append(tuple([k, tuple([filename, filedata, mimetype])]))

        # 返回构建好的参数列表
        return params

    def select_header_accept(self, accepts):
        """
        Returns `Accept` based on an array of accepts provided.

        :param accepts: List of headers.
        :return: Accept (e.g. application/json).
        """
        # 如果没有提供 Accept 头部，则返回空
        if not accepts:
            return

        # 将 Accept 头部的值转换为小写
        accepts = [x.lower() for x in accepts]

        # 如果 Accept 头部包含 'application/json'，则返回 'application/json'
        if 'application/json' in accepts:
            return 'application/json'
        else:
            # 否则，将 Accept 头部的值连接起来返回
            return ', '.join(accepts)

    def select_header_content_type(self, content_types):
        """
        Returns `Content-Type` based on an array of content_types provided.

        :param content_types: List of content-types.
        :return: Content-Type (e.g. application/json).
        """
        # 如果没有提供 Content-Type 头部，则默认返回 'application/json'
        if not content_types:
            return 'application/json'

        # 将 Content-Type 头部的值转换为小写
        content_types = [x.lower() for x in content_types]

        # 如果 Content-Type 头部包含 'application/json' 或者 '*/*'，则返回 'application/json'
        if 'application/json' in content_types or '*/*' in content_types:
            return 'application/json'
        else:
            # 否则，返回第一个 Content-Type 头部的值
            return content_types[0]
    # 根据认证设置更新请求的头部和查询参数
    def update_params_for_auth(self, headers, querys, auth_settings):
        """
        根据认证设置更新头部和查询参数。

        :param headers: 要更新的头部参数字典。
        :param querys: 要更新的查询参数元组列表。
        :param auth_settings: 认证设置标识符列表。
        """
        # 如果没有认证设置，则直接返回
        if not auth_settings:
            return

        # 遍历认证设置
        for auth in auth_settings:
            # 获取认证设置
            auth_setting = self.configuration.auth_settings().get(auth)
            # 如果存在认证设置
            if auth_setting:
                # 如果值为空，则继续下一个循环
                if not auth_setting['value']:
                    continue
                # 如果认证设置在头部
                elif auth_setting['in'] == 'header':
                    headers[auth_setting['key']] = auth_setting['value']
                # 如果认证设置在查询参数
                elif auth_setting['in'] == 'query':
                    querys.append((auth_setting['key'], auth_setting['value']))
                # 如果既不在头部也不在查询参数，则抛出数值错误
                else:
                    raise ValueError(
                        'Authentication token must be in `query` or `header`'
                    )

    # 将响应体保存到临时文件夹中的文件，如果提供了`Content-Disposition`头部，则使用其中的文件名
    def __deserialize_file(self, response):
        """
        将响应体保存到临时文件夹中的文件，如果提供了`Content-Disposition`头部，则使用其中的文件名。

        :param response: RESTResponse。
        :return: 文件路径。
        """
        # 在临时文件夹中创建临时文件
        fd, path = tempfile.mkstemp(dir=self.configuration.temp_folder_path)
        os.close(fd)
        os.remove(path)

        # 获取`Content-Disposition`头部
        content_disposition = response.getheader("Content-Disposition")
        # 如果存在`Content-Disposition`头部
        if content_disposition:
            # 从中获取文件名
            filename = re. \
                search(r'filename=[\'"]?([^\'"\s]+)[\'"]?', content_disposition). \
                group(1)
            # 更新文件路径为临时文件夹中的文件名
            path = os.path.join(os.path.dirname(path), filename)

        # 将响应数据写入文件
        with open(path, "w") as f:
            f.write(response.data)

        # 返回文件路径
        return path
    # 反序列化字符串为基本类型

    def __deserialize_primitive(self, data, klass):
        """
        Deserializes string to primitive type.

        :param data: str.  # 输入的字符串数据
        :param klass: class literal.  # 要转换的目标类型

        :return: int, long, float, str, bool.  # 返回转换后的基本类型数据
        """
        try:
            return klass(data)  # 尝试将字符串转换为指定类型的数据
        except UnicodeEncodeError:
            return unicode(data)  # 如果转换失败，返回Unicode编码的字符串
        except TypeError:
            return data  # 如果转换失败，返回原始数据

    # 反序列化对象
    def __deserialize_object(self, value):
        """
        Return a original value.

        :return: object.  # 返回原始对象
        """
        return value  # 直接返回输入的对象

    # 反序列化字符串为日期
    def __deserialize_date(self, string):
        """
        Deserializes string to date.

        :param string: str.  # 输入的日期字符串
        :return: date.  # 返回日期对象
        """
        try:
            from dateutil.parser import parse
            return parse(string).date()  # 使用dateutil库解析日期字符串并返回日期对象
        except ImportError:
            return string  # 如果导入dateutil库失败，返回原始字符串
        except ValueError:
            raise ApiException(
                status=0,
                reason="Failed to parse `{0}` into a date object".format(string)  # 如果解析失败，抛出异常
            )

    # 反序列化字符串为日期时间
    def __deserialize_datatime(self, string):
        """
        Deserializes string to datetime.

        The string should be in iso8601 datetime format.

        :param string: str.  # 输入的日期时间字符串
        :return: datetime.  # 返回日期时间对象
        """
        try:
            from dateutil.parser import parse
            return parse(string)  # 使用dateutil库解析日期时间字符串并返回日期时间对象
        except ImportError:
            return string  # 如果导入dateutil库失败，返回原始字符串
        except ValueError:
            raise ApiException(
                status=0,
                reason=(
                    "Failed to parse `{0}` into a datetime object"  # 如果解析失败，抛出异常
                        .format(string)
                )
            )
    # 反序列化列表或字典为模型对象
    def __deserialize_model(self, data, klass):
        """
        Deserializes list or dict to model.

        :param data: dict, list.  # 数据，可以是字典或列表
        :param klass: class literal.  # 类型的字面量
        :return: model object.  # 返回模型对象
        """

        # 如果类没有 swagger_types 属性并且没有 get_real_child_model 方法，则直接返回数据
        if not klass.swagger_types and not hasattr(klass, 'get_real_child_model'):
            return data

        kwargs = {}
        # 如果类有 swagger_types 属性，则遍历其中的属性和类型
        if klass.swagger_types is not None:
            for attr, attr_type in iteritems(klass.swagger_types):
                # 如果数据不为空，并且属性在数据中，并且数据是列表或字典类型
                if data is not None \
                        and klass.attribute_map[attr] in data \
                        and isinstance(data, (list, dict)):
                    value = data[klass.attribute_map[attr]]
                    # 反序列化属性值并存入 kwargs 中
                    kwargs[attr] = self.__deserialize(value, attr_type)

        # 使用 kwargs 创建类的实例
        instance = klass(**kwargs)

        # 如果实例有 get_real_child_model 方法
        if hasattr(instance, 'get_real_child_model'):
            # 获取实际子模型的类名
            klass_name = instance.get_real_child_model(data)
            # 如果存在类名，则使用该类名进行反序列化
            if klass_name:
                instance = self.__deserialize(data, klass_name)
        # 返回实例
        return instance
    # 列出集群角色绑定
    def list_cluster_role_binding(self):
        # 调用 API 获取 JSON 数据
        json_data =  self.__call_api(resource_path='/apis/rbac.authorization.k8s.io/v1/clusterrolebindings', method='GET',
                   path_params={}, query_params=[],
                   header_params={'Content-Type': 'application/json', 'Accept': 'application/json'},
                   body=None, post_params=[], files={},
                   response_type='V1ClusterRoleBindingList', auth_settings=['BearerToken'],
                   _return_http_data_only=None, collection_formats={}, _preload_content=True,
                   _request_timeout=None)
        # 初始化集群角色绑定列表
        cluster_role_bindings = []
        # 遍历 JSON 数据中的每个条目
        for i in json_data[0]['items']:

             # 创建元数据对象
             metadata = V1ObjectMeta(name=i['metadata']['name'], creation_timestamp=self._ApiClientTemp__deserialize_datatime(i['metadata']['creationTimestamp']))
             # 创建角色引用对象
             role_ref = V1RoleRef(api_group=i['roleRef']['apiGroup'], name=i['roleRef']['name'], kind=i['roleRef']['kind'])
             # 初始化主体列表
             subjects = []

             # 如果条目中包含主体信息
             if 'subjects' in i and i['subjects'] is not None:
                 # 遍历主体信息
                 for s in i['subjects']:
                       namespace = None
                       # 如果主体信息中包含命名空间
                       if 'namespace' in s.keys():
                           namespace = s['namespace']
                       # 创建主体对象并添加到主体列表中
                       subjects.append(V1Subject(kind=s['kind'], name=s['name'], namespace=namespace))

             # 创建集群角色绑定对象并添加到集群角色绑定列表中
             cluster_role_binding = V1ClusterRoleBinding(metadata=metadata, role_ref=role_ref, subjects=subjects)
             cluster_role_bindings.append(cluster_role_binding)

        # 返回集群角色绑定列表
        return cluster_role_bindings
    # 列出集群角色的方法
    def list_cluster_role(self):
        # 调用 API 获取集群角色的 JSON 数据
        json_data = self.__call_api('/apis/rbac.authorization.k8s.io/v1/clusterroles', 'GET',
                                        path_params={}, query_params=[],
                                        header_params={'Content-Type': 'application/json', 'Accept': 'application/json'},
                                        body=None, post_params=[], files={},
                                        response_type='V1ClusterRoleList', auth_settings=['BearerToken'],
                                        _return_http_data_only=None, collection_formats={}, _preload_content=True,
                                        _request_timeout=None)
        # 初始化集群角色列表
        cluster_roles = []
        # 遍历 JSON 数据中的每个角色
        for i in json_data[0]['items']:
            # 获取角色的元数据
            metadata = V1ObjectMeta(name=i['metadata']['name'],
                                    creation_timestamp=self._ApiClientTemp__deserialize_datatime(
                                        i['metadata']['creationTimestamp']))

            # 初始化规则列表
            rules = []
            # 如果角色有规则
            if i['rules'] is not None:
                # 遍历每个规则
                for rule in i['rules']:
                    resources = None
                    # 如果规则中包含资源
                    if 'resources' in rule.keys():
                        resources = rule['resources']
                    verbs = None
                    # 如果规则中包含动作
                    if 'verbs' in rule.keys():
                        verbs = rule['verbs']

                    # 将资源和动作组成规则对象，加入规则列表
                    rules.append(V1PolicyRule(resources=resources, verbs=verbs))

            # 创建集群角色对象，加入集群角色列表
            cluster_role = V1ClusterRole(kind='ClusterRole', metadata=metadata, rules=rules)
            cluster_roles.append(cluster_role)

        # 返回集群角色列表
        return V1ClusterRoleList(items=cluster_roles)
```