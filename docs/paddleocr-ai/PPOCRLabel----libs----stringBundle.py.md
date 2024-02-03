# `.\PaddleOCR\PPOCRLabel\libs\stringBundle.py`

```py
# 版权声明，允许在特定条件下使用和分发软件
# 获取本程序文件路径
__dir__ = os.path.dirname(os.path.abspath(__file__))
# 获取资源文件夹路径
__dirpath__ = os.path.abspath(os.path.join(__dir__, '../resources/strings'))

# 尝试导入 PyQt5.QtCore 模块，如果失败则导入 PyQt4.QtCore 模块
try:
    from PyQt5.QtCore import *
except ImportError:
    if sys.version_info.major >= 3:
        import sip
        sip.setapi('QVariant', 2)
    from PyQt4.QtCore import *

# 定义 StringBundle 类
class StringBundle:

    # 创建一个私有的标识对象，用于限制类的实例化
    __create_key = object()
    # 初始化 StringBundle 对象，确保使用 StringBundle.getBundle 方法创建
    def __init__(self, create_key, localeStr):
        assert(create_key == StringBundle.__create_key), "StringBundle must be created using StringBundle.getBundle"
        # 初始化 idToMessage 字典，用于存储字符串 id 到消息的映射
        self.idToMessage = {}
        # 创建查找路径列表
        paths = self.__createLookupFallbackList(localeStr)
        # 遍历路径列表，加载对应的资源文件
        for path in paths:
            self.__loadBundle(path)

    # 类方法，用于获取 StringBundle 对象
    @classmethod
    def getBundle(cls, localeStr=None):
        # 如果未指定 localeStr，则尝试获取系统默认 locale
        if localeStr is None:
            try:
                localeStr = locale.getlocale()[0] if locale.getlocale() and len(
                    locale.getlocale()) > 0 else os.getenv('LANG')
            except:
                print('Invalid locale')
                localeStr = 'en'
        # 返回创建的 StringBundle 对象
        return StringBundle(cls.__create_key, localeStr)

    # 根据字符串 id 获取对应的消息
    def getString(self, stringId):
        assert(stringId in self.idToMessage), "Missing string id : " + stringId
        return self.idToMessage[stringId]

    # 创建查找资源文件的路径列表
    def __createLookupFallbackList(self, localeStr):
        # 初始化结果路径列表
        resultPaths = []
        # 基础路径，根据操作系统不同使用不同的分隔符
        basePath = "\strings" if os.name == 'nt' else "/strings"
        resultPaths.append(basePath)
        # 如果指定了 localeStr
        if localeStr is not None:
            # 不遵循标准的 BCP47，简单的回退策略
            tags = re.split('[^a-zA-Z]', localeStr)
            # 遍历标签，构建路径
            for tag in tags:
                lastPath = resultPaths[-1]
                resultPaths.append(lastPath + '-' + tag)
            # 更新最后一个路径，加上文件后缀
            resultPaths[-1] = __dirpath__ + resultPaths[-1] + ".properties"

        return resultPaths
    # 加载指定路径的资源包
    def __loadBundle(self, path):
        # 属性分隔符
        PROP_SEPERATOR = '='
        # 创建 QFile 对象
        f = QFile(path)
        # 如果文件存在
        if f.exists():
            # 以只读和文本模式打开文件
            if f.open(QIODevice.ReadOnly | QFile.Text):
                # 创建 QTextStream 对象
                text = QTextStream(f)
                # 设置编解码为 UTF-8

            # 循环读取文件内容直到结束
            while not text.atEnd():
                # 读取一行文本并转换为 Unicode 字符串
                line = ustr(text.readLine())
                # 使用属性分隔符分割键值对
                key_value = line.split(PROP_SEPERATOR)
                # 获取键
                key = key_value[0].strip()
                # 获取值并去除首尾空格和引号
                value = PROP_SEPERATOR.join(key_value[1:]).strip().strip('"')
                # 将键值对存入字典中
                self.idToMessage[key] = value

            # 关闭文件
            f.close()
```