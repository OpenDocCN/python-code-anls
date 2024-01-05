# `d:/src/tocomm/basic-computer-games\07_Basketball\csharp\Resources\Resource.cs`

```
# 引入必要的模块
import System.Reflection
import System.Runtime.CompilerServices

# 定义命名空间
namespace Basketball.Resources;

# 定义内部静态类 Resource
internal static class Resource
{
    # 定义内部静态类 Streams
    internal static class Streams
    {
        # 定义静态属性 Introduction，返回流对象
        public static Stream Introduction => GetStream()
        # 定义静态属性 TwoMinutesLeft，返回流对象
        public static Stream TwoMinutesLeft => GetStream()
    }

    # 定义内部静态类 Formats
    internal static class Formats
    {
        # 定义静态属性 EndOfFirstHalf，返回字符串
        public static string EndOfFirstHalf => GetString()
        # 定义静态属性 EndOfGame，返回字符串
        public static string EndOfGame => GetString()
        # 定义静态属性 EndOfSecondHalf，返回字符串
        public static string EndOfSecondHalf => GetString()
        # 定义静态属性 Score，返回字符串
        public static string Score => GetString()
    }
}
    # 从调用方成员名获取字符串
    def GetString(name=None):
        # 获取流
        stream = GetStream(name)
        # 使用流创建读取器
        reader = StreamReader(stream)
        # 读取并返回流中的所有内容
        return reader.ReadToEnd()

    # 获取流
    def GetStream(name=None):
        # 从当前执行的程序集中获取嵌入资源流
        stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(f'Basketball.Resources.{name}.txt')
        # 如果找不到嵌入资源流，则抛出异常
        if stream is None:
            raise Exception(f"Could not find embedded resource stream '{name}'.")
        return stream
```