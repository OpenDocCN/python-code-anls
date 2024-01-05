# `30_Cube\csharp\Resources\Resource.cs`

```
# 导入所需的模块
import System.Reflection
import System.Runtime.CompilerServices

# 定义命名空间 Cube.Resources
namespace Cube.Resources;

# 定义内部静态类 Resource
internal static class Resource
{
    # 定义内部静态类 Streams
    internal static class Streams
    {
        # 定义静态属性 Introduction，返回 GetStream() 方法的结果
        public static Stream Introduction => GetStream()
        # 定义静态属性 Instructions，返回 GetStream() 方法的结果
        public static Stream Instructions => GetStream()
        # 定义静态属性 Wager，返回 GetStream() 方法的结果
        public static Stream Wager => GetStream()
        # 定义静态属性 IllegalMove，返回 GetStream() 方法的结果
        public static Stream IllegalMove => GetStream()
        # 定义静态属性 Bang，返回 GetStream() 方法的结果
        public static Stream Bang => GetStream()
        # 定义静态属性 Bust，返回 GetStream() 方法的结果
        public static Stream Bust => GetStream()
        # 定义静态属性 Congratulations，返回 GetStream() 方法的结果
        public static Stream Congratulations => GetStream()
        # 定义静态属性 Goodbye，返回 GetStream() 方法的结果
        public static Stream Goodbye => GetStream()
    }

    # 定义内部静态类 Prompts
    internal static class Prompts
    {
        public static string HowMuch => GetString();  # 返回字符串"HowMuch"对应的值
        public static string BetAgain => GetString();  # 返回字符串"BetAgain"对应的值
        public static string YourMove => GetString();  # 返回字符串"YourMove"对应的值
        public static string NextMove => GetString();  # 返回字符串"NextMove"对应的值
        public static string TryAgain => GetString();  # 返回字符串"TryAgain"对应的值
    }

    internal static class Formats
    {
        public static string Balance => GetString();  # 返回字符串"Balance"对应的值
    }

    private static string GetString([CallerMemberName] string? name = null)
    {
        using var stream = GetStream(name);  # 使用给定的名称获取对应的流
        using var reader = new StreamReader(stream);  # 使用流创建一个读取器
        return reader.ReadToEnd();  # 读取并返回读取器中的所有内容
    }
# 从调用者的成员名获取流
def GetStream(name=None):
    # 获取当前执行程序集的嵌入资源流
    stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(f"{typeof(Resource).Namespace}.{name}.txt")
    # 如果找不到嵌入资源流，则抛出异常
    if stream is None:
        raise Exception(f"Could not find embedded resource stream '{name}'.")
    return stream
```