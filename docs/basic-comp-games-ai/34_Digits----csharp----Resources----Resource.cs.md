# `34_Digits\csharp\Resources\Resource.cs`

```
# 导入必要的模块
import System.Reflection
import System.Runtime.CompilerServices

# 定义命名空间
namespace Digits.Resources;

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
        # 定义静态属性 TryAgain，返回 GetStream() 方法的结果
        public static Stream TryAgain => GetStream()
        # 定义静态属性 ItsATie，返回 GetStream() 方法的结果
        public static Stream ItsATie => GetStream()
        # 定义静态属性 IWin，返回 GetStream() 方法的结果
        public static Stream IWin => GetStream()
        # 定义静态属性 YouWin，返回 GetStream() 方法的结果
        public static Stream YouWin => GetStream()
        # 定义静态属性 Thanks，返回 GetStream() 方法的结果
        public static Stream Thanks => GetStream()
        # 定义静态属性 Headings，返回 GetStream() 方法的结果
        public static Stream Headings => GetStream()
    }

    # 定义内部静态类 Prompts
    internal static class Prompts
    {
        public static string ForInstructions => GetString();  # 返回指示的字符串
        public static string TenNumbers => GetString();  # 返回包含十个数字的字符串
        public static string WantToTryAgain => GetString();  # 返回提示再次尝试的字符串
    }

    internal static class Formats
    {
        public static string GuessResult => GetString();  # 返回猜测结果的字符串
    }

    private static string GetString([CallerMemberName] string? name = null)  # 定义一个方法，根据调用者的成员名获取对应的字符串
    {
        using var stream = GetStream(name);  # 使用获取流的方法获取对应的流
        using var reader = new StreamReader(stream);  # 使用流创建一个读取器
        return reader.ReadToEnd();  # 读取并返回流中的所有内容
    }


    private static Stream GetStream([CallerMemberName] string? name = null) =>  # 定义一个方法，根据调用者的成员名获取对应的流
# 使用 Assembly 类的 GetExecutingAssembly 方法获取当前执行的程序集，再使用 GetManifestResourceStream 方法获取嵌入资源的流
# 参数为嵌入资源的命名空间和文件名
# 如果找不到嵌入资源的流，则抛出异常
```