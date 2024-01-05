# `72_Queen\csharp\Resources\Resource.cs`

```
# 导入必要的模块
import System.Reflection
import System.Runtime.CompilerServices

# 定义命名空间 Queen.Resources
namespace Queen.Resources;

# 定义内部静态类 Resource
internal static class Resource
{
    # 定义内部静态类 Streams
    internal static class Streams
    {
        # 定义静态属性 Title，返回 GetStream() 方法的结果
        public static Stream Title => GetStream();
        # 定义静态属性 Instructions，返回 GetStream() 方法的结果
        public static Stream Instructions => GetStream();
        # 定义静态属性 YesOrNo，返回 GetStream() 方法的结果
        public static Stream YesOrNo => GetStream();
        # 定义静态属性 Board，返回 GetStream() 方法的结果
        public static Stream Board => GetStream();
        # 定义静态属性 IllegalStart，返回 GetStream() 方法的结果
        public static Stream IllegalStart => GetStream();
        # 定义静态属性 IllegalMove，返回 GetStream() 方法的结果
        public static Stream IllegalMove => GetStream();
        # 定义静态属性 Forfeit，返回 GetStream() 方法的结果
        public static Stream Forfeit => GetStream();
        # 定义静态属性 IWin，返回 GetStream() 方法的结果
        public static Stream IWin => GetStream();
        # 定义静态属性 Congratulations，返回 GetStream() 方法的结果
        public static Stream Congratulations => GetStream();
        # 定义静态属性 Thanks，返回 GetStream() 方法的结果
        public static Stream Thanks => GetStream();
    }
}
    // 定义内部静态类 Prompts
    internal static class Prompts
    {
        // 获取指令提示信息
        public static string Instructions => GetPrompt();
        // 获取开始提示信息
        public static string Start => GetPrompt();
        // 获取移动提示信息
        public static string Move => GetPrompt();
        // 获取任何人提示信息
        public static string Anyone => GetPrompt();
    }

    // 定义内部静态类 Strings
    internal static class Strings
    {
        // 获取计算机移动的字符串信息
        public static string ComputerMove(Position position) => string.Format(GetString(), position);
    }

    // 获取提示信息
    private static string GetPrompt([CallerMemberName] string? name = null) => GetString($"{name}Prompt");

    // 获取字符串信息
    private static string GetString([CallerMemberName] string? name = null)
    {
        // 使用流获取字符串信息
        using var stream = GetStream(name);
        // 使用读取器读取流中的信息
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }
```
这行代码的作用是读取流中的所有内容并返回。

```
    private static Stream GetStream([CallerMemberName] string? name = null) =>
        Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
            ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
```
这段代码定义了一个私有的静态方法GetStream，它接受一个可选的参数name，并返回一个流对象。在方法内部，它使用Assembly.GetExecutingAssembly().GetManifestResourceStream方法来获取嵌入资源的流对象，如果找不到对应的资源流，则抛出一个异常。
```