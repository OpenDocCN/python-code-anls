# `d:/src/tocomm/basic-computer-games\16_Bug\csharp\Resources\Resource.cs`

```
# 导入所需的模块
import System.Reflection
import System.Runtime.CompilerServices

# 定义命名空间 BugGame.Resources
namespace BugGame.Resources;

# 定义内部静态类 Resource
internal static class Resource
{
    # 定义内部静态类 Streams
    internal static class Streams
    {
        # 定义静态属性 Introduction，返回 GetStream() 方法的结果
        public static Stream Introduction => GetStream();
        # 定义静态属性 Instructions，返回 GetStream() 方法的结果
        public static Stream Instructions => GetStream();
        # 定义静态属性 PlayAgain，返回 GetStream() 方法的结果
        public static Stream PlayAgain => GetStream();
    }

    # 定义私有静态方法 GetStream，参数为调用者的成员名，默认为 null
    private static Stream GetStream([CallerMemberName] string? name = null) =>
        # 获取当前执行程序集，获取嵌入的资源流
        Assembly.GetExecutingAssembly()
            .GetManifestResourceStream($"Bug.Resources.{name}.txt")
            # 如果找不到资源流，则抛出异常
            ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
}
```