# `d:/src/tocomm/basic-computer-games\13_Bounce\csharp\Resources\Resource.cs`

```
# 导入必要的模块
import System.Reflection
import System.Runtime.CompilerServices

# 定义命名空间和类
namespace Bounce.Resources;

internal static class Resource
{
    # 定义内部静态类 Streams
    internal static class Streams
    {
        # 定义静态属性 Instructions，返回 GetStream() 方法的结果
        public static Stream Instructions => GetStream();
        # 定义静态属性 Title，返回 GetStream() 方法的结果
        public static Stream Title => GetStream();
    }

    # 定义私有静态方法 GetStream，参数为调用者的成员名
    private static Stream GetStream([CallerMemberName] string? name = null)
        # 获取当前执行程序集的资源流，资源名称为 "Bounce.Resources.{name}.txt"
        => Assembly.GetExecutingAssembly().GetManifestResourceStream($"Bounce.Resources.{name}.txt")
            # 如果资源流不存在，则抛出参数异常
            ?? throw new ArgumentException($"Resource stream {name} does not exist", nameof(name));
}
```