# `basic-computer-games\70_Poetry\csharp\Resources\Resource.cs`

```py
# 使用 System.Reflection 和 System.Runtime.CompilerServices 命名空间
using System.Reflection;
using System.Runtime.CompilerServices;

# 定义 Poetry.Resources 命名空间
namespace Poetry.Resources;

# 定义 Resource 类
internal static class Resource
{
    # 定义 Streams 类
    internal static class Streams
    {
        # 定义 Title 属性，返回 GetStream 方法的结果
        public static Stream Title => GetStream();
    }

    # 定义 GetStream 方法，使用 CallerMemberName 特性获取方法名，默认为 null
    private static Stream GetStream([CallerMemberName] string? name = null) =>
        # 获取当前执行的程序集，并获取嵌入资源流
        Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
            # 如果资源流为空，则抛出异常
            ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
}
```