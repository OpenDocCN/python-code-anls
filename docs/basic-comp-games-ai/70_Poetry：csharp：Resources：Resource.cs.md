# `d:/src/tocomm/basic-computer-games\70_Poetry\csharp\Resources\Resource.cs`

```
# 导入必要的模块
import System.Reflection
import System.Runtime.CompilerServices

# 定义命名空间为 Poetry.Resources 的内部静态类 Resource
namespace Poetry.Resources;

internal static class Resource
{
    # 定义 Streams 类
    internal static class Streams
    {
        # 定义 Title 属性，返回 GetStream() 方法的结果
        public static Stream Title => GetStream();
    }

    # 定义 GetStream 方法，参数为调用者的成员名，返回类型为 Stream
    private static Stream GetStream([CallerMemberName] string? name = null) =>
        # 获取当前执行的程序集，获取嵌入的资源流
        Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
            # 如果找不到资源流，则抛出异常
            ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
}
```