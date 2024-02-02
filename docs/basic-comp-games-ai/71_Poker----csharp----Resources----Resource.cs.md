# `basic-computer-games\71_Poker\csharp\Resources\Resource.cs`

```py
// 引入系统反射和运行时编译服务命名空间
using System.Reflection;
using System.Runtime.CompilerServices;

// 定义名为Poker.Resources的命名空间
namespace Poker.Resources
{
    // 定义名为Resource的静态类
    internal static class Resource
    {
        // 定义名为Streams的静态类
        internal static class Streams
        {
            // 定义名为Instructions的公共静态属性，返回GetStream()方法的结果
            public static Stream Instructions => GetStream();
            // 定义名为Title的公共静态属性，返回GetStream()方法的结果
            public static Stream Title => GetStream();
        }

        // 定义名为GetStream的私有静态方法，参数为调用者成员的名称，默认为null
        private static Stream GetStream([CallerMemberName] string? name = null)
            // 返回当前执行程序集的清单资源流，资源名称为"Poker.Resources.{name}.txt"，如果不存在则抛出ArgumentException异常
            => Assembly.GetExecutingAssembly().GetManifestResourceStream($"Poker.Resources.{name}.txt")
                ?? throw new ArgumentException($"Resource stream {name} does not exist", nameof(name));
    }
}
```