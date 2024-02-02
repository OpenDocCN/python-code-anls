# `basic-computer-games\58_Love\csharp\Resources\Resource.cs`

```py
// 引入需要使用的命名空间
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;

// 声明命名空间 Love.Resources
namespace Love.Resources
{
    // 声明内部静态类 Resource
    internal static class Resource
    {
        // 声明内部静态类 Streams
        internal static class Streams
        {
            // 声明 Intro 属性，返回 GetStream 方法的结果
            public static Stream Intro => GetStream();
        }

        // 声明私有的 GetStream 方法，使用 CallerMemberName 特性获取调用方法的名称，默认为 null
        private static Stream GetStream([CallerMemberName] string name = null)
            // 返回当前执行程序集的嵌入资源流
            => Assembly.GetExecutingAssembly().GetManifestResourceStream($"Love.Resources.{name}.txt");
    }
}
```