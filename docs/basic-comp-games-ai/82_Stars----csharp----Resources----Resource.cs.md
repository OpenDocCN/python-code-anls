# `basic-computer-games\82_Stars\csharp\Resources\Resource.cs`

```
// 引入所需的命名空间
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;

// 定义内部静态资源类
namespace Stars.Resources
{
    internal static class Resource
    {
        // 定义内部静态流类
        internal static class Streams
        {
            // 获取标题流
            public static Stream Title => GetStream();
        }

        // 定义内部静态格式类
        internal static class Formats
        {
            // 获取指令格式字符串
            public static string Instructions => GetString();
        }

        // 获取字符串资源的私有方法
        private static string GetString([CallerMemberName] string name = null)
        {
            // 使用获取流的私有方法获取指定名称的流
            using var stream = GetStream(name);
            // 使用流创建读取器
            using var reader = new StreamReader(stream);
            // 读取并返回读取器中的所有内容
            return reader.ReadToEnd();
        }

        // 获取流资源的私有方法
        private static Stream GetStream([CallerMemberName] string name = null)
            // 获取当前执行程序集中指定名称的嵌入资源流
            => Assembly.GetExecutingAssembly().GetManifestResourceStream($"Stars.Resources.{name}.txt");
    }
}
```