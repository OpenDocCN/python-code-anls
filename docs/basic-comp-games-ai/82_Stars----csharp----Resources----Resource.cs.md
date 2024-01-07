# `basic-computer-games\82_Stars\csharp\Resources\Resource.cs`

```

// 引入所需的命名空间
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;

// 定义资源类
namespace Stars.Resources
{
    // 内部静态类，用于存储流资源
    internal static class Resource
    {
        // 内部静态类，用于存储流资源
        internal static class Streams
        {
            // 获取标题流
            public static Stream Title => GetStream();
        }

        // 内部静态类，用于存储格式化的资源
        internal static class Formats
        {
            // 获取指令字符串
            public static string Instructions => GetString();
        }

        // 获取资源字符串
        private static string GetString([CallerMemberName] string name = null)
        {
            // 使用流获取资源字符串
            using var stream = GetStream(name);
            using var reader = new StreamReader(stream);
            return reader.ReadToEnd();
        }

        // 获取资源流
        private static Stream GetStream([CallerMemberName] string name = null)
            => Assembly.GetExecutingAssembly().GetManifestResourceStream($"Stars.Resources.{name}.txt");
    }
}

```