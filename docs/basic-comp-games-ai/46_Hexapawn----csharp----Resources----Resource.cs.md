# `basic-computer-games\46_Hexapawn\csharp\Resources\Resource.cs`

```

// 引入所需的命名空间
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;

// 命名空间 Hexapawn.Resources
namespace Hexapawn.Resources
{
    // 内部静态类 Resource
    internal static class Resource
    {
        // 内部静态类 Streams
        internal static class Streams
        {
            // 获取指令流
            public static Stream Instructions => GetStream();
            // 获取标题流
            public static Stream Title => GetStream();
        }

        // 获取流的私有静态方法
        private static Stream GetStream([CallerMemberName] string name = null)
            => Assembly.GetExecutingAssembly().GetManifestResourceStream($"Hexapawn.Resources.{name}.txt");
    }
}

```