# `basic-computer-games\46_Hexapawn\csharp\Resources\Resource.cs`

```py
// 使用 System.IO 命名空间，包含文件和流操作相关的类
// 使用 System.Reflection 命名空间，包含程序集和反射相关的类
// 使用 System.Runtime.CompilerServices 命名空间，包含用于编译器服务的类
// 定义 Hexapawn.Resources 命名空间
internal static class Resource
{
    // 定义 Streams 类
    internal static class Streams
    {
        // 获取 Instructions 流
        public static Stream Instructions => GetStream();
        // 获取 Title 流
        public static Stream Title => GetStream();
    }

    // 获取流的私有方法，使用 CallerMemberName 特性获取调用方法的名称
    private static Stream GetStream([CallerMemberName] string name = null)
        => Assembly.GetExecutingAssembly().GetManifestResourceStream($"Hexapawn.Resources.{name}.txt");
}
```