# `58_Love\csharp\Resources\Resource.cs`

```
using System.IO;  // 导入 IO 类库，用于处理文件和流
using System.Reflection;  // 导入 Reflection 类库，用于获取程序集信息
using System.Runtime.CompilerServices;  // 导入 CompilerServices 类库，用于调用者成员名称的特性

namespace Love.Resources;  // 命名空间声明

internal static class Resource  // 声明一个内部静态类 Resource
{
    internal static class Streams  // 声明一个内部静态类 Streams
    {
        public static Stream Intro => GetStream();  // 声明一个公共静态属性 Intro，返回 GetStream() 方法的结果
    }

    private static Stream GetStream([CallerMemberName] string name = null)  // 声明一个私有静态方法 GetStream，参数为调用者成员名称，默认值为 null
        => Assembly.GetExecutingAssembly().GetManifestResourceStream($"Love.Resources.{name}.txt");  // 获取当前执行的程序集的资源流

}
```