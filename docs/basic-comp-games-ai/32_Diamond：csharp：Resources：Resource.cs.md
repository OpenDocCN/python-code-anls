# `32_Diamond\csharp\Resources\Resource.cs`

```
# 使用 System.Reflection 和 System.Runtime.CompilerServices 模块
using System.Reflection;
using System.Runtime.CompilerServices;

# 定义 Diamond 资源的命名空间
namespace Diamond.Resources;

# 定义 Resource 类
internal static class Resource
{
    # 定义 Streams 类
    internal static class Streams
    {
        # 定义 Introduction 属性，返回流对象
        public static Stream Introduction => GetStream();
    }

    # 定义 Prompts 类
    internal static class Prompts
    {
        # 定义 TypeNumber 属性，返回字符串
        public static string TypeNumber => GetString();
    }

    # 定义 GetString 方法，用于获取字符串
    private static string GetString([CallerMemberName] string? name = null)
    {
        # 使用 GetStream 方法获取流对象
        using var stream = GetStream(name);
        using var reader = new StreamReader(stream);  # 使用 StreamReader 类创建一个从流中读取数据的对象
        return reader.ReadToEnd();  # 读取 StreamReader 对象中的所有数据并返回

    }

    private static Stream GetStream([CallerMemberName] string? name = null) =>  # 创建一个私有的静态方法，返回一个流对象，方法名为调用者的成员名
        Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")  # 获取当前执行程序集中嵌入资源的流对象，资源名称由 Resource 类的命名空间和传入的方法名组成
            ?? throw new Exception($"Could not find embedded resource stream '{name}'.");  # 如果找不到指定的嵌入资源流，则抛出异常
}
```