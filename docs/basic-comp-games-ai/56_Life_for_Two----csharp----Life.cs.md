# `basic-computer-games\56_Life_for_Two\csharp\Life.cs`

```
// 定义一个名为 Life 的类，实现了 IEnumerable 接口，泛型参数为 Generation 类型
internal class Life : IEnumerable<Generation>
{
    // 声明一个私有的 IReadWrite 类型的字段 _io
    private readonly IReadWrite _io;

    // Life 类的构造函数，接受一个 IReadWrite 类型的参数 io
    public Life(IReadWrite io)
    {
        // 将传入的 io 参数赋值给 _io 字段
        _io = io;
        // 调用 Generation 类的静态方法 Create，创建第一代生命
        FirstGeneration = Generation.Create(io);
    }

    // 声明一个公共的 FirstGeneration 属性，类型为 Generation
    public Generation FirstGeneration { get; }
    // 声明一个可空的字符串类型的 Result 属性，并设置私有的 set 方法
    public string? Result { get; private set; }
    
    // 实现 IEnumerable 接口的 GetEnumerator 方法
    public IEnumerator<Generation> GetEnumerator()
    {
        // 初始化当前生命代为第一代
        var current = FirstGeneration;
        // 当当前生命代的结果为空时，循环执行以下操作
        while (current.Result is null)
        {
            // 计算下一代生命
            current = current.CalculateNextGeneration();
            // 返回当前生命代
            yield return current;

            // 如果当前生命代的结果为空，则将其添加到 _io 中
            if (current.Result is null) { current.AddPieces(_io); }
        }

        // 将当前生命代的结果赋值给 Result 属性
        Result = current.Result;
    }

    // 实现 IEnumerable 接口的非泛型 GetEnumerator 方法，调用泛型 GetEnumerator 方法
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator(); 
}
```