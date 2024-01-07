# `basic-computer-games\56_Life_for_Two\csharp\Life.cs`

```

// 使用 System.Collections 命名空间
internal class Life : IEnumerable<Generation>
{
    // 私有字段，用于读写操作
    private readonly IReadWrite _io;

    // 构造函数，接受 IReadWrite 接口实例作为参数
    public Life(IReadWrite io)
    {
        _io = io;
        // 创建第一代生命
        FirstGeneration = Generation.Create(io);
    }

    // 公共属性，获取第一代生命
    public Generation FirstGeneration { get; }
    // 公共属性，获取或设置结果
    public string? Result { get; private set; }
    
    // 实现 IEnumerable 接口的 GetEnumerator 方法
    public IEnumerator<Generation> GetEnumerator()
    {
        // 获取当前生命代
        var current = FirstGeneration;
        // 当当前生命代的结果为空时循环
        while (current.Result is null)
        {
            // 计算下一代生命
            current = current.CalculateNextGeneration();
            // 返回当前生命代
            yield return current;

            // 如果当前生命代的结果为空，则添加生命片段
            if (current.Result is null) { current.AddPieces(_io); }
        }

        // 设置结果为当前生命代的结果
        Result = current.Result;
    }

    // 实现 IEnumerable 接口的 GetEnumerator 方法
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator(); 
}

```