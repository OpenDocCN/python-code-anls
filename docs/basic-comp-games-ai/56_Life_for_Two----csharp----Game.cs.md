# `basic-computer-games\56_Life_for_Two\csharp\Game.cs`

```

// 定义一个名为 Game 的内部类
internal class Game
{
    // 声明一个私有的 IReadWrite 接口类型的变量 _io
    private readonly IReadWrite _io;

    // Game 类的构造函数，接受一个 IReadWrite 类型的参数 io
    public Game(IReadWrite io)
    {
        // 将传入的 io 参数赋值给 _io 变量
        _io = io;
    }

    // 定义一个名为 Play 的方法
    public void Play()
    {
        // 使用 _io 对象的 Write 方法输出 Streams.Title
        _io.Write(Streams.Title);

        // 创建一个名为 life 的 Life 对象，传入 _io 对象作为参数
        var life = new Life(_io);

        // 使用 _io 对象的 Write 方法输出 life 对象的 FirstGeneration 属性值
        _io.Write(life.FirstGeneration);

        // 遍历 life 对象的每一代
        foreach (var generation in life)
        {
            // 使用 _io 对象的 WriteLine 方法输出空行
            _io.WriteLine();
            // 使用 _io 对象的 Write 方法输出当前代的数据
            _io.Write(generation);
        }

        // 使用 _io 对象的 WriteLine 方法输出 life 对象的 Result 属性值，如果为空则输出 "No result"
        _io.WriteLine(life.Result ?? "No result");
    }
}

```