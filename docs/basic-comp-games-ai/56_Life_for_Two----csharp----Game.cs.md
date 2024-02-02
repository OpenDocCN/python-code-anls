# `basic-computer-games\56_Life_for_Two\csharp\Game.cs`

```py
# 定义一个内部类 Game
internal class Game
{
    # 声明一个只读的 IReadWrite 接口类型的私有变量 _io
    private readonly IReadWrite _io;

    # 构造函数，接受一个 IReadWrite 接口类型的参数 io
    public Game(IReadWrite io)
    {
        # 将参数 io 赋值给私有变量 _io
        _io = io;
    }

    # 游戏进行方法
    public void Play()
    {
        # 调用 _io 的 Write 方法，输出游戏标题
        _io.Write(Streams.Title);

        # 创建一个 Life 对象，传入 _io
        var life = new Life(_io);

        # 调用 _io 的 Write 方法，输出第一代生命
        _io.Write(life.FirstGeneration);

        # 遍历每一代生命
        foreach (var generation in life)
        {
            # 调用 _io 的 WriteLine 方法，输出空行
            _io.WriteLine();
            # 调用 _io 的 Write 方法，输出当前生命代
            _io.Write(generation);
        }

        # 调用 _io 的 WriteLine 方法，输出生命结果或者 "No result"
        _io.WriteLine(life.Result ?? "No result");
    }
}
```