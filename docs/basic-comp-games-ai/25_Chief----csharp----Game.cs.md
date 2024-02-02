# `basic-computer-games\25_Chief\csharp\Game.cs`

```py
# 引入 Chief.Resources.Resource 命名空间中的静态资源
using static Chief.Resources.Resource;

# 声明 Chief 命名空间
namespace Chief;

# 声明 Game 类
internal class Game
{
    # 声明私有只读字段 _io，类型为 IReadWrite 接口
    private readonly IReadWrite _io;

    # Game 类的构造函数，接受一个 IReadWrite 接口类型的参数 io
    public Game(IReadWrite io)
    {
        # 将参数 io 赋值给私有字段 _io
        _io = io;
    }

    # Play 方法，用于执行游戏
    internal void Play()
    {
        # 执行游戏介绍
        DoIntroduction();

        # 读取用户输入的数字
        var result = _io.ReadNumber(Prompts.Answer);

        # 如果用户输入的数字经过计算后返回 true，则输出消息并结束游戏
        if (_io.ReadYes(Formats.Bet, Math.CalculateOriginal(result)))
        {
            _io.Write(Streams.Bye);
            return;
        }

        # 读取用户输入的原始数字
        var original = _io.ReadNumber(Prompts.Original);

        # 输出计算过程
        _io.WriteLine(Math.ShowWorking(original));

        # 如果用户确认相信计算过程，则输出消息并结束游戏
        if (_io.ReadYes(Prompts.Believe))
        {
            _io.Write(Streams.Bye);
            return;
        }

        # 输出消息
        _io.Write(Streams.Lightning);
    }

    # DoIntroduction 方法，用于执行游戏介绍
    private void DoIntroduction()
    {
        # 输出游戏标题
        _io.Write(Streams.Title);
        # 如果用户不确认准备好，则输出消息
        if (!_io.ReadYes(Prompts.Ready))
        {
            _io.Write(Streams.ShutUp);
        }

        # 输出游戏说明
        _io.Write(Streams.Instructions);
    }
}
```