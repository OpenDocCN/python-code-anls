# `75_Roulette\csharp\Game.cs`

```
namespace Roulette;  // 命名空间声明

internal class Game  // 内部类 Game
{
    private readonly IReadWrite _io;  // 只读字段 _io，类型为 IReadWrite 接口
    private readonly IRandom _random;  // 只读字段 _random，类型为 IRandom 接口
    private readonly Table _table;  // 只读字段 _table，类型为 Table 类
    private readonly Croupier _croupier;  // 只读字段 _croupier，类型为 Croupier 类

    public Game(IReadWrite io, IRandom random)  // Game 类的构造函数，接受 IReadWrite 和 IRandom 接口类型的参数
    {
        _io = io;  // 将传入的 io 参数赋值给 _io 字段
        _random = random;  // 将传入的 random 参数赋值给 _random 字段
        _croupier = new();  // 实例化 Croupier 类并赋值给 _croupier 字段
        _table = new(_croupier, io, random);  // 实例化 Table 类并赋值给 _table 字段，传入 _croupier, io, random 参数
    }

    public void Play()  // Play 方法
    {
        _io.Write(Streams.Title);  // 调用 _io 的 Write 方法，传入 Streams.Title 参数
```
        # 如果用户输入的字符串不是以 'n' 开头，就执行下面的代码块
        if (!_io.ReadString(Prompts.Instructions).ToLowerInvariant().StartsWith('n'))
        {
            # 向用户输出游戏指令
            _io.Write(Streams.Instructions);
        }

        # 当玩家还有钱时，一直执行下面的代码块
        while (_table.Play());

        # 如果玩家破产了，向用户输出最后一句话并结束游戏
        if (_croupier.PlayerIsBroke)
        {
            _io.Write(Streams.LastDollar);
            _io.Write(Streams.Thanks);
            return;
        }

        # 如果庄家破产了，向用户输出庄家破产的消息
        if (_croupier.HouseIsBroke)
        {
            _io.Write(Streams.BrokeHouse);
        }

        # 让庄家结账
        _croupier.CutCheck(_io, _random);
    }
```

这部分代码是一个缩进错误，应该删除。
```