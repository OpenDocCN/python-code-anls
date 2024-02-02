# `basic-computer-games\75_Roulette\csharp\Game.cs`

```py
// 命名空间 Roulette 下的内部类 Game
internal class Game
{
    // 只读字段 _io，用于输入输出
    private readonly IReadWrite _io;
    // 只读字段 _random，用于生成随机数
    private readonly IRandom _random;
    // 只读字段 _table，用于存储游戏桌信息
    private readonly Table _table;
    // 只读字段 _croupier，用于存储荷官信息

    // Game 类的构造函数，接受 IReadWrite 和 IRandom 作为参数
    public Game(IReadWrite io, IRandom random)
    {
        // 将传入的 io 赋值给 _io
        _io = io;
        // 将传入的 random 赋值给 _random
        _random = random;
        // 创建一个新的 Croupier 对象并赋值给 _croupier
        _croupier = new();
        // 创建一个新的 Table 对象并传入 _croupier, io, random 作为参数，赋值给 _table
        _table = new(_croupier, io, random);
    }

    // Play 方法，用于开始游戏
    public void Play()
    {
        // 输出游戏标题
        _io.Write(Streams.Title);
        // 如果用户输入的字符串不以 'n' 开头，则输出游戏说明
        if (!_io.ReadString(Prompts.Instructions).ToLowerInvariant().StartsWith('n'))
        {
            _io.Write(Streams.Instructions);
        }

        // 当游戏桌上还有游戏进行时，循环执行游戏
        while (_table.Play());

        // 如果玩家破产，则输出最后一块钱和感谢信息，然后返回
        if (_croupier.PlayerIsBroke)
        {
            _io.Write(Streams.LastDollar);
            _io.Write(Streams.Thanks);
            return;
        }

        // 如果庄家破产，则输出庄家破产信息
        if (_croupier.HouseIsBroke)
        {
            _io.Write(Streams.BrokeHouse);
        }

        // 让庄家结账
        _croupier.CutCheck(_io, _random);
    }
}
```