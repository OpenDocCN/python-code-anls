# `basic-computer-games\75_Roulette\csharp\Table.cs`

```py
// 命名空间 Roulette 下的内部类 Table
internal class Table
{
    // 只读字段 _io，_wheel，_croupier
    private readonly IReadWrite _io;
    private readonly Wheel _wheel;
    private readonly Croupier _croupier;

    // 构造函数，接受 Croupier，IReadWrite，IRandom 三个参数
    public Table(Croupier croupier, IReadWrite io, IRandom random)
    {
        // 将参数赋值给对应的字段
        _croupier = croupier;
        _io = io;
        _wheel = new Wheel(random);
    }

    // 游戏方法
    public bool Play()
    {
        // 接受玩家的赌注
        var bets = AcceptBets();
        // 旋转轮盘
        var slot = SpinWheel();
        // 结算赌注
        SettleBets(bets, slot);

        // 输出庄家的总数
        _io.Write(_croupier.Totals);

        // 如果玩家破产或庄家破产，则返回 false
        if (_croupier.PlayerIsBroke || _croupier.HouseIsBroke) { return false; }

        // 询问玩家是否再玩一次，如果是则返回 true，否则返回 false
        return _io.ReadString(Prompts.Again).ToLowerInvariant().StartsWith('y');
    }

    // 旋转轮盘
    private Slot SpinWheel()
    {
        // 输出旋转中
        _io.Write(Streams.Spinning);
        // 旋转轮盘并返回结果
        var slot = _wheel.Spin();
        // 输出结果
        _io.Write(slot.Name);
        return slot;
    }

    // 接受赌注
    private IReadOnlyList<Bet> AcceptBets()
    {
        // 读取赌注数量
        var betCount = _io.ReadBetCount();
        // 创建赌注类型的哈希集合
        var betTypes = new HashSet<BetType>();
        // 创建赌注列表
        var bets = new List<Bet>();
        // 循环读取赌注
        for (int i = 1; i <= betCount; i++)
        {
            // 如果赌注已存在，则继续读取
            while (!TryAdd(_io.ReadBet(i)))
            {
                _io.Write(Streams.BetAlready);
            }
        }

        // 返回只读的赌注列表
        return bets.AsReadOnly();

        // 尝试添加赌注到列表中
        bool TryAdd(Bet bet)
        {
            if (betTypes.Add(bet.Type))
            {
                bets.Add(bet);
                return true;
            }

            return false;
        }
    }

    // 结算赌注
    private void SettleBets(IReadOnlyList<Bet> bets, Slot slot)
    {
        // 遍历赌注列表，根据轮盘结果结算赌注
        foreach (var bet in bets)
        {
            _io.Write(slot.IsCoveredBy(bet) ? _croupier.Pay(bet) : _croupier.Take(bet));
        }
    }
}
```