# `basic-computer-games\75_Roulette\csharp\Table.cs`

```

// 命名空间 Roulette，定义了一个名为 Table 的内部类
internal class Table
{
    // 私有字段，用于读写操作
    private readonly IReadWrite _io;
    // 私有字段，轮盘对象
    private readonly Wheel _wheel;
    // 私有字段，荷官对象
    private readonly Croupier _croupier;

    // 构造函数，接受荷官、读写操作和随机数生成器作为参数
    public Table(Croupier croupier, IReadWrite io, IRandom random)
    {
        // 初始化荷官
        _croupier = croupier;
        // 初始化读写操作
        _io = io;
        // 初始化轮盘
        _wheel = new Wheel(random);
    }

    // 游戏进行方法
    public bool Play()
    {
        // 接受玩家下注
        var bets = AcceptBets();
        // 旋转轮盘
        var slot = SpinWheel();
        // 结算下注
        SettleBets(bets, slot);

        // 输出荷官的总结
        _io.Write(_croupier.Totals);

        // 如果玩家破产或庄家破产，游戏结束
        if (_croupier.PlayerIsBroke || _croupier.HouseIsBroke) { return false; }

        // 询问玩家是否再玩一次
        return _io.ReadString(Prompts.Again).ToLowerInvariant().StartsWith('y');
    }

    // 旋转轮盘方法
    private Slot SpinWheel()
    {
        // 输出旋转提示
        _io.Write(Streams.Spinning);
        // 旋转轮盘，获取结果
        var slot = _wheel.Spin();
        // 输出结果
        _io.Write(slot.Name);
        return slot;
    }

    // 接受下注方法
    private IReadOnlyList<Bet> AcceptBets()
    {
        // 读取下注数量
        var betCount = _io.ReadBetCount();
        // 下注类型集合
        var betTypes = new HashSet<BetType>();
        // 下注列表
        var bets = new List<Bet>();
        for (int i = 1; i <= betCount; i++)
        {
            // 循环读取下注，直到成功添加
            while (!TryAdd(_io.ReadBet(i)))
            {
                _io.Write(Streams.BetAlready);
            }
        }

        return bets.AsReadOnly();

        // 尝试添加下注到列表的方法
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

    // 结算下注方法
    private void SettleBets(IReadOnlyList<Bet> bets, Slot slot)
    {
        // 遍历下注列表，根据轮盘结果结算
        foreach (var bet in bets)
        {
            _io.Write(slot.IsCoveredBy(bet) ? _croupier.Pay(bet) : _croupier.Take(bet));
        }
    }
}

```