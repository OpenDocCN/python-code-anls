# `basic-computer-games\75_Roulette\csharp\Croupier.cs`

```

// 命名空间 Roulette 下的内部类 Croupier
internal class Croupier
{
    // 初始的庄家和玩家的初始资金
    private const int _initialHouse = 100_000;
    private const int _initialPlayer = 1_000;

    // 庄家和玩家当前的资金
    private int _house = _initialHouse;
    private int _player = _initialPlayer;

    // 返回当前庄家和玩家的资金情况
    public string Totals => Strings.Totals(_house, _player);
    // 判断玩家是否破产
    public bool PlayerIsBroke => _player <= 0;
    // 判断庄家是否破产
    public bool HouseIsBroke => _house <= 0;

    // 玩家赢得赌注，更新庄家和玩家的资金情况
    internal string Pay(Bet bet)
    {
        _house -= bet.Payout;
        _player += bet.Payout;

        // 如果庄家破产，重置玩家和庄家的资金
        if (_house <= 0)
        {
            _player = _initialHouse + _initialPlayer;
        }

        return Strings.Win(bet);
    }

    // 玩家输掉赌注，更新庄家和玩家的资金情况
    internal string Take(Bet bet)
    {
        _house += bet.Wager;
        _player -= bet.Wager;

        return Strings.Lose(bet);
    }

    // 发放支票，根据随机数和玩家的资金情况输出支票信息
    public void CutCheck(IReadWrite io, IRandom random)
    {
        var name = io.ReadString(Prompts.Check);
        io.Write(Strings.Check(random, name, _player));
    }
}

```