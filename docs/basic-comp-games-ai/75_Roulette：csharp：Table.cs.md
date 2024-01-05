# `75_Roulette\csharp\Table.cs`

```
namespace Roulette;  # 命名空间声明

internal class Table  # 声明一个内部类 Table
{
    private readonly IReadWrite _io;  # 声明一个只读字段 _io，类型为 IReadWrite 接口
    private readonly Wheel _wheel;  # 声明一个只读字段 _wheel，类型为 Wheel 类
    private readonly Croupier _croupier;  # 声明一个只读字段 _croupier，类型为 Croupier 类

    public Table(Croupier croupier, IReadWrite io, IRandom random)  # Table 类的构造函数，接受 Croupier、IReadWrite 和 IRandom 三个参数
    {
        _croupier = croupier;  # 将传入的 croupier 参数赋值给 _croupier 字段
        _io = io;  # 将传入的 io 参数赋值给 _io 字段
        _wheel = new(random);  # 使用传入的 random 参数创建一个新的 Wheel 对象，并赋值给 _wheel 字段
    }

    public bool Play()  # 声明一个公共方法 Play，返回布尔类型
    {
        var bets = AcceptBets();  # 调用 AcceptBets 方法，将返回值赋给变量 bets
        var slot = SpinWheel();  # 调用 SpinWheel 方法，将返回值赋给变量 slot
        SettleBets(bets, slot);  # 调用 SettleBets 方法，传入参数 bets 和 slot
        _io.Write(_croupier.Totals);  # 将 _croupier.Totals 写入输出流 _io 中

        if (_croupier.PlayerIsBroke || _croupier.HouseIsBroke) { return false; }  # 如果玩家破产或庄家破产，则返回 false

        return _io.ReadString(Prompts.Again).ToLowerInvariant().StartsWith('y');  # 从输入流 _io 中读取字符串，转换为小写后检查是否以 'y' 开头，然后返回结果
    }

    private Slot SpinWheel()
    {
        _io.Write(Streams.Spinning);  # 将字符串 Streams.Spinning 写入输出流 _io 中
        var slot = _wheel.Spin();  # 从轮盘 _wheel 中旋转并返回一个 Slot 对象
        _io.Write(slot.Name);  # 将 slot 的名称写入输出流 _io 中
        return slot;  # 返回生成的 Slot 对象
    }

    private IReadOnlyList<Bet> AcceptBets()
    {
        var betCount = _io.ReadBetCount();  # 从输入流 _io 中读取下注数量
        var betTypes = new HashSet<BetType>();  # 创建一个 BetType 的哈希集合
        // 创建一个名为bets的Bet类型的列表
        var bets = new List<Bet>();
        // 循环遍历betCount次
        for (int i = 1; i <= betCount; i++)
        {
            // 当无法成功添加赌注时，循环执行直到成功为止
            while (!TryAdd(_io.ReadBet(i)))
            {
                // 写入Streams.BetAlready的内容
                _io.Write(Streams.BetAlready);
            }
        }

        // 将bets列表转换为只读的形式并返回
        return bets.AsReadOnly();

        // 尝试添加赌注到列表中
        bool TryAdd(Bet bet)
        {
            // 如果赌注类型尚未存在于betTypes集合中
            if (betTypes.Add(bet.Type))
            {
                // 将赌注添加到bets列表中并返回true
                bets.Add(bet);
                return true;
            }

            // 如果赌注类型已经存在于betTypes集合中，则返回false
            return false;
    private void SettleBets(IReadOnlyList<Bet> bets, Slot slot)
    {
        // 遍历下注列表中的每一个下注对象
        foreach (var bet in bets)
        {
            // 判断该下注是否覆盖了当前的槽位
            // 如果覆盖了，则调用 _croupier.Pay 方法结算下注
            // 如果没有覆盖，则调用 _croupier.Take 方法收回下注
            _io.Write(slot.IsCoveredBy(bet) ? _croupier.Pay(bet) : _croupier.Take(bet));
        }
    }
}
```