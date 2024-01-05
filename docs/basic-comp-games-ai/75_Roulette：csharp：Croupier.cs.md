# `75_Roulette\csharp\Croupier.cs`

```
namespace Roulette;  // 命名空间声明

internal class Croupier  // 内部类声明
{
    private const int _initialHouse = 100_000;  // 声明私有常量 _initialHouse 并赋值为 100000
    private const int _initialPlayer = 1_000;  // 声明私有常量 _initialPlayer 并赋值为 1000

    private int _house = _initialHouse;  // 声明私有变量 _house 并赋初值为 _initialHouse
    private int _player = _initialPlayer;  // 声明私有变量 _player 并赋初值为 _initialPlayer

    public string Totals => Strings.Totals(_house, _player);  // 公共属性 Totals 返回 _house 和 _player 的字符串表示
    public bool PlayerIsBroke => _player <= 0;  // 公共属性 PlayerIsBroke 返回 _player 是否小于等于 0
    public bool HouseIsBroke => _house <= 0;  // 公共属性 HouseIsBroke 返回 _house 是否小于等于 0

    internal string Pay(Bet bet)  // 内部方法 Pay 接受一个 Bet 类型的参数并返回字符串
    {
        _house -= bet.Payout;  // _house 减去赌注的支付金额
        _player += bet.Payout;  // _player 增加赌注的支付金额

        if (_house <= 0)  // 如果 _house 小于等于 0
        {
            _player = _initialHouse + _initialPlayer;  # 计算玩家的初始金额和初始赌注的总和，并赋值给_player变量
        }

        return Strings.Win(bet);  # 返回一个表示赢得赌注的字符串
    }

    internal string Take(Bet bet)
    {
        _house += bet.Wager;  # 将赌注的金额加到_house变量中
        _player -= bet.Wager;  # 将赌注的金额从_player变量中减去

        return Strings.Lose(bet);  # 返回一个表示失去赌注的字符串
    }

    public void CutCheck(IReadWrite io, IRandom random)
    {
        var name = io.ReadString(Prompts.Check);  # 从输入输出接口中读取一个字符串，用作名字
        io.Write(Strings.Check(random, name, _player));  # 将一个表示检查的字符串写入输入输出接口
    }
# 关闭 ZIP 对象
zip.close()  # 关闭 ZIP 对象，释放资源，避免内存泄漏。
```