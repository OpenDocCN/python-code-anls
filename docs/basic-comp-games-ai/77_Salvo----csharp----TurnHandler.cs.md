# `77_Salvo\csharp\TurnHandler.cs`

```
using Salvo.Targetting;  # 导入 Salvo.Targetting 模块

namespace Salvo;  # 声明 Salvo 命名空间

internal class TurnHandler  # 定义名为 TurnHandler 的内部类
{
    private readonly IReadWrite _io;  # 声明私有只读字段 _io，类型为 IReadWrite 接口
    private readonly Fleet _humanFleet;  # 声明私有只读字段 _humanFleet，类型为 Fleet 类
    private readonly Fleet _computerFleet;  # 声明私有只读字段 _computerFleet，类型为 Fleet 类
    private readonly bool _humanStarts;  # 声明私有只读字段 _humanStarts，类型为布尔值
    private readonly HumanShotSelector _humanShotSelector;  # 声明私有只读字段 _humanShotSelector，类型为 HumanShotSelector 类
    private readonly ComputerShotSelector _computerShotSelector;  # 声明私有只读字段 _computerShotSelector，类型为 ComputerShotSelector 类
    private readonly Func<Winner?> _turnAction;  # 声明私有只读字段 _turnAction，类型为返回 Winner 类型的委托
    private int _turnNumber;  # 声明私有字段 _turnNumber，类型为整数

    public TurnHandler(IReadWrite io, IRandom random)  # 定义名为 TurnHandler 的构造函数，接受 IReadWrite 类型的参数 io 和 IRandom 类型的参数 random
    {
        _io = io;  # 将参数 io 的值赋给字段 _io
        _computerFleet = new Fleet(random);  # 使用参数 random 创建一个新的 Fleet 对象，并将其赋给字段 _computerFleet
        _humanFleet = new Fleet(io);  # 使用参数 io 创建一个新的 Fleet 对象，并将其赋给字段 _humanFleet
        _turnAction = AskWhoStarts()  # 调用AskWhoStarts函数，根据返回结果给_turnAction赋值
            ? () => PlayHumanTurn() ?? PlayComputerTurn()  # 如果AskWhoStarts返回true，则执行PlayHumanTurn函数，否则执行PlayComputerTurn函数
            : () => PlayComputerTurn() ?? PlayHumanTurn();  # 如果AskWhoStarts返回false，则执行PlayComputerTurn函数，否则执行PlayHumanTurn函数
        _humanShotSelector = new HumanShotSelector(_humanFleet, io);  # 创建一个HumanShotSelector对象，传入_humanFleet和io参数
        _computerShotSelector = new ComputerShotSelector(_computerFleet, random, io);  # 创建一个ComputerShotSelector对象，传入_computerFleet、random和io参数
    }

    public Winner? PlayTurn()  # 定义一个PlayTurn函数，返回Winner类型的可空值
    {
        _io.Write(Strings.Turn(++_turnNumber));  # 在_io上调用Write方法，传入Strings.Turn(++_turnNumber)作为参数
        return _turnAction.Invoke();  # 调用_turnAction的Invoke方法，并返回结果
    }

    private bool AskWhoStarts()  # 定义一个AskWhoStarts函数，返回布尔值
    {
        while (true)  # 进入一个无限循环
        {
            var startResponse = _io.ReadString(Prompts.Start);  # 调用_io的ReadString方法，传入Prompts.Start作为参数，并将结果赋值给startResponse变量
            if (startResponse.Equals(Strings.WhereAreYourShips, StringComparison.InvariantCultureIgnoreCase))  # 如果startResponse等于Strings.WhereAreYourShips（忽略大小写）
            {
                # 遍历计算机舰队中的船只
                foreach (var ship in _computerFleet.Ships)
                {
                    # 输出船只信息
                    _io.WriteLine(ship);
                }
            }
            else
            {
                # 返回是否开始游戏的响应，忽略大小写
                return startResponse.Equals("yes", StringComparison.InvariantCultureIgnoreCase);
            }
        }
    }

    # 计算机进行回合
    private Winner? PlayComputerTurn()
    {
        # 获取计算机可以射击的次数
        var numberOfShots = _computerShotSelector.NumberOfShots;
        # 输出计算机射击次数的信息
        _io.Write(Strings.IHaveShots(numberOfShots));
        # 如果没有射击次数，返回人类获胜
        if (numberOfShots == 0) { return Winner.Human; }
        # 如果计算机可以瞄准所有剩余的方块
        if (_computerShotSelector.CanTargetAllRemainingSquares)
        {
            # 输出计算机拥有比方块数更多的射击次数的信息
            _io.Write(Streams.IHaveMoreShotsThanSquares);
        return Winner.Computer;  # 返回计算机获胜

        _humanFleet.ReceiveShots(  # 人类舰队接收射击
            _computerShotSelector.GetShots(_turnNumber),  # 使用计算机射击选择器获取射击
            ship =>  # 对于每艘船
            { 
                _io.Write(Strings.IHit(ship.Name));  # 输出“我击中了（船名）”
                _computerShotSelector.RecordHit(ship, _turnNumber);  # 记录计算机射中的船和回合数
            });

        return null;  # 返回空值
    }

    private Winner? PlayHumanTurn()  # 执行人类回合
    {
        var numberOfShots = _humanShotSelector.NumberOfShots;  # 获取人类射击选择器的射击数量
        _io.Write(Strings.YouHaveShots(numberOfShots));  # 输出“你有（射击数量）次射击机会”
        if (numberOfShots == 0) { return Winner.Computer; }  # 如果射击数量为0，则返回计算机获胜
        if (_humanShotSelector.CanTargetAllRemainingSquares)  # 如果人类射击选择器可以瞄准所有剩余的方块
        { 
            # 输出消息，表示玩家的射击次数多于对方的船只数量
            _io.WriteLine(Streams.YouHaveMoreShotsThanSquares);
            # 返回人类玩家获胜
            return Winner.Human;
        }
        
        # 让计算机舰队接收玩家的射击
        _computerFleet.ReceiveShots(
            _humanShotSelector.GetShots(_turnNumber), 
            # 输出消息，表示击中了对方的船只
            ship => _io.Write(Strings.YouHit(ship.Name)));
        
        # 返回空值
        return null;
    }
}
```