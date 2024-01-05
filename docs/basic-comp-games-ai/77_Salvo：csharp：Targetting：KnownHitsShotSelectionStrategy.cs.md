# `d:/src/tocomm/basic-computer-games\77_Salvo\csharp\Targetting\KnownHitsShotSelectionStrategy.cs`

```
namespace Salvo.Targetting;  // 命名空间声明

internal class KnownHitsShotSelectionStrategy : ShotSelectionStrategy  // 声明一个内部类 KnownHitsShotSelectionStrategy，继承自 ShotSelectionStrategy 类
{
    private readonly List<(int Turn, Ship Ship)> _damagedShips = new();  // 声明一个私有只读字段 _damagedShips，类型为元组列表

    internal KnownHitsShotSelectionStrategy(ShotSelector shotSelector)  // 声明一个内部构造函数，接受一个 ShotSelector 类型的参数
        : base(shotSelector)  // 调用基类的构造函数，传入 shotSelector 参数
    {
    }

    internal bool KnowsOfDamagedShips => _damagedShips.Any();  // 声明一个内部属性，返回 _damagedShips 列表是否包含任何元素

    internal override IEnumerable<Position> GetShots(int numberOfShots)  // 声明一个重写的方法，返回一个 Position 类型的可枚举集合，接受一个整数类型的参数
    {
        var tempGrid = Position.All.ToDictionary(x => x, _ => 0);  // 声明一个临时变量 tempGrid，将 Position.All 转换为字典
        var shots = Enumerable.Range(1, numberOfShots).Select(x => new Position(x, x)).ToArray();  // 声明一个变量 shots，使用 LINQ 生成一个包含 numberOfShots 个 Position 对象的数组

        foreach (var (hitTurn, ship) in _damagedShips)  // 遍历 _damagedShips 列表中的元组
        {
# 遍历所有的位置对象
foreach (var position in Position.All)
{
    # 如果之前已经选择过这个位置
    if (WasSelectedPreviously(position))
    {  
        # 将临时网格中对应位置的值设为一个很大的负数
        tempGrid[position]=-10000000;
        # 继续下一个位置的遍历
        continue;
    }

    # 遍历当前位置的所有邻居位置
    foreach (var neighbour in position.Neighbours)    
    {
        # 如果邻居位置之前被选择过，并且是当前的回合
        if (WasSelectedPreviously(neighbour, out var turn) && turn == hitTurn)
        {
            # 在临时网格中对应位置的值增加当前回合数加上10减去位置的Y坐标乘以船只的射击次数
            tempGrid[position] += hitTurn + 10 - position.Y * ship.Shots;
        }
    }
}
# 继续下一个位置的遍历
}
# 初始化变量 Q9 为 0
var Q9=0;
# 遍历射击次数的循环
for (var i = 0; i < numberOfShots; i++)
{
    # 如果当前位置的网格值小于 Q9 位置的网格值
    if (tempGrid[shots[i]] < tempGrid[shots[Q9]]) 
    { 
        # 更新 Q9 为当前位置
        Q9 = i;
    }
}
# 如果位置的 X 坐标小于等于射击次数并且位置在对角线上，则继续下一次循环
if (position.X <= numberOfShots && position.IsOnDiagonal) { continue; }
# 如果当前位置的网格值小于 Q9 位置的网格值，则继续下一次循环
if (tempGrid[position]<tempGrid[shots[Q9]]) { continue; }
# 如果射击位置不在已射击的位置列表中
if (!shots.Contains(position))
{
    # 更新 Q9 位置为当前位置
    shots[Q9] = position;
}

# 返回射击位置列表
return shots;
} 

# 记录击中的船只和回合数
internal void RecordHit(Ship ship, int turn)
# 如果船只被摧毁，从受损船只列表中移除该船只
if (ship.IsDestroyed) 
{
    _damagedShips.RemoveAll(x => x.Ship == ship);
}
# 如果船只没有被摧毁，将其加入受损船只列表，包括回合数和船只对象
else
{
    _damagedShips.Add((turn, ship));
}
```