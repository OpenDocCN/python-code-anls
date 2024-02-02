# `basic-computer-games\77_Salvo\csharp\Targetting\KnownHitsShotSelectionStrategy.cs`

```py
namespace Salvo.Targetting;

internal class KnownHitsShotSelectionStrategy : ShotSelectionStrategy
{
    private readonly List<(int Turn, Ship Ship)> _damagedShips = new();  // 创建一个存储受损船只信息的列表

    internal KnownHitsShotSelectionStrategy(ShotSelector shotSelector)  // 构造函数，接受 ShotSelector 对象作为参数
        : base(shotSelector)  // 调用基类的构造函数，传入 ShotSelector 对象
    {
    }

    internal bool KnowsOfDamagedShips => _damagedShips.Any();  // 判断是否存在受损船只的属性

    internal override IEnumerable<Position> GetShots(int numberOfShots)  // 重写基类的方法，返回一个包含 Position 对象的集合
    {
        var tempGrid = Position.All.ToDictionary(x => x, _ => 0);  // 创建一个包含所有 Position 对象的字典，初始值为 0
        var shots = Enumerable.Range(1, numberOfShots).Select(x => new Position(x, x)).ToArray();  // 创建一个包含指定数量 Position 对象的数组

        foreach (var (hitTurn, ship) in _damagedShips)  // 遍历受损船只列表
        {
            foreach (var position in Position.All)  // 遍历所有 Position 对象
            {
                if (WasSelectedPreviously(position))  // 如果该位置之前已经被选择过
                {  
                    tempGrid[position]=-10000000;  // 将该位置在临时网格中的值设为一个极小的负数
                    continue;  // 继续下一次循环
                }

                foreach (var neighbour in position.Neighbours)  // 遍历当前位置的邻居位置
                {
                    if (WasSelectedPreviously(neighbour, out var turn) && turn == hitTurn)  // 如果邻居位置之前被选择过，并且是在受损船只被击中的回合
                    {
                        tempGrid[position] += hitTurn + 10 - position.Y * ship.Shots;  // 更新临时网格中当前位置的值
                    }
                }
            }
        }

        foreach (var position in Position.All)  // 遍历所有 Position 对象
        {
            var Q9=0;  // 初始化变量 Q9 为 0
            for (var i = 0; i < numberOfShots; i++)  // 遍历指定数量的射击位置
            {
                if (tempGrid[shots[i]] < tempGrid[shots[Q9]])  // 如果当前位置的值小于 Q9 位置的值
                { 
                    Q9 = i;  // 更新 Q9 的值为当前位置的索引
                }
            }
            if (position.X <= numberOfShots && position.IsOnDiagonal) { continue; }  // 如果当前位置在对角线上或者 X 坐标小于等于射击数量，继续下一次循环
            if (tempGrid[position]<tempGrid[shots[Q9]]) { continue; }  // 如果当前位置的值小于 Q9 位置的值，继续下一次循环
            if (!shots.Contains(position))  // 如果射击位置数组不包含当前位置
            {
                shots[Q9] = position;  // 更新射击位置数组中的 Q9 位置为当前位置
            }
        }

        return shots;  // 返回射击位置数组
    } 

    internal void RecordHit(Ship ship, int turn)  // 记录受损船只的方法
    {
        # 如果船只被摧毁
        if (ship.IsDestroyed) 
        {
            # 从受损船只列表中移除该船只
            _damagedShips.RemoveAll(x => x.Ship == ship);
        }
        # 如果船只没有被摧毁
        else
        {
            # 将当前回合和船只添加到受损船只列表中
            _damagedShips.Add((turn, ship));
        }
    }
# 闭合前面的函数定义
```