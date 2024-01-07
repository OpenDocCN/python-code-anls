# `basic-computer-games\77_Salvo\csharp\Targetting\KnownHitsShotSelectionStrategy.cs`

```

// 命名空间 Salvo.Targetting 下的内部类 KnownHitsShotSelectionStrategy 继承自 ShotSelectionStrategy
internal class KnownHitsShotSelectionStrategy : ShotSelectionStrategy
{
    // 存储已损坏的船只的列表
    private readonly List<(int Turn, Ship Ship)> _damagedShips = new();

    // 构造函数，接受 ShotSelector 对象作为参数
    internal KnownHitsShotSelectionStrategy(ShotSelector shotSelector)
        : base(shotSelector)
    {
    }

    // 判断是否已知道有损坏的船只
    internal bool KnowsOfDamagedShips => _damagedShips.Any();

    // 重写基类的方法，返回一组射击位置
    internal override IEnumerable<Position> GetShots(int numberOfShots)
    {
        // 创建临时网格，存储位置和对应的值
        var tempGrid = Position.All.ToDictionary(x => x, _ => 0);
        // 创建一组射击位置
        var shots = Enumerable.Range(1, numberOfShots).Select(x => new Position(x, x)).ToArray();

        // 遍历已损坏的船只
        foreach (var (hitTurn, ship) in _damagedShips)
        {
            // 遍历所有位置
            foreach (var position in Position.All)
            {
                // 如果位置已经被选择过，则将其值设为一个极小的负数
                if (WasSelectedPreviously(position))
                {  
                    tempGrid[position]=-10000000;
                    continue;
                }

                // 遍历位置的邻居
                foreach (var neighbour in position.Neighbours)    
                {
                    // 如果邻居位置在相同的回合被选择过，则更新位置的值
                    if (WasSelectedPreviously(neighbour, out var turn) && turn == hitTurn)
                    {
                        tempGrid[position] += hitTurn + 10 - position.Y * ship.Shots;
                    }
                }
            }
        }

        // 遍历所有位置
        foreach (var position in Position.All)
        {
            var Q9=0;
            // 遍历射击位置
            for (var i = 0; i < numberOfShots; i++)
            {
                // 如果当前位置的值小于 Q9 位置的值，则更新 Q9
                if (tempGrid[shots[i]] < tempGrid[shots[Q9]]) 
                { 
                    Q9 = i;
                }
            }
            // 如果位置在射击次数范围内且在对角线上，则跳过
            if (position.X <= numberOfShots && position.IsOnDiagonal) { continue; }
            // 如果位置的值小于 Q9 位置的值，则跳过
            if (tempGrid[position]<tempGrid[shots[Q9]]) { continue; }
            // 如果射击位置数组不包含当前位置，则更新 Q9 位置
            if (!shots.Contains(position))
            {
                shots[Q9] = position;
            }
        }

        return shots; // 返回射击位置数组
    } 

    // 记录船只被击中的回合数
    internal void RecordHit(Ship ship, int turn)
    {
        // 如果船只被摧毁，则从已损坏的船只列表中移除
        if (ship.IsDestroyed) 
        {
            _damagedShips.RemoveAll(x => x.Ship == ship);
        }
        else
        {
            // 否则将船只和回合数添加到已损坏的船只列表中
            _damagedShips.Add((turn, ship));
        }
    }
}

```