# `d:/src/tocomm/basic-computer-games\77_Salvo\csharp\Targetting\ShotSelectionStrategy.cs`

```
namespace Salvo.Targetting;

// 创建一个抽象类 ShotSelectionStrategy，用于定义射击选择策略
internal abstract class ShotSelectionStrategy
{
    private readonly ShotSelector _shotSelector; // 声明一个私有的 ShotSelector 类型的成员变量 _shotSelector
    protected ShotSelectionStrategy(ShotSelector shotSelector) // 定义一个受保护的构造函数，接受一个 ShotSelector 类型的参数 shotSelector
    {
        _shotSelector = shotSelector; // 将传入的 shotSelector 赋值给 _shotSelector
    }

    // 定义一个抽象方法 GetShots，用于获取一定数量的射击位置
    internal abstract IEnumerable<Position> GetShots(int numberOfShots);

    // 定义一个受保护的方法 WasSelectedPreviously，用于判断指定位置是否之前已经被选择过
    protected bool WasSelectedPreviously(Position position) => _shotSelector.WasSelectedPreviously(position);

    // 定义一个受保护的方法 WasSelectedPreviously，用于判断指定位置是否之前已经被选择过，并返回选择的轮次
    protected bool WasSelectedPreviously(Position position, out int turn)
        => _shotSelector.WasSelectedPreviously(position, out turn);
}
```