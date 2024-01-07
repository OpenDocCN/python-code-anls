# `basic-computer-games\51_Hurkle\csharp\GamePoint.cs`

```

// 定义名为 hurkle 的命名空间
namespace hurkle
{
    // 定义名为 GamePoint 的内部类
    internal class GamePoint
    {
        // 定义具有 get 和 init 访问器的 X 属性
        public int X {get;init;}
        // 定义具有 get 和 init 访问器的 Y 属性
        public int Y {get;init;}

        // 根据目标点返回方向
        public CardinalDirection GetDirectionTo(GamePoint target)
        {
            // 如果当前点的 X 坐标与目标点的 X 坐标相等
            if(X == target.X)
            {
                // 如果当前点的 Y 坐标大于目标点的 Y 坐标
                if(Y > target.Y)
                {
                    // 返回南方向
                    return CardinalDirection.South;
                }
                // 如果当前点的 Y 坐标小于目标点的 Y 坐标
                else if(Y < target.Y)
                {
                    // 返回北方向
                    return CardinalDirection.North;
                }
                // 如果当前点的 Y 坐标与目标点的 Y 坐标相等
                else
                {
                    // 返回无方向
                    return CardinalDirection.None;
                }
            }
            // 如果当前点的 X 坐标大于目标点的 X 坐标
            else if(X > target.X)
            {
                // 如果当前点的 Y 坐标与目标点的 Y 坐标相等
                if(Y == target.Y)
                {
                    // 返回西方向
                    return CardinalDirection.West;
                }
                // 如果当前点的 Y 坐标大于目标点的 Y 坐标
                else if(Y > target.Y)
                {
                    // 返回西南方向
                    return CardinalDirection.SouthWest;
                }
                // 如果当前点的 Y 坐标小于目标点的 Y 坐标
                else
                {
                    // 返回西北方向
                    return CardinalDirection.NorthWest;
                }
            }
            // 如果当前点的 X 坐标小于目标点的 X 坐标
            else
            {
                // 如果当前点的 Y 坐标与目标点的 Y 坐标相等
                if(Y == target.Y)
                {
                    // 返回东方向
                    return CardinalDirection.East;
                }
                // 如果当前点的 Y 坐标大于目标点的 Y 坐标
                else if(Y > target.Y)
                {
                    // 返回东南方向
                    return CardinalDirection.SouthEast;
                }
                // 如果当前点的 Y 坐标小于目标点的 Y 坐标
                else{
                    // 返回东北方向
                    return CardinalDirection.NorthEast;
                }
            }
        }
    }
}

```