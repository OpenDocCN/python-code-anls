# `basic-computer-games\51_Hurkle\csharp\GamePoint.cs`

```py
namespace hurkle
{
    internal class GamePoint
    {
        // 定义 X 坐标属性
        public int X {get;init;}
        // 定义 Y 坐标属性
        public int Y {get;init;}

        // 获取当前点到目标点的方向
        public CardinalDirection GetDirectionTo(GamePoint target)
        {
            // 如果 X 坐标相等
            if(X == target.X)
            {
                // 如果 Y 坐标大于目标点的 Y 坐标，返回南方向
                if(Y > target.Y)
                {
                    return CardinalDirection.South;
                }
                // 如果 Y 坐标小于目标点的 Y 坐标，返回北方向
                else if(Y < target.Y)
                {
                    return CardinalDirection.North;
                }
                // 如果 Y 坐标相等，返回无方向
                else
                {
                    return CardinalDirection.None;
                }
            }
            // 如果 X 坐标大于目标点的 X 坐标
            else if(X > target.X)
            {
                // 如果 Y 坐标相等，返回西方向
                if(Y == target.Y)
                {
                    return CardinalDirection.West;
                }
                // 如果 Y 坐标大于目标点的 Y 坐标，返回西南方向
                else if(Y > target.Y)
                {
                    return CardinalDirection.SouthWest;
                }
                // 如果 Y 坐标小于目标点的 Y 坐标，返回西北方向
                else
                {
                    return CardinalDirection.NorthWest;
                }
            }
            // 如果 X 坐标小于目标点的 X 坐标
            else
            {
                // 如果 Y 坐标相等，返回东方向
                if(Y == target.Y)
                {
                    return CardinalDirection.East;
                }
                // 如果 Y 坐标大于目标点的 Y 坐标，返回东南方向
                else if(Y > target.Y)
                {
                    return CardinalDirection.SouthEast;
                }
                // 如果 Y 坐标小于目标点的 Y 坐标，返回东北方向
                else{
                    return CardinalDirection.NorthEast;
                }
            }
        }
    }
}
```