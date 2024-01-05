# `51_Hurkle\csharp\GamePoint.cs`

```
namespace hurkle
{
    internal class GamePoint
    {
        public int X {get;init;}  // 定义一个公共属性 X，用于存储游戏点的横坐标
        public int Y {get;init;}  // 定义一个公共属性 Y，用于存储游戏点的纵坐标

        public CardinalDirection GetDirectionTo(GamePoint target)  // 定义一个方法，用于获取当前点到目标点的方向
        {
            if(X == target.X)  // 如果当前点的横坐标等于目标点的横坐标
            {
                if(Y > target.Y)  // 如果当前点的纵坐标大于目标点的纵坐标
                {
                    return CardinalDirection.South;  // 返回南方向
                }
                else if(Y < target.Y)  // 如果当前点的纵坐标小于目标点的纵坐标
                {
                    return CardinalDirection.North;  // 返回北方向
                }
                else  // 如果当前点的纵坐标等于目标点的纵坐标
                {
                    return CardinalDirection.None;  # 如果X和Y均等于目标点的坐标，则返回None
                }
            }
            else if(X > target.X)  # 如果X大于目标点的X坐标
            {
                if(Y == target.Y)  # 如果Y等于目标点的Y坐标
                {
                    return CardinalDirection.West;  # 返回West
                }
                else if(Y > target.Y)  # 如果Y大于目标点的Y坐标
                {
                    return CardinalDirection.SouthWest;  # 返回SouthWest
                }
                else  # 如果Y小于目标点的Y坐标
                {
                    return CardinalDirection.NorthWest;  # 返回NorthWest
                }
            }
            else  # 如果X小于目标点的X坐标
# 如果当前位置的 Y 坐标等于目标位置的 Y 坐标
if(Y == target.Y)
{
    # 返回东方作为方向
    return CardinalDirection.East;
}
# 如果当前位置的 Y 坐标大于目标位置的 Y 坐标
else if(Y > target.Y)
{
    # 返回东南方作为方向
    return CardinalDirection.SouthEast;
}
# 如果当前位置的 Y 坐标小于目标位置的 Y 坐标
else{
    # 返回东北方作为方向
    return CardinalDirection.NorthEast;
}
```