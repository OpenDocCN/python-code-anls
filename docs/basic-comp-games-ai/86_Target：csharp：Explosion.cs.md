# `d:/src/tocomm/basic-computer-games\86_Target\csharp\Explosion.cs`

```
namespace Target
{
    internal class Explosion
    {
        private readonly Point _position; // 保存爆炸位置的私有变量

        public Explosion(Point position, Offset targetOffset) // 构造函数，接受位置和目标偏移作为参数
        {
            _position = position; // 将传入的位置赋值给私有变量
            FromTarget = targetOffset; // 将传入的目标偏移赋值给属性
            DistanceToTarget = targetOffset.Distance; // 将目标偏移的距离赋值给属性
        }

        public Point Position => _position; // 获取爆炸位置的属性
        public Offset FromTarget { get; } // 获取目标偏移的属性
        public float DistanceToTarget { get; } // 获取到目标的距离的属性
        public string GetBearing() => _position.GetBearing(); // 获取爆炸位置的方位

        public bool IsHit => DistanceToTarget <= 20; // 判断是否击中目标
        public bool IsTooClose => _position.Distance < 20; // 判断爆炸位置是否离目标太近
    }
```

这部分代码是一个缩进错误，应该删除这两行代码。
```