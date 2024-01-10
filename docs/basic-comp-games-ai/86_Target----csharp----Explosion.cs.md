# `basic-computer-games\86_Target\csharp\Explosion.cs`

```
namespace Target
{
    internal class Explosion
    {
        private readonly Point _position; // 保存爆炸位置的私有字段

        public Explosion(Point position, Offset targetOffset) // 构造函数，接受爆炸位置和目标偏移作为参数
        {
            _position = position; // 初始化爆炸位置
            FromTarget = targetOffset; // 初始化目标偏移
            DistanceToTarget = targetOffset.Distance; // 初始化到目标的距离
        }

        public Point Position => _position; // 获取爆炸位置的属性
        public Offset FromTarget { get; } // 获取目标偏移的属性
        public float DistanceToTarget { get; } // 获取到目标的距离的属性
        public string GetBearing() => _position.GetBearing(); // 获取爆炸位置的方位

        public bool IsHit => DistanceToTarget <= 20; // 判断是否击中目标
        public bool IsTooClose => _position.Distance < 20; // 判断是否离目标太近
    }
}
```