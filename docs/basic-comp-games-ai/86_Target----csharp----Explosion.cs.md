# `basic-computer-games\86_Target\csharp\Explosion.cs`

```

// 命名空间 Target，包含爆炸类
namespace Target
{
    // 爆炸类，包含位置信息
    internal class Explosion
    {
        // 爆炸位置
        private readonly Point _position;

        // 构造函数，初始化爆炸位置和目标偏移量
        public Explosion(Point position, Offset targetOffset)
        {
            _position = position;
            FromTarget = targetOffset;
            DistanceToTarget = targetOffset.Distance;
        }

        // 获取爆炸位置
        public Point Position => _position;
        // 获取目标偏移量
        public Offset FromTarget { get; }
        // 获取到目标的距离
        public float DistanceToTarget { get; }
        // 获取爆炸位置相对于目标的方向
        public string GetBearing() => _position.GetBearing();

        // 判断是否击中目标
        public bool IsHit => DistanceToTarget <= 20;
        // 判断爆炸位置是否离目标太近
        public bool IsTooClose => _position.Distance < 20;
    }
}

```