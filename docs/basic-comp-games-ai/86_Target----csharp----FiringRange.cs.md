# `basic-computer-games\86_Target\csharp\FiringRange.cs`

```

# 使用 Games.Common.Randomness 命名空间中的随机数接口
using Games.Common.Randomness;

# 定义 FiringRange 类
namespace Target
{
    internal class FiringRange
    {
        # 保存随机数接口的实例
        private readonly IRandom _random;
        # 保存目标位置的坐标
        private Point _targetPosition;

        # 构造函数，接受随机数接口的实例作为参数
        public FiringRange(IRandom random)
        {
            _random = random;
        }

        # 生成下一个目标位置，并将其保存到 _targetPosition 中
        public Point NextTarget() =>  _targetPosition = _random.NextPosition();

        # 发射炮弹，计算爆炸位置和目标位置的偏移，并返回爆炸对象
        public Explosion Fire(Angle angleFromX, Angle angleFromZ, float distance)
        {
            # 计算爆炸位置
            var explosionPosition = new Point(angleFromX, angleFromZ, distance);
            # 计算目标位置和爆炸位置的偏移
            var targetOffset = explosionPosition - _targetPosition;
            # 返回爆炸对象
            return new (explosionPosition, targetOffset);
        }
    }
}

```