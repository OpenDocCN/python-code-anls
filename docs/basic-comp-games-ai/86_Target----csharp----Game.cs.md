# `basic-computer-games\86_Target\csharp\Game.cs`

```

using System;  // 引入 System 命名空间
using Games.Common.IO;  // 引入 Games.Common.IO 命名空间

namespace Target  // 定义 Target 命名空间
{
    internal class Game  // 定义 Game 类
    {
        private readonly IReadWrite _io;  // 声明私有只读字段 _io，类型为 IReadWrite 接口
        private readonly FiringRange _firingRange;  // 声明私有只读字段 _firingRange，类型为 FiringRange 类
        private int _shotCount;  // 声明私有字段 _shotCount，类型为整数

        public Game(IReadWrite io, FiringRange firingRange)  // 定义 Game 类的构造函数，接受 IReadWrite 和 FiringRange 类型的参数
        {
            _io = io;  // 初始化 _io 字段
            _firingRange = firingRange;  // 初始化 _firingRange 字段
        }

        public void Play()  // 定义 Play 方法
        {
            _shotCount = 0;  // 将 _shotCount 字段初始化为 0
            var target = _firingRange.NextTarget();  // 调用 _firingRange 对象的 NextTarget 方法，获取目标
            _io.WriteLine(target.GetBearing());  // 调用 _io 对象的 WriteLine 方法，输出目标的方位
            _io.WriteLine($"Target sighted: approximate coordinates:  {target}");  // 输出目标的大致坐标

            while (true)  // 进入循环
            {
                _io.WriteLine($"     Estimated distance: {target.EstimateDistance()}");  // 输出估计距离
                _io.WriteLine();  // 输出空行

                var explosion = Shoot();  // 调用 Shoot 方法，获取爆炸对象

                if (explosion.IsTooClose)  // 判断是否爆炸过近
                {
                    _io.WriteLine("You blew yourself up!!");  // 输出爆炸过近的提示
                    return;  // 结束方法
                }

                _io.WriteLine(explosion.GetBearing());  // 输出爆炸的方位

                if (explosion.IsHit)  // 判断是否击中目标
                {
                    ReportHit(explosion.DistanceToTarget);  // 调用 ReportHit 方法，报告击中情况
                    return;  // 结束方法
                }

                ReportMiss(explosion);  // 调用 ReportMiss 方法，报告未击中情况
            }
        }

        private Explosion Shoot()  // 定义 Shoot 方法，返回 Explosion 对象
        {
            var (xDeviation, zDeviation, distance) = _io.Read3Numbers(  // 调用 _io 对象的 Read3Numbers 方法，获取角度偏差和距离
                "Input angle deviation from X, angle deviation from Z, distance");  // 提示用户输入角度偏差和距离
            _shotCount++;  // 射击次数加一
            _io.WriteLine();  // 输出空行

            return _firingRange.Fire(Angle.InDegrees(xDeviation), Angle.InDegrees(zDeviation), distance);  // 调用 _firingRange 对象的 Fire 方法，发射炮弹
        }

        private void ReportHit(float distance)  // 定义 ReportHit 方法，接受距离参数
        {
            _io.WriteLine();  // 输出空行
            _io.WriteLine($" * * * HIT * * *   Target is non-functional");  // 输出击中提示
            _io.WriteLine();  // 输出空行
            _io.WriteLine($"Distance of explosion from target was {distance} kilometers.");  // 输出爆炸距离目标的距离
            _io.WriteLine();  // 输出空行
            _io.WriteLine($"Mission accomplished in {_shotCount} shots.");  // 输出完成任务所需的射击次数
        }

        private void ReportMiss(Explosion explosion)  // 定义 ReportMiss 方法，接受 Explosion 对象参数
        {
            ReportMiss(explosion.FromTarget);  // 调用 ReportMiss 方法，报告未击中情况
            _io.WriteLine($"Approx position of explosion:  {explosion.Position}");  // 输出爆炸的大致位置
            _io.WriteLine($"     Distance from target = {explosion.DistanceToTarget}");  // 输出爆炸距离目标的距离
            _io.WriteLine();  // 输出空行
            _io.WriteLine();  // 输出空行
            _io.WriteLine();  // 输出空行
        }

        private void ReportMiss(Offset targetOffset)  // 定义 ReportMiss 方法，接受 Offset 对象参数
        {
            ReportMiss(targetOffset.DeltaX, "in front of", "behind");  // 调用 ReportMiss 方法，报告 X 轴偏差
            ReportMiss(targetOffset.DeltaY, "to left of", "to right of");  // 调用 ReportMiss 方法，报告 Y 轴偏差
            ReportMiss(targetOffset.DeltaZ, "above", "below");  // 调用 ReportMiss 方法，报告 Z 轴偏差
        }

        private void ReportMiss(float delta, string positiveText, string negativeText)  // 定义 ReportMiss 方法，接受浮点数和两个字符串参数
            => _io.WriteLine(delta >= 0 ? GetOffsetText(positiveText, delta) : GetOffsetText(negativeText, -delta));  // 根据偏差值输出不同的提示信息

        private static string GetOffsetText(string text, float distance)  // 定义 GetOffsetText 方法，接受字符串和浮点数参数
            => $"Shot {text} target {distance} kilometers.";  // 返回偏差提示信息
    }
}

```