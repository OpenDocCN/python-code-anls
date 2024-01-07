# `basic-computer-games\91_Train\csharp\TrainTests\TrainGameTests.cs`

```

# 引入Train命名空间
using Train;
# 引入Xunit测试框架
using Xunit;

# 定义TrainGameTests类
namespace TrainTests
{
    public class TrainGameTests
    {
        # 测试生成的随机数是否大于等于给定的最小值
        [Fact]
        public void MiniumRandomNumber()
        {
            TrainGame game = new TrainGame();
            Assert.True(game.GenerateRandomNumber(10, 10) >= 10);
        }

        # 测试生成的随机数是否小于等于给定的最大值
        [Fact]
        public void MaximumRandomNumber()
        {
            TrainGame game = new TrainGame();
            Assert.True(game.GenerateRandomNumber(10, 10) <= 110);
        }

        # 测试输入是否为"y"时返回true
        [Fact]
        public void IsInputYesWhenY()
        {
            Assert.True(TrainGame.IsInputYes("y"));
        }

        # 测试输入不为"y"时返回false
        [Fact]
        public void IsInputYesWhenNotY()
        {
            Assert.False(TrainGame.IsInputYes("a"));
        }

        # 测试计算车辆行程时间
        [Fact]
        public void CarDurationTest()
        {
            Assert.Equal(1, TrainGame.CalculateCarJourneyDuration(30, 1, 15) );
        }

        # 测试两个数值是否在允许的误差范围内
        [Fact]
        public void IsWithinAllowedDifference()
        {
            Assert.True(TrainGame.IsWithinAllowedDifference(5,5));
        }

        # 测试两个数值是否不在允许的误差范围内
        [Fact]
        public void IsNotWithinAllowedDifference()
        {
            Assert.False(TrainGame.IsWithinAllowedDifference(6, 5));
        }
    }
}

```