# `basic-computer-games\29_Craps\csharp\CrapsTester\CrapsTests.cs`

```

// 引入 Craps 命名空间和 Microsoft.VisualStudio.TestTools.UnitTesting 命名空间
using Craps;
using Microsoft.VisualStudio.TestTools.UnitTesting;

// 声明一个名为 DiceTests 的类
namespace CrapsTester
{
    [TestClass]
    public class DiceTests
    {
        // 声明一个名为 SixSidedDiceReturnsValidRolls 的测试方法
        [TestMethod]
        public void SixSidedDiceReturnsValidRolls()
        {
            // 创建一个六面骰子对象
            var dice = new Dice();
            // 循环进行 100000 次骰子投掷，并验证结果
            for (int i = 0; i < 100000; i++)
            {
                var roll = dice.Roll();
                Assert.IsTrue(roll >= 1 && roll <= dice.sides);
            }
        }

        // 声明一个名为 TwentySidedDiceReturnsValidRolls 的测试方法
        [TestMethod]
        public void TwentySidedDiceReturnsValidRolls()
        {
            // 创建一个二十面骰子对象
            var dice = new Dice(20);
            // 循环进行 100000 次骰子投掷，并验证结果
            for (int i = 0; i < 100000; i++)
            {
                var roll = dice.Roll();
                Assert.IsTrue(roll >= 1 && roll <= dice.sides);
            }
        }

        // 声明一个名为 DiceRollsAreRandom 的测试方法
        [TestMethod]
        public void DiceRollsAreRandom()
        {
            // 抛掷 600,000 次骰子，并统计每个面出现的次数

            var dice = new Dice();

            int numOnes = 0;
            int numTwos = 0;
            int numThrees = 0;
            int numFours = 0;
            int numFives = 0;
            int numSixes = 0;
            int numErrors = 0;

            for (int i = 0; i < 600000; i++)
            {
                switch (dice.Roll())
                {
                    case 1:
                        numOnes++;
                        break;

                    case 2:
                        numTwos++;
                        break;

                    case 3:
                        numThrees++;
                        break;

                    case 4:
                        numFours++;
                        break;

                    case 5:
                        numFives++;
                        break;

                    case 6:
                        numSixes++;
                        break;

                    default:
                        numErrors++;
                        break;
                }
            }

            // 假设每个面的出现次数变化在 10% 以内是足够随机的
            // 完全随机的情况下，每个面的出现次数应该是 100000，加减 5% 的范围是 90000 到 110000
            const int minRolls = 95000;
            const int maxRolls = 105000;
            Assert.IsTrue(numOnes >= minRolls && numOnes <= maxRolls);
            Assert.IsTrue(numTwos >= minRolls && numTwos <= maxRolls);
            Assert.IsTrue(numThrees >= minRolls && numThrees <= maxRolls);
            Assert.IsTrue(numFours >= minRolls && numFours <= maxRolls);
            Assert.IsTrue(numFives >= minRolls && numFives <= maxRolls);
            Assert.IsTrue(numSixes >= minRolls && numSixes <= maxRolls);
            Assert.AreEqual(numErrors, 0);
        }
    }
}

```