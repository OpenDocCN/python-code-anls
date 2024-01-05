# `d:/src/tocomm/basic-computer-games\29_Craps\csharp\CrapsTester\CrapsTests.cs`

```
using Craps;  # 导入 Craps 模块
using Microsoft.VisualStudio.TestTools.UnitTesting;  # 导入 Microsoft.VisualStudio.TestTools.UnitTesting 模块

namespace CrapsTester:  # 定义 CrapsTester 命名空间
{
    [TestClass]  # 标记 DiceTests 类为测试类
    public class DiceTests  # 定义 DiceTests 类
    {
        [TestMethod]  # 标记 SixSidedDiceReturnsValidRolls 方法为测试方法
        public void SixSidedDiceReturnsValidRolls()  # 定义 SixSidedDiceReturnsValidRolls 方法
        {
            var dice = new Dice();  # 创建 Dice 类的实例对象 dice
            for (int i = 0; i < 100000; i++)  # 循环 100000 次
            {
                var roll = dice.Roll();  # 调用 dice 对象的 Roll 方法，将结果赋值给 roll
                Assert.IsTrue(roll >= 1 && roll <= dice.sides);  # 使用断言判断 roll 的值是否在 1 到 dice.sides 之间
            }
        }
        [TestMethod]  # 标记下面的方法为测试方法
        public void TwentySidedDiceReturnsValidRolls()  # 测试20面骰子返回有效的结果
        {
            var dice = new Dice(20);  # 创建一个20面骰子对象
            for (int i = 0; i < 100000; i++)  # 循环10万次
            {
                var roll = dice.Roll();  # 掷骰子
                Assert.IsTrue(roll >= 1 && roll <= dice.sides);  # 断言骰子的结果在1到20之间
            }
        }

        [TestMethod]  # 标记下面的方法为测试方法
        public void DiceRollsAreRandom()  # 测试骰子的结果是随机的
        {
            // Roll 600,000 dice and count how many rolls there are for each side.
            // 掷60万次骰子并计算每个面出现的次数

            var dice = new Dice();  # 创建一个骰子对象

            int numOnes = 0;  # 初始化计数器
            # 初始化变量，用于记录不同点数出现的次数和错误次数
            int numTwos = 0;
            int numThrees = 0;
            int numFours = 0;
            int numFives = 0;
            int numSixes = 0;
            int numErrors = 0;

            # 循环600000次，模拟掷骰子的结果
            for (int i = 0; i < 600000; i++)
            {
                # 调用骰子对象的Roll方法，获取随机点数
                switch (dice.Roll())
                {
                    # 如果点数为1，numOnes加1
                    case 1:
                        numOnes++;
                        break;

                    # 如果点数为2，numTwos加1
                    case 2:
                        numTwos++;
                        break;

                    # 如果点数为3，numThrees加1
# 增加变量 numThrees 的值
numThrees++;
# 跳出 switch 语句
break;

# 当 switch 语句的值为 4 时，增加变量 numFours 的值
case 4:
    numFours++;
    # 跳出 switch 语句
    break;

# 当 switch 语句的值为 5 时，增加变量 numFives 的值
case 5:
    numFives++;
    # 跳出 switch 语句
    break;

# 当 switch 语句的值为 6 时，增加变量 numSixes 的值
case 6:
    numSixes++;
    # 跳出 switch 语句
    break;

# 当 switch 语句的值不是 3、4、5、6 时，增加变量 numErrors 的值
default:
    numErrors++;
    # 跳出 switch 语句
    break;
// 假设不同数字的投掷变化在10%范围内是足够随机的。
// 完全随机的投掷会产生每个面的100000次投掷，加减5%得到范围90000..110000。
const int minRolls = 95000;  // 最小投掷次数
const int maxRolls = 105000;  // 最大投掷次数
Assert.IsTrue(numOnes >= minRolls && numOnes <= maxRolls);  // 确保投掷次数在范围内
Assert.IsTrue(numTwos >= minRolls && numTwos <= maxRolls);  // 确保投掷次数在范围内
Assert.IsTrue(numThrees >= minRolls && numThrees <= maxRolls);  // 确保投掷次数在范围内
Assert.IsTrue(numFours >= minRolls && numFours <= maxRolls);  // 确保投掷次数在范围内
Assert.IsTrue(numFives >= minRolls && numFives <= maxRolls);  // 确保投掷次数在范围内
Assert.IsTrue(numSixes >= minRolls && numSixes <= maxRolls);  // 确保投掷次数在范围内
Assert.AreEqual(numErrors, 0);  // 确保错误次数为0
```