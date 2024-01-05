# `d:/src/tocomm/basic-computer-games\22_Change\csharp\Program.cs`

```
# 引入系统模块
import System

# 定义程序的主类
class Program:
    # 打印标题
    static void Header():
        Console.WriteLine("Change".PadLeft(33))
        Console.WriteLine("Creative Computing Morristown, New Jersey".PadLeft(15))
        Console.WriteLine()
        Console.WriteLine()
        Console.WriteLine()
        Console.WriteLine("I, your friendly microcomputer, will determine\n"
        + "the correct change for items costing up to $100.")
        Console.WriteLine()
        Console.WriteLine()
        }

        /// <summary>
        /// 获取用户输入的价格和付款信息。
        /// </summary>
        /// <returns>
        /// 如果任何输入无法解析为双精度数，则返回 false。返回的价格和付款将为 0。
        /// 如果能够将输入解析为双精度数，则返回 true。返回的价格和付款将为用户提供的值。
	    /// </returns>
        static (bool status, double price, double payment) GetInput()
        {
            Console.Write("物品价格？ ");
            var priceString = Console.ReadLine();
            if (!double.TryParse(priceString, out double price))
            {
                Console.WriteLine($"{priceString} 不是一个数字！");
                return (false, 0, 0);
            }
            // 提示用户输入付款金额
            Console.Write("Amount of payment? ");
            // 读取用户输入的付款金额
            var paymentString = Console.ReadLine();
            // 如果无法将用户输入的付款金额转换为 double 类型
            if (!double.TryParse(paymentString, out double payment))
            {
                // 输出错误信息并返回错误标志和默认值
                Console.WriteLine($"{paymentString} isn't a number!");
                return (false, 0, 0);
            }

            // 返回正确标志和价格以及付款金额
            return (true, price, payment);
        }

        /// <summary>
        /// Prints bills and coins for given change.
        /// </summary>
        /// <param name="change"></param>
        // 打印找零的纸币和硬币
        static void PrintChange(double change)
        {
            // 计算需要的十元纸币数量
            var tens = (int)(change / 10);
            // 如果需要的十元纸币数量大于 0，则输出相应信息
            if (tens > 0)
                Console.WriteLine($"{tens} ten dollar bill(s)");
            // 计算找零金额中的十元纸币数量
            var tens = (int)(change / 10);
            if (tens > 0)
                Console.WriteLine($"{tens} ten dollar bill(s)");

            // 计算找零金额中的五元纸币数量
            var temp = change - (tens * 10);
            var fives = (int)(temp / 5);
            if (fives > 0)
                Console.WriteLine($"{fives} five dollar bill(s)");

            // 计算找零金额中的一元纸币数量
            temp -= fives * 5;
            var ones = (int)temp;
            if (ones > 0)
                Console.WriteLine($"{ones} one dollar bill(s)");

            // 计算找零金额中的零钱数量
            temp -= ones;
            var cents = temp * 100;

            // 计算找零金额中的五角硬币数量
            var half = (int)(cents / 50);
            if (half > 0)
                Console.WriteLine($"{half} one half dollar(s)");

            // 计算找零金额中的二角五分硬币数量
            temp = cents - (half * 50);
            var quarters = (int)(temp / 25);
            if (quarters > 0)
            # 计算需要的 quarter 数量，并打印输出
            Console.WriteLine($"{quarters} quarter(s)");

            # 从总金额中减去 quarters 的价值
            temp -= quarters * 25;
            # 计算需要的 dime 数量，并打印输出
            var dimes = (int)(temp / 10);
            if (dimes > 0)
                Console.WriteLine($"{dimes} dime(s)");

            # 从总金额中减去 dimes 的价值
            temp -= dimes * 10;
            # 计算需要的 nickel 数量，并打印输出
            var nickels = (int)(temp / 5);
            if (nickels > 0)
                Console.WriteLine($"{nickels} nickel(s)");

            # 从总金额中减去 nickels 的价值
            temp -= nickels * 5;
            # 计算需要的 penny 数量，并打印输出
            var pennies = (int)(temp + 0.5);
            if (pennies > 0)
                Console.WriteLine($"{pennies} penny(s)");
        }

        static void Main(string[] args)
        {
            Header();  # 调用名为Header的函数，可能用于显示程序的标题或者欢迎信息

            while (true)  # 进入一个无限循环
            {
                (bool result, double price, double payment) = GetInput();  # 调用名为GetInput的函数，获取用户输入的价格和付款金额，并将结果赋值给result, price, payment变量
                if (!result)  # 如果result为false，即用户输入不合法，跳过本次循环
                    continue;

                var change = payment - price;  # 计算找零金额
                if (change == 0)  # 如果找零金额为0
                {
                    Console.WriteLine("Correct amount, thank you!");  # 输出正确的找零金额
                    continue;  # 跳过本次循环
                }

                if (change < 0)  # 如果找零金额小于0
                {
                    Console.WriteLine($"Sorry, you have short-changed me ${price - payment:N2}!");  # 输出用户付款不足的信息
                    continue;  # 跳过本次循环
                }
# 打印找零金额
Console.WriteLine($"Your change ${change:N2}");
# 调用打印找零函数，打印找零金额的具体面额
PrintChange(change);
# 打印感谢信息
Console.WriteLine("Thank you, come again!");
# 打印空行
Console.WriteLine();
```