# `95_Weekday\csharp\Program.cs`

```
# 使用 System.Text 命名空间
using System.Text;

# Weekday 类
namespace Weekday
{
    class Weekday
    {
        # 显示介绍信息
        private void DisplayIntro()
        {
            Console.WriteLine("");  # 输出空行
            Console.WriteLine("SYNONYM".PadLeft(23));  # 输出左对齐的字符串
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  # 输出字符串
            Console.WriteLine("");  # 输出空行
            Console.WriteLine("Weekday is a computer demonstration that");  # 输出字符串
            Console.WriteLine("gives facts about a date of interest to you.");  # 输出字符串
            Console.WriteLine("");  # 输出空行
        }

        # 验证日期格式是否正确
        private bool ValidateDate(string InputDate, out DateTime ReturnDate)
        {
            # 期望输入的日期格式为 D,M,Y
            // 将输入的日期字符串中的逗号替换为斜杠，以便转换为日期格式
            string DateString = InputDate.Replace(",", "/");

            // 尝试将替换后的日期字符串转换为日期类型，如果成功则将结果赋值给ReturnDate并返回true，否则返回false
            return (DateTime.TryParse(DateString, out ReturnDate));
        }

        private DateTime PromptForADate(string Prompt)
        {
            bool Success = false;
            string LineInput = String.Empty;
            DateTime TodaysDate = DateTime.MinValue;

            // 获取输入的日期并验证
            while (!Success)
            {
                Console.Write(Prompt);
                LineInput = Console.ReadLine().Trim().ToLower();

                // 调用ValidateDate方法验证输入的日期，如果成功则将结果赋值给TodaysDate并将Success设置为true
                Success = ValidateDate(LineInput, out TodaysDate);
                if (!Success)
                {
                    Console.WriteLine("*** Invalid date.  Please try again.");  // 如果日期无效，则打印错误消息
                    Console.WriteLine("");  // 打印空行
                }
            }

            return TodaysDate;  // 返回计算出的日期
        }

        private void CalculateDateDiff(DateTime TodaysDate, DateTime BirthDate, Double Factor, out int AgeInYears, out int AgeInMonths, out int AgeInDays)
        {
            // 利用 Stack Overflow 的答案：https://stackoverflow.com/a/3055445

            // 计算出生日期到今天的时间差，乘以因子后存储为新的因子日期
            TimeSpan TimeDiff = TodaysDate.Subtract(BirthDate);  // 计算时间差
            Double NumberOfDays = TimeDiff.Days * Factor;  // 将时间差转换为天数并乘以因子
            DateTime FactorDate = BirthDate.AddDays(NumberOfDays);  // 根据计算出的天数得到新的日期

            // 计算因子日期（即今天的日期乘以因子）与出生日期之间的差异
            # 计算年龄的月份差
            AgeInMonths = FactorDate.Month - BirthDate.Month;
            # 计算年龄的年份差
            AgeInYears = FactorDate.Year - BirthDate.Year;

            # 如果当前日期的日小于出生日期的日，则月份差减一
            if (FactorDate.Day < BirthDate.Day)
            {
                AgeInMonths--;
            }

            # 如果月份差小于0，则年份差减一，月份差加12
            if (AgeInMonths < 0)
            {
                AgeInYears--;
                AgeInMonths += 12;
            }

            # 计算年龄的天数差
            AgeInDays = (FactorDate - BirthDate.AddMonths((AgeInYears * 12) + AgeInMonths)).Days;

        }

        # 输出年龄信息
        private void WriteColumnOutput(string Message, int Years, int Months, int Days)
        {
            Console.WriteLine("{0,-25} {1,-10:N0} {2,-10:N0} {3,-10:N0}", Message, Years, Months, Days);
            // 在控制台输出格式化的消息，包括消息、年、月、日的数据

        }

        private void DisplayOutput(DateTime TodaysDate, DateTime BirthDate)
        {
            Console.WriteLine("");
            // 在控制台输出空行

            // 如果当前年份早于1582年，则不允许进行操作
            if (TodaysDate.Year < 1582)
            {
                Console.WriteLine("Not prepared to give day of week prior to MDLXXXII.");
                return;
            }

            // 输出生日是星期几
            Console.Write(" {0} ", BirthDate.ToString("d"));

            string DateVerb = "";
            // 初始化一个字符串变量DateVerb
            // 检查出生日期是否在今天日期之前，如果是则设置 DateVerb 为 "was a "
            if (BirthDate.CompareTo(TodaysDate) < 0)
            {
                DateVerb = "was a ";
            }
            // 检查出生日期是否与今天日期相同，如果是则设置 DateVerb 为 "is a "
            else if (BirthDate.CompareTo(TodaysDate) == 0)
            {
                DateVerb = "is a ";
            }
            // 如果出生日期在今天日期之后，则设置 DateVerb 为 "will be a "
            else
            {
                DateVerb = "will be a ";
            }
            // 输出 DateVerb
            Console.Write("{0}", DateVerb);

            // 如果他们的生日是在星期五的13号，输出特殊警告
            if (BirthDate.DayOfWeek.ToString().Equals("Friday") && BirthDate.Day == 13)
            {
                Console.WriteLine("{0} the Thirteenth---BEWARE", BirthDate.DayOfWeek.ToString());
            }
            // 如果不是星期五的13号，则继续执行下面的代码
            else
            {
                // 打印出生日期是星期几
                Console.WriteLine("{0}", BirthDate.DayOfWeek.ToString());
            }

            // 如果今天的日期和生日日期的月份和日期相同，则祝他们生日快乐！
            if (BirthDate.Month == TodaysDate.Month && BirthDate.Day == TodaysDate.Day)
            {
                // 打印生日祝福
                Console.WriteLine("");
                Console.Write("***Happy Birthday***");
            }

            Console.WriteLine("");

            // 只有在生日日期在今天日期之前时才显示日期计算
            if (DateVerb.Trim().Equals("was a"))
            {
                // 打印表头
                Console.WriteLine("{0,-24} {1,-10} {2,-10} {3,-10}", " ", "Years", "Months", "Days");

                // 初始化年、月、日
                int TheYears = 0, TheMonths = 0, TheDays = 0;
                # 初始化灵活年、月、日为0
                int FlexYears = 0, FlexMonths = 0, FlexDays = 0;

                # 计算出生日期到今天的年龄，并将结果存储在TheYears, TheMonths, TheDays中
                CalculateDateDiff(TodaysDate, BirthDate, 1, out TheYears, out TheMonths, out TheDays);
                # 输出出生日期到今天的年龄
                WriteColumnOutput("Your age if birthdate", TheYears, TheMonths, TheDays);

                # 将灵活年、月、日设置为出生日期到今天的年龄
                FlexYears = TheYears;
                FlexMonths = TheMonths;
                FlexDays = TheDays;
                # 计算出生日期到今天的35%时间，并将结果存储在FlexYears, FlexMonths, FlexDays中
                CalculateDateDiff(TodaysDate, BirthDate, .35, out FlexYears, out FlexMonths, out FlexDays);
                # 输出出生日期到今天的35%时间
                WriteColumnOutput("You have slept", FlexYears, FlexMonths, FlexDays);

                # 将灵活年、月、日设置为出生日期到今天的年龄
                FlexYears = TheYears;
                FlexMonths = TheMonths;
                FlexDays = TheDays;
                # 计算出生日期到今天的17%时间，并将结果存储在FlexYears, FlexMonths, FlexDays中
                CalculateDateDiff(TodaysDate, BirthDate, .17, out FlexYears, out FlexMonths, out FlexDays);
                # 输出出生日期到今天的17%时间
                WriteColumnOutput("You have eaten", FlexYears, FlexMonths, FlexDays);

                # 将灵活年、月、日设置为出生日期到今天的年龄
                FlexYears = TheYears;
                FlexMonths = TheMonths;
                FlexDays = TheDays;
                # 调用 CalculateDateDiff 函数计算日期差，将结果存储在 FlexYears, FlexMonths, FlexDays 中
                CalculateDateDiff(TodaysDate, BirthDate, .23, out FlexYears, out FlexMonths, out FlexDays);
                # 初始化 FlexPhrase 变量
                string FlexPhrase = "You have played";
                # 如果 TheYears 大于 3，则修改 FlexPhrase 变量的值
                if (TheYears > 3)
                    FlexPhrase = "You have played/studied";
                # 如果 TheYears 大于 9，则修改 FlexPhrase 变量的值
                if (TheYears > 9)
                    FlexPhrase = "You have worked/played";
                # 调用 WriteColumnOutput 函数输出 FlexPhrase, FlexYears, FlexMonths, FlexDays 的值
                WriteColumnOutput(FlexPhrase, FlexYears, FlexMonths, FlexDays);

                # 将 FlexYears, FlexMonths, FlexDays 的值更新为 TheYears, TheMonths, TheDays 的值
                FlexYears = TheYears;
                FlexMonths = TheMonths;
                FlexDays = TheDays;
                # 调用 CalculateDateDiff 函数计算日期差，将结果存储在 FlexYears, FlexMonths, FlexDays 中
                CalculateDateDiff(TodaysDate, BirthDate, .25, out FlexYears, out FlexMonths, out FlexDays);
                # 调用 WriteColumnOutput 函数输出 "You have relaxed", FlexYears, FlexMonths, FlexDays 的值
                WriteColumnOutput("You have relaxed", FlexYears, FlexMonths, FlexDays);

                # 输出空行
                Console.WriteLine("");
                # 输出 "* You may retire in {0} *"，其中 {0} 为 BirthDate.Year + 65 的值
                Console.WriteLine("* You may retire in {0} *".PadLeft(38), BirthDate.Year + 65);
            }
        }

        # 定义 PlayTheGame 函数
        public void PlayTheGame()
        {
            // 声明并初始化变量TodaysDate和BirthDate为最小日期值
            DateTime TodaysDate = DateTime.MinValue;
            DateTime BirthDate = DateTime.MinValue;

            // 调用DisplayIntro函数，显示游戏介绍
            DisplayIntro();

            // 通过PromptForADate函数获取用户输入的今天日期和出生日期
            TodaysDate = PromptForADate("Enter today's date in the form: 3,24,1978  ? ");
            BirthDate = PromptForADate("Enter day of birth (or other day of interest)? ");

            // 调用DisplayOutput函数，显示计算结果
            DisplayOutput(TodaysDate, BirthDate);

        }
    }
    class Program
    {
        static void Main(string[] args)
        {
            // 创建Weekday对象并调用PlayTheGame方法开始游戏
            new Weekday().PlayTheGame();
抱歉，给定的代码片段不完整，无法为其添加注释。
```