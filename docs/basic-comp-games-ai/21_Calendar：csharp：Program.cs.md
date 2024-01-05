# `d:/src/tocomm/basic-computer-games\21_Calendar\csharp\Program.cs`

```
                //准备一个空字符串
                space = "";
                //循环指定的空格数，每次循环添加一个空格
                while (numspaces > 0)
                {
                    //添加一个空格
                    space += " ";
                //减少循环变量，这样我们就不会无限循环下去！
                numspaces--;
            }
            return space;
        }

        static void Main(string[] args)
        {
            // 打印程序的“标题”
            // 使用 Write*Line* 的用法意味着我们不必指定换行符（\n）
            Console.WriteLine(Tab(32) + "CALENDAR");
            Console.WriteLine(Tab(15) + "CREATE COMPUTING  MORRISTOWN, NEW JERSEY");
            //给我们一些空间。
            Console.WriteLine("");
            Console.WriteLine("");
            Console.WriteLine("");

            //建立一些打印计算器所需的变量

            //每个月的天数。在闰年中，这个值的开始会是
            // 0, 31, 29 to account for Feb. the 0 at the start is for days elapsed to work right in Jan.
            // 用于计算2月份的天数，0在开头是为了在1月份正确计算天数
            int[] monthLengths = { 0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}; // m in original source

            //the starting day of the month. in 1979 this was monday
            // 月份的起始日，1979年是星期一
            // 0 = sun, -1 = mon, -2 = tue, -3 = wed, etc.
            int day = -1; // called d in original source

            //how much time in the year has gone by?
            // 一年中经过了多少时间？
            int elapsed = 0; // called s in original source

            //loop through printing all the months.
            // 循环打印所有的月份
            for (int month = 1; month <= 12; month++) //month is called n in original source
            {
                //pad some space
                // 添加一些空格
                Console.WriteLine("");
                Console.WriteLine("");
                //increment days elapsed
                // 增加经过的天数
                elapsed += monthLengths[month - 1];
                //build our header for this month of the calendar
                // 为本月份的日历构建标题
                string header = "** " + elapsed;
// 根据需要添加填充
while (header.Length < 7)
{
    header += " ";
}
// 在头部添加18个星号
for (int i = 1; i <= 18; i++)
{
    header += "*";
}
// 确定是哪个月，然后相应地添加文本
switch (month) {
    case 1: header += " JANUARY "; break;
    case 2: header += " FEBRUARY"; break;
    case 3: header += "  MARCH  "; break;
    case 4: header += "  APRIL  "; break;
    case 5: header += "   MAY   "; break;
    case 6: header += "   JUNE  "; break;
    case 7: header += "   JULY  "; break;
    case 8: header += "  AUGUST "; break;
    case 9: header += "SEPTEMBER"; break;
                    case 10: header += " OCTOBER "; break;  // 如果月份是10，向header字符串添加" OCTOBER "
                    case 11: header += " NOVEMBER"; break;  // 如果月份是11，向header字符串添加" NOVEMBER"
                    case 12: header += " DECEMBER"; break;  // 如果月份是12，向header字符串添加" DECEMBER"
                }
                //more padding
                for (int i = 1; i <= 18; i++)  // 循环18次
                {
                    header += "*";  // 向header字符串添加"*"
                }
                header += "  ";  // 向header字符串添加两个空格
                // how many days left till the year's over?
                header += (365 - elapsed) + " **"; // on leap years 366  // 计算距离年底还有多少天，并添加到header字符串中，如果是闰年则为366
                Console.WriteLine(header);  // 输出header字符串
                //dates
                Console.WriteLine("     S       M       T       W       T       F       S");  // 输出星期的缩写
                Console.WriteLine(" ");  // 输出空行

                string weekOutput = "";  // 初始化weekOutput字符串为空
                for (int i = 1; i <= 59; i++)  // 循环59次
                {
                // 初始化一些变量
                int g = 0;
                int d2 = 0;
                // 遍历周和天
                for (int week = 1; week <= 6; week++)
                {
                    Console.WriteLine(weekOutput);
                    weekOutput = "    ";
                    for (g = 1; g <= 7; g++)
                    {
                        // 将星期几加一
                        day++;
                        d2 = day - elapsed;
                        // 检查是否已经完成了这个月的天数
                        if (d2 > monthLengths[month])
                        {
                            week = 6;
                            break;
                        }
                        // 是否应该打印这一天？
                        if (d2 > 0)
                        {
                            weekOutput += d2;
                        }
                        // 填充空格
                        while (weekOutput.Length < 4 + 8 * g)
                        {
                            weekOutput += " ";
                        }
                    }
                    if (d2 == monthLengths[month])
                    {
                        day += g;
                        break;
                    }
                }
                day -= g;
                Console.WriteLine(weekOutput);
抱歉，给定的代码片段不完整，无法为每个语句添加注释。如果您有完整的代码片段需要解释，请提供完整的代码。谢谢！
```