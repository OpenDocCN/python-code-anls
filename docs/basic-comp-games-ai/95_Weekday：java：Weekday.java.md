# `d:/src/tocomm/basic-computer-games\95_Weekday\java\Weekday.java`

```
import java.util.Scanner;  // 导入 Scanner 类，用于从控制台读取输入

/**
 * WEEKDAY
 *
 * Converted from BASIC to Java by Aldrin Misquitta (@aldrinm)
 *
 */
public class Weekday {

	//TABLE OF VALUES FOR THE MONTHS TO BE USED IN CALCULATIONS.
	//Dummy value added at index 0, so we can reference directly by the month number value
	private final static int[] t = new int[]{-1, 0, 3, 3, 6, 1, 4, 6, 2, 5, 0, 3, 5};  // 创建一个包含月份值的数组，用于计算

	public static void main(String[] args) {
		printIntro();  // 调用打印介绍信息的函数

		Scanner scanner = new Scanner(System.in);  // 创建一个 Scanner 对象，用于从控制台读取输入
		System.out.print("ENTER TODAY'S DATE IN THE FORM: 3,24,1979 ");  // 提示用户输入今天的日期
		DateStruct todaysDate = readDate(scanner);  // 调用读取日期的函数，将用户输入的日期存储在 DateStruct 对象中
		System.out.print("ENTER DAY OF BIRTH (OR OTHER DAY OF INTEREST) "); // 提示用户输入出生日期或其他感兴趣的日期
		DateStruct dateOfInterest = readDate(scanner); // 从用户输入中读取日期并存储在dateOfInterest变量中

		int I1 = (dateOfInterest.year - 1500) / 100; // 计算I1的值

		// 测试日期是否在当前日历之前
		if ((dateOfInterest.year - 1582) >= 0) {
			int A = I1 * 5 + (I1 + 3) / 4; // 计算A的值
			int I2 = (A - b(A) * 7); // 计算I2的值
			int Y2 = (dateOfInterest.year / 100); // 计算Y2的值
			int Y3 = (dateOfInterest.year - Y2 * 100); // 计算Y3的值
			A = Y3 / 4 + Y3 + dateOfInterest.day + t[dateOfInterest.month] + I2; // 重新计算A的值
			calculateAndPrintDayOfWeek(I1, A, todaysDate, dateOfInterest, Y3); // 调用函数计算并打印星期几

			// 检查日期是否与今天的日期相同，如果是则停止程序
			if ((todaysDate.year * 12 + todaysDate.month) * 31 + todaysDate.day
					== (dateOfInterest.year * 12 + dateOfInterest.month) * 31 + dateOfInterest.day) {
				return; // 停止程序
			}

			int I5 = todaysDate.year - dateOfInterest.year; // 计算I5的值
			System.out.print("\n"); // 打印换行符
			int I6 = todaysDate.month - dateOfInterest.month; // 计算当前日期的月份与感兴趣日期的月份之差
			int I7 = todaysDate.day - dateOfInterest.day; // 计算当前日期的天数与感兴趣日期的天数之差
			if (I7 < 0) { // 如果天数差小于0
				I6 = I6 - 1; // 月份差减1
				I7 = I7 + 30; // 天数差加30
			}
			if (I6 < 0) { // 如果月份差小于0
				I5 = I5 - 1; // 年份差减1
				I6 = I6 + 12; // 月份差加12
			}
			if (I5 < 0) { // 如果年份差小于0
				return; // 什么也不做，结束程序
			} else {
				if (I7 != 0) { // 如果天数差不等于0
					printHeadersAndAge(I5, I6, I7); // 调用函数打印标题和年龄
				} else {
					if (I6 != 0) { // 如果月份差不等于0
						printHeadersAndAge(I5, I6, I7); // 调用函数打印标题和年龄
					} else {
# 打印生日祝福信息
System.out.println("***HAPPY BIRTHDAY***");
# 调用函数打印头部信息和年龄
printHeadersAndAge(I5, I6, I7);
# 计算总天数
int A8 = (I5 * 365) + (I6 * 30) + I7 + (I6 / 2);
int K5 = I5;
int K6 = I6;
int K7 = I7;
# 计算退休日期
int E = dateOfInterest.year + 65;
# 计算在以下功能中花费的时间
float F = 0.35f;
# 打印睡眠时间统计信息
System.out.printf("%-28s", "YOU HAVE SLEPT");
# 创建一个临时日期结构
DateStruct scratchDate = new DateStruct(K6, K7, K5); #K5是临时年份，K6是月份，K7是日期
# 调用函数打印统计行
printStatisticRow(F, A8, scratchDate);
K5 = scratchDate.year;
K6 = scratchDate.month;
K7 = scratchDate.day;
			F = 0.17f; // 设置变量F的值为0.17

			System.out.printf("%-28s", "YOU HAVE EATEN"); // 格式化输出字符串"YOU HAVE EATEN"，并保证占用28个字符的宽度

			scratchDate = new DateStruct(K6, K7, K5); // 使用变量K6、K7、K5创建一个新的DateStruct对象，并赋值给scratchDate
			printStatisticRow(F, A8, scratchDate); // 调用printStatisticRow方法，传入参数F、A8、scratchDate
			K5 = scratchDate.year; // 将scratchDate对象的year属性赋值给K5
			K6 = scratchDate.month; // 将scratchDate对象的month属性赋值给K6
			K7 = scratchDate.day; // 将scratchDate对象的day属性赋值给K7

			F = 0.23f; // 设置变量F的值为0.23
			if (K5 > 3) { // 如果K5大于3
				if (K5 > 9) { // 如果K5大于9
					System.out.printf("%-28s", "YOU HAVE WORKED/PLAYED"); // 格式化输出字符串"YOU HAVE WORKED/PLAYED"，并保证占用28个字符的宽度
				} else {
					System.out.printf("%-28s", "YOU HAVE PLAYED/STUDIED"); // 格式化输出字符串"YOU HAVE PLAYED/STUDIED"，并保证占用28个字符的宽度
				}
			} else {
				System.out.printf("%-28s", "YOU HAVE PLAYED"); // 格式化输出字符串"YOU HAVE PLAYED"，并保证占用28个字符的宽度
			}
			// 使用 K6、K7、K5 创建一个新的 DateStruct 对象
			scratchDate = new DateStruct(K6, K7, K5);
			// 调用 printStatisticRow 方法，打印统计行信息
			printStatisticRow(F, A8, scratchDate);
			// 将 scratchDate 的年份赋值给 K5
			K5 = scratchDate.year;
			// 将 scratchDate 的月份赋值给 K6
			K6 = scratchDate.month;
			// 将 scratchDate 的日期赋值给 K7
			K7 = scratchDate.day;

			// 如果月份为 12，则年份加一，月份重置为 0
			if (K6 == 12) {
				K5 = K5 + 1;
				K6 = 0;
			}
			// 打印格式化的字符串，显示放松的信息和日期
			System.out.printf("%-28s%14s%14s%14s%n", "YOU HAVE RELAXED", K5, K6, K7);
			// 打印格式化的字符串，显示可以退休的信息和 E 的值
			System.out.printf("%16s***  YOU MAY RETIRE IN %s ***%n", " ", E);
			// 打印空行
			System.out.printf("%n%n%n%n%n");
		} else {
			// 如果条件不满足，打印提示信息
			System.out.println("NOT PREPARED TO GIVE DAY OF WEEK PRIOR TO MDLXXXII.");
		}
	}
# 计算并打印统计行
def printStatisticRow(F, A8, scratchDate):
    # 计算K1
    K1 = int(F * A8)
    # 计算I5
    I5 = K1 // 365
    # 更新K1
    K1 = K1 - (I5 * 365)
    # 计算I6
    I6 = K1 // 30
    # 计算I7
    I7 = K1 - (I6 * 30)
    # 计算K5
    K5 = scratchDate.year - I5
    # 计算K6
    K6 = scratchDate.month - I6
    # 计算K7
    K7 = scratchDate.day - I7
    # 如果K7小于0，进行调整
    if K7 < 0:
        K7 = K7 + 30
        K6 = K6 - 1
    # 如果K6小于等于0，进行调整
    if K6 <= 0:
        K6 = K6 + 12
        K5 = K5 - 1
    # 更新scratchDate中的年月日值
    scratchDate.year = K5
    scratchDate.month = K6
    scratchDate.day = K7
		# 设置scratchDate对象的day属性为K7
		scratchDate.day = K7;
		# 使用printf函数打印I5, I6, I7的值，格式为14个字符宽度，右对齐
		System.out.printf("%14s%14s%14s%n", I5, I6, I7);
	}

	# 打印表头和年龄
	private static void printHeadersAndAge(int I5, int I6, int I7) {
		# 使用printf函数打印表头，格式为14个字符宽度，右对齐
		System.out.printf("%14s%14s%14s%14s%14s%n", " ", " ", "YEARS", "MONTHS", "DAYS");
		# 使用printf函数打印分隔线
		System.out.printf("%14s%14s%14s%14s%14s%n", " ", " ", "-----", "------", "----");
		# 使用printf函数打印年龄信息
		System.out.printf("%-28s%14s%14s%14s%n", "YOUR AGE (IF BIRTHDATE)", I5, I6, I7);
	}

	# 计算并打印星期几
	private static void calculateAndPrintDayOfWeek(int i1, int a, DateStruct dateStruct, DateStruct dateOfInterest, int y3) {
		# 计算b的值
		int b = (a - b(a) * 7) + 1;
		# 如果dateOfInterest的月份大于2，则调用printDayOfWeek函数打印星期几
		if (dateOfInterest.month > 2) {
			printDayOfWeek(dateStruct, dateOfInterest, b);
		} else {
			# 如果y3等于0
			if (y3 == 0) {
				# 计算aa的值
				int aa = i1 - 1;
				# 计算t1的值
				int t1 = aa - a(aa) * 4;
				# 如果t1等于0
				if (t1 == 0) {
					# 如果b不等于0
					if (b != 0) {
# 减去1天
b = b - 1;
# 打印感兴趣日期的星期几
printDayOfWeek(dateStruct, dateOfInterest, b);
# 否则
else:
    # 将b设置为6
    b = 6;
    # 减去1天
    b = b - 1;
    # 打印感兴趣日期的星期几
    printDayOfWeek(dateStruct, dateOfInterest, b);
# 如果b等于0
if (b == 0) {
    # 将b设置为7
    b = 7;
}
# 如果日期结构的年份乘以12再加上月份再乘以31
if ((dateStruct.year * 12 + dateStruct.month) * 31
# 检查日期是否在给定日期之前，如果是则打印相应信息
if ((dateStruct.year * 12 + dateStruct.month) * 31 + dateStruct.day
    < (dateOfInterest.year * 12 + dateOfInterest.month) * 31 + dateOfInterest.day):
    System.out.printf("%s / %s / %s WILL BE A ", dateOfInterest.month, dateOfInterest.day, dateOfInterest.year)
# 检查日期是否与给定日期相同，如果是则打印相应信息
elif ((dateStruct.year * 12 + dateStruct.month) * 31 + dateStruct.day
      == (dateOfInterest.year * 12 + dateOfInterest.month) * 31 + dateOfInterest.day):
    System.out.printf("%s / %s / %s IS A ", dateOfInterest.month, dateOfInterest.day, dateOfInterest.year)
# 如果日期早于给定日期，则打印相应信息
else:
    System.out.printf("%s / %s / %s WAS A ", dateOfInterest.month, dateOfInterest.day, dateOfInterest.year)

# 根据变量 b 的值进行不同的操作
switch (b):
    case 1:
        System.out.println("SUNDAY.")
        break
    case 2:
        System.out.println("MONDAY.")
        break
    case 3:
    # 其他 case 语句...
				# 打印“TUESDAY.”，表示星期二
				System.out.println("TUESDAY.")
				# 跳出 switch 语句
				break
			# 如果 dateOfInterest.weekday 为 4
			case 4:
				# 打印“WEDNESDAY.”，表示星期三
				System.out.println("WEDNESDAY.")
				# 跳出 switch 语句
				break
			# 如果 dateOfInterest.weekday 为 5
			case 5:
				# 打印“THURSDAY.”，表示星期四
				System.out.println("THURSDAY.")
				# 跳出 switch 语句
				break
			# 如果 dateOfInterest.weekday 为 6
			case 6:
				# 如果 dateOfInterest.day 为 13
				if (dateOfInterest.day == 13):
					# 打印“FRIDAY THE THIRTEENTH---BEWARE!”，表示星期五的13号
					System.out.println("FRIDAY THE THIRTEENTH---BEWARE!")
				else:
					# 打印“FRIDAY.”，表示星期五
					System.out.println("FRIDAY.")
				# 跳出 switch 语句
				break
			# 如果 dateOfInterest.weekday 为 7
			case 7:
				# 打印“SATURDAY.”，表示星期六
				System.out.println("SATURDAY.")
				# 跳出 switch 语句
				break
		}
	}
```
```python
# 以上代码是一个 switch 语句，根据 dateOfInterest.weekday 的值来打印对应的星期几。如果是星期五并且日期是13号，则会打印“FRIDAY THE THIRTEENTH---BEWARE!”，否则打印“FRIDAY.”。其他日期则分别打印对应的星期几。每个 case 后面的 break 语句用于跳出 switch 语句。
	private static int a(int a) {  // 定义一个名为a的静态方法，接受一个整数参数a，返回a除以4的结果
		return a / 4;
	}

	private static int b(int a) {  // 定义一个名为b的静态方法，接受一个整数参数a，返回a除以7的结果
		return a / 7;
	}


	private static void printIntro() {  // 定义一个名为printIntro的静态方法，不返回任何值
		System.out.println("                                WEEKDAY");  // 打印"WEEKDAY"
		System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 打印"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
		System.out.println("\n\n\n");  // 打印三个空行
		System.out.println("WEEKDAY IS A COMPUTER DEMONSTRATION THAT");  // 打印"WEEKDAY IS A COMPUTER DEMONSTRATION THAT"
		System.out.println("GIVES FACTS ABOUT A DATE OF INTEREST TO YOU.");  // 打印"GIVES FACTS ABOUT A DATE OF INTEREST TO YOU."
		System.out.println("\n");  // 打印一个空行
	}
	 * 读取用户输入的日期，进行一些验证并返回一个简单的日期结构
	 */
	private static DateStruct readDate(Scanner scanner) {
		boolean done = false; // 标记是否完成输入
		int mm = 0, dd = 0, yyyy = 0; // 初始化月、日、年
		while (!done) { // 循环直到输入完成
			String input = scanner.next(); // 读取用户输入
			String[] tokens = input.split(","); // 将输入按逗号分割成数组
			if (tokens.length < 3) { // 如果输入不符合日期格式
				System.out.println("DATE EXPECTED IN FORM: 3,24,1979 - RETRY INPUT LINE"); // 提示用户重新输入
			} else {
				try {
					mm = Integer.parseInt(tokens[0]); // 将第一个元素转换为整数作为月
					dd = Integer.parseInt(tokens[1]); // 将第二个元素转换为整数作为日
					yyyy = Integer.parseInt(tokens[2]); // 将第三个元素转换为整数作为年
					done = true; // 输入完成
				} catch (NumberFormatException nfe) { // 捕获转换异常
					System.out.println("NUMBER EXPECTED - RETRY INPUT LINE"); // 提示用户重新输入
				}
			}
		}
		return new DateStruct(mm, dd, yyyy);  // 返回一个新的DateStruct对象，包含用户输入的月、日、年信息
	}

	/**
	 * Convenience date structure to hold user date input
	 */
	private static class DateStruct {  // 创建一个私有静态内部类DateStruct，用于存储用户输入的日期信息
		int month;  // 月份
		int day;  // 日
		int year;  // 年

		public DateStruct(int month, int day, int year) {  // DateStruct类的构造函数，用于初始化月、日、年的值
			this.month = month;  // 初始化月份
			this.day = day;  // 初始化日
			this.year = year;  // 初始化年
		}
	}

}
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```