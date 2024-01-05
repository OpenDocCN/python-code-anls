# `d:/src/tocomm/basic-computer-games\66_Number\java\2\Number.java`

```
import java.util.Scanner;  // 导入 Scanner 类，用于从控制台读取用户输入

public class Number {

	public static void main(String[] args) {
		printIntro();  // 调用 printIntro() 函数，打印游戏介绍

		int points = 100; //start with 100 points for the user  // 初始化用户的分数为 100

		Scanner scan = new Scanner(System.in);  // 创建 Scanner 对象，用于接收用户输入

		boolean done = false;  // 初始化一个布尔变量 done，用于控制循环
		while (!done) {  // 当 done 为 false 时执行循环

			System.out.print("GUESS A NUMBER FROM 1 TO 5? ");  // 打印提示信息，要求用户猜一个 1 到 5 之间的数字
			int g = scan.nextInt();  // 从控制台读取用户输入的整数，存储在变量 g 中

			//Initialize 5 random numbers between 1-5
			var r = randomNumber(1);  // 调用 randomNumber() 函数，生成一个 1 到 5 之间的随机数，存储在变量 r 中
			var s = randomNumber(1);  // 调用 randomNumber() 函数，生成一个 1 到 5 之间的随机数，存储在变量 s 中
			var t = randomNumber(1);  // 调用 randomNumber() 函数，生成一个 1 到 5 之间的随机数，存储在变量 t 中
			var u = randomNumber(1);  // 调用 randomNumber() 函数，生成一个 1 到 5 之间的随机数，存储在变量 u 中
			var v = randomNumber(1);  // 调用 randomNumber() 函数，生成一个 1 到 5 之间的随机数，存储在变量 v 中
			if (r == g) { // 如果猜测的数字与 r 相等
				points -= 5; // 减去 5 分
			} else if (s == g) { // 如果猜测的数字与 s 相等
				points += 5; // 加上 5 分
			} else if (t == g) { // 如果猜测的数字与 t 相等
				points += points; // 加上当前的分数
			} else if (u == g) { // 如果猜测的数字与 u 相等
				points += 1; // 加上 1 分
			} else if (v == g) { // 如果猜测的数字与 v 相等
				points -= points * 0.5; // 减去当前分数的一半
			} else {
				continue; // 不匹配任何随机数字，所以继续请求另一个猜测
			}

			if (points > 500) { // 如果分数大于 500
				done = true; // 完成游戏
			} else {
				System.out.println("YOU HAVE " + points + " POINTS."); // 打印当前分数
			}
		}

		System.out.println("!!!!YOU WIN!!!! WITH " + points + " POINTS.\n");
	}

	private static int randomNumber(int x) {
		// 生成一个1到5之间的随机整数并返回
		// 注意：'x'在原始基本列表中完全被忽略
		return (int) (5 * Math.random() + 1);
	}

	private static void printIntro() {
		// 打印游戏介绍
		System.out.println("                                NUMBER");
		System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
		System.out.println("\n\n\n");
		System.out.println("YOU HAVE 100 POINTS.  BY GUESSING NUMBERS FROM 1 TO 5, YOU");
		System.out.println("CAN GAIN OR LOSE POINTS DEPENDING UPON HOW CLOSE YOU GET TO");
		System.out.println("A RANDOM NUMBER SELECTED BY THE COMPUTER.");
		System.out.println("\n");
		System.out.println("YOU OCCASIONALLY WILL GET A JACKPOT WHICH WILL DOUBLE(!)");
		System.out.println("YOUR POINT COUNT.  YOU WIN WHEN YOU GET 500 POINTS.");
	} # 结束函数定义
} # 结束代码块
```