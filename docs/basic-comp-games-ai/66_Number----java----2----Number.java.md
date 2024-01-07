# `basic-computer-games\66_Number\java\2\Number.java`

```

// 导入 Scanner 类
import java.util.Scanner;

// 定义 Number 类
public class Number {

	// 主函数
	public static void main(String[] args) {
		// 打印游戏介绍
		printIntro();
		// 初始化用户的初始分数为100
		int points = 100;

		// 创建 Scanner 对象
		Scanner scan = new Scanner(System.in);
		// 设置循环标志
		boolean done = false;
		// 循环直到游戏结束
		while (!done) {
			// 提示用户猜一个1到5之间的数字
			System.out.print("GUESS A NUMBER FROM 1 TO 5? ");
			// 读取用户输入的数字
			int g = scan.nextInt();

			// 生成5个1到5之间的随机数
			var r = randomNumber(1);
			var s = randomNumber(1);
			var t = randomNumber(1);
			var u = randomNumber(1);
			var v = randomNumber(1);

			// 根据用户猜的数字和随机数进行计分
			if (r == g) {
				points -= 5;
			} else if (s == g) {
				points += 5;
			} else if (t == g) {
				points += points;
			} else if (u == g) {
				points += 1;
			} else if (v == g) {
				points -= points * 0.5;
			} else {
				continue; // 如果用户猜的数字不匹配任何随机数，则继续循环
			}

			// 判断分数是否达到500，如果是则游戏结束
			if (points > 500) {
				done = true;
			} else {
				// 打印当前分数
				System.out.println("YOU HAVE " + points + " POINTS.");
			}
		}

		// 打印游戏结束信息
		System.out.println("!!!!YOU WIN!!!! WITH " + points + " POINTS.\n");
	}

	// 生成随机数的方法
	private static int randomNumber(int x) {
		// 注意：'x' 在原始的基本列表中完全被忽略
		return (int) (5 * Math.random() + 1);
	}

	// 打印游戏介绍的方法
	private static void printIntro() {
		System.out.println("                                NUMBER");
		System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
		System.out.println("\n\n\n");
		System.out.println("YOU HAVE 100 POINTS.  BY GUESSING NUMBERS FROM 1 TO 5, YOU");
		System.out.println("CAN GAIN OR LOSE POINTS DEPENDING UPON HOW CLOSE YOU GET TO");
		System.out.println("A RANDOM NUMBER SELECTED BY THE COMPUTER.");
		System.out.println("\n");
		System.out.println("YOU OCCASIONALLY WILL GET A JACKPOT WHICH WILL DOUBLE(!)");
		System.out.println("YOUR POINT COUNT.  YOU WIN WHEN YOU GET 500 POINTS.");
	}
}

```