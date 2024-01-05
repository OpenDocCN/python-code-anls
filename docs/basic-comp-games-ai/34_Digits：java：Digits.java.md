# `d:/src/tocomm/basic-computer-games\34_Digits\java\Digits.java`

```
import java.util.Arrays;  // 导入 Arrays 类，用于操作数组
import java.util.InputMismatchException;  // 导入 InputMismatchException 类，用于处理输入不匹配异常
import java.util.Scanner;  // 导入 Scanner 类，用于接收用户输入

/**
 * DIGITS
 * <p>
 * Converted from BASIC to Java by Aldrin Misquitta (@aldrinm)
 */
public class Digits {

	public static void main(String[] args) {
		printIntro();  // 调用 printIntro 方法，打印介绍信息
		Scanner scan = new Scanner(System.in);  // 创建 Scanner 对象，用于接收用户输入

		boolean showInstructions = readInstructionChoice(scan);  // 调用 readInstructionChoice 方法，根据用户输入判断是否显示说明
		if (showInstructions) {  // 如果需要显示说明
			printInstructions();  // 调用 printInstructions 方法，打印说明
		}
		// 初始化变量a、b、c，分别赋值为0、1、3
		int a = 0, b = 1, c = 3;
		// 创建一个27行3列的二维数组m
		int[][] m = new int[27][3];
		// 创建一个3行3列的二维数组k
		int[][] k = new int[3][3];
		// 创建一个9行3列的二维数组l
		int[][] l = new int[9][3];

		// 初始化布尔变量continueGame，赋值为true
		boolean continueGame = true;
		// 当continueGame为true时执行循环
		while (continueGame) {
			// 遍历二维数组m的每一行，将每行元素填充为1
			for (int[] ints : m) {
				Arrays.fill(ints, 1);
			}
			// 遍历二维数组k的每一行，将每行元素填充为9
			for (int[] ints : k) {
				Arrays.fill(ints, 9);
			}
			// 遍历二维数组l的每一行，将每行元素填充为3
			for (int[] ints : l) {
				Arrays.fill(ints, 3);
			}

			// 将二维数组l的第一行第一列元素赋值为2
			l[0][0] = 2;
			// 将二维数组l的第五行第二列元素赋值为2
			l[4][1] = 2;
			// 将二维数组l的第九行第三列元素赋值为2
			l[8][2] = 2;
			# 初始化变量 z, z1, z2, runningCorrect
			z = 26
			z1 = 8
			z2 = 2
			runningCorrect = 0

			# 循环3次
			for t in range(1, 4):
				# 初始化 validNumbers 为 False
				validNumbers = False
				# 初始化空数组 numbers
				numbers = []
				# 当 validNumbers 为 False 时循环
				while not validNumbers:
					# 读取10个数字
					print()
					numbers = read10Numbers(scan)
					validNumbers = True
					# 遍历 numbers 数组
					for number in numbers:
						# 如果数字小于0或大于2，则输出错误信息并将 validNumbers 置为 False
						if number < 0 or number > 2:
							print("ONLY USE THE DIGITS '0', '1', OR '2'.")
							print("LET'S TRY AGAIN.")
							validNumbers = False
							break
				# 打印表头
				System.out.printf("\n%-14s%-14s%-14s%-14s", "MY GUESS", "YOUR NO.", "RESULT", "NO. RIGHT");
				# 遍历 numbers 列表
				for (int number : numbers) {
					# 初始化变量 s 和 myGuess
					int s = 0;
					int myGuess = 0;
					# 遍历 j 从 0 到 2
					for (int j = 0; j <= 2; j++) {
						# 计算 s1 的值
						# 第一个表达式总是得到 0，因为 a 总是 0，原作者的意图是什么？
						int s1 = a * k[z2][j] + b * l[z1][j] + c * m[z][j];
						# 如果 s1 大于 s，则更新 s 和 myGuess
						if (s < s1) {
							s = s1;
							myGuess = j;
						# 如果 s1 等于 s
						} else if (s1 == s) {
							# 如果随机数大于等于 0.5，则更新 myGuess
							if (Math.random() >= 0.5) {
								myGuess = j;
							}
						}
					}

					# 初始化 result 变量
					String result;
					# 如果 myGuess 不等于 number，则设置 result 为 "WRONG"
					if (myGuess != number) {
						result = "WRONG";
					} else {
						// 如果猜测正确，增加 runningCorrect 计数，设置 result 为 "RIGHT"
						runningCorrect++;
						result = "RIGHT";
						// 更新 m、l、k 数组中对应位置的值
						m[z][number] = m[z][number] + 1;
						l[z1][number] = l[z1][number] + 1;
						k[z2][number] = k[z2][number] + 1;
						// 更新 z 的值
						z = z - (z / 9) * 9;
						z = 3 * z + number;
					}
					// 打印结果
					System.out.printf("\n%-14d%-14d%-14s%-14d", myGuess, number, result, runningCorrect);

					// 更新 z1 和 z2 的值
					z1 = z - (z / 9) * 9;
					z2 = number;
				}
			}

			// 打印总结报告
			System.out.println();
			// 如果 runningCorrect 大于 10，打印空行
			if (runningCorrect > 10) {
				System.out.println();
# 如果程序猜对的数字超过你选择的数字的1/3，输出以下内容
System.out.println("I GUESSED MORE THAN 1/3 OF YOUR NUMBERS.");
System.out.println("I WIN.\u0007");
# 如果程序猜对的数字少于你选择的数字的1/3，输出以下内容
} else if (runningCorrect < 10) {
    System.out.println("I GUESSED LESS THAN 1/3 OF YOUR NUMBERS.");
    System.out.println("YOU BEAT ME.  CONGRATULATIONS *****");
# 如果程序猜对的数字正好是你选择的数字的1/3，输出以下内容
} else {
    System.out.println("I GUESSED EXACTLY 1/3 OF YOUR NUMBERS.");
    System.out.println("IT'S A TIE GAME.");
}

# 调用readContinueChoice方法，读取用户是否继续游戏的选择
continueGame = readContinueChoice(scan);
```
```java
# 读取用户是否继续游戏的选择
private static boolean readContinueChoice(Scanner scan) {
    System.out.print("\nDO YOU WANT TO TRY AGAIN (1 FOR YES, 0 FOR NO) ? ");
    int choice;
    try {
			// 从输入流中读取一个整数，存储在choice变量中
			choice = scan.nextInt();
			// 如果choice等于1，则返回true，否则返回false
			return choice == 1;
		} catch (InputMismatchException ex) {
			// 捕获输入不匹配异常，返回false
			return false;
		} finally {
			// 无论是否发生异常，都调用scan.nextLine()方法
			scan.nextLine();
		}
	}

	private static int[] read10Numbers(Scanner scan) {
		// 打印提示信息
		System.out.print("TEN NUMBERS, PLEASE ? ");
		// 创建一个包含10个整数的数组
		int[] numbers = new int[10];

		// 循环读取10个整数
		for (int i = 0; i < numbers.length; i++) {
			// 初始化validInput为false
			boolean validInput = false;
			// 当输入不合法时循环
			while (!validInput) {
				try {
					// 从输入流中读取一个整数
					int n = scan.nextInt();
					// 输入合法，将validInput设置为true
					validInput = true;
					// 将读取的整数存储在数组中
					numbers[i] = n;
				} catch (InputMismatchException ex) {  // 捕获输入不匹配异常
					System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE");  // 打印错误消息
				} finally {  // 无论是否发生异常都会执行的代码块
					scan.nextLine();  // 清空输入缓冲区
				}
			}
		}

		return numbers;  // 返回输入的数字数组
	}

	private static void printInstructions() {  // 打印游戏说明的方法
		System.out.println("\n");  // 打印空行
		System.out.println("PLEASE TAKE A PIECE OF PAPER AND WRITE DOWN");  // 打印提示信息
		System.out.println("THE DIGITS '0', '1', OR '2' THIRTY TIMES AT RANDOM.");  // 打印提示信息
		System.out.println("ARRANGE THEM IN THREE LINES OF TEN DIGITS EACH.");  // 打印提示信息
		System.out.println("I WILL ASK FOR THEN TEN AT A TIME.");  // 打印提示信息
		System.out.println("I WILL ALWAYS GUESS THEM FIRST AND THEN LOOK AT YOUR");  // 打印提示信息
		System.out.println("NEXT NUMBER TO SEE IF I WAS RIGHT. BY PURE LUCK,");  // 打印提示信息
		System.out.println("I OUGHT TO BE RIGHT TEN TIMES. BUT I HOPE TO DO BETTER");  // 打印提示信息
		System.out.println("THAN THAT *****");  // 打印字符串 "THAN THAT *****"

		System.out.println();  // 打印空行
	}

	private static boolean readInstructionChoice(Scanner scan) {
		System.out.print("FOR INSTRUCTIONS, TYPE '1', ELSE TYPE '0' ? ");  // 打印提示信息
		int choice;
		try {
			choice = scan.nextInt();  // 从输入中读取整数
			return choice == 1;  // 如果输入为1，则返回true，否则返回false
		} catch (InputMismatchException ex) {
			return false;  // 如果输入不是整数，则返回false
		} finally {
			scan.nextLine();  // 无论如何都会执行，用于清空输入缓冲区
		}
	}

	private static void printIntro() {
		System.out.println("                                DIGITS");  // 打印字符串 "DIGITS"
		System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 打印字符串 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
# 输出两个空行
System.out.println("\n\n")
# 输出提示信息
System.out.println("THIS IS A GAME OF GUESSING.")
```