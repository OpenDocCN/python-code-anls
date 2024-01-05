# `40_Gomoko\java\Gomoko.java`

```
import java.util.Arrays;  # 导入 Arrays 类，用于操作数组
import java.util.InputMismatchException;  # 导入 InputMismatchException 类，用于处理输入不匹配异常
import java.util.Scanner;  # 导入 Scanner 类，用于接收用户输入

/**
 * GOMOKO
 * <p>
 * Converted from BASIC to Java by Aldrin Misquitta (@aldrinm)
 */
public class Gomoko {

	private static final int MIN_BOARD_SIZE = 7;  # 定义最小棋盘大小为 7
	private static final int MAX_BOARD_SIZE = 19;  # 定义最大棋盘大小为 19

	public static void main(String[] args) {
		printIntro();  # 调用打印游戏介绍的函数
		Scanner scan = new Scanner(System.in);  # 创建 Scanner 对象，用于接收用户输入
		int boardSize = readBoardSize(scan);  # 调用函数从用户输入中读取棋盘大小

		boolean continuePlay = true;  # 定义变量，表示是否继续游戏
		while (continuePlay) {  # 当游戏继续时执行循环
			int[][] board = new int[boardSize][boardSize];  # 创建一个二维数组作为游戏棋盘
			//initialize the board elements to 0  # 将棋盘元素初始化为0
			for (int[] ints : board) {  # 遍历棋盘的每一行
				Arrays.fill(ints, 0);  # 将每一行的元素填充为0
			}

			System.out.println("\n\nWE ALTERNATE MOVES.  YOU GO FIRST...");  # 打印提示信息

			boolean doneRound = false;  # 初始化一个标志变量
			while (!doneRound) {  # 当游戏回合未结束时执行循环
				Move playerMove = null;  # 初始化玩家移动
				boolean validMove = false;  # 初始化一个标志变量
				while (!validMove) {  # 当移动无效时执行循环
					playerMove = readMove(scan);  # 从输入中读取玩家的移动
					if (playerMove.i == -1 || playerMove.j == -1) {  # 如果玩家选择结束游戏
						doneRound = true;  # 设置游戏回合结束标志
						System.out.println("\nTHANKS FOR THE GAME!!");  # 打印结束游戏的提示信息
						System.out.print("PLAY AGAIN (1 FOR YES, 0 FOR NO)? ");  # 打印提示信息
						final int playAgain = scan.nextInt();  # 从输入中读取玩家是否再玩一局
						scan.nextLine();  # 读取下一行输入
						if (playAgain == 1) {  # 如果玩家选择再玩一次
							continuePlay = true;  # 继续游戏标志设为true
							break;  # 跳出循环
						} else {  # 否则
							continuePlay = false;  # 继续游戏标志设为false
							break;  # 跳出循环
						}
					} else if (!isLegalMove(playerMove, boardSize)) {  # 如果玩家移动不合法
						System.out.println("ILLEGAL MOVE.  TRY AGAIN...");  # 输出提示信息
					} else if (board[playerMove.i - 1][playerMove.j - 1] != 0) {  # 如果玩家选择的位置已经被占据
						System.out.println("SQUARE OCCUPIED.  TRY AGAIN...");  # 输出提示信息
					} else {  # 否则
						validMove = true;  # 有效移动标志设为true
					}
				}

				if (!doneRound) {  # 如果游戏回合未结束
					board[playerMove.i - 1][playerMove.j - 1] = 1;  # 玩家选择的位置设为1
					Move computerMove = getComputerMove(playerMove, board, boardSize);  # 获取计算机的移动
					// 如果计算机移动为空，随机生成一个计算机移动
					if (computerMove == null) {
						computerMove = getRandomMove(board, boardSize);
					}
					// 在棋盘上将计算机移动的位置标记为2
					board[computerMove.i - 1][computerMove.j - 1] = 2;

					// 打印更新后的棋盘
					printBoard(board);
				}
			}

		}
	}

	//*** 计算机尝试智能移动 ***
	// 根据玩家的移动和棋盘状态，计算出计算机的下一步移动
	private static Move getComputerMove(Move playerMove, int[][] board, int boardSize) {
		// 遍历周围的九个格子
		for (int e = -1; e <= 1; e++) {
			for (int f = -1; f <= 1; f++) {
				// 排除中心格子
				if ((e + f - e * f) != 0) {
					// 计算新的移动位置
					var x = playerMove.i + f;
					var y = playerMove.j + f;
					// 创建新的移动对象
					final Move newMove = new Move(x, y);
					# 检查新移动是否合法
					if (isLegalMove(newMove, boardSize)) {
						# 如果新移动位置上已经有棋子
						if (board[newMove.i - 1][newMove.j - 1] != 0) {
							# 回退到上一个位置
							newMove.i = newMove.i - e;
							newMove.i = newMove.j - f;
							# 如果回退后位置不合法，则返回空
							if (!isLegalMove(newMove, boardSize)) {
								return null;
							} else {
								# 如果回退后位置合法且为空，则返回新移动位置
								if (board[newMove.i - 1][newMove.j - 1] == 0) {
									return newMove;
								}
							}
						}
					}
				}
			}
		}
		# 如果没有合适的位置，则返回空
		return null;
	}

	# 打印棋盘
	private static void printBoard(int[][] board) {
		for (int[] ints : board) {
			// 遍历二维数组board的每一行
			for (int cell : ints) {
				// 遍历每一行中的每个元素
				System.out.printf(" %s", cell);
			}
			// 换行
			System.out.println();
		}
	}

	//*** COMPUTER TRIES A RANDOM MOVE ***
	// 生成一个随机的合法移动
	private static Move getRandomMove(int[][] board, int boardSize) {
		// 初始化合法移动标志为false
		boolean legalMove = false;
		// 初始化随机移动为null
		Move randomMove = null;
		// 当没有找到合法移动时循环
		while (!legalMove) {
			// 生成一个随机移动
			randomMove = randomMove(boardSize);
			// 检查移动是否合法并且目标位置为空
			legalMove = isLegalMove(randomMove, boardSize) && board[randomMove.i - 1][randomMove.j - 1] == 0;
		}
		// 返回随机移动
		return randomMove;
	}
	private static Move randomMove(int boardSize) {
		// 生成随机的x坐标
		int x = (int) (boardSize * Math.random() + 1);
		// 生成随机的y坐标
		int y = (int) (boardSize * Math.random() + 1);
		// 返回一个包含随机坐标的Move对象
		return new Move(x, y);
	}

	private static boolean isLegalMove(Move move, int boardSize) {
		// 判断移动是否合法，即坐标是否在棋盘范围内
		return (move.i >= 1) && (move.i <= boardSize) && (move.j >= 1) && (move.j <= boardSize);
	}

	private static void printIntro() {
		// 打印游戏介绍
		System.out.println("                                GOMOKO");
		System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
		System.out.println("\n\n");
		System.out.println("WELCOME TO THE ORIENTAL GAME OF GOMOKO.");
		System.out.println("\n");
		System.out.println("THE GAME IS PLAYED ON AN N BY N GRID OF A SIZE");
		System.out.println("THAT YOU SPECIFY.  DURING YOUR PLAY, YOU MAY COVER ONE GRID");
		System.out.println("INTERSECTION WITH A MARKER. THE OBJECT OF THE GAME IS TO GET");
		System.out.println("5 ADJACENT MARKERS IN A ROW -- HORIZONTALLY, VERTICALLY, OR");
	}
		System.out.println("DIAGONALLY.  ON THE BOARD DIAGRAM, YOUR MOVES ARE MARKED");  // 打印提示信息
		System.out.println("WITH A '1' AND THE COMPUTER MOVES WITH A '2'.");  // 打印提示信息
		System.out.println("\nTHE COMPUTER DOES NOT KEEP TRACK OF WHO HAS WON.");  // 打印提示信息
		System.out.println("TO END THE GAME, TYPE -1,-1 FOR YOUR MOVE.\n ");  // 打印提示信息
	}

	private static int readBoardSize(Scanner scan) {  // 定义一个静态方法，用于读取棋盘大小
		System.out.print("WHAT IS YOUR BOARD SIZE (MIN 7/ MAX 19)? ");  // 打印提示信息

		boolean validInput = false;  // 定义一个布尔变量，用于判断输入是否有效
		int input = 0;  // 定义一个整型变量，用于存储输入的值
		while (!validInput) {  // 循环直到输入有效
			try {  // 尝试读取输入
				input = scan.nextInt();  // 从输入中读取整数
				if (input < MIN_BOARD_SIZE || input > MAX_BOARD_SIZE) {  // 如果输入小于最小值或大于最大值
					System.out.printf("I SAID, THE MINIMUM IS %s, THE MAXIMUM IS %s.\n", MIN_BOARD_SIZE, MAX_BOARD_SIZE);  // 打印提示信息
				} else {
					validInput = true;  // 输入有效，设置标志为true
				}
			} catch (InputMismatchException ex) {  // 捕获输入类型不匹配的异常
				System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE\n"); // 打印错误消息，提示用户输入数字
				validInput = false; // 将 validInput 设置为 false，表示输入无效
			} finally {
				scan.nextLine(); // 清空输入缓冲区
			}
		}
		return input; // 返回用户输入的字符串
	}

	private static Move readMove(Scanner scan) { // 定义一个静态方法 readMove，接受一个 Scanner 对象作为参数
		System.out.print("YOUR PLAY (I,J)? "); // 打印提示信息，要求用户输入坐标
		boolean validInput = false; // 初始化 validInput 为 false，表示输入无效
		Move move = new Move(); // 创建一个 Move 对象
		while (!validInput) { // 进入循环，直到输入有效
			String input = scan.nextLine(); // 读取用户输入的字符串
			final String[] split = input.split(","); // 将输入字符串按逗号分割成数组
			try {
				move.i = Integer.parseInt(split[0]); // 将数组第一个元素转换为整数，赋值给 move 对象的 i 属性
				move.j = Integer.parseInt(split[1]); // 将数组第二个元素转换为整数，赋值给 move 对象的 j 属性
				validInput = true; // 输入有效，将 validInput 设置为 true
			} catch (NumberFormatException nfe) {
				System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE\n? ");
			}
```
这段代码是一个异常处理块，用来捕获NumberFormatException异常。如果在尝试将字符串转换为数字时出现异常，会打印"!NUMBER EXPECTED - RETRY INPUT LINE\n? "，然后重新提示用户输入。

```java
		}
		return move;
	}
```
这段代码是一个方法的结束和返回语句。它表示方法的结束，并返回move变量的值。

```java
	private static class Move {
		int i;
		int j;

		public Move() {
		}

		public Move(int i, int j) {
			this.i = i;
			this.j = j;
		}
	}
```
这段代码定义了一个内部静态类Move，它有两个整型变量i和j。它还包括一个无参构造函数和一个带有两个参数的构造函数。
		@Override  # 重写父类的 toString 方法
		public String toString() {  # 定义一个公共的返回字符串类型的方法
			return "Move{" +  # 返回字符串 "Move{"
					"i=" + i +  # 将变量 i 的值转换为字符串并拼接到返回字符串中
					", j=" + j +  # 将变量 j 的值转换为字符串并拼接到返回字符串中
					'}';  # 返回字符串 "}"
		}
	}  # 结束 toString 方法的定义

}  # 结束类的定义
```