# `d:/src/tocomm/basic-computer-games\71_Poker\java\Poker.java`

```
import java.util.Random;  // 导入 Random 类，用于生成随机数
import java.util.Scanner;  // 导入 Scanner 类，用于接收用户输入

import static java.lang.System.out;  // 静态导入 System 类的 out 对象，用于输出信息

/**
 * Port of CREATIVE COMPUTING Poker written in Commodore 64 Basic to plain Java
 *
 * Original source scanned from magazine: https://www.atariarchives.org/basicgames/showpage.php?page=129
 *
 * I based my port on the OCR'ed source code here: https://github.com/coding-horror/basic-computer-games/blob/main/71_Poker/poker.bas
 *
 * Why? Because I remember typing this into my C64 when I was a tiny little developer and having great fun playing it!
 *
 * Goal: Keep the algorithms and UX more or less as-is; Improve the control flow a bit (no goto in Java!) and rename some stuff to be easier to follow.
 *
 * Result: There are probably bugs, please let me know.
 */
public class Poker {
	public static void main(String[] args) {
		new Poker().run(); // 创建一个新的Poker对象并调用其run方法
	}

	float[] cards = new float[50]; // 创建一个长度为50的浮点数数组，用于存储扑克牌的值，索引1-5为玩家手牌，索引6-10为电脑手牌
	float[] B = new float[15]; // 创建一个长度为15的浮点数数组B

	float playerValuables = 1; // 玩家的财产价值
	float computerMoney = 200; // 电脑的金钱
	float humanMoney = 200; // 玩家的金钱
	float pot = 0; // 奖池的金额

	String J$ = ""; // 一个空字符串J$
	float computerHandValue = 0; // 电脑手牌的价值

	int K = 0; // 整数K的值为0
	float G = 0; // 浮点数G的值为0
	float T = 0; // 浮点数T的值为0
	int M = 0; // 整数M的值为0
	int D = 0; // 整数D的值为0
# 初始化整型变量 U 为 0
int U = 0;
# 初始化浮点型变量 N 为 1
float N = 1;

# 初始化浮点型变量 I 为 0
float I = 0;

# 初始化浮点型变量 X 为 0
float X = 0;

# 初始化整型变量 Z 为 0
int Z = 0;

# 初始化字符串变量 handDescription 为空字符串
String handDescription = "";

# 初始化浮点型变量 V，未赋初值

# 定义函数 run，包含打印欢迎信息、进行一轮游戏、重新开始游戏的操作
void run() {
    printWelcome();
    playRound();
    startAgain();
}
	// 打印欢迎信息
	void printWelcome() {
		// 跳到第33个字符位置
		tab(33);
		// 打印标题
		out.println("POKER");
		// 跳到第15个字符位置
		tab(15);
		// 打印公司信息
		out.print("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
		out.println();
		out.println();
		out.println();
		// 打印欢迎词
		out.println("WELCOME TO THE CASINO.  WE EACH HAVE $200.");
		out.println("I WILL OPEN THE BETTING BEFORE THE DRAW; YOU OPEN AFTER.");
		out.println("TO FOLD BET 0; TO CHECK BET .5.");
		out.println("ENOUGH TALK -- LET'S GET DOWN TO BUSINESS.");
		out.println();
	}

	// 打印指定数量的制表符
	void tab(int number) {
		// 打印指定数量的制表符
		System.out.print("\t".repeat(number));
	}

	// 生成一个0到10之间的随机数
	int random0to10() {
		return new Random().nextInt(10);  # 返回一个0到9之间的随机整数

	int removeHundreds(long x) {  # 定义一个函数，用于去除参数x中的百位数
		return _int(x - (100F * _int(x / 100F)));  # 返回x减去百位数后的结果
	}

	void startAgain() {  # 定义一个函数，用于重新开始游戏
		pot = 0;  # 将pot变量重置为0
		playRound();  # 调用playRound函数开始新的一轮游戏
	}

	void playRound() {  # 定义一个函数，用于进行游戏的一轮
		if (computerMoney <= 5) {  # 如果电脑的钱少于或等于5
			computerBroke();  # 调用computerBroke函数
		}

		out.println("THE ANTE IS $5.  I WILL DEAL:");  # 输出文本信息
		out.println();  # 输出空行
		# 如果玩家的钱少于等于5，调用playerBroke()函数
		if (humanMoney <= 5) {
			playerBroke();
		}

		# 将底池增加10
		pot = pot + 10;
		# 玩家的钱减少5
		humanMoney = humanMoney - 5;
		# 电脑的钱减少5
		computerMoney = computerMoney - 5;
		# 循环10次，生成牌
		for (int Z = 1; Z < 10; Z++) {
			generateCards(Z);
		}
		# 输出"YOUR HAND:"
		out.println("YOUR HAND:");
		# 设置N为1，展示手牌
		N = 1;
		showHand();
		# 设置N为6，I为2，描述手牌
		N = 6;
		I = 2;
		describeHand();
		# 输出空行
		out.println();
		# 如果 I 不等于 6
		if (I != 6) {
			# 如果 U 大于等于 13
			if (U >= 13) {
				# 如果 U 小于等于 16
				if (U <= 16) {
					# 设置 Z 为 35
					Z = 35;
				} else {
					# 设置 Z 为 2
					Z = 2;
					# 如果随机数小于 1
					if (random0to10() < 1) {
						# 设置 Z 为 35
						Z = 35;
					}
				}
				# 让计算机进行开局
				computerOpens();
				# 玩家进行移动
				playerMoves();
			} else if (random0to10() >= 2) {
				# 让计算机进行检查
				computerChecks();
			} else {
				# 设置 I 为 7
				I = 7;
				# 设置 Z 为 23
				Z = 23;
				# 让计算机进行开局
				computerOpens();
				# 玩家进行移动
				playerMoves();
			}
		} else if (random0to10() <= 7) {  # 如果随机数小于等于7
			if (random0to10() <= 7) {  # 如果随机数小于等于7
				if (random0to10() >= 1) {  # 如果随机数大于等于1
					Z = 1;  # 设置 Z 为 1
					K = 0;  # 设置 K 为 0
					out.print("I CHECK. ");  # 输出 "I CHECK."
					playerMoves();  # 调用 playerMoves() 函数
				} else {
					X = 11111;  # 设置 X 为 11111
					I = 7;  # 设置 I 为 7
					Z = 23;  # 设置 Z 为 23
					computerOpens();  # 调用 computerOpens() 函数
					playerMoves();  # 调用 playerMoves() 函数
				}
			} else {
				X = 11110;  # 设置 X 为 11110
				I = 7;  # 设置 I 为 7
				Z = 23;  # 设置 Z 为 23
				computerOpens();  # 调用 computerOpens() 函数
				playerMoves();  # 调用 playerMoves() 函数
		}
		# 如果条件不满足，则执行以下代码
		else:
			# 设置变量X为11100
			X = 11100;
			# 设置变量I为7
			I = 7;
			# 设置变量Z为23
			Z = 23;
			# 调用computerOpens函数
			computerOpens();
			# 调用playerMoves函数
			playerMoves();
	}

	# 定义playerMoves函数
	void playerMoves() {
		# 调用playersTurn函数
		playersTurn();
		# 调用checkWinnerAfterFirstBet函数
		checkWinnerAfterFirstBet();
		# 调用promptPlayerDrawCards函数
		promptPlayerDrawCards();
	}

	# 定义computerOpens函数
	void computerOpens() {
		# 将变量V设置为Z加上一个0到10的随机数
		V = Z + random0to10();
		# 调用computerMoves函数
		computerMoves();
		# 打印"I'LL OPEN WITH $"和V的值
		out.print("I'LL OPEN WITH $" + V);
		K = _int(V);  # 将变量 V 转换为整数并赋值给变量 K
	}

	@SuppressWarnings("StatementWithEmptyBody")
	void computerMoves() {
		if (computerMoney - G - V >= 0) {  # 如果计算机的钱减去 G 和 V 的差大于等于 0
		} else if (G != 0) {  # 否则如果 G 不等于 0
			if (computerMoney - G >= 0) {  # 如果计算机的钱减去 G 大于等于 0
				computerSees();  # 计算机进行看牌操作
			} else {
				computerBroke();  # 否则计算机宣布破产
			}
		} else {
			V = computerMoney;  # 否则将计算机的钱赋值给 V
		}
	}

	void promptPlayerDrawCards() {
		out.println();  # 输出空行
		out.println("NOW WE DRAW -- HOW MANY CARDS DO YOU WANT");  # 输出提示信息
		inputPlayerDrawCards();  # 调用函数inputPlayerDrawCards()

	void inputPlayerDrawCards() {  # 定义函数inputPlayerDrawCards()
		T = Integer.parseInt(readString());  # 从输入中读取一个整数赋值给变量T
		if (T == 0) {  # 如果T等于0
			computerDrawing();  # 调用函数computerDrawing()
		} else {  # 否则
			Z = 10;  # 将变量Z赋值为10
			if (T < 4) {  # 如果T小于4
				playerDrawsCards();  # 调用函数playerDrawsCards()
			} else {  # 否则
				out.println("YOU CAN'T DRAW MORE THAN THREE CARDS.");  # 输出提示信息
				inputPlayerDrawCards();  # 重新调用函数inputPlayerDrawCards()
			}
		}
	}

	// line # 980
	void computerDrawing() {  # 定义函数computerDrawing()
		# 将整数10与T相加，并赋值给变量Z
		Z = _int(10 + T);
		# 循环，从U=6开始，直到U=10结束
		for (U = 6; U <= 10; U++) {
			# 如果X除以10的(U-6)次方的结果取整等于10乘以(X除以10的(U-5)次方)的结果取整
			if (_int((float) (X / Math.pow(10F, (U - 6F)))) == (10 * (_int((float) (X / Math.pow(10, (U - 5))))))) {
				# 调用drawNextCard函数
				drawNextCard();
			}
		}
		# 打印"I AM TAKING "加上Z-10-T的整数，再加上"CARD"
		out.print("I AM TAKING " + _int(Z - 10 - T) + " CARD");
		# 如果Z等于11+T，则换行打印
		if (Z == 11 + T) {
			out.println();
		} else {
			# 否则打印"S"
			out.println("S");
		}

		# 将变量N赋值为6
		N = 6;
		# 将变量V赋值为I的值
		V = I;
		# 将变量I赋值为1
		I = 1;
		# 调用describeHand函数
		describeHand();
		# 调用startPlayerBettingAndReaction函数
		startPlayerBettingAndReaction();
	}
	void drawNextCard() {  // 定义一个函数，用于绘制下一张卡片
		Z = Z + 1;  // 将 Z 的值加一
		drawCard();  // 调用 drawCard 函数
	}

	@SuppressWarnings("StatementWithEmptyBody")  // 忽略空语句的警告
	void drawCard() {  // 定义一个函数，用于绘制卡片
		cards[Z] = 100 * new Random().nextInt(4) + new Random().nextInt(100);  // 生成一个随机数并赋值给 cards[Z]
		if (_int(cards[Z] / 100) > 3) {  // 如果 cards[Z] 除以 100 的商大于 3
			drawCard();  // 调用 drawCard 函数
		} else if (cards[Z] - 100 * _int(cards[Z] / 100) > 12) {  // 如果 cards[Z] 减去 100 乘以 cards[Z] 除以 100 的商大于 12
			drawCard();  // 调用 drawCard 函数
		} else if (Z == 1) {  // 如果 Z 等于 1
		} else {  // 否则
			for (K = 1; K <= Z - 1; K++) {  // 循环，K 从 1 到 Z-1
				if (cards[Z] == cards[K]) {  // 如果 cards[Z] 等于 cards[K]
					drawCard();  // 调用 drawCard 函数
				}
			}
			if (Z <= 10) {  // 如果 Z 小于等于 10
		} else {  # 如果条件不满足
			N = cards[U];  # 将变量N赋值为cards[U]的值
			cards[U] = cards[Z];  # 将cards[U]的值赋值为cards[Z]的值
			cards[Z] = N;  # 将cards[Z]的值赋值为N的值
		}
	}

	void playerDrawsCards() {  # 定义一个名为playerDrawsCards的函数
		out.println("WHAT ARE THEIR NUMBERS:");  # 输出字符串"WHAT ARE THEIR NUMBERS:"
		for (int Q = 1; Q <= T; Q++) {  # 循环，从1到T
			U = Integer.parseInt(readString());  # 将读取的字符串转换为整数并赋值给变量U
			drawNextCard();  # 调用drawNextCard函数
		}

		out.println("YOUR NEW HAND:");  # 输出字符串"YOUR NEW HAND:"
		N = 1;  # 将变量N赋值为1
		showHand();  # 调用showHand函数
		computerDrawing();  # 调用computerDrawing函数
	}
		# 初始化计算机手牌的值为U
		computerHandValue = U
		# 初始化M为D
		M = D

		# 如果V不等于7
		if (V != 7):
			# 如果I不等于6
			if (I != 6):
				# 如果U大于等于13
				if (U >= 13):
					# 如果U大于等于16
					if (U >= 16):
						# 设置Z为2
						Z = 2
						# 调用playerBetsAndComputerReacts函数
						playerBetsAndComputerReacts()
					else:
						# 设置Z为19
						Z = 19
						# 如果random0to10()的返回值为8
						if (random0to10() == 8):
							# 设置Z为11
							Z = 11
						# 调用playerBetsAndComputerReacts函数
						playerBetsAndComputerReacts()
				else:
					# 设置Z为2
					Z = 2
# 如果随机数等于6，则将Z赋值为19
if (random0to10() == 6) {
    Z = 19;
}
# 调用playerBetsAndComputerReacts()函数
playerBetsAndComputerReacts();
# 如果条件不满足，则执行下面的代码
} else {
    Z = 1;
    # 调用playerBetsAndComputerReacts()函数
    playerBetsAndComputerReacts();
}
# 如果条件不满足，则执行下面的代码
} else {
    Z = 28;
    # 调用playerBetsAndComputerReacts()函数
    playerBetsAndComputerReacts();
}

# 定义playerBetsAndComputerReacts()函数
void playerBetsAndComputerReacts() {
    K = 0;
    # 调用playersTurn()函数
    playersTurn();
    # 如果条件不满足，则执行下面的代码
    if (T != .5) {
        # 调用checkWinnerAfterFirstBetAndCompareHands()函数
        checkWinnerAfterFirstBetAndCompareHands();
		} else if (V == 7 || I != 6) {  # 如果 V 等于 7 或者 I 不等于 6
			computerOpens();  # 调用 computerOpens() 函数
			promptAndInputPlayerBet();  # 调用 promptAndInputPlayerBet() 函数
			checkWinnerAfterFirstBetAndCompareHands();  # 调用 checkWinnerAfterFirstBetAndCompareHands() 函数
		} else {  # 否则
			out.println("I'LL CHECK");  # 输出 "I'LL CHECK"
			compareHands();  # 调用 compareHands() 函数
		}
	}

	void checkWinnerAfterFirstBetAndCompareHands() {  # 定义 checkWinnerAfterFirstBetAndCompareHands() 函数
		checkWinnerAfterFirstBet();  # 调用 checkWinnerAfterFirstBet() 函数
		compareHands();  # 调用 compareHands() 函数
	}

	void compareHands() {  # 定义 compareHands() 函数
		out.println("NOW WE COMPARE HANDS:");  # 输出 "NOW WE COMPARE HANDS:"
		J$ = handDescription;  # 将 handDescription 赋值给 J$
		out.println("MY HAND:");  # 输出 "MY HAND:"
		N = 6;  # 将 N 赋值为 6
		showHand();  # 调用函数显示当前手牌
		N = 1;  # 将变量N赋值为1
		describeHand();  # 调用函数描述手牌
		out.print("YOU HAVE ");  # 输出字符串"YOU HAVE "
		K = D;  # 将变量K赋值为D
		printHandDescriptionResult();  # 调用函数打印手牌描述结果
		handDescription = J$;  # 将handDescription赋值为J$
		K = M;  # 将变量K赋值为M
		out.print(" AND I HAVE ");  # 输出字符串" AND I HAVE "
		printHandDescriptionResult();  # 调用函数打印手牌描述结果
		out.print(". ");  # 输出字符串". "
		if (computerHandValue > U) {  # 如果computerHandValue大于U
			computerWins();  # 调用函数表示电脑获胜
		} else if (U > computerHandValue) {  # 否则如果U大于computerHandValue
			humanWins();  # 调用函数表示玩家获胜
		} else if (handDescription.contains("A FLUS")) {  # 否则如果handDescription包含"A FLUS"
			someoneWinsWithFlush();  # 调用函数表示有人以同花牌型获胜
		} else if (removeHundreds(M) < removeHundreds(D)) {  # 否则如果去除百位后的M小于去除百位后的D
			humanWins();  # 调用函数表示玩家获胜
		} else if (removeHundreds(M) > removeHundreds(D)) {  # 否则如果去除百位后的M大于去除百位后的D
			computerWins();  # 调用名为computerWins的函数，表示电脑获胜
		} else {
			handIsDrawn();  # 调用名为handIsDrawn的函数，表示手牌平局
		}
	}

	void printHandDescriptionResult() {
		out.print(handDescription);  # 打印手牌描述
		if (!handDescription.contains("A FLUS")) {  # 如果手牌描述中不包含"A FLUS"
			K = removeHundreds(K);  # 调用名为removeHundreds的函数，移除K中的百位数
			printCardValue();  # 调用名为printCardValue的函数，打印卡牌数值
			if (handDescription.contains("SCHMAL")) {  # 如果手牌描述中包含"SCHMAL"
				out.print(" HIGH");  # 打印" HIGH"
			} else if (!handDescription.contains("STRAIG")) {  # 如果手牌描述中不包含"STRAIG"
				out.print("'S");  # 打印"'S"
			} else {
				out.print(" HIGH");  # 打印" HIGH"
			}
		} else {
			K = K / 100;  # 将K除以100
			printCardColor(); // 调用printCardColor方法，打印卡片颜色
			out.println(); // 打印空行
		}
	}

	void handIsDrawn() {
		out.print("THE HAND IS DRAWN."); // 打印“THE HAND IS DRAWN.”
		out.print("ALL $" + pot + " REMAINS IN THE POT."); // 打印“ALL $”加上pot变量的值再加上“ REMAINS IN THE POT.”
		playRound(); // 调用playRound方法
	}

	void someoneWinsWithFlush() {
		if (removeHundreds(M) > removeHundreds(D)) { // 如果removeHundreds(M)的值大于removeHundreds(D)的值
			computerWins(); // 调用computerWins方法
		} else if (removeHundreds(D) > removeHundreds(M)) { // 否则如果removeHundreds(D)的值大于removeHundreds(M)的值
			humanWins(); // 调用humanWins方法
		} else { // 否则
			handIsDrawn(); // 调用handIsDrawn方法
		}
	}
	@SuppressWarnings("StatementWithEmptyBody")  // 使用注解标记此方法可能包含空语句，避免编译器警告
	void checkWinnerAfterFirstBet() {  // 定义一个名为checkWinnerAfterFirstBet的方法
		if (I != 3) {  // 如果变量I的值不等于3
			if (I != 4) {  // 如果变量I的值不等于4
			} else {  // 否则
				humanWins();  // 调用humanWins方法
			}
		} else {  // 如果变量I的值等于3
			out.println();  // 输出空行
			computerWins();  // 调用computerWins方法
		}
	}

	void computerWins() {  // 定义一个名为computerWins的方法
		out.print(". I WIN. ");  // 输出". I WIN. "
		computerMoney = computerMoney + pot;  // 计算computerMoney的新值
		potStatusAndNextRoundPrompt();  // 调用potStatusAndNextRoundPrompt方法
	}
	// 输出计算机和玩家的资金情况
	void potStatusAndNextRoundPrompt() {
		out.println("NOW I HAVE $" + computerMoney + " AND YOU HAVE $" + humanMoney);
		// 提示玩家是否继续游戏
		out.print("DO YOU WISH TO CONTINUE");

		// 如果玩家选择继续游戏，则重新开始游戏
		if (yesFromPrompt()) {
			startAgain();
		} else {
			// 如果玩家选择退出游戏，则退出程序
			System.exit(0);
		}
	}

	// 从用户输入中判断是否选择继续游戏
	private boolean yesFromPrompt() {
		// 读取用户输入
		String h = readString();
		if (h != null) {
			// 如果用户输入是肯定的，则返回true
			if (h.toLowerCase().matches("y|yes|yep|affirmative|yay")) {
				return true;
			} else if (h.toLowerCase().matches("n|no|nope|fuck off|nay")) {
				// 如果用户输入是否定的，则返回false
				return false;
			}
		}
	}
		out.println("ANSWER YES OR NO, PLEASE.");  // 打印提示信息要求用户回答是或否
		return yesFromPrompt();  // 调用函数从用户输入中获取回答
	}

	void computerChecks() {
		Z = 0;  // 初始化变量 Z 为 0
		K = 0;  // 初始化变量 K 为 0
		out.print("I CHECK. ");  // 打印信息表示电脑进行检查
		playerMoves();  // 调用玩家移动的函数
	}

	void humanWins() {
		out.println("YOU WIN.");  // 打印信息表示玩家获胜
		humanMoney = humanMoney + pot;  // 更新玩家的金钱数量
		potStatusAndNextRoundPrompt();  // 调用函数显示奖池状态并提示下一轮
	}

	// line # 1740
	void generateCards(int Z) {
		cards[Z] = (100 * new Random().nextInt(4)) + new Random().nextInt(100);  // 生成一张卡片并将其存储在指定位置
		# 如果卡片值除以100的商大于3，则调用generateCards函数并返回
		if (_int(cards[Z] / 100) > 3) {
			generateCards(Z);
			return;
		}
		# 如果卡片值减去100乘以商大于12，则调用generateCards函数并返回
		if (cards[Z] - 100 * (_int(cards[Z] / 100)) > 12) {
			generateCards(Z);
			return;
		}
		# 如果Z等于1，则直接返回
		if (Z == 1) {return;}
		# 遍历1到Z-1的卡片值，如果有重复的值，则调用generateCards函数并返回
		for (int K = 1; K <= Z - 1; K++) {// TO Z-1
			if (cards[Z] == cards[K]) {
				generateCards(Z);
				return;
			}
		}
		# 如果Z小于等于10，则直接返回
		if (Z <= 10) {return;}
		# 交换U位置和Z位置的卡片值
		float N = cards[U];
		cards[U] = cards[Z];
		cards[Z] = N;
	}
// line # 1850
// 定义一个名为 showHand 的函数，用于展示玩家的手牌
void showHand() {
    // 循环遍历玩家手牌中的每张牌
    for (int cardNumber = _int(N); cardNumber <= N + 4; cardNumber++) {
        // 打印牌号
        out.print(cardNumber + "--  ");
        // 调用函数打印牌的数值
        printCardValueAtIndex(cardNumber);
        // 打印牌的花色
        out.print(" OF");
        printCardColorAtIndex(cardNumber);
        // 如果是偶数张牌，换行
        if (cardNumber / 2 == (cardNumber / 2)) {
            out.println();
        }
    }
}

// line # 1950
// 定义一个名为 printCardValueAtIndex 的函数，用于打印指定位置牌的数值
void printCardValueAtIndex(int Z) {
    // 移除牌的百位数
    K = removeHundreds(_int(cards[Z]));
    // 调用函数打印牌的数值
    printCardValue();
}
// 打印卡片的值
void printCardValue() {
    // 如果卡片的值为9，打印"JACK"
    if (K == 9) {
        out.print("JACK");
    } 
    // 如果卡片的值为10，打印"QUEEN"
    else if (K == 10) {
        out.print("QUEEN");
    } 
    // 如果卡片的值为11，打印"KING"
    else if (K == 11) {
        out.print("KING");
    } 
    // 如果卡片的值为12，打印"ACE"
    else if (K == 12) {
        out.print("ACE");
    } 
    // 如果卡片的值小于9，打印卡片的值加2
    else if (K < 9) {
        out.print(K + 2);
    }
}

// line # 2070
// 打印给定索引处卡片的颜色
void printCardColorAtIndex(int Z) {
    // 获取卡片的值
    K = _int(cards[Z] / 100);
    // 调用打印卡片颜色的函数
    printCardColor();
}
	void printCardColor() { // 定义一个打印卡片花色的函数
		if (K == 0) { // 如果 K 的值为 0
			out.print(" CLUBS"); // 打印出 " CLUBS"
		} else if (K == 1) { // 如果 K 的值为 1
			out.print(" DIAMONDS"); // 打印出 " DIAMONDS"
		} else if (K == 2) { // 如果 K 的值为 2
			out.print(" HEARTS"); // 打印出 " HEARTS"
		} else if (K == 3) { // 如果 K 的值为 3
			out.print(" SPADES"); // 打印出 " SPADES"
		}
	}

	// line # 2170
	void describeHand() { // 定义一个描述手牌的函数
		U = 0; // 初始化 U 的值为 0
		for (Z = _int(N); Z <= N + 4; Z++) { // 循环遍历从 N 到 N + 4 的值
			B[Z] = removeHundreds(_int(cards[Z])); // 将 cards[Z] 的百位数去除后赋值给 B[Z]
			if (Z == N + 4) {continue;} // 如果 Z 的值等于 N + 4，则继续下一次循环
			if (_int(cards[Z] / 100) != _int(cards[Z + 1] / 100)) {continue;} // 如果 cards[Z] 除以 100 的商不等于 cards[Z + 1] 除以 100 的商，则继续下一次循环
			U = U + 1; // U 的值加 1
		}
		if (U != 4):  # 如果 U 不等于 4
			for (Z = _int(N); Z <= N + 3; Z++):  # 循环 Z 从 N 取整到 N+3
				for (K = Z + 1; K <= N + 4; K++):  # 循环 K 从 Z+1 到 N+4
					if (B[Z] <= B[K]):  # 如果 B[Z] 小于等于 B[K]
						continue  # 继续下一次循环
					X = cards[Z]  # 将 cards[Z] 赋值给 X
					cards[Z] = cards[K]  # 将 cards[K] 赋值给 cards[Z]
					B[Z] = B[K]  # 将 B[K] 赋值给 B[Z]
					cards[K] = X  # 将 X 赋值给 cards[K]
					B[K] = cards[K] - 100 * _int(cards[K] / 100)  # 计算并赋值给 B[K]
			X = 0  # 将 X 置零
			for (Z = _int(N); Z <= N + 3; Z++):  # 循环 Z 从 N 取整到 N+3
				if (B[Z] != B[Z + 1]):  # 如果 B[Z] 不等于 B[Z+1]
					continue  # 继续下一次循环
				X = (float) (X + 11 * Math.pow(10, (Z - N)))  # 计算并赋值给 X
				D = _int(cards[Z])  # 将 cards[Z] 取整并赋值给 D
				if (U >= 11):  # 如果 U 大于等于 11
					if (U != 11):  # 如果 U 不等于 11
# 如果 U 大于 12
if (U > 12):
    # 如果当前牌不等于上一张牌
    if (B[Z] != B[Z - 1]):
        # 调用 fullHouse() 函数
        fullHouse()
    else:
        # 将 U 设为 17
        U = 17
        # 将手牌描述设为 "FOUR "
        handDescription = "FOUR "
# 如果 U 不大于 12
else:
    # 调用 fullHouse() 函数
    fullHouse()
# 如果 B[Z] 不等于 B[Z - 1]
if (B[Z] != B[Z - 1]):
    # 将手牌描述设为 "TWO PAIR, "
    handDescription = "TWO PAIR, "
    # 将 U 设为 12
    U = 12
else:
    # 将手牌描述设为 "THREE "
    handDescription = "THREE "
    # 将 U 设为 13
    U = 13
# 如果不满足上述条件
else:
    # 将 U 设为 11
    U = 11
    # 将手牌描述设为 "A PAIR OF "
    handDescription = "A PAIR OF "
				}
			}

			if (X != 0):  # 如果 X 不等于 0
				schmaltzHand()  # 调用 schmaltzHand() 函数
			else:
				if (B[_int(N)] + 3 == B[_int(N + 3)]):  # 如果 B[N] + 3 等于 B[N + 3]
					X = 1111  # 将 X 赋值为 1111
					U = 10  # 将 U 赋值为 10
				if (B[_int(N + 1)] + 3 != B[_int(N + 4)]):  # 如果 B[N + 1] + 3 不等于 B[N + 4]
					schmaltzHand()  # 调用 schmaltzHand() 函数
				elif (U != 10):  # 否则如果 U 不等于 10
					U = 10  # 将 U 赋值为 10
					X = 11110  # 将 X 赋值为 11110
					schmaltzHand()  # 调用 schmaltzHand() 函数
				else:
					U = 14  # 将 U 赋值为 14
					handDescription = "STRAIGHT"  # 将 handDescription 赋值为 "STRAIGHT"
					X = 11111  # 将 X 赋值为 11111
					D = _int(cards[_int(N + 4)]);  # 从列表cards中获取索引为N+4的元素，将其转换为整数赋值给变量D
				}
			}
		} else {
			X = 11111;  # 将变量X赋值为11111
			D = _int(cards[_int(N)]);  # 从列表cards中获取索引为N的元素，将其转换为整数赋值给变量D
			handDescription = "A FLUSH IN";  # 将handDescription赋值为"A FLUSH IN"
			U = 15;  # 将变量U赋值为15
		}
	}

	void schmaltzHand() {
		if (U >= 10) {  # 如果变量U大于等于10
			if (U != 10) {  # 如果变量U不等于10
				if (U > 12) {return;}  # 如果变量U大于12，则结束函数
				if (removeHundreds(D) <= 6) {  # 如果removeHundreds(D)的结果小于等于6
					I = 6;  # 将变量I赋值为6
				}
			} else {
				if (I == 1) {  # 如果变量I等于1
					I = 6;  # 将变量I的值设置为6
				}
			}
		} else {
			D = _int(cards[_int(N + 4)]);  # 将变量D的值设置为cards[N + 4]的整数值
			handDescription = "SCHMALTZ, ";  # 将handDescription的值设置为"SCHMALTZ, "
			U = 9;  # 将变量U的值设置为9
			X = 11000;  # 将变量X的值设置为11000
			I = 6;  # 将变量I的值设置为6
		}
	}

	void fullHouse() {
		U = 16;  # 将变量U的值设置为16
		handDescription = "FULL HOUSE, ";  # 将handDescription的值设置为"FULL HOUSE, "
	}

	void playersTurn() {
		G = 0;  # 将变量G的值设置为0
		promptAndInputPlayerBet();  # 调用promptAndInputPlayerBet()函数
	}

	// 读取用户输入的字符串
	String readString() {
		Scanner sc = new Scanner(System.in);
		return sc.nextLine();
	}

	// 提示用户输入赌注并进行处理
	@SuppressWarnings("StatementWithEmptyBody")
	void promptAndInputPlayerBet() {
		out.println("WHAT IS YOUR BET");
		T = readFloat();
		// 如果用户输入的赌注是整数，则处理赌注
		if (T - _int(T) == 0) {
			processPlayerBet();
		} 
		// 如果用户输入的赌注不是整数且K不等于0，则提示用户赌注无效
		else if (K != 0) {
			playerBetInvalidAmount();
		} 
		// 如果用户输入的赌注不是整数且G不等于0，则提示用户赌注无效
		else if (G != 0) {
			playerBetInvalidAmount();
		} 
		// 如果用户输入的赌注是0.5，则不进行任何处理
		else if (T == .5) {
		} 
		// 如果用户输入的赌注不符合以上条件，则提示用户赌注无效
		else {
			playerBetInvalidAmount();
		}
	}

	# 读取一个浮点数并返回
	private float readFloat() {
		try:
			# 尝试将读取的字符串转换为浮点数并返回
			return Float.parseFloat(readString());
		except (Exception ex):
			# 如果出现异常，打印错误信息并提示用户重新输入浮点数
			System.out.println("INVALID INPUT, PLEASE TYPE A FLOAT. ");
			return readFloat();
	}

	# 提示玩家下注金额无效
	void playerBetInvalidAmount() {
		# 打印提示信息
		out.println("NO SMALL CHANGE, PLEASE.");
		# 提示并输入玩家下注金额
		promptAndInputPlayerBet();
	}

	# 处理玩家下注
	void processPlayerBet() {
		# 如果玩家的余额减去赌注和保险费后仍大于等于0
		if (humanMoney - G - T >= 0) {
			# 玩家能够承担赌注
			humanCanAffordBet();
		} else {
			// 如果玩家无法支付赌注，则调用playerBroke函数
			playerBroke();
			// 提示玩家输入赌注并进行输入
			promptAndInputPlayerBet();
		}
	}

	// 判断玩家是否能够支付赌注
	void humanCanAffordBet() {
		// 如果T不等于0
		if (T != 0) {
			// 如果玩家的总资产加上赌注大于等于底池
			if (G + T >= K) {
				// 处理电脑的移动
				processComputerMove();
			} else {
				// 输出提示信息
				out.println("IF YOU CAN'T SEE MY BET, THEN FOLD.");
				// 提示玩家输入赌注并进行输入
				promptAndInputPlayerBet();
			}
		} else {
			// 将I设置为3
			I = 3;
			// 将赌注移动到底池
			moveMoneyToPot();
		}
	}
	# 计算电脑的移动
	def processComputerMove():
		# 将 G 和 T 相加
		G = G + T
		# 如果 G 等于 K
		if (G == K):
			# 将钱移动到奖池
			moveMoneyToPot()
		# 如果 Z 不等于 1
		elif (Z != 1):
			# 如果 G 大于 3 * Z
			if (G > 3 * Z):
				# 电脑加注或跟注
				computerRaisesOrSees()
			else:
				# 电脑加注
				computerRaises()
		# 如果 Z 等于 1 且 G 大于 5
		elif (G > 5):
			# 如果 T 小于等于 25
			if (T <= 25):
				# 电脑加注或跟注
				computerRaisesOrSees()
			else:
				# 电脑弃牌
				computerFolds()
		else:
			# V 等于 5
			V = 5
			# 如果 G 大于 3 * Z
			if (G > 3 * Z):
				# 电脑加注或跟注
				computerRaisesOrSees()
		} else {
			# 如果玩家下注金额大于电脑的余额，电脑选择加注
			computerRaises();
		}
	}
	
	# 电脑加注操作
	void computerRaises() {
		# 计算加注金额，公式为：加注金额 = 筹码池金额 - 上一次加注金额 + 0到10的随机数
		V = G - K + random0to10();
		# 电脑进行加注操作
		computerMoves();
		# 输出电脑的加注信息
		out.println("I'LL SEE YOU, AND RAISE YOU" + V);
		# 更新上一次加注金额
		K = _int(G + V);
		# 提示玩家输入下注金额
		promptAndInputPlayerBet();
	}

	# 电脑弃牌操作
	void computerFolds() {
		# 将电脑的状态设置为弃牌
		I = 4;
		# 输出电脑的弃牌信息
		out.println("I FOLD.");
	}

	# 电脑选择加注或跟注操作
	void computerRaisesOrSees() {
		if (Z == 2) {  # 如果 Z 的值为 2
			computerRaises();  # 则调用 computerRaises() 函数
		} else {  # 否则
			computerSees();  # 调用 computerSees() 函数
		}
	}

	void computerSees() {  # 定义 computerSees() 函数
		out.println("I'LL SEE YOU.");  # 输出 "I'LL SEE YOU."
		K = _int(G);  # 将 G 转换为整数并赋值给 K
		moveMoneyToPot();  # 调用 moveMoneyToPot() 函数
	}

	void moveMoneyToPot() {  # 定义 moveMoneyToPot() 函数
		humanMoney = humanMoney - G;  # 从 humanMoney 中减去 G
		computerMoney = computerMoney - K;  # 从 computerMoney 中减去 K
		pot = pot + G + K;  # 将 G 和 K 的值加到 pot 中
	}

	void computerBusted() {  # 定义 computerBusted() 函数
		out.println("I'M BUSTED.  CONGRATULATIONS!");  // 打印出玩家破产的消息
		System.exit(0);  // 立即退出程序
	}

	@SuppressWarnings("StatementWithEmptyBody")  // 忽略空语句的警告
	private void computerBroke() {  // 定义一个名为computerBroke的方法
		if ((playerValuables / 2) == _int(playerValuables / 2) && playerBuyBackWatch()) {  // 如果玩家财物除以2的结果是整数并且玩家可以买回手表
		} else if (playerValuables / 3 == _int(playerValuables / 3) && playerBuyBackTieRack()) {  // 否则如果玩家财物除以3的结果是整数并且玩家可以买回领带架
		} else {
			computerBusted();  // 否则调用computerBusted方法
		}
	}

	private int _int(float v) {  // 定义一个名为_int的方法，将浮点数转换为整数
		return (int) Math.floor(v);  // 返回向下取整的结果
	}

	private boolean playerBuyBackWatch() {  // 定义一个名为playerBuyBackWatch的方法，询问玩家是否愿意花50美元买回手表
		out.println("WOULD YOU LIKE TO BUY BACK YOUR WATCH FOR $50");  // 打印出询问消息
		if (yesFromPrompt()) {  // 如果玩家从提示中选择了“是”
			# 增加玩家的金钱，增加50
			computerMoney = computerMoney + 50;
			# 玩家贵重物品减半
			playerValuables = playerValuables / 2;
			# 返回真值
			return true;
		} else:
			# 返回假值
			return false;
		}
	}

	# 玩家买回领带夹
	private boolean playerBuyBackTieRack() {
		# 打印询问是否花50美元买回领带夹
		out.println("WOULD YOU LIKE TO BUY BACK YOUR TIE TACK FOR $50");
		# 如果从提示中得到肯定的回答
		if (yesFromPrompt()) {
			# 增加玩家的金钱，增加50
			computerMoney = computerMoney + 50;
			# 玩家贵重物品减少三分之一
			playerValuables = playerValuables / 3;
			# 返回真值
			return true;
		} else:
			# 返回假值
			return false;
		}
	}

	# 行号3830
	// 定义一个方法，当玩家破产时调用
	void playerBroke() {
		// 输出提示信息
		out.println("YOU CAN'T BET WITH WHAT YOU HAVEN'T GOT.");
		// 如果玩家财物除以2不等于整数并且玩家卖掉手表，则执行下面的代码块
		if (playerValuables / 2 != _int(playerValuables / 2) && playerSellWatch()) {
		} 
		// 如果玩家财物除以3不等于整数并且玩家卖掉领带夹，则执行下面的代码块
		else if (playerValuables / 3 != _int(playerValuables / 3) && playerSellTieTack()) {
		} 
		// 如果以上条件都不满足，则执行下面的代码块
		else {
			playerBusted();
		}
	}

	// 定义一个方法，当玩家破产时调用
	private void playerBusted() {
		// 输出提示信息
		out.println("YOUR WAD IS SHOT. SO LONG, SUCKER!");
		// 退出程序
		System.exit(0);
	}

	// 定义一个方法，用于询问玩家是否卖掉手表
	private boolean playerSellWatch() {
		// 输出提示信息
		out.println("WOULD YOU LIKE TO SELL YOUR WATCH");
		// 如果玩家选择是，则执行下面的代码块
		if (yesFromPrompt()) {
			// 如果随机数小于7，则执行下面的代码块
			if (random0to10() < 7) {
				// 输出提示信息
				out.println("I'LL GIVE YOU $75 FOR IT.");
				humanMoney = humanMoney + 75;  // 如果随机数小于6，玩家获得75美元
			} else {
				out.println("THAT'S A PRETTY CRUMMY WATCH - I'LL GIVE YOU $25.");  // 如果随机数大于等于6，输出信息并玩家获得25美元
				humanMoney = humanMoney + 25;  // 玩家获得25美元
			}
			playerValuables = playerValuables * 2;  // 玩家财产翻倍
			return true;  // 返回true
		} else {
			return false;  // 如果玩家不愿意出售，返回false
		}
	}

	private boolean playerSellTieTack() {
		out.println("WILL YOU PART WITH THAT DIAMOND TIE TACK");  // 输出信息询问玩家是否愿意出售钻石领带夹

		if (yesFromPrompt()) {  // 如果玩家选择是
			if (random0to10() < 6) {  // 生成0到10的随机数，如果小于6
				out.println("YOU ARE NOW $100 RICHER.");  // 输出信息并玩家获得100美元
				humanMoney = humanMoney + 100;  // 玩家获得100美元
			} else {
				out.println("IT'S PASTE.  $25.");  // 打印字符串 "IT'S PASTE.  $25." 到控制台
				humanMoney = humanMoney + 25;  // 将 humanMoney 变量的值增加 25
			}
			playerValuables = playerValuables * 3;  // 将 playerValuables 变量的值乘以 3
			return true;  // 返回 true
		} else {
			return false;  // 如果条件不满足，返回 false
		}
	}

}
```