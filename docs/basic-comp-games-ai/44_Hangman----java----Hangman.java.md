# `44_Hangman\java\Hangman.java`

```
import java.util.Arrays; // 导入 Arrays 类，用于操作数组
import java.util.LinkedHashSet; // 导入 LinkedHashSet 类，用于创建不重复元素的集合
import java.util.List; // 导入 List 类，用于创建列表
import java.util.Scanner; // 导入 Scanner 类，用于读取用户输入
import java.util.Set; // 导入 Set 类，用于创建集合
import java.util.stream.Collectors; // 导入 Collectors 类，用于在流中生成列表或其他集合

/**
 * HANGMAN
 *
 * Converted from BASIC to Java by Aldrin Misquitta (@aldrinm)
 */

public class Hangman {

	//50 word list
	private final static List<String> words = List.of( // 创建包含50个单词的列表
			"GUM", "SIN", "FOR", "CRY", "LUG", "BYE", "FLY",
			"UGLY", "EACH", "FROM", "WORK", "TALK", "WITH", "SELF",
			"PIZZA", "THING", "FEIGN", "FIEND", "ELBOW", "FAULT", "DIRTY",
			# 创建一个包含单词的列表
			words = Arrays.asList("BUDGET", "SPIRIT", "QUAINT", "MAIDEN", "ESCORT", "PICKAX",
			"EXAMPLE", "TENSION", "QUININE", "KIDNEY", "REPLICA", "SLEEPER",
			"TRIANGLE", "KANGAROO", "MAHOGANY", "SERGEANT", "SEQUENCE",
			"MOUSTACHE", "DANGEROUS", "SCIENTIST", "DIFFERENT", "QUIESCENT",
			"MAGISTRATE", "ERRONEOUSLY", "LOUDSPEAKER", "PHYTOTOXIC",
			"MATRIMONIAL", "PARASYMPATHOMIMETIC", "THIGMOTROPISM");

	public static void main(String[] args) {
		# 创建一个用于从控制台读取输入的 Scanner 对象
		Scanner scan = new Scanner(System.in);

		# 打印游戏介绍
		printIntro();

		# 创建一个包含已使用单词的数组
		int[] usedWords = new int[50];
		# 初始化游戏轮数和总单词数
		int roundNumber = 1;
		int totalWords = words.size();
		# 初始化是否继续游戏的标志
		boolean continueGame = false;

		# 开始游戏循环
		do {
			# 如果轮数大于总单词数，打印提示信息
			if (roundNumber > totalWords) {
				System.out.println("\nYOU DID ALL THE WORDS!!");
			// 如果用户选择不再玩游戏，则跳出循环
			if (anotherWordChoice.toUpperCase().equals("NO") || anotherWordChoice.toUpperCase().equals("N")) {
				break;
			}

			// 生成一个随机单词的索引
			int randomWordIndex;
			do {
				randomWordIndex = ((int) (totalWords * Math.random())) + 1;
			} while (usedWords[randomWordIndex] == 1);
			usedWords[randomWordIndex] = 1;

			// 调用playRound方法进行游戏回合，并判断是否获胜
			boolean youWon = playRound(scan, words.get(randomWordIndex - 1));
			if (!youWon) {
				System.out.print("\nYOU MISSED THAT ONE.  DO YOU WANT ANOTHER WORD? ");
			} else {
				System.out.print("\nWANT ANOTHER WORD? ");
			}
			final String anotherWordChoice = scan.next();

			// 如果用户选择继续游戏，则设置continueGame为true
			if (anotherWordChoice.toUpperCase().equals("YES") || anotherWordChoice.toUpperCase().equals("Y")) {
				continueGame = true;
			}
			roundNumber++;  // 增加游戏轮数
		} while (continueGame);  // 当游戏继续时执行循环

		System.out.println("\nIT'S BEEN FUN!  BYE FOR NOW.");  // 打印结束游戏的消息
	}

	private static boolean playRound(Scanner scan, String word) {  // 定义一个名为playRound的静态方法，接受Scanner对象和字符串word作为参数
		char[] letters;  // 声明字符数组变量letters
		char[] discoveredLetters;  // 声明字符数组变量discoveredLetters
		int misses = 0;  // 初始化整型变量misses为0
		Set<Character> lettersUsed = new LinkedHashSet<>();  // 创建一个LinkedHashSet对象lettersUsed，用于存储已使用的字母，并保持插入顺序

		String[][] hangmanPicture = new String[12][12];  // 创建一个12x12的二维字符串数组hangmanPicture，用于存储“hangman”游戏的图像
		//initialize the hangman picture
		for (int i = 0; i < hangmanPicture.length; i++) {  // 循环遍历hangmanPicture数组的行
			for (int j = 0; j < hangmanPicture[i].length; j++) {  // 循环遍历hangmanPicture数组的列
				hangmanPicture[i][j] = " ";  // 将数组元素初始化为空格
			}
		}
		for (int i = 0; i < hangmanPicture.length; i++) {  // 循环遍历hangmanPicture数组的行
			hangmanPicture[i][0] = "X";  // 在hangmanPicture数组中的第i行第0列设置为"X"
		}
		for (int i = 0; i < 7; i++) {  // 循环7次
			hangmanPicture[0][i] = "X";  // 在hangmanPicture数组中的第0行的第i列设置为"X"
		}
		hangmanPicture[1][6] = "X";  // 在hangmanPicture数组中的第1行的第6列设置为"X"

		int totalWordGuesses = 0;  // 初始化单词猜测次数为0

		int len = word.length();  // 获取单词的长度
		letters = word.toCharArray();  // 将单词转换为字符数组

		discoveredLetters = new char[len];  // 创建一个与单词长度相同的字符数组
		Arrays.fill(discoveredLetters, '-');  // 将discoveredLetters数组填充为'-'

		boolean validNextGuess = false;  // 初始化validNextGuess为false
		char guessLetter = ' ';  // 初始化guessLetter为一个空格

		while (misses < 10) {  // 当misses小于10时执行循环
			while (!validNextGuess) {  // 当validNextGuess为false时执行循环
				// 调用 printLettersUsed 方法，打印已使用的字母
				printLettersUsed(lettersUsed);
				// 调用 printDiscoveredLetters 方法，打印已发现的字母
				printDiscoveredLetters(discoveredLetters);

				// 打印提示信息，要求用户输入猜测的字母
				System.out.print("WHAT IS YOUR GUESS? ");
				// 读取用户输入的字符串
				var tmpRead = scan.next();
				// 将用户输入的字符串转换为大写字母
				guessLetter = Character.toUpperCase(tmpRead.charAt(0));
				// 如果已使用的字母中包含用户猜测的字母
				if (lettersUsed.contains(guessLetter)) {
					// 打印提示信息，告知用户已经猜测过该字母
					System.out.println("YOU GUESSED THAT LETTER BEFORE!");
				} else {
					// 将用户猜测的字母添加到已使用的字母列表中
					lettersUsed.add(guessLetter);
					// 增加总猜测次数
					totalWordGuesses++;
					// 设置下一次猜测为有效
					validNextGuess = true;
				}
			}

			// 如果单词中包含用户猜测的字母
			if (word.indexOf(guessLetter) >= 0) {
				// 在已发现的字母列表中替换所有出现的 D$ 为 G$
				for (int i = 0; i < letters.length; i++) {
					if (letters[i] == guessLetter) {
						discoveredLetters[i] = guessLetter;
				}
				}
				// 检查单词是否完全被猜出
				boolean isWordDiscovered = true;
				for (char discoveredLetter : discoveredLetters) {
					if (discoveredLetter == '-') {
						isWordDiscovered = false;
						break;
					}
				}
				// 如果单词已经被完全猜出，则输出提示信息并返回true
				if (isWordDiscovered) {
					System.out.println("YOU FOUND THE WORD!");
					return true;
				}

				// 输出已经猜出的字母
				printDiscoveredLetters(discoveredLetters);
				System.out.print("WHAT IS YOUR GUESS FOR THE WORD? ");
				final String wordGuess = scan.next();
				// 如果猜测的单词与答案相同，则输出提示信息
				if (wordGuess.toUpperCase().equals(word)) {
					System.out.printf("RIGHT!!  IT TOOK YOU %s GUESSES!", totalWordGuesses);
					return true;  # 如果猜测的字母在单词中出现，返回 true
				} else {
					System.out.println("WRONG.  TRY ANOTHER LETTER.");  # 如果猜测的字母不在单词中，提示用户尝试另一个字母
				}
			} else {
				misses = misses + 1;  # 如果猜测的字母不在单词中，增加错误次数
				System.out.println("\n\nSORRY, THAT LETTER ISN'T IN THE WORD.");  # 提示用户猜测的字母不在单词中
				drawHangman(misses, hangmanPicture);  # 调用函数绘制 hangman 图案
			}
			validNextGuess = false;  # 设置下一个猜测为无效
		}

		System.out.printf("SORRY, YOU LOSE.  THE WORD WAS %s", word);  # 输出用户猜测错误后的提示信息
		return false;  # 返回 false，表示用户猜测错误
	}

	private static void drawHangman(int m, String[][] hangmanPicture) {  # 定义绘制 hangman 图案的函数
		switch (m) {  # 根据错误次数选择不同的绘制方式
			case 1:
				System.out.println("FIRST, WE DRAW A HEAD");  # 绘制 hangman 图案的第一步
				# 在hangmanPicture的特定位置添加字符“-”
				hangmanPicture[2][5] = "-";
				hangmanPicture[2][6] = "-";
				hangmanPicture[2][7] = "-";
				hangmanPicture[3][4] = "(";
				hangmanPicture[3][5] = ".";
				hangmanPicture[3][7] = ".";
				hangmanPicture[3][8] = ")";
				hangmanPicture[4][5] = "-";
				hangmanPicture[4][6] = "-";
				hangmanPicture[4][7] = "-";
				break;
			case 2:
				# 打印“NOW WE DRAW A BODY.”
				System.out.println("NOW WE DRAW A BODY.");
				# 在hangmanPicture的特定位置添加字符“X”
				for (var i = 5; i <= 8; i++) {
					hangmanPicture[i][6] = "X";
				}
				break;
			case 3:
				# 打印“NEXT WE DRAW AN ARM.”
				System.out.println("NEXT WE DRAW AN ARM.");
				# 在hangmanPicture的特定位置添加字符“X”
				for (int i = 3; i <= 6; i++) {
# 画第一条横线
hangmanPicture[0][0] = "-";
# 画第二条横线
hangmanPicture[1][0] = "-";
# 画竖线
hangmanPicture[0][1] = "|";
# 画头部
hangmanPicture[2][1] = "O";
# 画身体
hangmanPicture[3][1] = "|";
# 画左臂
hangmanPicture[3][0] = "/";
# 画右臂
hangmanPicture[3][2] = "\\";
# 画左腿
hangmanPicture[4][0] = "/";
hangmanPicture[5][0] = "/";
# 画右腿
hangmanPicture[4][2] = "\\";
hangmanPicture[5][2] = "\\";
			case 7: // 如果猜错次数为7，打印提示信息
				System.out.println("NOW WE PUT UP A HAND.");
				hangmanPicture[2][10] = "\\"; // 在hangmanPicture数组中放置一个手臂的图形
				break; // 结束该case
			case 8: // 如果猜错次数为8，打印提示信息
				System.out.println("NEXT THE OTHER HAND.");
				hangmanPicture[2][2] = "/"; // 在hangmanPicture数组中放置另一个手臂的图形
				break; // 结束该case
			case 9: // 如果猜错次数为9，打印提示信息
				System.out.println("NOW WE DRAW ONE FOOT");
				hangmanPicture[11][9] = "\\"; // 在hangmanPicture数组中放置一只脚的图形
				hangmanPicture[11][10] = "-"; // 在hangmanPicture数组中放置一只脚的图形
				break; // 结束该case
			case 10: // 如果猜错次数为10，打印提示信息
				System.out.println("HERE'S THE OTHER FOOT -- YOU'RE HUNG!!");
				hangmanPicture[11][2] = "-"; // 在hangmanPicture数组中放置另一只脚的图形
				hangmanPicture[11][3] = "/"; // 在hangmanPicture数组中放置另一只脚的图形
				break; // 结束该case
		}
		for (int i = 0; i <= 11; i++) { // 循环遍历hangmanPicture数组的每一行
			# 遍历二维数组 hangmanPicture 的每一行
			for (int j = 0; j <= 11; j++) {
				# 打印当前行的元素
				System.out.print(hangmanPicture[i][j]);
			}
			# 换行
			System.out.print("\n");
		}

	}

	# 打印已经猜测到的字母
	private static void printDiscoveredLetters(char[] D$) {
		# 将字符数组转换为字符串并打印
		System.out.println(new String(D$));
		# 打印空行
		System.out.println("\n");
	}

	# 打印已经使用过的字母
	private static void printLettersUsed(Set<Character> lettersUsed) {
		# 打印提示信息
		System.out.println("\nHERE ARE THE LETTERS YOU USED:");
		# 将使用过的字母集合转换为字符串并打印
		System.out.println(lettersUsed.stream()
				.map(Object::toString).collect(Collectors.joining(",")));
		# 打印空行
		System.out.println("\n");
	}
# 打印游戏介绍
def printIntro():
    # 打印游戏标题
    print("                                HANGMAN")
    # 打印游戏制作方信息
    print("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    # 打印空行
    print("\n\n\n")
```
这段代码是一个打印游戏介绍的函数，首先打印游戏标题，然后打印游戏制作方的信息，最后打印三行空行。
```