# `basic-computer-games\20_Buzzword\java\src\UserInterface.java`

```

// 导入所需的类
import java.io.PrintStream;
import java.util.Scanner;
import java.util.function.Supplier;

/**
 * 一个命令行用户界面，每次用户请求时输出一个流行词。
 */
public class UserInterface implements Runnable {

	/**
	 * 用户输入
	 */
	private final Scanner input;

	/**
	 * 用户输出
	 */
	private final PrintStream output;

	/**
	 * 流行词生成器
	 */
	private final Supplier<String> buzzwords;

	/**
	 * 创建一个新的用户界面。
	 *
	 * @param input 用户输入扫描器，用户用它给出命令。
	 * @param output 用于向用户显示消息的输出。
	 * @param buzzwords 流行词供应商。
	 */
	public UserInterface(final Scanner input,
			final PrintStream output,
			final Supplier<String> buzzwords) {
		this.input = input;
		this.output = output;
		this.buzzwords = buzzwords;
	}

	@Override
	public void run() {
		output.println("              BUZZWORD GENERATOR");
		output.println("   CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
		output.println();
		output.println();
		output.println();
		output.println("THIS PROGRAM PRINTS HIGHLY ACCEPTABLE PHRASES IN");
		output.println("'EDUCATOR-SPEAK' THAT YOU CAN WORK INTO REPORTS");
		output.println("AND SPEECHES.  WHENEVER A QUESTION MARK IS PRINTED,");
		output.println("TYPE A 'Y' FOR ANOTHER PHRASE OR 'N' TO QUIT.");
		output.println();
		output.println();
		output.println("HERE'S THE FIRST PHRASE:");

		// 循环直到用户输入 'N'
		do {
			output.println(buzzwords.get()); // 输出一个流行词
			output.println();
			output.print("?"); // 提示用户输入
		} while ("Y".equals(input.nextLine().toUpperCase())); // 当用户输入不是 'Y' 时结束循环

		output.println("COME BACK WHEN YOU NEED HELP WITH ANOTHER REPORT!");
	}
}

```