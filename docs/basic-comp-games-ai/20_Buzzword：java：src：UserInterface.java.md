# `d:/src/tocomm/basic-computer-games\20_Buzzword\java\src\UserInterface.java`

```
import java.io.PrintStream;  // 导入打印流类，用于输出信息
import java.util.Scanner;  // 导入扫描器类，用于接收用户输入
import java.util.function.Supplier;  // 导入供应商函数接口，用于提供一个值或者生成一个值
/**
 * A command line user interface that outputs a buzzword every
 * time the user requests a new one.
 */
public class UserInterface implements Runnable {  // 创建一个用户界面类，实现可运行接口

	/**
	 * Input from the user.
	 */
	private final Scanner input;  // 用户输入的扫描器对象

	/**
	 * Output to the user.
	 */
	private final PrintStream output;  // 用户输出的打印流对象
	/**
	 * The buzzword generator.
	 * 生成流行语的供应商
	 */
	private final Supplier<String> buzzwords;

	/**
	 * Create a new user interface.
	 * 创建一个新的用户界面
	 *
	 * @param input The input scanner with which the user gives commands.
	 *              用户输入命令的输入扫描器
	 * @param output The output to show messages to the user.
	 *               用于向用户显示消息的输出
	 * @param buzzwords The buzzword supplier.
	 *                  流行语供应商
	 */
	public UserInterface(final Scanner input,
			final PrintStream output,
			final Supplier<String> buzzwords) {
		this.input = input;
		this.output = output;
		this.buzzwords = buzzwords;
	}
	@Override
	// 重写 run 方法
	public void run() {
		// 输出标题
		output.println("              BUZZWORD GENERATOR");
		// 输出创意计算的地点
		output.println("   CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
		output.println();
		output.println();
		output.println();
		// 输出程序介绍
		output.println("THIS PROGRAM PRINTS HIGHLY ACCEPTABLE PHRASES IN");
		output.println("'EDUCATOR-SPEAK' THAT YOU CAN WORK INTO REPORTS");
		output.println("AND SPEECHES.  WHENEVER A QUESTION MARK IS PRINTED,");
		output.println("TYPE A 'Y' FOR ANOTHER PHRASE OR 'N' TO QUIT.");
		output.println();
		output.println();
		output.println("HERE'S THE FIRST PHRASE:");

		// 循环生成短语
		do {
			// 输出短语
			output.println(buzzwords.get());
			output.println();
			// 提示用户输入
			output.print("?");
		} while ("Y".equals(input.nextLine().toUpperCase()));  // 当用户输入为Y时继续循环
	}
# 输出提示信息到控制台
output.println("COME BACK WHEN YOU NEED HELP WITH ANOTHER REPORT!")
# 结束程序
```
这段代码是在Java中，用于输出一条提示信息到控制台，并结束程序。
```