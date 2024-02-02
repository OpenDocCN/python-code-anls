# `basic-computer-games\20_Buzzword\java\src\UserInterface.java`

```py
import java.io.PrintStream;  // 导入打印流类
import java.util.Scanner;  // 导入扫描器类
import java.util.function.Supplier;  // 导入供应商函数接口

/**
 * 一个命令行用户界面，每次用户请求时输出一个流行词。
 */
public class UserInterface implements Runnable {

    /**
     * 用户输入。
     */
    private final Scanner input;

    /**
     * 用户输出。
     */
    private final PrintStream output;

    /**
     * 流行词生成器。
     */
    private final Supplier<String> buzzwords;

    /**
     * 创建一个新的用户界面。
     *
     * @param input 用户使用的输入扫描器。
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
        output.println("              BUZZWORD GENERATOR");  // 打印流行词生成器标题
        output.println("   CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 打印创意计算和地点
        output.println();  // 打印空行
        output.println();  // 打印空行
        output.println();  // 打印空行
        output.println("THIS PROGRAM PRINTS HIGHLY ACCEPTABLE PHRASES IN");  // 打印程序功能介绍
        output.println("'EDUCATOR-SPEAK' THAT YOU CAN WORK INTO REPORTS");  // 打印程序功能介绍
        output.println("AND SPEECHES.  WHENEVER A QUESTION MARK IS PRINTED,");  // 打印程序功能介绍
        output.println("TYPE A 'Y' FOR ANOTHER PHRASE OR 'N' TO QUIT.");  // 打印程序功能介绍
        output.println();  // 打印空行
        output.println();  // 打印空行
        output.println("HERE'S THE FIRST PHRASE:");  // 打印提示语句

        do {
            output.println(buzzwords.get());  // 打印流行词
            output.println();  // 打印空行
            output.print("?");  // 打印问号
        } while ("Y".equals(input.nextLine().toUpperCase()));  // 当用户输入为Y时继续循环

        output.println("COME BACK WHEN YOU NEED HELP WITH ANOTHER REPORT!");  // 打印结束语
    }
}
```