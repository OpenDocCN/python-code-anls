# `basic-computer-games\10_Blackjack\java\src\UserIo.java`

```py
import java.io.BufferedReader;
import java.io.EOFException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Reader;
import java.io.UncheckedIOException;
import java.io.Writer;
import java.util.stream.IntStream;

/**
 * This class is responsible for printing output to the screen and reading input
 * from the user. It must be initialized with a reader to get input data from
 * and a writer to send output to. Typically these will wrap System.in and
 * System.out respectively, but can be a StringReader and StringWriter when
 * running in test code.
 */
public class UserIo {

    private BufferedReader in;  // 用于从用户获取输入数据的缓冲读取器
    private PrintWriter out;  // 用于发送输出到屏幕的打印写入器

    /**
     * Initializes the UserIo with the given reader/writer. The reader will be
     * wrapped in a BufferedReader and so should <i>not</i> be a BufferedReader
     * already (to avoid double buffering).
     * 
     * @param in Typically an InputStreamReader wrapping System.in or a StringReader
     * @param out Typically an OuputStreamWriter wrapping System.out or a StringWriter
     */
    public UserIo(Reader in, Writer out) {
        this.in = new BufferedReader(in);  // 使用给定的读取器初始化缓冲读取器
        this.out = new PrintWriter(out, true);  // 使用给定的写入器初始化打印写入器，设置自动刷新
    }

    /**
     * Print the line of text to output including a trailing linebreak.
     * 
     * @param text the text to print
     */
    public void println(String text) {
        out.println(text);  // 打印带有换行符的文本到输出
    }

    /**
     * Print the given text left padded with spaces.
     * 
     * @param text The text to print
     * @param leftPad The number of spaces to pad with.
     */
    public void println(String text, int leftPad) {
        IntStream.range(0, leftPad).forEach((i) -> out.print(' '));  // 使用流生成指定数量的空格并打印
        out.println(text);  // 打印带有换行符的文本到输出
    }

    /**
     * Print the given text <i>without</i> a trailing linebreak.
     * 
     * @param text The text to print.
     */
    public void print(String text) {
        out.print(text);  // 打印文本到输出
        out.flush();  // 刷新输出缓冲区
    }
}
    /**
     * 从输入中读取一行文本。
     * 
     * @return 输入的行。
     * @throws UncheckedIOException 如果行为null（按下了CTRL+D或CTRL+Z）
     */
    private String readLine() {
        try {
            String line = in.readLine();  // 从输入中读取一行文本
            if(line == null) {  // 如果行为null
                throw new UncheckedIOException("!END OF INPUT", new EOFException());  // 抛出未经检查的IO异常
            }
            return line;  // 返回行
        } catch (IOException e) {
            throw new UncheckedIOException(e);  // 抛出未经检查的IO异常
        }
    }

    /**
     * 通过输入提示用户。
     * 
     * @param prompt 要显示为提示的文本。问号和空格将被添加到末尾，所以如果prompt = "EXAMPLE"，用户将看到"EXAMPLE? "
     * @return 从输入中读取的行。
     */
    public String prompt(String prompt) {
        print(prompt + "? ");  // 打印提示
        return readLine();  // 返回从输入中读取的行
    }

    /**
     * 提示用户选择“是”或“否”。
     * @param prompt 要在标准输出上显示给用户的提示。
     * @return 如果用户输入以“N”或“n”开头的值，则返回false；否则返回true。
     */
    public boolean promptBoolean(String prompt) {
        print(prompt + "? ");  // 打印提示

        String input = readLine();  // 从输入中读取行

        if(input.toLowerCase().startsWith("n")) {  // 如果输入以“n”开头
            return false;  // 返回false
        } else {
            return true;  // 返回true
        }
    }

    /**
     * 提示用户输入整数。与Vintage Basic一样，“可选提示字符串后面跟着一个问号和一个空格。”，如果输入非数字，“将生成错误并重新提示用户。”
     *
     * @param prompt 要显示给用户的提示。
     * @return 用户给出的数字。
     */
    // 提示用户输入一个整数，并返回输入的整数值
    public int promptInt(String prompt) {
        // 打印提示信息
        print(prompt + "? ");

        // 循环直到用户输入正确的整数
        while(true) {
            // 读取用户输入
            String input = readLine();
            try {
                // 尝试将输入转换为整数并返回
                return Integer.parseInt(input);
            } catch(NumberFormatException e) {
                // 输入不是数字
                println("!NUMBER EXPECTED - RETRY INPUT LINE");
                // 重新提示用户输入
                print("? ");
                continue;
            }
        }
    }

    /**
     * 提示用户输入一个浮点数。与 Vintage Basic 中的行为类似，"可选的提示字符串后面跟着一个问号和一个空格"，如果输入不是数字，"将生成错误并要求用户重新输入。"
     *
     * @param prompt 要显示给用户的提示信息
     * @return 用户输入的数字
     */
    public double promptDouble(String prompt) {
        // 打印提示信息
        print(prompt + "? ");

        // 循环直到用户输入正确的浮点数
        while(true) {
            // 读取用户输入
            String input = readLine();
            try {
                // 尝试将输入转换为浮点数并返回
                return Double.parseDouble(input);
            } catch(NumberFormatException e) {
                // 输入不是数字
                println("!NUMBER EXPECTED - RETRY INPUT LINE");
                // 重新提示用户输入
                print("? ");
                continue;
            }
        }
    }
# 闭合前面的函数定义
```