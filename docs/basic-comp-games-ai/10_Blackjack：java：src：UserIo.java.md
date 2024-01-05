# `d:/src/tocomm/basic-computer-games\10_Blackjack\java\src\UserIo.java`

```
import java.io.BufferedReader; // 导入用于读取字符流的类
import java.io.EOFException; // 导入用于指示输入流已经结束的异常类
import java.io.IOException; // 导入用于处理输入输出异常的类
import java.io.PrintWriter; // 导入用于打印输出的类
import java.io.Reader; // 导入用于读取字符流的类
import java.io.UncheckedIOException; // 导入用于指示未检查的输入输出异常的类
import java.io.Writer; // 导入用于写入字符流的类
import java.util.stream.IntStream; // 导入用于处理整数流的类

/**
 * This class is responsible for printing output to the screen and reading input
 * from the user. It must be initialized with a reader to get input data from
 * and a writer to send output to. Typically these will wrap System.in and
 * System.out respectively, but can be a StringReader and StringWriter when
 * running in test code.
 */
public class UserIo {

    private BufferedReader in; // 用于从输入流中读取字符的缓冲输入流
    private PrintWriter out; // 用于将格式化对象写入字符输出流
    /**
     * Initializes the UserIo with the given reader/writer. The reader will be
     * wrapped in a BufferedReader and so should <i>not</i> be a BufferedReader
     * already (to avoid double buffering).
     * 
     * @param in Typically an InputStreamReader wrapping System.in or a StringReader
     * @param out Typically an OuputStreamWriter wrapping System.out or a StringWriter
     */
    public UserIo(Reader in, Writer out) {
        this.in = new BufferedReader(in); // 使用给定的Reader初始化一个BufferedReader对象，用于读取输入
        this.out = new PrintWriter(out, true); // 使用给定的Writer初始化一个PrintWriter对象，用于输出
    }

    /**
     * Print the line of text to output including a trailing linebreak.
     * 
     * @param text the text to print
     */
    public void println(String text) {
        // 将给定的文本输出到输出流，并在末尾添加换行符
		out.println(text);  # 打印给定的文本并换行

	/**
	 * Print the given text left padded with spaces.
	 * 
	 * @param text The text to print  # 要打印的文本
	 * @param leftPad The number of spaces to pad with.  # 要填充的空格数
	 */
	public void println(String text, int leftPad) {  # 打印给定的文本并在左侧填充指定数量的空格
		IntStream.range(0, leftPad).forEach((i) -> out.print(' '));  # 使用IntStream生成指定数量的空格并打印
		out.println(text);  # 打印给定的文本并换行
	}

	/**
	 * Print the given text <i>without</i> a trailing linebreak.
	 * 
	 * @param text The text to print.  # 要打印的文本
	 */
	public void print(String text) {  # 打印给定的文本
		out.print(text); // 将文本输出到输出流
		out.flush(); // 刷新输出流，确保所有缓冲的输出都被写出

	/**
	 * 从输入流中读取一行文本。
	 * 
	 * @return 输入流中读取的一行文本。
	 * @throws UncheckedIOException 如果读取的行为null（按下了CTRL+D或CTRL+Z）
	 */
	private String readLine() {
		try {
			String line = in.readLine(); // 从输入流中读取一行文本
			if(line == null) { // 如果读取的行为null
				throw new UncheckedIOException("!END OF INPUT", new EOFException()); // 抛出未经检查的IO异常
			}
			return line; // 返回读取的行
		} catch (IOException e) {
			throw new UncheckedIOException(e); // 抛出未经检查的IO异常
		}
	}

	/**
	 * Prompt the user via input.
	 * 
	 * @param prompt The text to display as a prompt. A question mark and space will be added to the end, so if prompt = "EXAMPLE" then the user will see "EXAMPLE? ".
	 * @return The line read from input.
	 */
	public String prompt(String prompt) {
		print(prompt + "? ");  // 显示提示信息并加上问号和空格
		return readLine();  // 从输入中读取一行并返回
	}

	/**
	 * Prompts the user for a "Yes" or "No" answer.
	 * @param prompt The prompt to display to the user on STDOUT.
	 * @return false if the user enters a value beginning with "N" or "n"; true otherwise.
	 */
	public boolean promptBoolean(String prompt) {
		print(prompt + "? ");  // 显示提示信息
		# 从用户输入中读取一行
		String input = readLine();

		# 如果输入的字符串转换为小写后以 "n" 开头，则返回 False
		if(input.toLowerCase().startsWith("n")) {
			return false;
		# 否则返回 True
		} else {
			return true;
		}
	}

	/**
	 * 提示用户输入一个整数。与 Vintage Basic 中一样，"可选的提示字符串后面跟着一个问号和一个空格。" 如果输入的内容不是数字，"将生成错误并提示用户重新输入。"
	 *
	 * @param prompt 要显示给用户的提示信息。
	 * @return 用户输入的数字。
	 */
	public int promptInt(String prompt) {
		# 打印提示信息
		print(prompt + "? ");

		# 进入循环，等待用户输入
		while(true) {
			# 读取用户输入
			String input = readLine();
			# 尝试将输入转换为整数
			try {
				return Integer.parseInt(input);
			} catch(NumberFormatException e) {
				# 如果输入不是数字，捕获异常并提示用户重新输入
				println("!NUMBER EXPECTED - RETRY INPUT LINE");
				print("? ");
				continue;
			}
		}
	}

	/**
	 * 提示用户输入一个双精度浮点数。与Vintage Basic一样，“可选的提示字符串后面跟着一个问号和一个空格。”如果输入不是数字，“将生成错误并提示用户重新输入。”
# 定义一个方法，用于提示用户输入一个 double 类型的数值
public double promptDouble(String prompt) {
    # 打印提示信息
    print(prompt + "? ");

    # 循环，直到用户输入一个有效的 double 类型的数值
    while(true) {
        # 读取用户输入的字符串
        String input = readLine();
        try {
            # 尝试将用户输入的字符串转换为 double 类型的数值
            return Double.parseDouble(input);
        } catch(NumberFormatException e) {
            # 捕获异常，说明用户输入的不是一个有效的数值
            # 打印错误信息
            println("!NUMBER EXPECTED - RETRY INPUT LINE");
            # 重新提示用户输入
            print("? ");
            # 继续循环，等待用户输入
            continue;
        }
    }
}
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 使用 open 函数读取文件内容，BytesIO 将其封装成字节流
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用 BytesIO 封装的字节流创建 ZIP 对象
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象的文件名列表，读取文件数据，组成字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
```