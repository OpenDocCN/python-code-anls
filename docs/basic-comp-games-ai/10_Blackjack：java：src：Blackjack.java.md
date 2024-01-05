# `10_Blackjack\java\src\Blackjack.java`

```
import java.io.InputStreamReader; // 导入 InputStreamReader 类，用于读取字节流并将其解码为字符流
import java.io.OutputStreamWriter; // 导入 OutputStreamWriter 类，用于将字符流编码为字节流
import java.io.Reader; // 导入 Reader 类，用于读取字符流
import java.io.Writer; // 导入 Writer 类，用于写入字符流
import java.util.Collections; // 导入 Collections 类，用于操作集合类的静态方法
 * 
 * Another "unnecessary" aspect of this codebase is good Javadoc. Again, this is
 * meant to be educational, but another often overlooked benefit is that most
 * IDEs will display Javadoc in "autocomplete" suggestions. This is remarkably
 * helpful when using a class as a quick reminder of what you coded earlier.
 * This is true even if no one ever publishes or reads the HTML output of the
 * javadoc.
 * 
 */
public class Blackjack {
	public static void main(String[] args) {
		// Intuitively it might seem like the main program logic should be right
		// here in 'main' and that we should just use System.in and System.out
		// directly whenever we need them.  However, notice that System.out and
		// System.in are just an OutputStream and InputStream respectively. By
		// allowing alternate streams to be specified to Game at runtime, we can
		// write non-interactive tests of the code. See UserIoTest as an
		// example.
		// Likewise, by allowing an alternative "shuffle" algorithm, test code
		// can provide a deterministic card ordering.
		// 使用 try-with-resources 语句创建一个 Reader 对象，用于从标准输入流中读取数据
		try (Reader in = new InputStreamReader(System.in)) {
			// 创建一个 Writer 对象，用于向标准输出流写入数据
			Writer out = new OutputStreamWriter(System.out);
			// 创建一个 UserIo 对象，用于处理输入输出
			UserIo userIo = new UserIo(in, out);
			// 创建一个 Deck 对象，传入一个 lambda 表达式，用于洗牌操作
			Deck deck = new Deck(cards -> {
				// 在洗牌前向用户输出信息
				userIo.println("RESHUFFLING");
			    // 使用 Collections 类的 shuffle 方法对牌组进行洗牌操作
			    Collections.shuffle(cards);
			    // 返回洗好的牌组
			    return cards;
			});
			// 创建一个 Game 对象，传入 Deck 对象和 UserIo 对象
			Game game = new Game(deck, userIo);
			// 运行游戏
			game.run();
		} catch (Exception e) {
			// 捕获异常，处理 CTRL+D / CTRL+Z 操作，输出异常信息
			System.out.println(e.getMessage());
			// 退出程序
			System.exit(1);
		}
	}
}
```