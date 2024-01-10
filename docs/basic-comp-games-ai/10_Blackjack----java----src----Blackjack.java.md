# `basic-computer-games\10_Blackjack\java\src\Blackjack.java`

```
# 导入所需的 Java 类
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Reader;
import java.io.Writer;
import java.util.Collections;

/**
 * 在终端上玩二十一点游戏。从代码来看，读者可能会得出这个实现是“过度设计”的结论。我们使用了许多为更大的代码库开发的技术和模式，以创建更易维护的代码，这对于一个简单的二十一点游戏可能不太相关。换句话说，规则和要求可能永远不会改变，因此使代码灵活性不那么有价值。
 * 
 * 尽管如此，这是一个示例，读者可以从中学习良好的 Java 编码技巧。此外，许多“过度设计”的策略既关乎可维护性，也关乎可测试性。想象一下，如果没有任何自动化特定场景的能力，就要手动测试像二十一点、保险或分牌这样不经常发生的情况，单元测试的价值立即显而易见。
 * 
 * 这个代码库的另一个“不必要”的方面是良好的 Javadoc。同样，这是为了教育，但通常被忽视的另一个好处是，大多数集成开发环境将在“自动完成”建议中显示 Javadoc。当将类用作快速提醒您之前编写的内容时，这非常有帮助。即使没有人发布或阅读 javadoc 的 HTML 输出，这也是真实的。
 * 
 */
public class Blackjack {
    public static void main(String[] args) {
        // 程序的入口，通过命令行参数传入参数
        // 直观上，似乎主程序逻辑应该在这里的 'main' 中，并且我们应该直接使用 System.in 和 System.out
        // 但是请注意，System.out 和 System.in 只是一个输出流和输入流。通过允许在运行时指定替代流给 Game，
        // 我们可以编写非交互式的代码测试。参见 UserIoTest 作为一个例子。
        // 同样，通过允许替代的“洗牌”算法，测试代码可以提供确定性的卡片排序。
        try (Reader in = new InputStreamReader(System.in)) {
            // 创建输出流写入器
            Writer out = new OutputStreamWriter(System.out);
            // 创建用户输入输出对象
            UserIo userIo = new UserIo(in, out);
            // 创建一副牌，并在洗牌时打印信息
            Deck deck = new Deck(cards -> {
                userIo.println("RESHUFFLING");
                Collections.shuffle(cards);
                return cards;
            });
            // 创建游戏对象
            Game game = new Game(deck, userIo);
            // 运行游戏
            game.run();
        } catch (Exception e) {
            // 这允许我们通过抛出异常来优雅地处理 CTRL+D / CTRL+Z
            System.out.println(e.getMessage());
            System.exit(1);
        }
    }
# 闭合前面的函数定义
```