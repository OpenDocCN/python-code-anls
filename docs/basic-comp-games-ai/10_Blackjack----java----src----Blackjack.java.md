# `basic-computer-games\10_Blackjack\java\src\Blackjack.java`

```

// 导入所需的类
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Reader;
import java.io.Writer;
import java.util.Collections;

/**
 * 在终端上玩21点游戏。从代码来看，读者可能会得出这个实现是“过度设计”的结论。我们使用了许多为更大的代码库开发的技术和模式来创建更易维护的代码，这对于一个简单的21点游戏可能并不那么重要。事实上，规则和要求可能永远不会改变，因此使代码灵活性并不那么有价值。
 * 
 * 尽管如此，这是一个示例，读者可以从中学习良好的Java编码技巧。此外，许多“过度设计”的策略既关乎可维护性，也关乎可测试性。想象一下，如果没有任何自动化测试特定场景的能力，试图手动测试21点、保险或分牌等不经常发生的情况时，单元测试的价值立即显现出来。
 * 
 * 这个代码库的另一个“不必要”的方面是良好的Javadoc。同样，这是为了教育目的，但通常被忽视的另一个好处是，大多数IDE将在“自动完成”建议中显示Javadoc。当使用类作为对你之前编码的快速提醒时，这是非常有帮助的。
 * 即使没有人发布或阅读Javadoc的HTML输出，这也是真实的。
 * 
 */
public class Blackjack {
    public static void main(String[] args) {
        // 直观上，似乎主程序逻辑应该在这里的 'main' 中，每当需要时直接使用 System.in 和 System.out。然而，请注意，System.out 和 System.in 只是分别是一个 OutputStream 和 InputStream。通过允许在运行时指定替代流给游戏，我们可以编写代码的非交互式测试。参见 UserIoTest 作为一个示例。
        // 同样，通过允许替代的“洗牌”算法，测试代码可以提供确定性的牌序。
        try (Reader in = new InputStreamReader(System.in)) {
            Writer out = new OutputStreamWriter(System.out);
            UserIo userIo = new UserIo(in, out);
            Deck deck = new Deck(cards -> {
                userIo.println("RESHUFFLING");
                Collections.shuffle(cards);
                return cards;
            });
            Game game = new Game(deck, userIo);
            game.run();
        } catch (Exception e) {
            // 这使我们能够优雅地通过抛出异常来处理CTRL+D / CTRL+Z。
            System.out.println(e.getMessage());
            System.exit(1);
        }
    }
}

```