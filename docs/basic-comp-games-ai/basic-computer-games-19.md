# BasicComputerGames源码解析 19

# `10_Blackjack/java/test/ScoringUtilsTest.java`

This is a Java class that simulates a game of cards. The class includes methods for comparing the hands of two players.

The `compareHands` method takes two lists of cards and compares them. It returns `-1` if the first hand wins, `0` if the game is a tie, and `1` if the second hand wins.

The `compareHandsABressed` method compares the hands of two players. It returns `-1` if the first hand wins, `0` if the game is a tie, and `-1` if the second hand wins.

The `compareHandsBBusted` method compares the hands of two players. It returns `1` if the first hand wins, `0` if the game is a tie, and `-1` if the second hand wins.

All of these methods take a single parameter, which is an instance of the `Card` class. This class represents a single card in a hand. It has a `getSuit` method for getting the suit of the card, and a `getFace` method for getting the face value of the card.

The `ScoringUtils.compareHands` method is a utility method that compares the hands of two players. It takes two instances of the `Card` class and returns `-1` if the first hand wins, `0` if the game is a tie, and `1` if the second hand wins.


```
import org.junit.jupiter.api.Test;

import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.LinkedList;

public class ScoringUtilsTest {

    @Test
    @DisplayName("scoreHand should score aces as 1 when using 11 would bust")
    public void scoreHandHardAce() {
        // Given
        LinkedList<Card> hand = new LinkedList<>();
        hand.add(new Card(10, Card.Suit.SPADES));
        hand.add(new Card(9, Card.Suit.SPADES));
        hand.add(new Card(1, Card.Suit.SPADES));

        // When
        int result = ScoringUtils.scoreHand(hand);

        // Then
        assertEquals(20, result);
    }

    @Test
    @DisplayName("scoreHand should score 3 aces as 13")
    public void scoreHandMultipleAces() {
        // Given
        LinkedList<Card> hand = new LinkedList<>();
        hand.add(new Card(1, Card.Suit.SPADES));
        hand.add(new Card(1, Card.Suit.CLUBS));
        hand.add(new Card(1, Card.Suit.HEARTS));

        // When
        int result = ScoringUtils.scoreHand(hand);

        // Then
        assertEquals(13, result);
    }

    @Test
    @DisplayName("compareHands should return 1 meaning A beat B, 20 to 12")
    public void compareHandsAWins() {
        LinkedList<Card> handA = new LinkedList<>();
        handA.add(new Card(10, Card.Suit.SPADES));
        handA.add(new Card(10, Card.Suit.CLUBS));

        LinkedList<Card> handB = new LinkedList<>();
        handB.add(new Card(1, Card.Suit.SPADES));
        handB.add(new Card(1, Card.Suit.CLUBS));

        int result = ScoringUtils.compareHands(handA,handB);

        assertEquals(1, result);
    }

    @Test
    @DisplayName("compareHands should return -1 meaning B beat A, 18 to 4")
    public void compareHandsBwins() {
        LinkedList<Card> handA = new LinkedList<>();
        handA.add(new Card(2, Card.Suit.SPADES));
        handA.add(new Card(2, Card.Suit.CLUBS));

        LinkedList<Card> handB = new LinkedList<>();
        handB.add(new Card(5, Card.Suit.SPADES));
        handB.add(new Card(6, Card.Suit.HEARTS));
        handB.add(new Card(7, Card.Suit.CLUBS));

        int result = ScoringUtils.compareHands(handA,handB);

        assertEquals(-1, result);
    }

    @Test
    @DisplayName("compareHands should return 1 meaning A beat B, natural Blackjack to Blackjack")
    public void compareHandsAWinsWithNaturalBlackJack() {
        //Hand A wins with natural BlackJack, B with Blackjack
        LinkedList<Card> handA = new LinkedList<>();
        handA.add(new Card(10, Card.Suit.SPADES));
        handA.add(new Card(1, Card.Suit.CLUBS));

        LinkedList<Card> handB = new LinkedList<>();
        handB.add(new Card(6, Card.Suit.SPADES));
        handB.add(new Card(7, Card.Suit.HEARTS));
        handB.add(new Card(8, Card.Suit.CLUBS));

        int result = ScoringUtils.compareHands(handA,handB);

        assertEquals(1, result);
    }

    @Test
    @DisplayName("compareHands should return -1 meaning B beat A, natural Blackjack to Blackjack")
    public void compareHandsBWinsWithNaturalBlackJack() {
        LinkedList<Card> handA = new LinkedList<>();
        handA.add(new Card(6, Card.Suit.SPADES));
        handA.add(new Card(7, Card.Suit.HEARTS));
        handA.add(new Card(8, Card.Suit.CLUBS));
        
        LinkedList<Card> handB = new LinkedList<>();
        handB.add(new Card(10, Card.Suit.SPADES));
        handB.add(new Card(1, Card.Suit.CLUBS));

        int result = ScoringUtils.compareHands(handA,handB);

        assertEquals(-1, result);
    }

    @Test
    @DisplayName("compareHands should return 0, hand A and B tied with a Blackjack")
    public void compareHandsTieBothBlackJack() {
        LinkedList<Card> handA = new LinkedList<>();
        handA.add(new Card(11, Card.Suit.SPADES));
        handA.add(new Card(10, Card.Suit.CLUBS));
        
        LinkedList<Card> handB = new LinkedList<>();
        handB.add(new Card(10, Card.Suit.SPADES));
        handB.add(new Card(11, Card.Suit.CLUBS));

        int result = ScoringUtils.compareHands(handA,handB);

        assertEquals(0, result);
    }

    @Test
    @DisplayName("compareHands should return 0, hand A and B tie without a Blackjack")
    public void compareHandsTieNoBlackJack() {
        LinkedList<Card> handA = new LinkedList<>();
        handA.add(new Card(10, Card.Suit.DIAMONDS));
        handA.add(new Card(10, Card.Suit.HEARTS));
        
        LinkedList<Card> handB = new LinkedList<>();
        handB.add(new Card(10, Card.Suit.SPADES));
        handB.add(new Card(10, Card.Suit.CLUBS));

        int result = ScoringUtils.compareHands(handA,handB);

        assertEquals(0, result);
    }

    @Test
    @DisplayName("compareHands should return 0, hand A and B tie when both bust")
    public void compareHandsTieBust() {
        LinkedList<Card> handA = new LinkedList<>();
        handA.add(new Card(10, Card.Suit.DIAMONDS));
        handA.add(new Card(10, Card.Suit.HEARTS));
        handA.add(new Card(3, Card.Suit.HEARTS));
        
        LinkedList<Card> handB = new LinkedList<>();
        handB.add(new Card(10, Card.Suit.SPADES));
        handB.add(new Card(11, Card.Suit.SPADES));
        handB.add(new Card(4, Card.Suit.SPADES));

        int result = ScoringUtils.compareHands(handA,handB);

        assertEquals(0, result);
    }
    @Test
    @DisplayName("compareHands should return -1, meaning B beat A, A busted")
    public void compareHandsABusted() {
        LinkedList<Card> handA = new LinkedList<>();
        handA.add(new Card(10, Card.Suit.DIAMONDS));
        handA.add(new Card(10, Card.Suit.HEARTS));
        handA.add(new Card(3, Card.Suit.HEARTS));
        
        LinkedList<Card> handB = new LinkedList<>();
        handB.add(new Card(10, Card.Suit.SPADES));
        handB.add(new Card(10, Card.Suit.SPADES));

        int result = ScoringUtils.compareHands(handA,handB);

        assertEquals(-1, result);
    }

    @Test
    @DisplayName("compareHands should return 1, meaning A beat B, B busted")
    public void compareHandsBBusted() {
        LinkedList<Card> handA = new LinkedList<>();
        handA.add(new Card(10, Card.Suit.DIAMONDS));
        handA.add(new Card(3, Card.Suit.HEARTS));
        
        LinkedList<Card> handB = new LinkedList<>();
        handB.add(new Card(10, Card.Suit.SPADES));
        handB.add(new Card(10, Card.Suit.SPADES));
        handB.add(new Card(5, Card.Suit.SPADES));

        int result = ScoringUtils.compareHands(handA,handB);

        assertEquals(1, result);
    }
}

```

# `10_Blackjack/java/test/UserIoTest.java`

This is a test case that uses the `UserIo` class to interact with the `System.in` and `System.out` streams. It tests two methods of the `UserIo` class, `promptBoolean` and `promptInt`, in the following order:

1. The `testPromptBooleanAcceptsYes` method tests whether the `promptBoolean` method can correctly interpret a "yes" or "Y" response from the user. It achieves this by creating a new `Reader` object with the input response, and then calling the `promptBoolean` method on the `UserIo` object, passing in the argument "TEST". It then checks the output of the method, making sure it matches the expected "yes", and asserts that the method returns the expected result.
2. The `testPromptIntAcceptsNumbers` method tests whether the `promptInt` method can correctly interpret a numeric or non-numeric response from the user. It achieves this by creating a new `Reader` object with the input response, and then calling the `promptInt` method on the `UserIo` object, passing in the argument "TEST". It then checks the output of the method, making sure it matches the expected result, and asserts that the method returns the expected value. If the input is non-numeric, it will print an error and then repeat the input, repeating the process until it returns a numeric value.


```
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.Reader;
import java.io.StringReader;
import java.io.StringWriter;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

public class UserIoTest {

    @ParameterizedTest(name = "''{0}'' is accepted as ''no''")
    @ValueSource(strings = {"N", "n", "No", "NO", "no"})
    public void testPromptBooleanAcceptsNo(String response) {
        // Given
        Reader in = new StringReader(response + "\n");
        StringWriter out = new StringWriter();
        UserIo userIo = new UserIo(in, out);

        // When
        boolean result = userIo.promptBoolean("TEST");

        // Then
        assertEquals("TEST? ", out.toString());
        assertFalse(result);
    }

    @ParameterizedTest(name = "''{0}'' is accepted as ''yes''")
    @ValueSource(strings = {"Y", "y", "Yes", "YES", "yes", "", "foobar"})
    public void testPromptBooleanAcceptsYes(String response) {
        // Given
        Reader in = new StringReader(response + "\n");
        StringWriter out = new StringWriter();
        UserIo userIo = new UserIo(in, out);

        // When
        boolean result = userIo.promptBoolean("TEST");

        // Then
        assertEquals("TEST? ", out.toString());
        assertTrue(result);
    }

    @ParameterizedTest(name = "''{0}'' is accepted as number")
    @CsvSource({
        "1,1",
        "0,0",
        "-1,-1",
    })
    public void testPromptIntAcceptsNumbers(String response, int expected) {
        // Given
        Reader in = new StringReader(response + "\n");
        StringWriter out = new StringWriter();
        UserIo userIo = new UserIo(in, out);

        // When
        int result = userIo.promptInt("TEST");

        // Then
        assertEquals("TEST? ", out.toString());
        assertEquals(expected, result);
    }

    @Test
    @DisplayName("promptInt should print an error and reprompt if given a non-numeric response")
    public void testPromptIntRepromptsOnNonNumeric() {
        // Given
        Reader in = new StringReader("foo" + System.lineSeparator() +"1"); // word, then number
        StringWriter out = new StringWriter();
        UserIo userIo = new UserIo(in, out);

        // When
        int result = userIo.promptInt("TEST");

        // Then
        assertEquals("TEST? !NUMBER EXPECTED - RETRY INPUT LINE" + System.lineSeparator() +"? ", out.toString());
        assertEquals(1, result);
    }
}

```

# `10_Blackjack/javascript/blackjack.js`

这段代码定义了两个函数，分别是`print()`和`input()`。

`print()`函数的作用是打印字符串`str`到页面上，并将结果添加到页面上一个由`document.getElementById("output")`产生的`appendChild()`添加到页面上。

`input()`函数的作用是从用户接收输入框`input_element`中获取输入值，并在获取到输入值后将其存储在变量`input_str`中。然后使用户可以通过点击输入框上的`enter`键来触发函数内部的事件处理程序，该程序会读取用户输入的值，将其存储在`input_str`中，并将其打印到页面上，然后清除当前输入的内容。


```
// BLACKJACK
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       input_element = document.createElement("INPUT");

                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      input_str = input_element.value;
                                                      document.getElementById("output").removeChild(input_element);
                                                      print(input_str);
                                                      print("\n");
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

```

这段代码定义了一个名为 `tab` 的函数，它会接收一个参数 `space`，并在这个参数上产生一个空格字符串。函数的主要作用是输出一个空格字符串，它会在这个空格字符串中填充一些字符，然后返回这个空格字符串。

接下来，代码创建了一个名为 `da` 的数组，一个名为 `pa` 的数组，一个名为 `qa` 的数组，一个名为 `ca` 的数组，一个名为 `ta` 的数组和一个名为 `sa` 的数组。这些数组的具体内容没有在函数中进行使用，所以它们只是被创建了出来。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var da = [];

var pa = [];
var qa = [];
var ca = [];
var ta = [];
var sa = [];
```

这段代码定义了一个名为"af"的函数，接受一个参数"q"，并返回一个整数。函数的作用是通过以下操作来调整给定的参数"q"：

1. 如果"q"的值大于或等于22，则将"q"减去11，否则保持不变。

2. 对"q"进行下取整操作，将小数部分舍去，得到一个整数。

3. 如果"q"的整数部分等于22，则将"q"减去23，否则保持不变。

4. 返回对"q"进行调整后的整数部分。

"af"函数的具体实现可以看作是对给定参数"q"进行数学运算，以实现对参数"q"的平滑处理。


```
var ba = [];
var za = [];
var ra = [];

var ds = "N A  2  3  4  5  6  7N 8  9 10  J  Q  K";
var is = "H,S,D,/,"

var q;
var aa;
var ab;
var ac;
var h;
var h1;

function af(q) {
    return q >= 22 ? q - 11 : q;
}

```

这两函数 together 可以用来摇匀一个数组，将数组中的元素随机交换到数组的另一端，从而达到一组数据的重新分布。

第一个函数 `reshuffle()` ，它使用了一个 `for` 循环来遍历数组 `ca`，并且在循环变量 `d` 逐渐减少的过程中，将数组中的元素 `ca` 进行交换，使得循环变量的值在不断的变化过程中被重置。

在 `reshuffle()` 的循环体内，还使用了一个 `for` 循环来遍历数组 `c`，并且使用 `Math.random()` 函数来生成一个 0 到 1 之间的随机数，然后将其加到数组 `c` 中，使得数组 `c` 中的元素更加均匀地分布。

第二个函数 `get_card()` 是一个辅助函数，用于从数组 `ca` 中获取一个随机元素，由于这个函数需要使用 `Math.random()` 函数来生成随机数，因此它只能在运行时调用，不能被函数式地返回。

这两个函数可以组合使用，reshuffle() 函数用来将数组 `ca` 中的元素随机交换，get_card() 函数用来从数组 `ca` 中获取一个随机元素，两个函数可以一起被调用，从而实现一组数据的重新分布。


```
function reshuffle()
{
    print("RESHUFFLING\n");
    for (; d >= 1; d--)
        ca[--c] = da[d];
    for (c1 = 52; c1 >= c; c1--) {
        c2 = Math.floor(Math.random() * (c1 - c + 1)) + c;
        c3 = ca[c2];
        ca[c2] = ca[c1];
        ca[c1] = c3;
    }
}

// Subroutine to get a card.
function get_card()
{
    if (c >= 51)
        reshuffle();
    return ca[c++];
}

```

这段代码定义了两个函数card_print和alt_card_print，以及一个名为add_card的子函数。它们都接受一个参数x，并在控制台上打印出带有"which"的卡片。

card_print函数采用一个子字符串ds.substr(3 * x - 3, 3)来在控制台上打印出x的3个数字。然后加上"  "字符，以及在x的位置上打印出两个空格。

alt_card_print函数在card_print的基础上，将两个空格改为一个带有" "字符的整数。

add_card函数接受一个名为which的整数，并将其加到变量q上。如果q的值小于11，则将此整数加到q上，并将整个过程返回。如果q的值大于或等于11，则将11加到q上，并在添加后返回。如果q的值介于1和11之间，则将11加到q上，并在添加后返回。如果q的值大于21且大于或等于21，则将21加到q上，并在添加后返回。否则，将21加到q上，并在添加后返回。如果q的值大于等于33，则执行-1操作。

总的来说，这段代码定义了两个函数，用于在控制台上打印带有"which"的卡片。在主函数中，通过调用add_card函数来添加实际的卡片。


```
// Card printing subroutine
function card_print(x)
{
    print(ds.substr(3 * x - 3, 3) + "  ");
}

// Alternate card printing subroutine
function alt_card_print(x)
{
    print(" " + ds.substr(3 * x - 2, 2) + "   ");
}

// Subroutine to add card 'which' to total 'q'
function add_card(which)
{
    x1 = which;
    if (x1 > 10)
        x1 = 10;
    q1 = q + x1;
    if (q < 11) {
        if (which <= 1) {
            q += 11;
            return;
        }
        if (q1 >= 11)
            q = q1 + 11;
        else
            q = q1;
        return;
    }
    if (q <= 21 && q1 > 21)
        q = q1 + 1;
    else
        q = q1;
    if (q >= 33)
        q = -1;
}

```

这段代码是一个Subroutine，它的作用是评估一副扑牌中的手牌。

函数参数为hand，也就是一副扑牌中的牌的编号。函数中使用了两个数组qa和ra，它们都用于存储评估结果。

函数首先将hand的值存储在q中，然后使用for循环，从hand的第二个牌开始，逐步向前遍历。在遍历过程中，对于当前牌，将其添加到pa数组中。

最后，函数将qa数组中存储的值存储在hand变量中，即评估结果。

当函数完成执行后，pa数组中存储的值将反映hand所代表的牌的难度级别。


```
// Subroutine to evaluate hand 'which'. Total is put into
// qa[which]. Totals have the following meaning:
//  2-10...hard 2-10
// 11-21...soft 11-21
// 22-32...hard 11-21
//  33+....busted
function evaluate_hand(which)
{
    q = 0;
    for (q2 = 1; q2 <= ra[which]; q2++) {
        add_card(pa[i][q2]);
    }
    qa[which] = q;
}

```

这段代码定义了两个子函数，add_card_to_row()和discard_row()。add_card_to_row()的作用是将一张卡片添加到指定的行中，即将pa数组中的一项与i行中的一个元素x相加，并更新ra数组和pa数组。然后将新的qa数组中的元素与x连接起来，并将添加的卡片添加到队列中。如果队列中的元素数量已经达到了最大值，那么程序会打印一个错误消息，并将所有card从该行中删除。

discard_row()的作用是在指定的行中处理行中的元素。它通过一个while循环来遍历该行中的所有元素，并将pa数组中与元素d相加的行号存储在变量d中。然后，它将变量d的值存储在pa数组中的d行，并删除ra数组中与元素d相加的元素。


```
// Subroutine to add a card to row i
function add_card_to_row(i, x) {
    ra[i]++;
    pa[i][ra[i]] = x;
    q = qa[i];
    add_card(x);
    qa[i] = q;
    if (q < 0) {
        print("...BUSTED\n");
        discard_row(i);
    }
}

// Subroutine to discard row i
function discard_row(i) {
    while (ra[i]) {
        d++;
        da[d] = pa[i][ra[i]];
        ra[i]--;
    }
}

```

这段代码定义了三个函数，分别是 print_total、total_aa 和 total_ab。print_total 的作用是打印 Hand i 的总数。total_aa 和 total_ab 都是 if 语句，当 aa 或 ab 达到 22 时，执行相应的减法操作，否则输出 "TOTAL IS " 和 aa 或 ab 的值。


```
// Prints total of hand i
function print_total(i) {
    print("\n");
    aa = qa[i];
    total_aa();
    print("TOTAL IS " + aa + "\n");
}

function total_aa()
{
    if (aa >= 22)
        aa -= 11;
}

function total_ab()
{
    if (ab >= 22)
        ab -= 11;
}

```

这两段代码定义了一个函数 `total_ac()` 和一个函数 `process_input(str)`。

1. `function total_ac()` 定义了一个全局函数 `total_ac()`，其作用是减去一个变量 `ac` 的值，并将结果赋值给 `ac`。具体来说，如果 `ac` 的值大于等于 22，则减去 11，否则不执行减法操作。

2. `function process_input(str)` 定义了一个全局函数 `process_input(str)`，其作用是处理一个字符串 `str`。具体来说，函数将 `str` 中的前两个字符与一个预定义的正则表达式 `is` 比较，如果两个字符串相等，则跳出比较，否则计算字符串长度的一半，并将结果打印出来。如果两种字符串中只有一种相等，则返回 0，否则返回 1。函数的实现使用了 Python 中的字符串操作函数 `substr()` 和正则表达式函数 `is.substr()`。


```
function total_ac()
{
    if (ac >= 22)
        ac -= 11;
}

function process_input(str)
{
    str = str.substr(0, 1);
    for (h = 1; h <= h1; h += 2) {
        if (str == is.substr(h - 1, 1))
            break;
    }
    if (h <= h1) {
        h = (h + 1) / 2;
        return 0;
    }
    print("TYPE " + is.substr(0, h1 - 1) + " OR " + is.substr(h1 - 1, 2) + " PLEASE");
    return 1;
}

```

This is a Python program that appears to simulate a game of P住户 against D住户. It uses two libraries, 'colors' and '坚持下去', to display the game board and 'pandas' for data manipulation. The game board is defined by a 6x6 matrix, where 'a' stands for 'already', 'b' stands for 'beta', 'c' stands for 'candidates', 'd' stands for 'difficulty', and 'e' stands for 'electorate'. The 'player' and 'sign' variables are used to store the player's and the signal for the current player. The 'sa' and 'ta' arrays are used to store the player's score and the total score for each player, respectively. The program uses three different 'if' statements to check if the player has won, lost, or tied. The program also uses a 'discard\_row()' function to clear a certain number of rows based on the player's score. The 'pandas' library is also imported to use the 'qa' data structure, which is a 6x6 matrix that is used to store the player's score for each day.


```
// Main program
async function main()
{
    print(tab(31) + "BLACK JACK\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // --pa[i][j] IS THE JTH CARD IN HAND I, qa[i] IS TOTAL OF HAND I
    // --C IS THE DECK BEING DEALT FROM, D IS THE DISCARD PILE,
    // --ta[i] IS THE TOTAL FOR PLAYER I, sa[i] IS THE TOTAL THIS HAND FOR
    // --PLAYER I, ba[i] IS TH BET FOR HAND I
    // --ra[i] IS THE LENGTH OF pa[I,*]

    // --Program starts here
    // --Initialize
    for (i = 1; i <= 15; i++)
        pa[i] = [];
    for (i = 1; i <= 13; i++)
        for (j = 4 * i - 3; j <= 4 * i; j++)
            da[j] = i;
    d = 52;
    c = 53;
    print("DO YOU WANT INSTRUCTIONS");
    str = await input();
    if (str.toUpperCase().substr(0, 1) != "N") {
        print("THIS IS THE GAME OF 21. AS MANY AS 7 PLAYERS MAY PLAY THE\n");
        print("GAME. ON EACH DEAL, BETS WILL BE ASKED FOR, AND THE\n");
        print("PLAYERS' BETS SHOULD BE TYPED IN. THE CARDS WILL THEN BE\n");
        print("DEALT, AND EACH PLAYER IN TURN PLAYS HIS HAND. THE\n");
        print("FIRST RESPONSE SHOULD BE EITHER 'D', INDICATING THAT THE\n");
        print("PLAYER IS DOUBLING DOWN, 'S', INDICATING THAT HE IS\n");
        print("STANDING, 'H', INDICATING HE WANTS ANOTHER CARD, OR '/',\n");
        print("INDICATING THAT HE WANTS TO SPLIT HIS CARDS. AFTER THE\n");
        print("INITIAL RESPONSE, ALL FURTHER RESPONSES SHOULD BE 'S' OR\n");
        print("'H', UNLESS THE CARDS WERE SPLIT, IN WHICH CASE DOUBLING\n");
        print("DOWN IS AGAIN PERMITTED. IN ORDER TO COLLECT FOR\n");
        print("BLACKJACK, THE INITIAL RESPONSE SHOULD BE 'S'.\n");
    }
    while (1) {
        print("NUMBER OF PLAYERS");
        n = parseInt(await input());
        print("\n");
        if (n < 1 || n > 7)
            continue;
        else
            break;
    }
    for (i = 1; i <= 8; i++)
        ta[i] = 0;
    d1 = n + 1;
    while (1) {
        if (2 * d1 + c >= 52) {
            reshuffle();
        }
        if (c == 2)
            c--;
        for (i = 1; i <= n; i++)
            za[i] = 0;
        for (i = 1; i <= 15; i++)
            ba[i] = 0;
        for (i = 1; i <= 15; i++)
            qa[i] = 0;
        for (i = 1; i <= 7; i++)
            sa[i] = 0;
        for (i = 1; i <= 15; i++)
            ra[i] = 0;
        print("BETS:\n");
        for (i = 1; i <= n; i++) {
            do {
                print("#" + i + " ");
                za[i] = parseFloat(await input());
            } while (za[i] <= 0 || za[i] > 500) ;
        }
        for (i = 1; i <= n; i++)
            ba[i] = za[i];
        print("PLAYER");
        for (i = 1; i <= n; i++) {
            print(" " + i + "    ");
        }
        print("DEALER\n");
        for (j = 1; j <= 2; j++) {
            print(tab(5));
            for (i = 1; i <= d1; i++) {
                pa[i][j] = get_card();
                if (j == 1 || i <= n)
                    alt_card_print(pa[i][j]);
            }
            print("\n");
        }
        for (i = 1; i <= d1; i++)
            ra[i] = 2;
        // --Test for insurance
        if (pa[d1][1] <= 1) {
            print("ANY INSURANCE");
            str = await input();
            if (str.substr(0, 1) == "Y") {
                print("INSURANCE BETS\n");
                for (i = 1; i <= n; i++) {
                    do {
                        print("#" + i + " ");
                        za[i] = parseFloat(await input());
                    } while (za[i] < 0 || za[i] > ba[i] / 2) ;
                }
                for (i = 1; i <= n; i++)
                    sa[i] = za[i] * ((pa[d1][2] >= 10 ? 3 : 0) - 1);
            }
        }
        // --Test for dealer blackjack
        l1 = 1;
        l2 = 1;
        if (pa[d1][1] == 1 && pa[d1][2] > 9) {
            l1 = 0;
            l2 = 0;
        }
        if (pa[d1][2] == 1 && pa[d1][1] > 9) {
            l1 = 0;
            l2 = 0;
        }
        if (l1 == 0 && l2 == 0) {
            print("\n");
            print("DEALER HAS A" + ds.substr(3 * pa[d1][2] - 3, 3) + " IN THE HOLE FOR BLACKJACK\n");
            for (i = 1; i <= d1; i++)
                evaluate_hand(i);
        } else {
            // --No dealer blackjack
            if (pa[d1][1] <= 1 || pa[d1][1] >= 10) {
                print("\n");
                print("NO DEALER BLACKJACK.\n");
            }
            // --Now play the hands
            for (i = 1; i <= n; i++) {
                print("PLAYER " + i + " ");
                h1 = 7;
                do {
                    str = await input();
                } while (process_input(str)) ;
                if (h == 1) {   // Player wants to be hit
                    evaluate_hand(i);
                    h1 = 3;
                    x = get_card();
                    print("RECEIVED A");
                    card_print(x);
                    add_card_to_row(i, x);
                    if (q > 0)
                        print_total(i);
                } else if (h == 2) {    // Player wants to stand
                    evaluate_hand(i);
                    if (qa[i] == 21) {
                        print("BLACKJACK\n");
                        sa[i] = sa[i] + 1.5 * ba[i];
                        ba[i] = 0;
                        discard_row(i);
                    } else {
                        print_total(i);
                    }
                } else if (h == 3) {    // Player wants to double down
                    evaluate_hand(i);
                    h1 = 3;
                    h = 1;
                    while (1) {
                        if (h == 1) {   // Hit
                            x = get_card();
                            print("RECEIVED A");
                            card_print(x);
                            add_card_to_row(i, x);
                            if (q < 0)
                                break;
                            print("HIT");
                        } else if (h == 2) {    // Stand
                            print_total(i);
                            break;
                        }
                        do {
                            str = await input();
                        } while (process_input(str)) ;
                        h1 = 3;
                    }
                } else if (h == 4) {    // Player wants to split
                    l1 = pa[i][1];
                    if (l1 > 10)
                        l1 = 10;
                    l2 = pa[i][2];
                    if (l2 > 10)
                        l2 = 10;
                    if (l1 != l2) {
                        print("SPLITTING NOT ALLOWED.\n");
                        i--;
                        continue;
                    }
                    // --Play out split
                    i1 = i + d1;
                    ra[i1] = 2;
                    pa[i1][1] = pa[i1][2];
                    ba[i + d1] = ba[i];
                    x = get_card();
                    print("FIRST HAND RECEIVES A");
                    card_print(x);
                    pa[i][2] = x;
                    evaluate_hand(i);
                    print("\n");
                    x = get_card();
                    print("SECOND HAND RECEIVES A");
                    i = i1;
                    card_print(x);
                    pa[i][2] = x;
                    evaluate_hand(i);
                    print("\n");
                    i = i1 - d1;
                    if (pa[i][1] != 1) {
                        // --Now play the two hands
                        do {

                            print("HAND " + (i > d1 ? 2 : 1) + " ");
                            h1 = 5;
                            while (1) {
                                do {
                                    str = await input();
                                } while (process_input(str)) ;
                                h1 = 3;
                                if (h == 1) {   // Hit
                                    x = get_card();
                                    print("RECEIVED A");
                                    card_print(x);
                                    add_card_to_row(i, x);
                                    if (q < 0)
                                        break;
                                    print("HIT");
                                } else if (h == 2) {    // Stand
                                    print_total(i);
                                    break;
                                } else {    // Double
                                    x = get_card();
                                    ba[i] *= 2;
                                    print("RECEIVED A");
                                    card_print(x);
                                    add_card_to_row(i, x);
                                    if (q > 0)
                                        print_total(i);
                                    break;
                                }
                            }
                            i += d1;
                        } while (i == i1) ;
                        i = i1 - d1;
                    }
                }
            }
            // --Test for playing dealer's hand
            evaluate_hand(i);
            for (i = 1; i <= n; i++) {
                if (ra[i] > 0 || ra[i + d1] > 0)
                    break;
            }
            if (i > n) {
                print("DEALER HAD A");
                x = pa[d1][2];
                card_print(x);
                print(" CONCEALED.\n");
            } else {
                print("DEALER HAS A" + ds.substr(3 * pa[d1][2] - 3, 3) + " CONCEALED ");
                i = d1;
                aa = qa[i];
                total_aa();
                print("FOR A TOTAL OF " + aa + "\n");
                if (aa <= 16) {
                    print("DRAWS");
                    do {

                        x = get_card();
                        alt_card_print(x);
                        add_card_to_row(i, x);
                        aa = q;
                        total_aa();
                    } while (q > 0 && aa < 17) ;
                    if (q < 0) {
                        qa[i] = q + 0.5;
                    } else {
                        qa[i] = q;
                    }
                    if (q >= 0) {
                        aa = q;
                        total_aa();
                        print("---TOTAL IS " + aa + "\n");
                    }
                }
                print("\n");
            }
        }
        // --TALLY THE RESULT
        str = "LOSES PUSHES WINS "
        print("\n");
        for (i = 1; i <= n; i++) {
            aa = qa[i]
            total_aa();
            ab = qa[i + d1];
            total_ab();
            ac = qa[d1];
            total_ac();
            signaaac = aa - ac;
            if (signaaac) {
                if (signaaac < 0)
                    signaaac = -1;
                else
                    signaaac = 1;
            }
            signabac = ab - ac;
            if (signabac) {
                if (signabac < 0)
                    signabac = -1;
                else
                    signabac = 1;
            }
            sa[i] = sa[i] + ba[i] * signaaac + ba[i + d1] * signabac;
            ba[i + d1] = 0;
            print("PLAYER " + i + " ");
            signsai = sa[i];
            if (signsai) {
                if (signsai < 0)
                    signsai = -1;
                else
                    signsai = 1;
            }
            print(str.substr(signsai * 6 + 6, 6) + " ");
            if (sa[i] == 0)
                print("      ");
            else
                print(" " + Math.abs(sa[i]) + " ");
            ta[i] = ta[i] + sa[i];
            print("TOTAL= " + ta[i] + "\n");
            discard_row(i);
            ta[d1] = ta[d1] - sa[i];
            i += d1;
            discard_row(i);
            i -= d1;
        }
        print("DEALER'S TOTAL= " + ta[d1] + "\n");
        print("\n");
        discard_row(i);
    }
}

```

这道题是一个简单的 Python 代码，包含一个名为 "main()" 的函数。然而，没有函数体，说明此代码没有具体的功能。我们需要了解一些基本知识来分析这个问题。

1. Python 是一种高级编程语言，具有简洁易懂的语法。其中，标量（value）和变量（variable）是两种基本的变量类型。

2. 函数是 Python 中实现特定功能的部分。一个函数通常包含一个或多个字符串（称为函数体），用于描述函数的行为。

3. 当我们给函数起一个名字时，通常将其称为"main"函数。然而，这个名字并没有实际的功能。

4. 任何程序在开始执行之前都需要加载。在 Python 中，这通常是通过 "import" 语句来完成的。但是，对于 "main()" 函数而言，我们不需要关心如何加载程序。

综上所述，以上代码可能是一个简单的 Python 程序，但缺乏具体的功能。我们需要了解更多信息来获取此代码的实际功能。


```
main();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


# `10_Blackjack/python/blackjack.py`

这段代码定义了一个名为 "PlayerType" 的枚举类型，它有两个枚举值，分别对应 "Player" 和 "Dealer" 两个名称。

在代码中，它导入了 "enum"、"random"、"dataclasses" 和 "typing" 四个库，分别用于定义枚举类型、生成随机数、定义变量和提供类型声明等。

接着，它定义了一个名为 "Blackjack" 的类，它从 "PlayerType" 枚举类型中获取了一个名为 "Player" 的成员，并定义了一个名为 "PortedByMartinThomam2022" 的元数据，表示该类是年由 Martin Thoma 创建的，使用了 AnthonyMichaelTDM 实现的 Blackjack 游戏。

最后，代码中还有一行代码，导入了 "NamedTuple" 类型，但是并没有使用它，所以这一行代码可能是无效的。


```
"""
Blackjack

Ported by Martin Thoma in 2022,
using the rust implementation of AnthonyMichaelTDM
"""

import enum
import random
from dataclasses import dataclass
from typing import List, NamedTuple


class PlayerType(enum.Enum):
    Player = "Player"
    Dealer = "Dealer"


```

这段代码定义了一个名为 "Play" 的类，该类使用了一个枚举类型 "enum.Enum"。枚举类型是一种简单的数据类型，它可以定义常量，变量和常量方法。在这个例子中，定义了四个枚举常量：Stand, Hit, DoubleDown, Split。这些常量的值将会被存储为整数类型。

接着定义了一个名为 "Card" 的类，该类使用了一个名为 "NamedTuple" 的子类型。NamedTuple 是 Python 3.6 引入的类型，它是一个简单类型的组合对象，可以方便地在属性中使用属性名称来引用对象成员。在这个例子中，使用 NamedTuple 创建了一个 "Card" 类的实例，该类包含了一个名为 "name" 的属性。

接着在 Play 类中定义了四个枚举常量：Stand, Hit, DoubleDown, Split。这些常量的值将会被存储为整数类型。这里，枚举常量的默认值是使用枚举类型自动生成的。在函数内部，还可以使用这些枚举常量来设置游戏中的状态。

最后，定义了一个 "Card" 类，该类使用了一个名为 "enum.Enum" 的枚举类型。这里，枚举类型定义了一些常量，例如 "ACE" 和 "QUEEN"。这些常量的值将会被存储为整数类型。


```
class Play(enum.Enum):
    Stand = enum.auto()
    Hit = enum.auto()
    DoubleDown = enum.auto()
    Split = enum.auto()


class Card(NamedTuple):
    name: str

    @property
    def value(self) -> int:
        """
        returns the value associated with a card with the passed name
        return 0 if the passed card name doesn't exist
        """
        return {
            "ACE": 11,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "10": 10,
            "JACK": 10,
            "QUEEN": 10,
            "KING": 10,
        }.get(self.name, 0)


```

这段代码定义了一个名为“Hand”的类，该类使用“NamedTuple”作为元类，表示每个卡片实例都是一个名为“Card”的元组，包含一个名为“cards”的列表。

该类有两个方法：

- add\_card(self)：将传入的卡片添加到这张手工中。
- get\_total(self)：返回手中的所有牌的点数之和（不考虑ACE的点数）。

在get\_total方法中，通过遍历手中的所有牌，计算出所有牌点数之和，然后根据是否为ACE来决定是否从总点数中减去10。

在discard\_hand(self)方法中，如果手中的牌为空，会抛出一个“ValueError”。如果剩余牌，会将它们添加到“discard pile”中。


```
class Hand(NamedTuple):
    cards: List[Card]

    def add_card(self, card: Card) -> None:
        """add a passed card to this hand"""
        self.cards.append(card)

    def get_total(self) -> int:
        """returns the total points of the cards in this hand"""
        total: int = 0
        for card in self.cards:
            total += int(card.value)

        # if there is an ACE, and the hand would otherwise bust,
        # treat the ace like it's worth 1
        if total > 21 and any(card.name == "ACE" for card in self.cards):
            total -= 10

        return total

    def discard_hand(self, deck: "Decks") -> None:
        """adds the cards in hand into the discard pile"""
        _len = len(self.cards)
        for _i in range(_len):
            if len(self.cards) == 0:
                raise ValueError("hand empty")
            deck.discard_pile.append(self.cards.pop())


```

This is a class called `Decks` that inherits from the `NamedTuple` class. It contains two lists, `deck` and `discard_pile`, which are both lists of cards. The `draw_card` method is used to draw a card from the deck and return it. The `shuffle` method is used to shuffle the cards in the deck. The `draw_card` method tries to draw a card from the deck, but if the deck is empty or the deck is full, it will try to draw a card from the discard pile.


```
class Decks(NamedTuple):
    deck: List[Card]
    discard_pile: List[Card]

    @classmethod
    def new(cls) -> "Decks":
        """creates a new full and shuffled deck, and an empty discard pile"""
        # returns a number of full decks of 52 cards, shuffles them
        deck = Decks(deck=[], discard_pile=[])
        number_of_decks = 3

        # fill deck
        for _n in range(number_of_decks):
            # fill deck with number_of_decks decks worth of cards
            for card_name in CARD_NAMES:
                # add 4 of each card, totaling one deck with 4 of each card
                for _ in range(4):
                    deck.deck.append(Card(name=card_name))

        deck.shuffle()
        return deck

    def shuffle(self) -> None:
        """shuffles the deck"""
        random.shuffle(self.deck)

    def draw_card(self) -> Card:
        """
        draw card from deck, and return it
        if deck is empty, shuffles discard pile into it and tries again
        """
        if len(self.deck) == 0:
            _len = len(self.discard_pile)

            if _len > 0:
                # deck is empty, shuffle discard pile into deck and try again
                print("deck is empty, shuffling")
                for _i in range(_len):
                    if len(self.discard_pile) == 0:
                        raise ValueError("discard pile is empty")
                    self.deck.append(self.discard_pile.pop())
                self.shuffle()
                return self.draw_card()
            else:
                # discard pile and deck are empty, should never happen
                raise Exception("discard pile empty")
        else:
            card = self.deck.pop()
            return card


```

This is a class definition for a game of cards. It contains a player interface with methods for getting the player's turn, hitting, and splitting. It also includes a get\_play method that is used by the player interface methods.

The get\_play method takes a player object and returns the play for that player. If the player object has not specified a player type when instantiating the class, it defaults to being a dealer. If the player object is a dealer, the get\_play method uses an algorithm to determine the player's play. If the player object is a player, the method asks the user to input their play. The valid options for the play methods are "s", "h", "d", and "/".

ThePlayer class represents a player in the game. It contains a hand property that is a list of the player's cards, as well as a player type property that specifies whether the player is a dealer or a player. ThePlayer class also has methods for getting the player's turn, hitting, and splitting.

TheDealer class represents a dealer in the game. It has methods for getting the player's turn, hitting, and splitting, as well as a method for getting the total points of the hand.

ThePlayer class has methods for getting the player's turn, hitting, and splitting, as well as a method for getting the total points of the hand.

Theget\_play method is used by the player interface methods to determine the play for the player. It takes a player object and returns the play for that player. If the player object has not specified a player type when instantiating the class, it defaults to being a dealer. If the player object is a dealer, the get\_play method uses an algorithm to determine the player's play. If the player object is a player, the method asks the user to input their play. The valid options for the play methods are "s", "h", "d", and "/".


```
@dataclass
class Player:
    hand: Hand
    balance: int
    bet: int
    wins: int
    player_type: PlayerType
    index: int

    @classmethod
    def new(cls, player_type: PlayerType, index: int) -> "Player":
        """creates a new player of the given type"""
        return Player(
            hand=Hand(cards=[]),
            balance=STARTING_BALANCE,
            bet=0,
            wins=0,
            player_type=player_type,
            index=index,
        )

    def get_name(self) -> str:
        return f"{self.player_type}{self.index}"

    def get_bet(self) -> None:
        """gets a bet from the player"""
        if PlayerType.Player == self.player_type:
            if self.balance < 1:
                print(f"{self.get_name()} is out of money :(")
                self.bet = 0
            self.bet = get_number_from_user_input(
                f"{self.get_name()}\tWhat is your bet", 1, self.balance
            )

    def hand_as_string(self, hide_dealer: bool) -> str:
        """
        returns a string of the players hand

        if player is a dealer, returns the first card in the hand followed
        by *'s for every other card
        if player is a player, returns every card and the total
        """
        if not hide_dealer:
            s = ""
            for cards_in_hand in self.hand.cards[::-1]:
                s += f"{cards_in_hand.name}\t"
            s += f"total points = {self.hand.get_total()}"
            return s
        else:
            if self.player_type == PlayerType.Dealer:
                s = ""
                for c in self.hand.cards[1::-1]:
                    s += f"{c.name}\t"
                return s
            elif self.player_type == PlayerType.Player:
                s = ""
                for cards_in_hand in self.hand.cards[::-1]:
                    s += f"{cards_in_hand.name}\t"
                s += f"total points = {self.hand.get_total()}"
                return s
        raise Exception("This is unreachable")

    def get_play(self) -> Play:
        """get the players 'play'"""
        # do different things depending on what type of player this is:
        # if it's a dealer, use an algorithm to determine the play
        # if it's a player, ask user for input
        if self.player_type == PlayerType.Dealer:
            if self.hand.get_total() > 16:
                return Play.Stand
            else:
                return Play.Hit
        elif self.player_type == PlayerType.Player:
            valid_results: List[str]
            if len(self.hand.cards) > 2:
                # if there are more than 2 cards in the hand,
                # at least one turn has happened, so splitting and
                # doubling down are not allowed
                valid_results = ["s", "h"]
            else:
                valid_results = ["s", "h", "d", "/"]
            play = get_char_from_user_input("\tWhat is your play?", valid_results)
            if play == "s":
                return Play.Stand
            elif play == "h":
                return Play.Hit
            elif play == "d":
                return Play.DoubleDown
            elif play == "/":
                return Play.Split
            else:
                raise ValueError(f"got invalid character {play}")
        raise Exception("This is unreachable")


```

`player_hands_message` is a string that displays the player's hands for each player. It starts with the player's name and then displays the player's hand as a string (with some additional formatting).

The function determines the winner by counting the points of all players who have less points than the winner's. The winner is shown at the end of the message.

If there are multiple winners with the same number of points, the function displays only the first winner's name and increments their wins by 1.

If there are no winners, the function displays a message indicating that.

The function also handles bets by removing money from losers and adding it to the winner's balance.

Finally, the function updates the `player_hand` object with the player's current hands and resets the `player_hand` object for each player.


```
@dataclass
class Game:
    players: List[Player]  # last item in this is the dealer
    decks: Decks
    games_played: int

    @classmethod
    def new(cls, num_players: int) -> "Game":
        players: List[Player] = []

        # add dealer
        players.append(Player.new(PlayerType.Dealer, 0))
        # create human player(s) (at least one)
        players.append(Player.new(PlayerType.Player, 1))
        for i in range(2, num_players):  # one less than num_players players
            players.append(Player.new(PlayerType.Player, i))

        if get_char_from_user_input("Do you want instructions", ["y", "n"]) == "y":
            print_instructions()
        print()

        return Game(players=players, decks=Decks.new(), games_played=0)

    def _print_stats(self) -> None:
        """prints the score of every player"""
        print(f"{self.stats_as_string()}")

    def stats_as_string(self) -> str:
        """returns a string of the wins, balance, and bets of every player"""
        s = ""
        for p in self.players:
            # format the presentation of player stats
            if p.player_type == PlayerType.Dealer:
                s += f"{p.get_name()} Wins:\t{p.wins}\n"
            elif p.player_type == PlayerType.Player:
                s += f"{p.get_name()} "
                s += f"Wins:\t{p.wins}\t\t"
                s += f"Balance:\t{p.balance}\t\tBet\t{p.bet}\n"
        return f"Scores:\n{s}"

    def play_game(self) -> None:
        """plays a round of blackjack"""
        game = self.games_played
        player_hands_message: str = ""

        # deal two cards to each player
        for _i in range(2):
            for player in self.players:
                player.hand.add_card(self.decks.draw_card())

        # get everyones bets
        for player in self.players:
            player.get_bet()
        scores = self.stats_as_string()

        # play game for each player
        for player in self.players:
            # turn loop, ends when player finishes their turn
            while True:
                clear()
                print_welcome_screen()
                print(f"\n\t\t\tGame {game}")
                print(scores)
                print(player_hands_message)
                print(f"{player.get_name()} Hand:\t{player.hand_as_string(True)}")

                if PlayerType.Player == player.player_type and player.bet == 0:
                    break

                # play through turn
                # check their hand value for a blackjack(21) or bust
                score = player.hand.get_total()
                if score >= 21:
                    if score == 21:
                        print("\tBlackjack! (21 points)")
                    else:
                        print(f"\tBust      ({score} points)")
                    break

                # get player move
                play = player.get_play()
                # process play
                if play == Play.Stand:
                    print(f"\t{play}")
                    break
                elif play == Play.Hit:
                    print(f"\t{play}")
                    player.hand.add_card(self.decks.draw_card())
                elif play == Play.DoubleDown:
                    print(f"\t{play}")

                    # double their balance if there's enough money,
                    # othewise go all-in
                    if player.bet * 2 < player.balance:
                        player.bet *= 2
                    else:
                        player.bet = player.balance
                    player.hand.add_card(self.decks.draw_card())
                elif play == Play.Split:
                    pass

            # add player to score cache thing
            player_hands_message += (
                f"{player.get_name()} Hand:\t{player.hand_as_string(True)}\n"
            )

        # determine winner
        top_score = 0

        # player with the highest points
        num_winners = 1

        non_burst_players = [
            player for player in self.players if player.hand.get_total() <= 21
        ]
        for player in non_burst_players:
            score = player.hand.get_total()
            if score > top_score:
                top_score = score
                num_winners = 1
            elif score == top_score:
                num_winners += 1

        # show winner(s)
        top_score_players = [
            player
            for player in non_burst_players
            if player.hand.get_total() == top_score
        ]
        for x in top_score_players:
            print(f"{x.get_name()} ")
            x.wins += 1
            # increment their wins
        if num_winners > 1:
            print(f"all tie with {top_score}\n\n\n")
        else:
            print(
                f"wins with {top_score}!\n\n\n",
            )

        # handle bets
        # remove money from losers
        losers = [
            player for player in self.players if player.hand.get_total() != top_score
        ]
        for loser in losers:
            loser.balance -= loser.bet
        # add money to winner
        winners = [
            player for player in self.players if player.hand.get_total() == top_score
        ]
        for winner in winners:
            winner.balance += winner.bet

        # discard hands
        for player in self.players:
            player.hand.discard_hand(self.decks)

        # increment games_played
        self.games_played += 1


```

这段代码创建了一个名为 "CARD_NAMES" 的列表类型，并向其中添加了一系列字符串元素，这些元素将作为信用卡名字。具体来说，这段代码将以下信用卡名字添加到了 "CARD_NAMES" 列表中：

```
"ACE",
"2",
"3",
"4",
"5",
"6",
"7",
"8",
"9",
"10",
"JACK",
"QUEEN",
"KING"
```

这段代码还使用了 Python 的字符串格式化语法，将每个信用卡名字转换为小写字母，并将其添加到了 "CARD_NAMES" 列表中。


```
CARD_NAMES: List[str] = [
    "ACE",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "JACK",
    "QUEEN",
    "KING",
]
```

这段代码是一个Python程序，它的作用是让用户创建一个游戏，并运行一个游戏循环，直到用户想要停止为止。

在程序开始时，先定义了一个名为STARTING_BALANCE的整型变量，其值为100。然后定义了一个名为main的函数，这个函数内部有一个game变量，它是一个Game类的实例。

接下来程序输出一个欢迎屏幕，然后进入游戏循环。在循环中，程序会首先调用game类的new函数，这个函数会询问用户要创建多少个玩家，程序会记录下用户输入的值，然后创建一个新游戏对象。

循环内部还有一个while循环，它会不断地运行游戏循环和用户交互。在while循环内部，程序会首先获取用户的输入，如果是"y"，那么程序会再次调用game类的play_game函数，让游戏继续运行；如果是"n"，那么游戏循环就会停止。

总结起来，这段代码的作用是让用户创建一个游戏对象，并运行一个游戏循环，直到用户想要停止为止。


```
STARTING_BALANCE: int = 100


def main() -> None:
    game: Game

    print_welcome_screen()

    # create game
    game = Game.new(
        get_number_from_user_input("How many players should there be", 1, 7)
    )

    # game loop, play game until user wants to stop
    char = "y"
    while char == "y":
        game.play_game()
        char = get_char_from_user_input("Play Again?", ["y", "n"])


```

这两函数脚本解释如下：

1. print_welcome_screen()：
```less
BLACK JACK
 CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY
```
这个函数脚本的作用是在欢迎屏幕上输出游戏规则，类似于在赌场海报上看到的“不要离开，否则你会输掉你的筹码”的标语。

2. print_instructions()：
```less
THIS IS THE GAME OF 21. AS MANY AS 7 PLAYERS MAY PLAY THE
   GAME. ON EACH DEAL, BETS WILL BE ASKED FOR, AND THE
   PLAYERS' BETS SHOULD BE TYPED IN. THE CARDS WILL THEN
   BE DEALT, AND EACH PLAYER IN TURN PLAYS HIS HAND.
   THE FIRST RESPONSE SHOULD BE EITHER 'D', INDICATING THAT THE
   PLAYER IS DOUBLING DOWN, 'S', INDICATING THAT HE IS STANDING,
   'H', INDICATING HE WANDS ANOTHER CARD, OR '/', INDICATING
   THAT HE WANTS TO SPLIT HIS CARDS. AFTER THE INITIAL
   RESPONSE, ALL FURTHER RESPONSE SHOULD BE 'S' OR 'H', UNLESS THE
   CARDS WERE SPLIT, IN WHICH CASE DOUBLING DOWN IS AGAIN
   PERMITTED. IN ORDER TO COLLECT FOR BLACKJACK, THE
   INITIAL RESPONSE SHOULD BE 'S'.
   NUMBER OF PLAYERS

   NOTE:'/' (splitting) is not currently implemented, and does nothing
```
这个函数脚本的作用是在游戏开始时告诉玩家有几个玩家以及游戏规则。

在游戏过程中，每个玩家会依次叫出自己的手牌，然后可以选择“D”表示加倍，“S”表示 stand，“/”表示 split，或者“H”表示拿牌。如果有人选择了“/”，那么他可以继续选择“/”，除非有人先选择了“H”。

此外，如果有人选择了“H”来拿牌，那么游戏就会重新开始并继续。

整个游戏的目的是让7个玩家参与，通过各种手牌组合获得最高分数。


```
def print_welcome_screen() -> None:
    print(
        """
                            BLACK JACK
              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY
    \n\n"""
    )


def print_instructions() -> None:
    print(
        """
    THIS IS THE GAME OF 21. AS MANY AS 7 PLAYERS MAY PLAY THE
    GAME. ON EACH DEAL, BETS WILL BE ASKED FOR, AND THE
    PLAYERS' BETS SHOULD BE TYPED IN. THE CARDS WILL THEN BE
    DEALT, AND EACH PLAYER IN TURN PLAYS HIS HAND. THE
    FIRST RESPONSE SHOULD BE EITHER 'D', INDICATING THAT THE
    PLAYER IS DOUBLING DOWN, 'S', INDICATING THAT HE IS
    STANDING, 'H', INDICATING HE WANTS ANOTHER CARD, OR '/',
    INDICATING THAT HE WANTS TO SPLIT HIS CARDS. AFTER THE
    INITIAL RESPONSE, ALL FURTHER RESPONSES SHOULD BE 'S' OR
    'H', UNLESS THE CARDS WERE SPLIT, IN WHICH CASE DOUBLING
    DOWN IS AGAIN PERMITTED. IN ORDER TO COLLECT FOR
    BLACKJACK, THE INITIAL RESPONSE SHOULD BE 'S'.
    NUMBER OF PLAYERS

    NOTE:'/' (splitting) is not currently implemented, and does nothing

    PRESS ENTER TO CONTINUE
    """
    )
    input()


```

这段代码定义了一个名为 `get_number_from_user_input` 的函数，用于从用户输入中获取一个整数并返回。函数接受两个参数，一个是提示字符串 `prompt`，一个是最小值和最大值，分别用 `min_value` 和 `max_value` 表示。

函数内部包含一个 while 循环，该循环会在用户输入任何内容后都继续执行，直到用户输入的内容符合要求。循环内部包含一个 f-string 格式化子，该子将提示字符串中的内容与 `min_value` 和 `max_value` 之间的差值赋值给变量 `user_input`。如果 `user_input` 的值小于 `min_value` 或者大于 `max_value`，函数会打印出错误消息并退出循环。

当用户输入符合要求后，函数返回 `user_input` 的值。如果 `user_input` 的值出现任何错误，函数也会打印出错误消息并退出循环。


```
def get_number_from_user_input(prompt: str, min_value: int, max_value: int) -> int:
    """gets a int integer from user input"""
    # input loop
    user_input = None
    while user_input is None or user_input < min_value or user_input > max_value:
        raw_input = input(prompt + f" ({min_value}-{max_value})? ")

        try:
            user_input = int(raw_input)
            if user_input < min_value or user_input > max_value:
                print("Invalid input, please try again")
        except ValueError:
            print("Invalid input, please try again")
    return user_input


```



该代码定义了一个名为 `get_char_from_user_input` 的函数，用于从用户输入中获取第一个字符并返回给调用者。函数接受两个参数：一个字符串 `prompt`，和一个有效的结果列表 `valid_results`。在函数内部，使用了一个 while 循环和一个 input() 函数来获取用户输入。如果用户输入的字符串不在有效的结果列表中，函数会提示用户重新输入。

函数内部还有一个名为 `clear` 的函数，用于清空标准输出并输出一个 `J` 字符。该函数使用了 `print` 函数和平行控制 `\x1b[2J` 和 `\x1b[0;0H`，这些字符可以用来在控制台输出。


```
def get_char_from_user_input(prompt: str, valid_results: List[str]) -> str:
    """returns the first character they type"""
    user_input = None
    while user_input not in valid_results:
        user_input = input(prompt + f" {valid_results}? ").lower()
        if user_input not in valid_results:
            print("Invalid input, please try again")
    assert user_input is not None
    return user_input


def clear() -> None:
    """clear std out"""
    print("\x1b[2J\x1b[0;0H")


```

这段代码是一个if语句，它的判断条件是(__name__ == "__main__")。

"__name__"是一个特殊的属性，它是用来保存程序名称的，格式为"程序名称__"。当程序运行时，操作系统会为程序分配一个内存空间，程序名称就是被保存在这个内存空间中的。

"__main__"也是一个特殊的属性，它是用来保存程序名称的，格式为"程序名称__"。这个属性同样会被保存在内存空间中，只不过它是在程序名称之后。

if语句的条件是，如果当前程序运行时 __name__ == "__main__"，那么程序就会执行if语句块内的语句。

如果当前程序的名称与__main__相同，那么if语句块内的语句就会被执行。举个例子，如果当前程序名为"myprogram"，那么程序将先执行if语句块内的语句，输出"myprogram"，然后才会执行if语句块外的语句，不会输出"myprogram"。


```
if __name__ == "__main__":
    main()

```