# BasicComputerGames源码解析 64

# `69_Pizza/java/src/PizzaGame.java`

这段代码定义了一个名为 "PizzaGame" 的类，其包含一个名为 "main" 的方法。在 "main" 方法中，创建了一个新的 "Pizza" 对象，并调用其 "play" 方法。

具体来说，"Pizza" 类可能是一个包含组件的游戏，而 "play" 方法可能是用于游戏中的某种行为或操作。由于没有提供更多的上下文和信息，无法进一步解释 "PizzaGame" 类和 "play" 方法的确切含义。


```
public class PizzaGame {

    public static void main(String[] args) {

        Pizza pizza = new Pizza();
        pizza.play();
    }
}

```

# `69_Pizza/javascript/pizza.js`

这段代码定义了两个函数，分别是`print()`和`input()`。

`print()`函数的作用是接收一个字符串参数（`str`），将其显示在页面上，并返回。具体来说，通过创建一个`document.getElementById("output")`中的`appendChild()`操作，将接收到的字符串插入到页面中的一个`<textarea>`元素中，从而实现了将字符串打印到页面上。

`input()`函数的作用是接收一个字符（`str`），并返回其哈希码（一种数据类型，通常是一个字符串的ASCII码）。具体来说，该函数会提示用户输入一个字符，然后返回该字符的哈希码，以便在将来的代码中对其进行匹配或者搜索等操作。函数中使用了`document.createElement("INPUT")`来创建一个`<input>`元素，并设置了其属性，然后将其添加到页面中的一个元素上，使其具有输入框的样式。接着，函数监控该元素的`keydown`事件，当用户按下了`<Keyboard>`上的`13`键（通常是`回车`键）时，函数会将用户输入的字符串存储到`input_str`变量中，并将其显示在页面上的输出区域中，以便于对用户输入的字符进行操作。函数还添加了一个`print()`函数来将`input_str`字符串打印到页面上。


```
// PIZZA
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

This is a script written in JavaScript that simulates a job interview process. The interviewer is given a task to deliver pizzas to a specific address. The script randomly generates six tasks, and for each task, it asks the driver to deliver a specific pizza and then asks if the driver wants to deliver more.

The code starts by printing a message to the console if the task is too difficult for the driver and then explains that the driver has completed the task. After that, it prints a message asking if the driver wants to deliver more pizzas. If the driver enters "YES", the script will continue by printing a message for each additional task and then end the loop.

It is important to note that the code may not work correctly if the address or the number of tasks are not defined or if the input from the user is not processed correctly.


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var sa = [, "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P"];
var ma = [, "1","2","3","4"];
var a = [];

// Main program
async function main()
{
    print(tab(33) + "PIZZA\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("PIZZA DELIVERY GAME\n");
    print("\n");
    print("WHAT IS YOUR FIRST NAME");
    ns = await input();
    print("\n");
    print("HI, " + ns + ". IN THIS GAME YOU ARE TO TAKE ORDERS\n");
    print("FOR PIZZAS.  THEN YOU ARE TO TELL A DELIVERY BOY\n");
    print("WHERE TO DELIVER THE ORDERED PIZZAS.\n");
    print("\n");
    print("\n");
    print("MAP OF THE CITY OF HYATTSVILLE\n");
    print("\n");
    print(" -----1-----2-----3-----4-----\n");
    k = 4;
    for (i = 1; i <= 4; i++) {
        print("-\n");
        print("-\n");
        print("-\n");
        print("-\n");
        print(ma[k]);
        s1 = 16 - 4 * i + 1;
        print("     " + sa[s1] + "     " + sa[s1 + 1] + "     " + sa[s1 + 2] + "     ");
        print(sa[s1 + 3] + "     " + ma[k] + "\n");
        k--;
    }
    print("-\n");
    print("-\n");
    print("-\n");
    print("-\n");
    print(" -----1-----2-----3-----4-----\n");
    print("\n");
    print("THE OUTPUT IS A MAP OF THE HOMES WHERE\n");
    print("YOU ARE TO SEND PIZZAS.\n");
    print("\n");
    print("YOUR JOB IS TO GIVE A TRUCK DRIVER\n");
    print("THE LOCATION OR COORDINATES OF THE\n");
    print("HOME ORDERING THE PIZZA.\n");
    print("\n");
    while (1) {
        print("DO YOU NEED MORE DIRECTIONS");
        str = await input();
        if (str == "YES" || str == "NO")
            break;
        print("'YES' OR 'NO' PLEASE, NOW THEN, ");
    }
    if (str == "YES") {
        print("\n");
        print("SOMEBODY WILL ASK FOR A PIZZA TO BE\n");
        print("DELIVERED.  THEN A DELIVERY BOY WILL\n");
        print("ASK YOU FOR THE LOCATION.\n");
        print("     EXAMPLE:\n");
        print("THIS IS J.  PLEASE SEND A PIZZA.\n");
        print("DRIVER TO " + ns + ".  WHERE DOES J LIVE?\n");
        print("YOUR ANSWER WOULD BE 2,3\n");
        print("\n");
        print("UNDERSTAND");
        str = await input();
        if (str != "YES") {
            print("THIS JOB IS DEFINITELY TOO DIFFICULT FOR YOU. THANKS ANYWAY");
            return;
        }
        print("GOOD.  YOU ARE NOW READY TO START TAKING ORDERS.\n");
        print("\n");
        print("GOOD LUCK!!\n");
        print("\n");
    }
    while (1) {
        for (i = 1; i <= 5; i++) {
            s = Math.floor(Math.random() * 16 + 1);
            print("\n");
            print("HELLO " + ns + "'S PIZZA.  THIS IS " + sa[s] + ".\n");
            print("  PLEASE SEND A PIZZA.\n");
            while (1) {
                print("  DRIVER TO " + ns + ":  WHERE DOES " + sa[s] + " LIVE");
                str = await input();
                a[1] = parseInt(str);
                a[2] = parseInt(str.substr(str.indexOf(",") + 1));
                t = a[1] + (a[2] - 1) * 4;
                if (t != s) {
                    print("THIS IS " + sa[t] + ". I DID NOT ORDER A PIZZA.\n");
                    print("I LIVE AT " + a[1] + "," + a[2] + "\n");
                } else {
                    break;
                }
            }
            print("HELLO " + ns + ".  THIS IS " + sa[s] + ", THANKS FOR THE PIZZA.\n");
        }
        print("\n");
        print("DO YOU WANT TO DELIVER MORE PIZZAS");
        str = await input();
        if (str != "YES")
            break;
    }
    print("\n");
    print("O.K. " + ns + ", SEE YOU LATER!\n");
    print("\n");
}

```

这道题是一个简单的 Python 代码，包含一个名为 `main()` 的函数。在这个函数中，只有一个语句，即 `return` 并输出了一个整数。

根据 Python 编程规范，`return` 语句是用来返回一个表达式的。在这里，表达式是 `None`，它是一个特殊的值，代表着没有值或没有结果。

因此，这个 `main()` 函数的作用是返回一个名为 `None` 的值。


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


# `69_Pizza/python/pizza.py`

这段代码是一个Python程序，是一个 pizza 配送模拟游戏。程序的主要作用是模拟一个送披萨的业务，模拟了送餐员在送货过程中可能会遇到的各种情况。

具体来说，程序的作用可以分为以下几个步骤：

1. 导入必要的模块和函数，包括`random`模块中的`randint()`函数，用于生成随机整数。

2. 定义了`PAGE_WIDTH`和`customer_names`、`street_names`变量，分别表示披萨的页宽和客户的姓名、街道的名称。

3. 程序使用`chr()`函数生成了16个随机整数，每个整数都对应一个客户的姓名。然后使用`str()`函数将每个整数转换成一个字符串，存储客户的姓名。

4. 程序使用`range()`函数生成0到4的随机整数，用于生成街道的名称。然后使用字符串格式化将0到4转换成字符串，存储街道的名称。

5. 程序使用`while`循环，在循环中调用`customer_names.append(chr(65 + x))`和`street_names.append(str(n))`函数。前者将生成的随机整数存储到`customer_names`列表中，后者将生成的随机整数转换成字符串并存储到`street_names`列表中。

6. 程序使用`print()`函数输出模拟结果。


```
"""
PIZZA

A pizza delivery simulation

Ported by Dave LeCompte
"""

import random

PAGE_WIDTH = 64

customer_names = [chr(65 + x) for x in range(16)]
street_names = [str(n) for n in range(1, 5)]


```



这些函数的主要目的是在控制台输出中使用不同的格式和中心对齐方式。

1. `print_centered` 函数接收一个字符串参数 `msg`，并使用 `spaces` 变量来获取适当的字符空间位置。然后，通过 `print` 函数将 `msg` 和 `spaces` 组合在一起输出，并在字符的中心对齐。这里注意到，`PAGE_WIDTH` 是计算页面宽度的变量，而不是页面大小。

2. `print_header` 函数接收一个字符串参数 `title`，并使用 `print_centered` 函数来输出标题。然后，它再次使用 `print_centered` 函数来输出 "CREATIVE COMPUTING" 和 "MORRISTOWN, NEW JERSEY"。

3. `print_ticks` 函数接收一个整数参数 `num_ticks`，并使用循环来输出 `num_ticks` 个 ".-" 字符。


```
def print_centered(msg: str) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    print(spaces + msg)


def print_header(title: str) -> None:
    print_centered(title)
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print()
    print()
    print()


def print_ticks() -> None:
    for _ in range(4):
        print("-")


```

这两函数的主要目的是打印一个街名和一条街道的信息。

第一个函数 `print_street` 接收一个整数参数 `i`，然后打印出一个街名，其中 `i` 是多少。街名是由变量 `street_number` 计算出来的，根据这个数字，打印出相应的街名。打印中使用了 `print_ruler` 中定义的 `line` 变量，这个变量未来还会被使用。

第二个函数 `print_street` 接收一个整数参数 `i`，然后打印出一个街名，其中 `i` 是多少。街名是由变量 `street_number` 计算出来的，根据这个数字，打印出相应的街名。打印中使用了 `print_ruler` 中定义的 `line` 变量，这个变量未来还会被使用。


```
def print_ruler() -> None:
    print(" -----1-----2-----3-----4-----")


def print_street(i: int) -> None:
    street_number = 3 - i

    street_name = street_names[street_number]
    line = street_name

    space = " " * 5
    for customer_index in range(4):
        line += space
        customer_name = customer_names[4 * street_number + customer_index]
        line += customer_name
    line += space
    line += street_name
    print(line)


```

这道题目是一个Python编写的函数，它的目的是让玩家输入他们的名字并告知他们在这个Pizzadeliverygame中的任务。一旦玩家输入了他们的名字，程序将打印出包含他们名字的地图。接下来，程序将打印出一些提示消息，告诉玩家他们的任务是在城市中为他们的订单送货。最后，程序将输出一个包含玩家名字的地图，以及一个地图，其中包含城市中所有已知的房屋位置。

总之，这段代码的主要目的是让玩家输入他们的名字并告诉他们在这个游戏中的任务，然后让他们开始工作。


```
def print_map() -> None:
    print("MAP OF THE CITY OF HYATTSVILLE")
    print()
    print_ruler()
    for i in range(4):
        print_ticks()
        print_street(i)
    print_ticks()
    print_ruler()
    print()


def print_instructions() -> str:
    print("PIZZA DELIVERY GAME")
    print()
    print("WHAT IS YOUR FIRST NAME?")
    player_name = input()
    print()
    print(f"HI, {player_name}.  IN THIS GAME YOU ARE TO TAKE ORDERS")
    print("FOR PIZZAS.  THEN YOU ARE TO TELL A DELIVERY BOY")
    print("WHERE TO DELIVER THE ORDERED PIZZAS.")
    print()
    print()

    print_map()

    print("THE OUTPUT IS A MAP OF THE HOMES WHERE")
    print("YOU ARE TO SEND PIZZAS.")
    print()
    print("YOUR JOB IS TO GIVE A TRUCK DRIVER")
    print("THE LOCATION OR COORDINATES OF THE")
    print("HOME ORDERING THE PIZZA.")
    print()

    return player_name


```

这两段代码是在创建一个名为“yes_no_prompt”的函数和一个名为“print_more_directions”的函数。

函数yes_no_prompt的作用是询问用户一个“YES”或“NO”的问题，并在得到用户输入后根据用户回答的结果返回一个布尔值(True或False)。

函数print_more_directions的作用是在玩家到达指定位置时，向玩家询问他们到达该位置的交通工具，如“出租车”或“自行车”。它将提示玩家两次，然后根据玩家的回答提示他们输入一个数字(1或2)，表示他们到达的位置是第几栋房子。最后，它将向玩家提供两个选项，告诉玩家他们应该往东还是往西。


```
def yes_no_prompt(msg: str) -> bool:
    while True:
        print(msg)
        response = input().upper()

        if response == "YES":
            return True
        elif response == "NO":
            return False
        print("'YES' OR 'NO' PLEASE, NOW THEN,")


def print_more_directions(player_name: str) -> None:
    print()
    print("SOMEBODY WILL ASK FOR A PIZZA TO BE")
    print("DELIVERED.  THEN A DELIVERY BOY WILL")
    print("ASK YOU FOR THE LOCATION.")
    print("     EXAMPLE:")
    print("THIS IS J.  PLEASE SEND A PIZZA.")
    print(f"DRIVER TO {player_name}.  WHERE DOES J LIVE?")
    print("YOUR ANSWER WOULD BE 2,3")
    print()


```

这段代码定义了两个函数，分别是 `calculate_customer_index` 和 `deliver_to`。这两个函数的功能如下：

1. `calculate_customer_index(x: int, y: int) -> int` - 函数用于计算一个定制顾客的编号。传递给这个函数的参数 `x` 和 `y` 是整数，它们将作为输入参数。函数返回一个整数，代表定制顾客的编号。

2. `deliver_to(customer_index, customer_name, player_name) -> bool` - 函数用于将货物从司机处送到指定的顾客处。传递给这个函数的参数 `customer_index` 是定制顾客的编号，`customer_name` 是顾客的名字，`player_name` 是球员的名字。函数返回一个布尔值，表示是否成功将货物送到指定的顾客处。

函数 `deliver_to` 的实现如下：

```python
def deliver_to(customer_index, customer_name, player_name) -> bool:
   print(f"  DRIVER TO {player_name}:  WHERE DOES {customer_name} LIVE?")

   coords = input()
   xc, yc = (int(c) for c in coords.split(","))
   delivery_index = calculate_customer_index(xc, yc)
   if delivery_index == customer_index:
       print(f"HELLO {player_name}.  THIS IS {customer_name}, THANKS FOR THE PIZZA.")
       return True
   else:
       delivery_name = customer_names[delivery_index]
       print(f"THIS IS {delivery_name}.  I DID NOT ORDER A PIZZA.")
       print(f"I LIVE AT {xc},{yc}")
       return False
```

函数 `calculate_customer_index` 的作用是计算一个定制顾客的编号，这个编号与给定的 `x` 和 `y` 值有关。函数的实现是通过 `x` 和 `y` 变量来获取给定的 `x` 和 `y` 值，然后使用这两个值来计算一个与 `customer_index` 相关的值，最后将计算出的值作为函数的返回值。

函数 `deliver_to` 的作用是将货物从司机处送到指定的顾客处，这个函数接收一个定制顾客的编号、顾客的名字和球员的名字作为参数，然后将这些信息用于计算，最后将计算出的结果作为函数的返回值。函数会尝试将货物送到指定的顾客处，如果成功，则返回 `True`，否则返回 `False`。


```
def calculate_customer_index(x: int, y: int) -> int:
    return 4 * (y - 1) + x - 1


def deliver_to(customer_index, customer_name, player_name) -> bool:
    print(f"  DRIVER TO {player_name}:  WHERE DOES {customer_name} LIVE?")

    coords = input()
    xc, yc = (int(c) for c in coords.split(","))
    delivery_index = calculate_customer_index(xc, yc)
    if delivery_index == customer_index:
        print(f"HELLO {player_name}.  THIS IS {customer_name}, THANKS FOR THE PIZZA.")
        return True
    else:
        delivery_name = customer_names[delivery_index]
        print(f"THIS IS {delivery_name}.  I DID NOT ORDER A PIZZA.")
        print(f"I LIVE AT {xc},{yc}")
        return False


```

这段代码定义了一个名为 `play_game` 的函数，它接受两个参数 `num_turns` 和 `player_name`。函数内部进行了一系列的循环操作，用于生成随机的客户订单号和客户姓名。然后，它根据客户订单号在 `customer_names` 字典中查找客户姓名，并输出确认消息，询问客户是否要下订单。如果客户下订单，函数会递归调用 `deliver_to` 函数将订单发送给客户。在循环体内，若客户未能下订单，则可能会继续生成新的客户订单号，并重复此过程。

虽然函数内部没有输出代码，但你可以根据需要自行添加输出语句，例如将结果输出到文件或屏幕上。


```
def play_game(num_turns, player_name) -> None:
    for _turn in range(num_turns):
        x = random.randint(1, 4)
        y = random.randint(1, 4)
        customer_index = calculate_customer_index(x, y)
        customer_name = customer_names[customer_index]

        print()
        print(
            f"HELLO {player_name}'S PIZZA.  THIS IS {customer_name}.  PLEASE SEND A PIZZA."
        )
        while True:
            success = deliver_to(customer_index, customer_name, player_name)
            if success:
                break


```

这段代码是一个Python程序，名为“main”。程序的主要目的是让用户决定是否要接收更多的订单，并按照他们的意愿进行游戏。

以下是程序的详细解释：

1. 首先定义了一个名为“main”的函数，它接受一个空括号“()”作为参数，这意味着这个函数不会返回任何值。

2. 在函数内部，首先调用了一个名为“print_header”的函数，这个函数的作用是输出一个“PIZZA”字符串，作为游戏开始前的提示。

3. 然后，程序调用了名为“print_instructions”的函数，这个函数的作用是询问玩家是否需要更多的指导。

4. 接下来，程序调用了一个名为“yes_no_prompt”的函数，这个函数接受两个参数：一个是要询问用户的问题，另一个是一个布尔值，表示回答是True还是False。

5. 如果需要更多的指导，程序将调用名为“print_more_directions”的函数，并将“player_name”参数作为第一个参数。

6. 如果不需要更多的指导，程序将调用名为“yes_no_prompt”的函数，并无论如何都会调用它。第一个参数将包含一个提示，表示游戏是否很难。

7. 然后，程序打印“GOOD. YOU ARE NOW READY TO START TAKING ORDERS.”，表示游戏即将开始。

8. 接着，程序打印“GOOD LUNCH”，表示开始向用户接收订单。

9. 在游戏的正式部分，程序将循环5次，每次循环将调用一个名为“play_game”的函数，并将“num_turns”和“player_name”作为参数。

10. 循环内部，程序将打印提示信息，然后调用一个名为“print_request”的函数，并将不同的参数（数字游戏轮数，玩家名字等）作为第一个和第二个参数。

11. 如果玩家需要更多的指导，程序将调用名为“print_more_directions”的函数，并将“player_name”参数作为第一个参数。

12. 如果不需要更多的指导，程序将调用“yes_no_prompt”的函数，并将“player_name”参数作为第一个参数。

13. 程序将在每次循环结束后，打印“YES. YOUR ORDER IS READY TO collect.”，表示用户已准备好接收订单。

14. 最后，程序将打印“GOOD. INTO意想不到的世界......”，表示程序将随机选择一个方向。


```
def main() -> None:
    print_header("PIZZA")

    player_name = print_instructions()

    more_directions = yes_no_prompt("DO YOU NEED MORE DIRECTIONS?")

    if more_directions:
        print_more_directions(player_name)

        understand = yes_no_prompt("UNDERSTAND?")

        if not understand:
            print("THIS JOB IS DEFINITELY TOO DIFFICULT FOR YOU. THANKS ANYWAY")
            return

    print("GOOD.  YOU ARE NOW READY TO START TAKING ORDERS.")
    print()
    print("GOOD LUCK!!")
    print()

    while True:
        num_turns = 5
        play_game(num_turns, player_name)

        print()
        more = yes_no_prompt("DO YOU WANT TO DELIVER MORE PIZZAS?")
        if not more:
            print(f"O.K. {player_name}, SEE YOU LATER!")
            print()
            return


```

这段代码是一个if语句，它会判断当前脚本是否作为主程序运行。如果脚本作为主程序运行，则会执行if语句中的代码。

在if语句中，代码会块内写入了main()函数。main()函数是Python中的一个标准函数，它会在脚本作为主程序运行时被调用，从而使整个脚本可以被正确地执行。

因此，这段代码的作用是检查当前脚本是否作为主程序运行，如果是，就执行if语句中的代码，即调用main()函数使脚本正确地执行。


```
if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/).


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Poetry

This program produces random verse which might loosely be considered in the Japanese Haiku style. It uses 20 phrases in four groups of five phrases each and generally cycles through the groups in order. It inserts commas (random — 19% of the time), indentation (random — 22% of the time), and starts new paragraphs (18% probability but at least once every 20 phrases).

The phrases in POETRY are somewhat suggestive of Edgar Allen Poe. Try it with phrases from computer technology, from love and romance, from four-year-old children, or from some other project. Send us the output.

Here are some phrases from nature to try:
```
Carpet of ferns     Mighty Oaks
Morning dew         Grace and beauty
Tang of dawn        Silently singing
Swaying pines       Nature speaking

Entrances me        Untouched, unspoiled
Soothing me         Shades of green
Rustling leaves     Tranquility
Radiates calm       …so peaceful
```

The original author of this program is unknown. It was modified and reworked by Jim Bailey, Peggy Ewing, and Dave Ahl at DEC.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=128)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=143)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

- The program begins by switching on `I`, which has not been initialized.  We should probably initialize this to 0, though this means the output always begins with the phrase "midnight dreary".

- Though the program contains an END statement (line 999), it is unreachable.  The program continues to generate output until it is forcibly interrupted.


# `70_Poetry/csharp/Context.cs`

This is a code snippet written in a language that uses the ESFAs with拓展 character classes.

The `Phrase` class is a POI (Position Independent Information) class that holds information about a paragraph, including its text and associated metadata.

The `WritePhrase` method is a method for the `Phrase` class that writes the paragraph's text to the `_io` object and sets the `_atStartOfLine` flag to indicate whether the paragraph's text starts on the first character.

The `MaybeWriteComma` method is a method for the `Phrase` class that checks whether the paragraph is already at the end of its text and whether it contains a comma or contains multiple commas. If it contains a comma or multiple commas, the method writes a comma to the `_io` object and sets the `_atStartOfLine` flag to indicate that the text does start on the first character. If the paragraph does not contain a comma or multiple commas, the method ends the line and resets the `_atStartOfLine` flag.

The `WriteSpaceOrNewLine` method is a method for the `Phrase` class that writes a space or a new line to the `_io` object based on the random number generated by the `Random` class.

The `Update` method is a method for the `Phrase` class that updates its internal state by incrementing the `_phraseNumber`, `_groupNumber`, and `_lineCount` properties.

The `MaybeIndent` method is a method for the `Phrase` class that adds indentation to the paragraph's text based on the current value of `_groupNumber`.

The `ResetGroup` method is a method for the `Phrase` class that resets its internal state by resetting the `_groupNumber` property to 0 and ending the current line.

The `MaybeCompleteStanza` method is a method for the `Phrase` class that checks whether the paragraph is already complete. If the line count is greater than 20, the method writes a new line to the `_io` object and sets the `_atStartOfLine` flag to indicate that the text does start on the first character. If the line count is less than or equal to 20, the method returns the previous result.


```
namespace Poetry;

internal class Context
{
    private readonly IReadWrite _io;
    private readonly IRandom _random;
    private int _phraseNumber;
    private int _groupNumber;
    private bool _skipComma;
    private int _lineCount;
    private bool _useGroup2;
    private bool _atStartOfLine = true;

    public Context(IReadWrite io, IRandom random)
    {
        _io = io;
        _random = random;
    }

    public int PhraseNumber => Math.Max(_phraseNumber - 1, 0); 

    public int GroupNumber 
    { 
        get
        {
            var value = _useGroup2 ? 2 : _groupNumber;
            _useGroup2 = false;
            return Math.Max(value - 1, 0);
        }
    }

    public int PhraseCount { get; set; }
    public bool GroupNumberIsValid => _groupNumber < 5;

    public void WritePhrase()
    {
        Phrase.GetPhrase(this).Write(_io, this);
        _atStartOfLine = false;
    }

    public void MaybeWriteComma()
    {
        if (!_skipComma && _random.NextFloat() <= 0.19F && PhraseCount != 0)
        {
            _io.Write(",");
            PhraseCount = 2;
        }
        _skipComma = false;
    }

    public void WriteSpaceOrNewLine()
    {
        if (_random.NextFloat() <= 0.65F)
        {
            _io.Write(" ");
            PhraseCount += 1;
        }
        else
        {
            EndLine();
            PhraseCount = 0;
        }
    }

    public void Update(IRandom random)
    {
        _phraseNumber = random.Next(1, 6);
        _groupNumber += 1;
        _lineCount += 1;
    }

    public void MaybeIndent()
    {
        if (PhraseCount == 0 && _groupNumber % 2 == 0)
        {
            _io.Write("     ");
        }
    }
    
    public void ResetGroup()
    {
        _groupNumber = 0;
        EndLine();
    }

    public bool MaybeCompleteStanza()
    {
        if (_lineCount > 20)
        {
            _io.WriteLine();
            PhraseCount = _lineCount = 0;
            _useGroup2 = true;
            return true;
        }

        return false;
    }

    internal string MaybeCapitalise(string text) =>
        _atStartOfLine ? (char.ToUpper(text[0]) + text[1..]) : text;

    public void SkipNextComma() => _skipComma = true;

    public void EndLine()
    {
        _io.WriteLine();
        _atStartOfLine = true;
    }
}

```

# `70_Poetry/csharp/Phrase.cs`

This is a Goan language prophète that defines a set of phrasings with different meanings and interpretations. These phrasings are used to describe various situations and events, and the meanings of the phrasings can vary depending on the context in which they are used.

The phrasings are grouped together into different sections, such as " thing of evil " and " prophet " , and each section has its own set of phrasings. For example, the section " thing of evil " has the following phrasings: " beguiling me", " thrilled me", " still sitting....", " never flitting", and " burned ".

The phrasings are predicated on a given context, which means that they can only be invoked if the context passes a certain criteria. The predicate is defined using the `Predicate` interface, which specifies the type of the predicate, the type of the argument, and the type of the return type.

The `Phrase` class represents a single phraction, which is either a new Phrase constructor that takes a predicate and a text parameter and invokes the predicate on a given context, or a `Phrase` instance that takes a predicate and an action parameter and invokes the action on the given context.

The `GetPhrase` method returns a `Phrase` instance with the given predicate and text, while the `Write` method writes the given text to the input stream.


```
namespace Poetry;

internal class Phrase
{
    private readonly static Phrase[][] _phrases = new Phrase[][]
    {
        new Phrase[]
        {
            new("midnight dreary"),
            new("fiery eyes"),
            new("bird or fiend"),
            new("thing of evil"),
            new("prophet")
        },
        new Phrase[]
        {
            new("beguiling me", ctx => ctx.PhraseCount = 2),
            new("thrilled me"),
            new("still sitting....", ctx => ctx.SkipNextComma()),
            new("never flitting", ctx => ctx.PhraseCount = 2),
            new("burned")
        },
        new Phrase[]
        {
            new("and my soul"),
            new("darkness there"),
            new("shall be lifted"),
            new("quoth the raven"),
            new(ctx => ctx.PhraseCount != 0, "sign of parting")
        },
        new Phrase[]
        {
            new("nothing more"),
            new("yet again"),
            new("slowly creeping"),
            new("...evermore"),
            new("nevermore")
        }
    };

    private readonly Predicate<Context> _condition;
    private readonly string _text;
    private readonly Action<Context> _update;

    private Phrase(Predicate<Context> condition, string text)
        : this(condition, text, _ => { })
    {
    }

    private Phrase(string text, Action<Context> update)
        : this(_ => true, text, update)
    {
    }

    private Phrase(string text)
        : this(_ => true, text, _ => { })
    {
    }

    private Phrase(Predicate<Context> condition, string text, Action<Context> update)
    {
        _condition = condition;
        _text = text;
        _update = update;
    }

    public static Phrase GetPhrase(Context context) => _phrases[context.GroupNumber][context.PhraseNumber];

    public void Write(IReadWrite io, Context context)
    {
        if (_condition.Invoke(context))
        {
            io.Write(context.MaybeCapitalise(_text));
        }

        _update.Invoke(context);
    }
}
```

# `70_Poetry/csharp/Poem.cs`



该代码是一个名为 `Poem` 的类，它实现了 `Compose` 方法，该方法接受两个参数 `io` 和 `random`，它们都是 `IReadWrite` 和 `IRandom` 接口的实例。

`Poem.Compose` 方法的主要目的是在两个循环中生成诗歌行。具体来说，它首先在 `io` 中写入标题，然后在 `random` 的作用下生成诗句。接着，它创建了一个 `Context` 对象，该对象用于管理诗句的生成过程。

在生成诗句的过程中，`Poem.Compose` 方法首先在不断地循环中生成短语。如果需要，它会生成逗号。然后，它会尝试在当前诗句中添加一个新单词，或者在句子末尾添加一个新的空行。

此外，`Poem.Compose` 方法还包含一个 `while` 循环，该循环用于处理诗句的编写。在每次循环中，它使用 `context.WritePhrase()` 方法在 `io` 中写入一行诗句，然后使用 `context.MaybeWriteComma()` 方法在诗句中是否包含逗号。如果包含逗号，则使用 `context.WriteSpaceOrNewLine()` 方法在 `io` 中写入一个空行。在循环的每次迭代中，它还使用 `context.Update(random)` 方法更新诗句的生成概率，并使用 `context.MaybeIndent()` 方法在诗句中是否包含新单词。如果包含新单词，则使用 `context.ResetGroup()` 方法重置诗句的组别。

最后，如果 `context.MaybeCompleteStanza()` 方法在本次循环中没有检测到诗句的结尾，则会发生什么呢？它会继续生成诗句，直到检测到一个新的 stanza。一旦检测到新单词，该方法就会跳出循环，并在生成诗句的顶部添加一个新的空行。


```
using static Poetry.Resources.Resource;

namespace Poetry;

internal class Poem
{
    internal static void Compose(IReadWrite io, IRandom random)
    {
        io.Write(Streams.Title);

        var context = new Context(io, random);

        while (true)
        {
            context.WritePhrase();
            context.MaybeWriteComma();
            context.WriteSpaceOrNewLine();

            while (true)
            {
                context.Update(random);
                context.MaybeIndent();

                if (context.GroupNumberIsValid) { break; }

                context.ResetGroup();

                if (context.MaybeCompleteStanza()) { break; }
            }
        }
    }
}
```

# `70_Poetry/csharp/Program.cs`

这段代码的作用是：

1. 全局引入了三个库：Games.Common.IO、Games.Common.Randomness 和 Poetry。
2. 通过 using 关键字引入了这三个库的常用类和接口。
3. 通过 Composite 方法全局地将它们组合在一起，并暴露出其接口。
4. 创建了一个 ConsoleIO 类的实例，和一个 RandomNumberGenerator 类的实例，并将它们作为参数传递给 Composite 的构造函数。
5. 通过调用 Composite 的方法，将两个随机数生成器实例合并在一起，形成一个新的 instance，从而实现将两个随机数生成器组合在一起使用。


```
global using Games.Common.IO;
global using Games.Common.Randomness;
global using Poetry;

Poem.Compose(new ConsoleIO(), new RandomNumberGenerator());

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `70_Poetry/csharp/Resources/Resource.cs`



这段代码是一个自定义的 .NET 类，名为 `Resource`。它从 `System.Reflection` 和 `System.Runtime.CompilerServices` 包中获取类和接口的引用，定义了一个名为 `Streams` 的内部类，以及一个名为 `Title` 的静态内部类。

`Streams` 类包含一个名为 `GetStream` 的方法，这个方法返回一个 `Stream` 类型的实例，它从嵌入在资源文件中的命名空间中获取内容。这里使用了一个 CallerMemberName 特性，以便在需要时获取声明时信息。

`GetStream` 方法使用 `Assembly.GetExecutingAssembly().GetManifestResourceStream` 方法从包含资源文件的程序集获取资源文件的内容。如果资源文件不存在，它将抛出异常。

最后，这个类的实例被声明为 `public static class Streams` 类。这个类的实例在整个程序集引用中可见，因此可以在程序的任何地方使用它。


```
using System.Reflection;
using System.Runtime.CompilerServices;

namespace Poetry.Resources;

internal static class Resource
{
    internal static class Streams
    {
        public static Stream Title => GetStream();
    }

    private static Stream GetStream([CallerMemberName] string? name = null) =>
        Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
            ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
}
```

# `70_Poetry/java/Poetry.java`

This looks like a Java class called `Poetry`, which defines a method `startGame()`. `startGame()` takes no arguments and returns no value, but it generates some random punctuation words and prints them in a group according to a predetermined pattern. Here's a summary of the code:

1. The class `Poetry` has a static method `main()` which serves as the entry point for the program.
2. Inside the `main()` method, there's a new `Poetry` object called `poetry`.
3. The `poetry` object has a `play()` method, which probably sends the contents of the `Poetry` object to the console or terminal.

The `startGame()` method is not doing anything in particular, but it generates some random punctuation words according to the specified pattern, and prints them in a group. The pattern appears to be based on the positions of the punctuation marks.


```
/**
 * Game of Poetry
 * <p>
 * Based on the BASIC game of Poetry here
 * https://github.com/coding-horror/basic-computer-games/blob/main/70%20Poetry/poetry.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

public class Poetry {

  private final static double COMMA_RATE = 0.19;
  private final static double SPACE_RATE = 0.65;
  private final static int PARAGRAPH_RATE = 20;

  private enum Step {
    WORD_GROUP1, WORD_GROUP2, WORD_GROUP3, WORD_GROUP4, RANDOMIZE_COMMA,
    RANDOMIZE_WHITESPACE, RANDOMIZE_COUNTERS
  }

  public void play() {

    showIntro();
    startGame();

  }  // End of method play

  private void showIntro() {

    System.out.println(" ".repeat(29) + "POETRY");
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    System.out.println("\n\n");

  }  // End of method showIntro

  private void startGame() {

    int groupIndex = 0;
    int paragraphIndex = 0;
    int punctuationIndex = 0;
    int wordIndex = 1;

    Step nextStep = Step.WORD_GROUP1;

    // Begin outer while loop
    while (true) {

      switch (nextStep) {

        case WORD_GROUP1:

          if (wordIndex == 1) {

            System.out.print("MIDNIGHT DREARY");
            nextStep = Step.RANDOMIZE_COMMA;

          } else if (wordIndex == 2) {

            System.out.print("FIERY EYES");
            nextStep = Step.RANDOMIZE_COMMA;

          } else if (wordIndex == 3) {

            System.out.print("BIRD OR FIEND");
            nextStep = Step.RANDOMIZE_COMMA;

          } else if (wordIndex == 4) {

            System.out.print("THING OF EVIL");
            nextStep = Step.RANDOMIZE_COMMA;

          } else if (wordIndex == 5) {

            System.out.print("PROPHET");
            nextStep = Step.RANDOMIZE_COMMA;
          }
          break;

        case WORD_GROUP2:

          if (wordIndex == 1) {

            System.out.print("BEGUILING ME");
            nextStep = Step.RANDOMIZE_COMMA;

          } else if (wordIndex == 2) {

            System.out.print("THRILLED ME");
            nextStep = Step.RANDOMIZE_COMMA;

          } else if (wordIndex == 3) {

            System.out.print("STILL SITTING....");
            nextStep = Step.RANDOMIZE_WHITESPACE;

          } else if (wordIndex == 4) {

            System.out.print("NEVER FLITTING");
            nextStep = Step.RANDOMIZE_COMMA;

          } else if (wordIndex == 5) {

            System.out.print("BURNED");
            nextStep = Step.RANDOMIZE_COMMA;
          }
          break;

        case WORD_GROUP3:

          if (wordIndex == 1) {

            System.out.print("AND MY SOUL");
            nextStep = Step.RANDOMIZE_COMMA;

          } else if (wordIndex == 2) {

            System.out.print("DARKNESS THERE");
            nextStep = Step.RANDOMIZE_COMMA;

          } else if (wordIndex == 3) {

            System.out.print("SHALL BE LIFTED");
            nextStep = Step.RANDOMIZE_COMMA;

          } else if (wordIndex == 4) {

            System.out.print("QUOTH THE RAVEN");
            nextStep = Step.RANDOMIZE_COMMA;

          } else if (wordIndex == 5) {

            if (punctuationIndex != 0) {

              System.out.print("SIGN OF PARTING");
            }

            nextStep = Step.RANDOMIZE_COMMA;
          }
          break;

        case WORD_GROUP4:

          if (wordIndex == 1) {

            System.out.print("NOTHING MORE");
            nextStep = Step.RANDOMIZE_COMMA;

          } else if (wordIndex == 2) {

            System.out.print("YET AGAIN");
            nextStep = Step.RANDOMIZE_COMMA;

          } else if (wordIndex == 3) {

            System.out.print("SLOWLY CREEPING");
            nextStep = Step.RANDOMIZE_WHITESPACE;

          } else if (wordIndex == 4) {

            System.out.print("...EVERMORE");
            nextStep = Step.RANDOMIZE_COMMA;

          } else if (wordIndex == 5) {

            System.out.print("NEVERMORE");
            nextStep = Step.RANDOMIZE_COMMA;
          }
          break;

        case RANDOMIZE_COMMA:

          // Insert commas
          if ((punctuationIndex != 0) && (Math.random() <= COMMA_RATE)) {

            System.out.print(",");
            punctuationIndex = 2;
          }
          nextStep = Step.RANDOMIZE_WHITESPACE;
          break;


        case RANDOMIZE_WHITESPACE:

          // Insert spaces
          if (Math.random() <= SPACE_RATE) {

            System.out.print(" ");
            punctuationIndex++;

          }
          // Insert newlines
          else {

            System.out.println("");
            punctuationIndex = 0;
          }
          nextStep = Step.RANDOMIZE_COUNTERS;
          break;

        case RANDOMIZE_COUNTERS:

          wordIndex = (int)((int)(10 * Math.random()) / 2) + 1;

          groupIndex++;
          paragraphIndex++;

          if ((punctuationIndex == 0) && (groupIndex % 2 == 0)) {

            System.out.print("     ");
          }

          if (groupIndex == 1) {

            nextStep = Step.WORD_GROUP1;

          } else if (groupIndex == 2) {

            nextStep = Step.WORD_GROUP2;

          } else if (groupIndex == 3) {

            nextStep = Step.WORD_GROUP3;

          } else if (groupIndex == 4) {

            nextStep = Step.WORD_GROUP4;

          } else if (groupIndex == 5) {

            groupIndex = 0;
            System.out.println("");

            if (paragraphIndex > PARAGRAPH_RATE) {

              System.out.println("");
              punctuationIndex = 0;
              paragraphIndex = 0;
              nextStep = Step.WORD_GROUP2;

            } else {

              nextStep = Step.RANDOMIZE_COUNTERS;
            }
          }
          break;

        default:
          System.out.println("INVALID STEP");
          break;
      }

    }  // End outer while loop

  }  // End of method startGame

  public static void main(String[] args) {

    Poetry poetry = new Poetry();
    poetry.play();

  }  // End of method main

}  // End of class Poetry

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `70_Poetry/javascript/poetry.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

`print` 函数的作用是在文档中创建一个 `textarea` 元素，然后将其内容插入到指定的 `id` 为 `output` 的元素中。

`input` 函数的作用是从用户那里获取一个字符串，然后将其存储在变量 `input_str` 中。该函数通过使用 `document.createElement("INPUT")` 创建一个 `textarea` 元素，并设置其 `type` 属性为 `text` 和 `length` 属性为 `50`。然后，该函数将 `input_str` 存储在变量中，并使用 `document.getElementById("output").appendChild(input_element)` 将 `textarea` 元素添加到文档中指定的位置。接着，该函数将 `input_str` 的内容插入到 `output` 元素中，并使用 `print(input_str)` 将内容输出到页面上。在 `input` 函数中，还添加了一个事件监听器来监听 `keydown` 事件，当用户按下 `13` 时，将 `input_str` 的值存储在变量中，并使用 `print(input_str)` 将内容输出到页面上，并使用 `print("\n")` 打印一个换行符。


```
// POETRY
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

This appears to be a program that generates random text based on a given category and number of cases. The text is generated using a series of if-else statements that check the current category and case, and outputs相应的 text. The text is generated in the form of multiple lines, with each line being either a single word or a phrase followed by a space. The program also includes a randomizing factor that changes each time the program is run.


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// Main program
async function main()
{
    print(tab(30) + "POETRY\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");

    times = 0;

    i = 1;
    j = 1;
    k = 0;
    u = 0;
    while (1) {
        if (j == 1) {
            switch (i) {
                case 1:
                    print("MIDNIGHT DREARY");
                    break;
                case 2:
                    print("FIERY EYES");
                    break;
                case 3:
                    print("BIRD OF FIEND");
                    break;
                case 4:
                    print("THING OF EVIL");
                    break;
                case 5:
                    print("PROPHET");
                    break;
            }
        } else if (j == 2) {
            switch (i) {
                case 1:
                    print("BEGUILING ME");
                    u = 2;
                    break;
                case 2:
                    print("THRILLED ME");
                    break;
                case 3:
                    print("STILL SITTING....");
                    u = 0;
                    break;
                case 4:
                    print("NEVER FLITTING");
                    u = 2;
                    break;
                case 5:
                    print("BURNED");
                    break;
            }
        } else if (j == 3) {
            switch (i) {
                case 1:
                    print("AND MY SOUL");
                    break;
                case 2:
                    print("DARKNESS THERE");
                    break;
                case 3:
                    print("SHALL BE LIFTED");
                    break;
                case 4:
                    print("QUOTH THE RAVEN");
                    break;
                case 5:
                    if (u == 0)
                        break;
                    print("SIGN OF PARTING");
                    break;
            }
        } else if (j == 4) {
            switch (i) {
                case 1:
                    print("NOTHING MORE");
                    break;
                case 2:
                    print("YET AGAIN");
                    break;
                case 3:
                    print("SLOWLY CREEPING");
                    break;
                case 4:
                    print("...EVERMORE");
                    break;
                case 5:
                    print("NEVERMORE");
                    break;
            }
        }
        if (u != 0 && Math.random() <= 0.19) {
            print(",");
            u = 2;
        }
        if (Math.random() <= 0.65) {
            print(" ");
            u++;
        } else {
            print("\n");
            u = 0;
        }
        while (1) {
            i = Math.floor(Math.floor(10 * Math.random()) / 2) + 1;
            j++;
            k++;
            if (u == 0 && j % 2 == 0)
                print("     ");
            if (j != 5)
                break;
            j = 0;
            print("\n");
            if (k <= 20)
                continue;
            print("\n");
            u = 0;
            k = 0;
            j = 2;
            break;
        }
        if (u == 0 && k == 0 && j == 2 && ++times == 10)
            break;
    }
}

```

这是C++中的一个标准库函数，名为“main()”。在C++程序中，当需要运行程序时，通常会将所有代码都保存在一个名为“main”的函数中。

当创建一个名为“main”的函数时，它会在程序启动时首先执行。因此，在“main()”函数内，程序会将所有代码块的代码全部执行一次。这包括函数内外的代码，但不会输出任何函数或返回任何值。

“main()”函数通常是C++程序的入口点。当程序运行完毕时，操作系统会将程序的所有权限恢复，并允许用户在退出程序之前进行任何更改。


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


# `70_Poetry/python/poetry.py`

这段代码定义了一个名为 `POperation` 的类，它是一个诗歌生成器。在这个类中，定义了一个名为 `generate_poetry` 的方法，这个方法使用 `random` 模块从类中的诗歌元素中产生随机的押韵和韵脚。

此外，定义了一个名为 `P slam poetry` 的类，这个类继承自 `POperation` 类，重写了 `generate_slam_poetry` 方法，使得这个方法在开始时直接产生 Slam poetry，而不是随机产生。


```
"""
POETRY

A poetry generator

Ported by Dave LeCompte
"""

import random
from dataclasses import dataclass

PAGE_WIDTH = 64


@dataclass
```

这段代码定义了一个名为State的类，该类包含以下成员变量：

- u: 整数，初始值为0
- i: 整数，初始值为0
- j: 整数，初始值为0
- k: 整数，初始值为0
- phrase: 整数，初始值为1
- line: 字符串，初始值为空字符串

该代码还定义了一个名为print_centered的函数，该函数接受一个字符串参数msg，并将其打印成居中格式。

- process_phrase_1函数，该函数接受State类型的实例作为参数。该函数使用State实例中的line和i成员变量，以及line和i成员变量上的索引，来打印出一个新的字符串。然后，它将返回该新字符串。由于该函数没有明确的返回类型，因此它不会在函数中被调用，也不会返回任何值。


```
class State:
    u: int = 0
    i: int = 0
    j: int = 0
    k: int = 0
    phrase: int = 1
    line: str = ""


def print_centered(msg: str) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    print(spaces + msg)


def process_phrase_1(state: State) -> str:
    line_1_options = [
        "MIDNIGHT DREARY",
        "FIERY EYES",
        "BIRD OR FIEND",
        "THING OF EVIL",
        "PROPHET",
    ]
    state.line = state.line + line_1_options[state.i]
    return state.line


```

这两函数的主要作用是处理文本中的标点符号和数字，并根据给定的上下文修改句子中词语的权重。

1. `process_phrase_2`函数接受一个状态对象（State），并处理句子中的第二个参数（即`words`）。在这个函数中，首先定义了一个可选项的列表（options）`line_2_options`，它包含了多个选项，如`BEGUILING ME`，`THRILLED ME`，`STILL SITTING....`和`NEVER FLITTING`。然后，从`line_2_options`中选择相应选项中的一项，并将其添加到状态中的句子中。接着，如果选项中的一项不是`None`，那么将其值赋给状态中的`u`变量。

2. `process_phrase_3`函数同样接受一个状态对象（State），并处理句子中提到的所有词语。在这个函数中，首先定义了一个包含多个选项的列表（phrases）。然后，遍历phrases列表，获取给定句子的选项，并将其值存储在状态中的句子中。接下来，根据给定的上下文，只有当`only_if_u`为`False`，且`state.u`的值大于0时，才会更新状态中的句子。


```
def process_phrase_2(state: State) -> None:
    line_2_options = [
        ("BEGUILING ME", 2),
        ("THRILLED ME", None),
        ("STILL SITTING....", None),
        ("NEVER FLITTING", 2),
        ("BURNED", None),
    ]
    words, u_modifier = line_2_options[state.i]
    state.line += words
    if not (u_modifier is None):
        state.u = u_modifier


def process_phrase_3(state: State) -> None:
    phrases = [
        (False, "AND MY SOUL"),
        (False, "DARKNESS THERE"),
        (False, "SHALL BE LIFTED"),
        (False, "QUOTH THE RAVEN"),
        (True, "SIGN OF PARTING"),
    ]

    only_if_u, words = phrases[state.i]
    if (not only_if_u) or (state.u > 0):
        state.line = state.line + words


```

这两函数是处理文本中的标点符号， `process_phrase_4()` 函数将带有标点符号的字符串存回其原始字符串中，然后在一个新的字符串中添加原始字符串中的标点符号。而 `maybe_comma()` 函数则是尝试在当前字符串中添加逗号或句号，如果添加成功则返回，否则将当前字符串打印并重置为空字符串。


```
def process_phrase_4(state: State) -> None:
    phrases = [
        ("NOTHING MORE"),
        ("YET AGAIN"),
        ("SLOWLY CREEPING"),
        ("...EVERMORE"),
        ("NEVERMORE"),
    ]

    state.line += phrases[state.i]


def maybe_comma(state: State) -> None:
    if len(state.line) > 0 and state.line[-1] == ".":
        # don't follow a period with a comma, ever
        return

    if state.u != 0 and random.random() <= 0.19:
        state.line += ", "
        state.u = 2
    if random.random() <= 0.65:
        state.line += " "
        state.u += 1
    else:
        print(state.line)
        state.line = ""
        state.u = 0


```

这段代码定义了一个名为 `pick_phrase` 的函数，它接受一个 `State` 对象作为参数，并返回 `None`。

`State` 对象包含以下属性：

- `i`: 状态中的项目的编号，从0开始。
- `j`: 状态中的另一个项目的编号，从0开始。
- `k`: 状态中的另一个项目的编号，从0开始。
- `u`: 状态中的一个指示符，决定是否继续添加新项目。
- `line`: 状态中的一个字符串，已经定义了一个循环，每次增加5个空格。
- `phrase`: 状态中的一个整数，递增的编号。

`pick_phrase` 函数的功能是，在给定的 `State` 对象上运行以下操作：

1. 生成一个随机整数，并将其分配给 `i` 属性。
2. 将 `j` 属性加1。
3. 将 `k` 属性加1。
4. 如果 `u` 小于0，并且 `j` 不是0，就在 `line` 上添加5个空格。
5. 将 `phrase` 属性加1。

`main` 函数的作用是，打印出以下内容：

- "POETRY"
- "CREATIVE COMPUTING"
- 5个空格
- 输入输出，依次打印出四个不同的状态。


```
def pick_phrase(state: State) -> None:
    state.i = random.randint(0, 4)
    state.j += 1
    state.k += 1

    if state.u <= 0 and (state.j % 2) != 0:
        # random indentation is fun!
        state.line += " " * 5
    state.phrase = state.j + 1


def main() -> None:
    print_centered("POETRY")
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")

    state = State()

    phrase_processors = {
        1: process_phrase_1,
        2: process_phrase_2,
        3: process_phrase_3,
        4: process_phrase_4,
    }

    while True:
        if state.phrase >= 1 and state.phrase <= 4:
            phrase_processors[state.phrase](state)
            maybe_comma(state)
        elif state.phrase == 5:
            state.j = 0
            print(state.line)
            state.line = ""
            if state.k > 20:
                print()
                state.u = 0
                state.k = 0
            else:
                state.phrase = 2
                continue
        pick_phrase(state)


```

这段代码是一个if语句，它的判断条件是(__name__ == "__main__")。如果这个条件为真，代码块内的内容将被执行。

"__name__"是一个特殊变量，它是用来保存当前程序的名称的。在这段代码中， "__name__"被赋值为 "__main__"，这意味着程序的名称是"__main__"。

if语句会在程序运行时检查(__name__ == "__main__")是否为真。如果为真，代码块内的内容将被执行。如果为假，代码块内的内容将不会被执行。

这段代码的作用是判断当前程序是否为"__main__"，如果是，就执行代码块内的内容，否则不执行。


```
if __name__ == "__main__":
    main()

```