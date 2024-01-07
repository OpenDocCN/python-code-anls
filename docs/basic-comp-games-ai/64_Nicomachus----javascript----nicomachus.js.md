# `basic-computer-games\64_Nicomachus\javascript\nicomachus.js`

```

// 定义一个打印函数，将字符串输出到指定的元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个输入函数，返回一个 Promise 对象
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       // 创建一个输入框元素
                       input_element = document.createElement("INPUT");

                       // 输出提示符
                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件，当按下回车键时，将输入的值传递给 resolve 函数
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

// 定义一个生成指定数量空格的函数
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var str;
var b;

// 主程序
async function main()
{
    // 输出标题
    print(tab(33) + "NICOMA\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("BOOMERANG PUZZLE FROM ARITHMETICA OF NICOMACHUS -- A.D. 90!\n");
    while (1) {
        print("\n");
        print("PLEASE THINK OF A NUMBER BETWEEN 1 AND 100.\n");
        print("YOUR NUMBER DIVIDED BY 3 HAS A REMAINDER OF");
        // 获取用户输入的值并转换为整数
        a = parseInt(await input());
        print("YOUR NUMBER DIVIDED BY 5 HAS A REMAINDER OF");
        // 获取用户输入的值并转换为整数
        b = parseInt(await input());
        print("YOUR NUMBER DIVIDED BY 7 HAS A REMAINDER OF");
        // 获取用户输入的值并转换为整数
        c = parseInt(await input());
        print("\n");
        print("LET ME THINK A MOMENT...\n");
        print("\n");
        // 根据用户输入的值进行计算
        d = 70 * a + 21 * b + 15 * c;
        while (d > 105)
            d -= 105;
        // 输出计算结果
        print("YOUR NUMBER WAS " + d + ", RIGHT");
        while (1) {
            // 获取用户输入的值
            str = await input();
            print("\n");
            // 根据用户输入的值进行不同的处理
            if (str == "YES") {
                print("HOW ABOUT THAT!!\n");
                break;
            } else if (str == "NO") {
                print("I FEEL YOUR ARITHMETIC IS IN ERROR.\n");
                break;
            } else {
                print("EH?  I DON'T UNDERSTAND '" + str + "'  TRY 'YES' OR 'NO'.\n");
            }
        }
        print("\n");
        print("LET'S TRY ANOTHER.\n");
    }
}

// 调用主程序
main();

```