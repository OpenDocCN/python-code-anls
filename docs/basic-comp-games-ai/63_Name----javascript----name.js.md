# `basic-computer-games\63_Name\javascript\name.js`

```

// 定义一个打印函数，将字符串添加到输出元素中
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

                       // 打印提示符
                       print("? ");
                       // 设置输入框类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到输出元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件，当按下回车键时，获取输入值并解析 Promise
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

// 生成指定数量的空格字符串
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
    // 打印名称和信息
    print(tab(34) + "NAME\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("HELLO.\n");
    print("MY NAME IS CREATIVE COMPUTER.\n");
    print("WHAT'S YOUR NAME (FIRST AND LAST)");
    // 获取用户输入的名字
    str = await input();
    l = str.length;
    print("\n");
    print("THANK YOU, ");
    // 倒序打印用户输入的名字
    for (i = l; i >= 1; i--)
        print(str[i - 1]);
    print(".\n");
    print("OOPS!  I GUESS I GOT IT BACKWARDS.  A SMART\n");
    print("COMPUTER LIKE ME SHOULDN'T MAKE A MISTAKE LIKE THAT!\n");
    print("\n");
    print("BUT I JUST NOTICED YOUR LETTERS ARE OUT OF ORDER.\n");
    print("LET'S PUT THEM IN ORDER LIKE THIS: ");
    b = [];
    // 将用户输入的名字转换为 ASCII 码并排序
    for (i = 1; i <= l; i++)
        b[i - 1] = str.charCodeAt(i - 1);
    b.sort();
    for (i = 1; i <= l; i++)
        // 打印排序后的字符
        print(String.fromCharCode(b[i - 1]));
    print("\n");
    print("\n");
    print("DON'T YOU LIKE THAT BETTER");
    // 获取用户输入
    ds = await input();
    if (ds == "YES") {
        print("\n");
        print("I KNEW YOU'D AGREE!!\n");
    } else {
        print("\n");
        print("I'M SORRY YOU DON'T LIKE IT THAT WAY.\n");
    }
    print("\n");
    print("I REALLY ENJOYED MEETING YOU " + str + ".\n");
    print("HAVE A NICE DAY!\n");
}

// 调用主程序
main();

```