# `basic-computer-games\45_Hello\javascript\hello.js`

```

// 定义一个打印函数，用于在页面上输出字符串
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

                       // 在页面上输出提示符
                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件，当按下回车键时，获取输入的值并解析为字符串
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

// 主控制部分，使用 async 函数定义
async function main()
{
    // 在页面上输出欢迎信息
    print(tab(33) + "HELLO\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("HELLO.  MY NAME IS CREATIVE COMPUTER.\n");
    print("\n");
    print("\n");
    print("WHAT'S YOUR NAME");
    // 获取用户输入的名字
    ns = await input();
    print("\n");
    print("HI THERE, " + ns + ", ARE YOU ENJOYING YOURSELF HERE");
    // 循环直到用户输入合法的回答
    while (1) {
        bs = await input();
        print("\n");
        if (bs == "YES") {
            print("I'M GLAD TO HEAR THAT, " + ns + ".\n");
            print("\n");
            break;
        } else if (bs == "NO") {
            print("OH, I'M SORRY TO HEAR THAT, " + ns + ". MAYBE WE CAN\n");
            print("BRIGHTEN UP YOUR VISIT A BIT.\n");
            break;
        } else {
            print("PLEASE ANSWER 'YES' OR 'NO'.  DO YOU LIKE IT HERE");
        }
    }
    // ...（以下部分类似，依次进行用户输入和输出）
}

// 调用主控制部分的函数
main();

```