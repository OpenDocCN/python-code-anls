# `basic-computer-games\58_Love\javascript\love.js`

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
                       // 创建一个输入元素
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       // 设置输入元素的类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入元素添加到输出元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入元素获得焦点
                       input_element.focus();
                       // 初始化输入字符串
                       input_str = undefined;
                       // 监听输入元素的按键事件
                       input_element.addEventListener("keydown", function (event) {
                                                      // 如果按下的是回车键
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 移除输入元素
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      // 打印换行符
                                                      print("\n");
                                                      // 解析 Promise 对象
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义一个制表符函数，返回指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}
// 定义一个包含数字的数组
var data = [60,1,12,26,9,12,3,8,24,17,8,4,6,23,21,6,4,6,22,12,5,6,5,
            4,6,21,11,8,6,4,4,6,21,10,10,5,4,4,6,21,9,11,5,4,
            4,6,21,8,11,6,4,4,6,21,7,11,7,4,4,6,21,6,11,8,4,
            4,6,19,1,1,5,11,9,4,4,6,19,1,1,5,10,10,4,4,6,18,2,1,6,8,11,4,
            4,6,17,3,1,7,5,13,4,4,6,15,5,2,23,5,1,29,5,17,8,
            1,29,9,9,12,1,13,5,40,1,1,13,5,40,1,4,6,13,3,10,6,12,5,1,
            5,6,11,3,11,6,14,3,1,5,6,11,3,11,6,15,2,1,
            6,6,9,3,12,6,16,1,1,6,6,9,3,12,6,7,1,10,
            7,6,7,3,13,6,6,2,10,7,6,7,3,13,14,10,8,6,5,3,14,6,6,2,10,
            8,6,5,3,14,6,7,1,10,9,6,3,3,15,6,16,1,1,
            9,6,3,3,15,6,15,2,1,10,6,1,3,16,6,14,3,1,10,10,16,6,12,5,1,
            11,8,13,27,1,11,8,13,27,1,60];

// 主程序
async function main()
{
    // 打印"LOVE"，并在前面加上33个空格
    print(tab(33) + "LOVE\n");
    // 打印"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，并在前面加上15个空格
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    // 打印三个空行
    print("\n");
    print("\n");
    print("\n");
    // 打印一段致敬的话
    print("A TRIBUTE TO THE GREAT AMERICAN ARTIST, ROBERT INDIANA.\n");
    print("HIS GREATEST WORK WILL BE REPRODUCED WITH A MESSAGE OF\n");
    print("YOUR CHOICE UP TO 60 CHARACTERS.  IF YOU CAN'T THINK OF\n");
    print("A MESSAGE, SIMPLE TYPE THE WORD 'LOVE'\n");
    print("\n");
    print("YOUR MESSAGE, PLEASE");
    // 等待用户输入
    str = await input();
    // 获取输入字符串的长度
    l = str.length;
    // 初始化一个空数组
    ts = [];
    // 打印10个空行
    for (i = 1; i <= 10; i++)
        print("\n");
    // 将输入字符串重复拼接，直到长度达到60
    ts = "";
    do {
        ts += str;
    } while (ts.length < 60) ;
    // 初始化位置和计数变量
    pos = 0;
    c = 0;
    // 循环37次
    while (++c < 37) {
        // 初始化变量
        a1 = 1;
        p = 1;
        // 打印一个空行
        print("\n");
        // 循环直到a1的值大于60
        do {
            // 获取数组中的值
            a = data[pos++];
            // 更新a1的值
            a1 += a;
            // 判断p的值
            if (p != 1) {
                // 如果p不等于1，打印a个空格
                for (i = 1; i <= a; i++)
                    print(" ");
                p = 1;
            } else {
                // 如果p等于1，打印ts中指定范围的字符
                for (i = a1 - a; i <= a1 - 1; i++)
                    print(ts[i]);
                p = 0;
            }
        } while (a1 <= 60) ;
    }
    // 打印10个空行
    for (i = 1; i <= 10; i++)
        print("\n");
}

// 调用主程序
main();
```