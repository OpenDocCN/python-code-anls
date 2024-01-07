# `basic-computer-games\39_Golf\javascript\golf.js`

```

// 定义打印函数，用于在页面上输出字符串
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义输入函数，返回一个 Promise 对象，用于获取用户输入
function input()
{
    // 声明变量
    var input_element;
    var input_str;

    // 返回一个 Promise 对象
    return new Promise(function (resolve) {
        // 创建输入框元素
        input_element = document.createElement("INPUT");

        // 在页面上输出提示符
        print("? ");

        // 设置输入框属性
        input_element.setAttribute("type", "text");
        input_element.setAttribute("length", "50");

        // 将输入框添加到页面上
        document.getElementById("output").appendChild(input_element);

        // 让输入框获得焦点
        input_element.focus();

        // 初始化输入字符串
        input_str = undefined;

        // 监听输入框的键盘事件
        input_element.addEventListener("keydown", function (event) {
            if (event.keyCode == 13) {
                // 当用户按下回车键时，获取输入的字符串并移除输入框
                input_str = input_element.value;
                document.getElementById("output").removeChild(input_element);
                // 在页面上输出用户输入的字符串
                print(input_str);
                print("\n");
                // 将输入的字符串传递给 resolve 函数
                resolve(input_str);
            }
        });
    });
}

// 定义缩进函数，返回指定数量空格组成的字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 声明变量
var la = [];
var f;
var s1;
var g2;
var g3;
var x;

// 存储球洞数据的数组
var hole_data = [
    361,4,4,2,389,4,3,3,206,3,4,2,500,5,7,2,
    408,4,2,4,359,4,6,4,424,4,4,2,388,4,4,4,
    196,3,7,2,400,4,7,2,560,5,7,2,132,3,2,2,
    357,4,4,4,294,4,2,4,475,5,2,3,375,4,4,2,
    180,3,6,2,550,5,6,6,
];

// 显示障碍物信息的函数
function show_obstacle()
{
    switch (la[x]) {
        case 1:
            print("FAIRWAY.\n");
            break;
        case 2:
            print("ROUGH.\n");
            break;
        case 3:
            print("TREES.\n");
            break;
        case 4:
            print("ADJACENT FAIRWAY.\n");
            break;
        case 5:
            print("TRAP.\n");
            break;
        case 6:
            print("WATER.\n");
            break;
    }
}

// 显示得分信息的函数
function show_score()
{
    g2 += s1;
    print("TOTAL PAR FOR " + (f - 1) + " HOLES IS " + g3 + "  YOUR TOTAL IS " + g2 + "\n");
}

// 主程序
async function main()
{
    // TODO: 缺少主程序的具体实现
}

// 调用主程序
main();

```