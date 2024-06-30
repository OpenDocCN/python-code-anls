# `D:\src\scipysrc\scikit-learn\build_tools\azure\pytest-pyodide.js`

```
// 引入 Node.js 的文件系统 promises 模块中的 opendir 函数
// 引入 pyodide 库中的 loadPyodide 函数
const { opendir } = require('node:fs/promises');
const { loadPyodide } = require("pyodide");

// 异步函数 main，用于执行主程序逻辑
async function main() {
    // 初始化退出代码为 0
    let exit_code = 0;
    try {
        // 全局变量 pyodide 被赋值为从 loadPyodide 函数中加载的对象
        global.pyodide = await loadPyodide();
        let pyodide = global.pyodide;
        const FS = pyodide.FS;
        const NODEFS = FS.filesystems.NODEFS;

        // 定义挂载目录为 "/mnt"，并在 pyodide 中创建该目录
        let mountDir = "/mnt";
        pyodide.FS.mkdir(mountDir);
        // 将 NODEFS 文件系统挂载到指定目录
        pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, { root: "." }, mountDir);

        // 异步加载 micropip 包
        await pyodide.loadPackage(["micropip"]);
        
        // 异步执行 Python 代码块，安装指定路径下的所有 .whl 包
        await pyodide.runPythonAsync(`
            import glob
            import micropip

            wheels = glob.glob('/mnt/dist/*.whl')
            wheels = [f'emfs://{wheel}' for wheel in wheels]
            print(f'installing wheels: {wheels}')
            await micropip.install(wheels);

            pkg_list = micropip.list()
            print(pkg_list)
        `);

        // 设置环境变量 SKLEARN_SKIP_OPENMP_TEST 为 'true'，跳过相关的测试
        await pyodide.runPythonAsync(`
            import os
            os.environ['SKLEARN_SKIP_OPENMP_TEST'] = 'true'
        `);

        // 异步安装 pytest 包
        await pyodide.runPythonAsync("import micropip; micropip.install('pytest')");
        // 从 pyodide 中导入 pytest 模块
        let pytest = pyodide.pyimport("pytest");
        // 获取命令行参数，排除前两个参数（通常是执行 Node.js 脚本的路径和文件名）
        let args = process.argv.slice(2);
        console.log('pytest args:', args);
        // 调用 pytest.main 函数执行测试，并将结果赋值给 exit_code
        exit_code = pytest.main(pyodide.toPy(args));
    } catch (e) {
        // 捕获异常并输出错误信息
        console.error(e);
        // 设置退出代码为 66，表示遇到了异常情况
        exit_code = 66;
    } finally {
        // 最终执行，退出进程，使用之前设置的退出代码
        process.exit(exit_code);
    }
}

// 调用 main 函数启动主程序逻辑
main();
```