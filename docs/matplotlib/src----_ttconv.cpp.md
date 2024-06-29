# `D:\src\scipysrc\matplotlib\src\_ttconv.cpp`

```
/*
  _ttconv.c

  Python wrapper for TrueType conversion library in ../ttconv.
 */

#include <pybind11/pybind11.h>  // 引入 pybind11 库，用于 Python 和 C++ 的交互
#include "pprdrv.h"             // 引入 pprdrv.h 头文件
#include <vector>               // 引入 vector 容器类

namespace py = pybind11;        // 命名空间简化别名

/**
 * An implementation of TTStreamWriter that writes to a Python
 * file-like object.
 */
class PythonFileWriter : public TTStreamWriter {
  py::function _write_method;    // Python 中的函数对象，用于写操作

  public:
    PythonFileWriter(py::object& file_object)
    : _write_method(file_object.attr("write")) {}  // 初始化函数对象

    virtual void write(const char *a)  // 虚函数，实现写操作
    {
        PyObject* decoded = PyUnicode_DecodeLatin1(a, strlen(a), "");  // 解码为 Unicode 字符串
        if (decoded == NULL) {
            throw py::error_already_set();  // 如果解码失败，抛出 Python 异常
        }
        _write_method(py::handle(decoded));  // 调用 Python 中的 write 方法写入数据
        Py_DECREF(decoded);  // 释放 Python 对象的引用计数
    }
};

static void convert_ttf_to_ps(
    const char *filename,     // TTF 字体文件名
    py::object &output,       // Python 文件对象，用于写入 Postscript 字体数据
    int fonttype,             // 字体类型：3 或 42
    py::iterable* glyph_ids)  // 可迭代对象，包含要保留的字形 ID
{
    PythonFileWriter output_(output);  // 创建 PythonFileWriter 对象，关联输出对象

    std::vector<int> glyph_ids_;  // 整型向量，用于存储字形 ID

    if (glyph_ids) {
        for (py::handle glyph_id: *glyph_ids) {  // 遍历传入的字形 ID
            glyph_ids_.push_back(glyph_id.cast<int>());  // 将字形 ID 转换并存入向量
        }
    }

    if (fonttype != 3 && fonttype != 42) {
        throw py::value_error(  // 如果字体类型不是 3 或 42，抛出值错误异常
            "fonttype must be either 3 (raw Postscript) or 42 (embedded Truetype)");
    }

    try {
        insert_ttfont(filename, output_, static_cast<font_type_enum>(fonttype), glyph_ids_);
        // 调用 C++ 函数 insert_ttfont 将 TTF 字体转换为 Postscript，并写入到输出对象中
    } catch (TTException &e) {
        throw std::runtime_error(e.getMessage());  // 捕获 TTException 异常，抛出运行时错误
    } catch (...) {
        throw std::runtime_error("Unknown C++ exception");  // 捕获其他异常，抛出未知的 C++ 异常错误
    }
}

PYBIND11_MODULE(_ttconv, m) {
    m.doc() = "Module to handle converting and subsetting TrueType "
              "fonts to Postscript Type 3, Postscript Type 42 and "
              "Pdf Type 3 fonts.";
    m.def("convert_ttf_to_ps", &convert_ttf_to_ps,
        "filename"_a,                          // TTF 字体文件名
        "output"_a,                            // Python 文件对象，用于输出 Postscript 数据
        "fonttype"_a,                          // 字体类型：3 或 42
        "glyph_ids"_a = py::none(),            // 可选参数，要保留的字形 ID 列表
        "Converts the Truetype font into a Type 3 or Type 42 Postscript font, "
        "optionally subsetting the font to only the desired set of characters.\n"
        "\n"
        "filename is the path to a TTF font file.\n"
        "output is a Python file-like object with a write method that the Postscript "
        "font data will be written to.\n"
        "fonttype may be either 3 or 42.  Type 3 is a \"raw Postscript\" font. "
        "Type 42 is an embedded Truetype font.  Glyph subsetting is not supported "
        "for Type 42 fonts within this module (needs to be done externally).\n"
        "glyph_ids (optional) is a list of glyph ids (integers) to keep when "
        "subsetting to a Type 3 font.  If glyph_ids is not provided or is None, "
        "then all glyphs will be included.  If any of the glyphs specified are "
        "composite glyphs, then the component glyphs will also be included."
    );
}
```