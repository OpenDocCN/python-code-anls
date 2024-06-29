# `D:\src\scipysrc\matplotlib\src\ft2font_wrapper.cpp`

```
#include "mplutils.h"
#include "ft2font.h"
#include "py_converters.h"
#include "py_exceptions.h"

// From Python
#include <structmember.h>

#include <set>

// 定义静态函数 convert_xys_to_array，将 C++ 向量 xys 转换为 NumPy 数组
static PyObject *convert_xys_to_array(std::vector<double> &xys)
{
    // 计算数组维度
    npy_intp dims[] = {(npy_intp)xys.size() / 2, 2 };
    // 如果数组非空，使用给定数据创建 NumPy 双精度数组
    if (dims[0] > 0) {
        return PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, &xys[0]);
    } else {
        // 否则创建空的 NumPy 双精度数组
        return PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    }
}

/**********************************************************************
 * FT2Image
 * */

// 定义 Python 对象 PyFT2Image 结构体
typedef struct
{
    PyObject_HEAD
    FT2Image *x; // 指向 FT2Image 对象的指针
    Py_ssize_t shape[2]; // 图像的形状
    Py_ssize_t strides[2]; // 数据步长
    Py_ssize_t suboffsets[2]; // 子偏移量
} PyFT2Image;

static PyTypeObject PyFT2ImageType; // 定义 PyFT2ImageType 类型对象

// 创建 PyFT2Image 对象的新实例
static PyObject *PyFT2Image_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyFT2Image *self;
    self = (PyFT2Image *)type->tp_alloc(type, 0);
    self->x = NULL; // 初始化 FT2Image 指针为空
    return (PyObject *)self;
}

// 初始化 PyFT2Image 对象
static int PyFT2Image_init(PyFT2Image *self, PyObject *args, PyObject *kwds)
{
    double width;
    double height;

    // 解析传入的参数，设置图像的宽度和高度
    if (!PyArg_ParseTuple(args, "dd:FT2Image", &width, &height)) {
        return -1;
    }

    CALL_CPP_INIT("FT2Image", (self->x = new FT2Image(width, height)));

    return 0;
}

// 释放 PyFT2Image 对象
static void PyFT2Image_dealloc(PyFT2Image *self)
{
    delete self->x; // 删除 FT2Image 对象
    Py_TYPE(self)->tp_free((PyObject *)self);
}

// 定义函数 PyFT2Image_draw_rect，绘制空矩形到图像中
const char *PyFT2Image_draw_rect__doc__ =
    "draw_rect(self, x0, y0, x1, y1)\n"
    "--\n\n"
    "Draw an empty rectangle to the image.\n"
    "\n"
    ".. deprecated:: 3.8\n";
;

static PyObject *PyFT2Image_draw_rect(PyFT2Image *self, PyObject *args)
{
    // 发出警告，指出 draw_rect 已自 Matplotlib 3.8 起弃用
    char const* msg =
        "FT2Image.draw_rect is deprecated since Matplotlib 3.8 and will be removed "
        "in Matplotlib 3.10 as it is not used in the library. If you rely on it, "
        "please let us know.";
    if (PyErr_WarnEx(PyExc_DeprecationWarning, msg, 1)) {
        return NULL;
    }

    double x0, y0, x1, y1;

    // 解析传入的参数，获取矩形的坐标信息
    if (!PyArg_ParseTuple(args, "dddd:draw_rect", &x0, &y0, &x1, &y1)) {
        return NULL;
    }

    // 调用 C++ 函数 draw_rect 绘制空矩形
    CALL_CPP("draw_rect", (self->x->draw_rect(x0, y0, x1, y1)));

    Py_RETURN_NONE;
}

// 定义函数 PyFT2Image_draw_rect_filled，绘制填充矩形到图像中
const char *PyFT2Image_draw_rect_filled__doc__ =
    "draw_rect_filled(self, x0, y0, x1, y1)\n"
    "--\n\n"
    "Draw a filled rectangle to the image.\n";

static PyObject *PyFT2Image_draw_rect_filled(PyFT2Image *self, PyObject *args)
{
    double x0, y0, x1, y1;

    // 解析传入的参数，获取填充矩形的坐标信息
    if (!PyArg_ParseTuple(args, "dddd:draw_rect_filled", &x0, &y0, &x1, &y1)) {
        return NULL;
    }

    // 调用 C++ 函数 draw_rect_filled 绘制填充矩形
    CALL_CPP("draw_rect_filled", (self->x->draw_rect_filled(x0, y0, x1, y1)));

    Py_RETURN_NONE;
}

// 获取 PyFT2Image 对象的缓冲区信息
static int PyFT2Image_get_buffer(PyFT2Image *self, Py_buffer *buf, int flags)
{
    FT2Image *im = self->x;

    Py_INCREF(self); // 增加对象的引用计数
    buf->obj = (PyObject *)self;
    buf->buf = im->get_buffer(); // 获取图像数据的缓冲区
    buf->len = im->get_width() * im->get_height(); // 设置缓冲区长度
    buf->readonly = 0; // 设置缓冲区可写
    buf->format = (char *)"B"; // 设置数据格式为无符号字节
    buf->ndim = 2; // 设置数据维度为二维
    self->shape[0] = im->get_height(); // 设置图像高度
    # 设置 self 对象的第二个维度为图像宽度
    self->shape[1] = im->get_width();
    # 将 buf 对象的 shape 属性设置为 self 对象的 shape 属性
    buf->shape = self->shape;
    # 设置 self 对象的第一个维度步进为图像宽度
    self->strides[0] = im->get_width();
    # 设置 self 对象的第二个维度步进为 1
    self->strides[1] = 1;
    # 将 buf 对象的 strides 属性设置为 self 对象的 strides 属性
    buf->strides = self->strides;
    # 将 buf 对象的 suboffsets 属性设置为 NULL，表示无子偏移
    buf->suboffsets = NULL;
    # 设置 buf 对象的每个项目的大小为 1 字节
    buf->itemsize = 1;
    # 将 buf 对象的 internal 属性设置为 NULL，表示无内部数据
    buf->internal = NULL;

    # 返回值为 1，表示设置完成
    return 1;
}

static PyTypeObject* PyFT2Image_init_type()
{
    // 定义方法数组，每个方法包括函数名、对应的C函数、调用方式和文档字符串
    static PyMethodDef methods[] = {
        {"draw_rect", (PyCFunction)PyFT2Image_draw_rect, METH_VARARGS, PyFT2Image_draw_rect__doc__},
        {"draw_rect_filled", (PyCFunction)PyFT2Image_draw_rect_filled, METH_VARARGS, PyFT2Image_draw_rect_filled__doc__},
        {NULL}  // 方法数组以NULL结尾
    };

    // 定义缓冲区处理器，设置获取缓冲区的函数为PyFT2Image_get_buffer
    static PyBufferProcs buffer_procs;
    buffer_procs.bf_getbuffer = (getbufferproc)PyFT2Image_get_buffer;

    // 设置PyFT2ImageType的类型对象的各种属性
    PyFT2ImageType.tp_name = "matplotlib.ft2font.FT2Image";  // 类型对象的名称
    PyFT2ImageType.tp_basicsize = sizeof(PyFT2Image);  // 类型对象的基本大小
    PyFT2ImageType.tp_dealloc = (destructor)PyFT2Image_dealloc;  // 析构函数
    PyFT2ImageType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;  // 类型对象的标志
    PyFT2ImageType.tp_methods = methods;  // 类型对象的方法列表
    PyFT2ImageType.tp_new = PyFT2Image_new;  // 类型对象的新建函数
    PyFT2ImageType.tp_init = (initproc)PyFT2Image_init;  // 类型对象的初始化函数
    PyFT2ImageType.tp_as_buffer = &buffer_procs;  // 缓冲区处理器

    return &PyFT2ImageType;  // 返回PyFT2ImageType的类型对象指针
}

/**********************************************************************
 * Glyph
 * */

typedef struct
{
    PyObject_HEAD  // 定义Python对象的头部信息
    size_t glyphInd;  // 字形索引
    long width;  // 宽度
    long height;  // 高度
    long horiBearingX;  // 水平轴上的起始位置X
    long horiBearingY;  // 水平轴上的起始位置Y
    long horiAdvance;  // 水平方向上的进度
    long linearHoriAdvance;  // 线性水平进度
    long vertBearingX;  // 垂直轴上的起始位置X
    long vertBearingY;  // 垂直轴上的起始位置Y
    long vertAdvance;  // 垂直方向上的进度
    FT_BBox bbox;  // 字形的包围盒
} PyGlyph;  // Python字形对象的结构体

static PyTypeObject PyGlyphType;  // 定义Python字形对象的类型对象

static PyObject *PyGlyph_from_FT2Font(const FT2Font *font)
{
    // 获取字体的相关信息
    const FT_Face &face = font->get_face();
    const long hinting_factor = font->get_hinting_factor();
    const FT_Glyph &glyph = font->get_last_glyph();

    PyGlyph *self;
    // 分配PyGlyph对象的内存空间
    self = (PyGlyph *)PyGlyphType.tp_alloc(&PyGlyphType, 0);

    // 设置PyGlyph对象的属性值
    self->glyphInd = font->get_last_glyph_index();
    FT_Glyph_Get_CBox(glyph, ft_glyph_bbox_subpixels, &self->bbox);

    self->width = face->glyph->metrics.width / hinting_factor;
    self->height = face->glyph->metrics.height;
    self->horiBearingX = face->glyph->metrics.horiBearingX / hinting_factor;
    self->horiBearingY = face->glyph->metrics.horiBearingY;
    self->horiAdvance = face->glyph->metrics.horiAdvance;
    self->linearHoriAdvance = face->glyph->linearHoriAdvance / hinting_factor;
    self->vertBearingX = face->glyph->metrics.vertBearingX;
    self->vertBearingY = face->glyph->metrics.vertBearingY;
    self->vertAdvance = face->glyph->metrics.vertAdvance;

    return (PyObject *)self;  // 返回Python对象指针
}

static void PyGlyph_dealloc(PyGlyph *self)
{
    Py_TYPE(self)->tp_free((PyObject *)self);  // 释放Python对象
}

static PyObject *PyGlyph_get_bbox(PyGlyph *self, void *closure)
{
    // 返回字形的包围盒信息
    return Py_BuildValue(
        "llll", self->bbox.xMin, self->bbox.yMin, self->bbox.xMax, self->bbox.yMax);
}

static PyTypeObject *PyGlyph_init_type()
{
    // 定义结构体成员列表，描述了 PyGlyph 对象的各个成员变量，包括名称、类型、偏移量和只读属性
    static PyMemberDef members[] = {
        {(char *)"width", T_LONG, offsetof(PyGlyph, width), READONLY, (char *)""},
        {(char *)"height", T_LONG, offsetof(PyGlyph, height), READONLY, (char *)""},
        {(char *)"horiBearingX", T_LONG, offsetof(PyGlyph, horiBearingX), READONLY, (char *)""},
        {(char *)"horiBearingY", T_LONG, offsetof(PyGlyph, horiBearingY), READONLY, (char *)""},
        {(char *)"horiAdvance", T_LONG, offsetof(PyGlyph, horiAdvance), READONLY, (char *)""},
        {(char *)"linearHoriAdvance", T_LONG, offsetof(PyGlyph, linearHoriAdvance), READONLY, (char *)""},
        {(char *)"vertBearingX", T_LONG, offsetof(PyGlyph, vertBearingX), READONLY, (char *)""},
        {(char *)"vertBearingY", T_LONG, offsetof(PyGlyph, vertBearingY), READONLY, (char *)""},
        {(char *)"vertAdvance", T_LONG, offsetof(PyGlyph, vertAdvance), READONLY, (char *)""},
        {NULL}  // 成员列表结束标记
    };

    // 定义获取器和设置器列表，用于访问 PyGlyph 对象的属性，目前只定义了 "bbox" 属性的获取器
    static PyGetSetDef getset[] = {
        {(char *)"bbox", (getter)PyGlyph_get_bbox, NULL, NULL, NULL},
        {NULL}  // 列表结束标记
    };

    // 设置 PyGlyphType 类型对象的名称
    PyGlyphType.tp_name = "matplotlib.ft2font.Glyph";
    // 设置 PyGlyphType 类型对象的基本大小，即 PyGlyph 结构体的大小
    PyGlyphType.tp_basicsize = sizeof(PyGlyph);
    // 设置 PyGlyphType 类型对象的析构函数，用于释放对象内存
    PyGlyphType.tp_dealloc = (destructor)PyGlyph_dealloc;
    // 设置 PyGlyphType 类型对象的标志，包括默认标志和基础类型标志
    PyGlyphType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    // 设置 PyGlyphType 类型对象的成员变量列表
    PyGlyphType.tp_members = members;
    // 设置 PyGlyphType 类型对象的获取器和设置器列表
    PyGlyphType.tp_getset = getset;

    // 返回指向 PyGlyphType 类型对象的指针
    return &PyGlyphType;
}

/**********************************************************************
 * FT2Font
 * */

// 定义 Python 对象 PyFT2Font 结构体
struct PyFT2Font
{
    PyObject_HEAD
    FT2Font *x;                 // FreeType 中的字体对象指针
    PyObject *py_file;          // Python 文件对象
    FT_StreamRec stream;        // FreeType 流对象
    Py_ssize_t shape[2];        // 对象形状数组
    Py_ssize_t strides[2];      // 步幅数组
    Py_ssize_t suboffsets[2];   // 子偏移数组
    std::vector<PyObject *> fallbacks;  // 备用对象向量
};

// 定义 PyFT2Font 类型对象
static PyTypeObject PyFT2FontType;

// 从文件中读取回调函数
static unsigned long read_from_file_callback(FT_Stream stream,
                                             unsigned long offset,
                                             unsigned char *buffer,
                                             unsigned long count)
{
    PyObject *py_file = ((PyFT2Font *)stream->descriptor.pointer)->py_file;
    PyObject *seek_result = NULL, *read_result = NULL;
    Py_ssize_t n_read = 0;
    if (!(seek_result = PyObject_CallMethod(py_file, "seek", "k", offset))
        || !(read_result = PyObject_CallMethod(py_file, "read", "k", count))) {
        goto exit;
    }
    char *tmpbuf;
    if (PyBytes_AsStringAndSize(read_result, &tmpbuf, &n_read) == -1) {
        goto exit;
    }
    memcpy(buffer, tmpbuf, n_read);
exit:
    Py_XDECREF(seek_result);
    Py_XDECREF(read_result);
    if (PyErr_Occurred()) {
        PyErr_WriteUnraisable(py_file);
        if (!count) {
            return 1;  // 当 count == 0 时，非零值表示错误
        }
    }
    return (unsigned long)n_read;
}

// 关闭文件的回调函数
static void close_file_callback(FT_Stream stream)
{
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    PyFT2Font *self = (PyFT2Font *)stream->descriptor.pointer;
    PyObject *close_result = NULL;
    if (!(close_result = PyObject_CallMethod(self->py_file, "close", ""))) {
        goto exit;
    }
exit:
    Py_XDECREF(close_result);
    Py_CLEAR(self->py_file);
    if (PyErr_Occurred()) {
        PyErr_WriteUnraisable((PyObject*)self);
    }
    PyErr_Restore(type, value, traceback);
}

// 创建 PyFT2Font 对象的构造函数
static PyObject *PyFT2Font_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyFT2Font *self;
    self = (PyFT2Font *)type->tp_alloc(type, 0);
    self->x = NULL;
    self->py_file = NULL;
    memset(&self->stream, 0, sizeof(FT_StreamRec));
    return (PyObject *)self;
}

// PyFT2Font 初始化文档字符串
const char *PyFT2Font_init__doc__ =
    "FT2Font(filename, hinting_factor=8, *, _fallback_list=None, _kerning_factor=0)\n"
    "--\n\n"
    "Create a new FT2Font object.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "filename : str or file-like\n"
    "    The source of the font data in a format (ttf or ttc) that FreeType can read\n"
    "\n"
    "hinting_factor : int, optional\n"
    "    Must be positive. Used to scale the hinting in the x-direction\n"
    "_fallback_list : list of FT2Font, optional\n"
    "    A list of FT2Font objects used to find missing glyphs.\n"
    "\n"
    "    .. warning::\n"
    "        This API is both private and provisional: do not use it directly\n"
    "\n"
    "_kerning_factor : int, optional\n"
    "    Used to adjust the degree of kerning.\n";
    "\n"
    "    .. warning::\n"
    "        This API is private: do not use it directly\n"
    "\n"
    "Attributes\n"
    "----------\n"
    "num_faces : int\n"
    "    Number of faces in file.\n"
    "face_flags, style_flags : int\n"
    "    Face and style flags; see the ft2font constants.\n"
    "num_glyphs : int\n"
    "    Number of glyphs in the face.\n"
    "family_name, style_name : str\n"
    "    Face family and style name.\n"
    "num_fixed_sizes : int\n"
    "    Number of bitmap in the face.\n"
    "scalable : bool\n"
    "    Whether face is scalable; attributes after this one are only\n"
    "    defined for scalable faces.\n"
    "bbox : tuple[int, int, int, int]\n"
    "    Face global bounding box (xmin, ymin, xmax, ymax).\n"
    "units_per_EM : int\n"
    "    Number of font units covered by the EM.\n"
    "ascender, descender : int\n"
    "    Ascender and descender in 26.6 units.\n"
    "height : int\n"
    "    Height in 26.6 units; used to compute a default line spacing\n"
    "    (baseline-to-baseline distance).\n"
    "max_advance_width, max_advance_height : int\n"
    "    Maximum horizontal and vertical cursor advance for all glyphs.\n"
    "underline_position, underline_thickness : int\n"
    "    Vertical position and thickness of the underline bar.\n"
    "postscript_name : str\n"
    "    PostScript name of the font.\n";



# 以下是一个包含文本块的字符串，描述了字体文件的各种属性
# 这些属性包括字体文件的面数、面和样式标志、字形数、字体家族和样式名称等
# 还包括一些特定于可缩放字体的属性，如全局边界框、EM 覆盖的字体单位数等
# 最后列出了一些与字体设计相关的度量值，如升高、降低、高度等
# 这些信息用于描述字体的视觉特征和版面设计的相关细节
    // 初始化函数，用于初始化 PyFT2Font 对象
    static int PyFT2Font_init(PyFT2Font *self, PyObject *args, PyObject *kwds)
    {
        // 初始化变量
        PyObject *filename = NULL, *open = NULL, *data = NULL, *fallback_list = NULL;
        FT_Open_Args open_args;
        long hinting_factor = 8;
        int kerning_factor = 0;
        const char *names[] = {
            "filename", "hinting_factor", "_fallback_list", "_kerning_factor", NULL
        };
        // 用于存储 fallback 字体对象的向量
        std::vector<FT2Font *> fallback_fonts;
        
        // 解析参数并关键字
        if (!PyArg_ParseTupleAndKeywords(
                 args, kwds, "O|l$Oi:FT2Font", (char **)names, &filename,
                 &hinting_factor, &fallback_list, &kerning_factor)) {
            return -1;
        }
        
        // 检查 hinting_factor 的有效性
        if (hinting_factor <= 0) {
          PyErr_SetString(PyExc_ValueError,
                          "hinting_factor must be greater than 0");
          goto exit;
        }

        // 初始化流对象
        self->stream.base = NULL;
        self->stream.size = 0x7fffffff;  // 未知大小
        self->stream.pos = 0;
        self->stream.descriptor.pointer = self;
        self->stream.read = &read_from_file_callback;
        
        // 清空并设置 open_args 结构体
        memset((void *)&open_args, 0, sizeof(FT_Open_Args));
        open_args.flags = FT_OPEN_STREAM;
        open_args.stream = &self->stream;

        // 如果提供了 fallback_list
        if (fallback_list) {
            // 检查 fallback_list 是否为列表类型
            if (!PyList_Check(fallback_list)) {
                PyErr_SetString(PyExc_TypeError, "Fallback list must be a list");
                goto exit;
            }
            Py_ssize_t size = PyList_Size(fallback_list);

            // 遍历 fallback_list 以确保类型正确
            for (Py_ssize_t i = 0; i < size; ++i) {
                // 获取列表项，这里是借用引用
                PyObject* item = PyList_GetItem(fallback_list, i);
                // 检查是否为 FT2Font 对象
                if (!PyObject_IsInstance(item, PyObject_Type(reinterpret_cast<PyObject *>(self)))) {
                    PyErr_SetString(PyExc_TypeError, "Fallback fonts must be FT2Font objects.");
                    goto exit;
                }
            }
            // 第二次遍历 fallback_list 将对象添加到 self->fallbacks 和 fallback_fonts 中
            for (Py_ssize_t i = 0; i < size; ++i) {
                // 获取列表项，这里是借用引用
                PyObject* item = PyList_GetItem(fallback_list, i);
                // 增加引用计数，dealloc 时会撤销这一操作，确保对象在使用期间不会被 GC
                Py_INCREF(item);
                self->fallbacks.push_back(item);
                // 缓存底层的 FT2Font 对象到 fallback_fonts 中
                FT2Font *fback = reinterpret_cast<PyFT2Font *>(item)->x;
                fallback_fonts.push_back(fback);
            }
        }

        // 检查 filename 类型并打开文件
        if (PyBytes_Check(filename) || PyUnicode_Check(filename)) {
            // 获取内置的 open 函数
            if (!(open = PyDict_GetItemString(PyEval_GetBuiltins(), "open"))  // 借用引用
                || !(self->py_file = PyObject_CallFunction(open, "Os", filename, "rb"))) {
                goto exit;
            }
            self->stream.close = &close_file_callback;
    } else if (!PyObject_HasAttrString(filename, "read")
               || !(data = PyObject_CallMethod(filename, "read", "i", 0))
               || !PyBytes_Check(data)) {
        // 检查 filename 对象是否有 "read" 方法或者无法调用 "read" 方法，或者返回的数据不是字节对象
        PyErr_SetString(PyExc_TypeError,
                        "First argument must be a path to a font file or a binary-mode file object");
        // 设置类型错误异常并清理 data 变量
        Py_CLEAR(data);
        // 跳转到 exit 标签处结束函数执行
        goto exit;
    } else {
        // 如果 filename 对象满足条件，则将其赋值给 self->py_file
        self->py_file = filename;
        // 关闭 self->stream.close 指针（设为 NULL）
        self->stream.close = NULL;
        // 增加 filename 的引用计数
        Py_INCREF(filename);
    }
    // 清理 data 变量
    Py_CLEAR(data);

    // 调用 C++ 函数 "FT2Font" 来创建 self->x 对象，传入参数 open_args, hinting_factor, fallback_fonts
    CALL_CPP_FULL(
        "FT2Font", (self->x = new FT2Font(open_args, hinting_factor, fallback_fonts)),
        // 清理 self->py_file 变量
        Py_CLEAR(self->py_file), -1);

    // 调用 C++ 函数 "FT2Font->set_kerning_factor" 来设置 self->x 对象的 kerning_factor
    CALL_CPP_INIT("FT2Font->set_kerning_factor", (self->x->set_kerning_factor(kerning_factor)));
exit:
    // 返回一个整数，如果发生异常（PyErr_Occurred() 非空），返回-1，否则返回0
    return PyErr_Occurred() ? -1 : 0;
}

static void PyFT2Font_dealloc(PyFT2Font *self)
{
    // 删除 self 指针所指向的 x 对象
    delete self->x;
    // 循环释放 fallbacks 容器中的每个对象的引用计数
    for (size_t i = 0; i < self->fallbacks.size(); i++) {
        Py_DECREF(self->fallbacks[i]);
    }

    // 释放 py_file 对象的引用计数
    Py_XDECREF(self->py_file);
    // 释放 self 对象的内存
    Py_TYPE(self)->tp_free((PyObject *)self);
}

const char *PyFT2Font_clear__doc__ =
    "clear(self)\n"
    "--\n\n"
    "Clear all the glyphs, reset for a new call to `.set_text`.\n";

static PyObject *PyFT2Font_clear(PyFT2Font *self, PyObject *args)
{
    // 调用 C++ 函数 self->x->clear() 来清除所有的字形（glyphs）
    CALL_CPP("clear", (self->x->clear()));

    // 返回 None
    Py_RETURN_NONE;
}

const char *PyFT2Font_set_size__doc__ =
    "set_size(self, ptsize, dpi)\n"
    "--\n\n"
    "Set the point size and dpi of the text.\n";

static PyObject *PyFT2Font_set_size(PyFT2Font *self, PyObject *args)
{
    double ptsize;
    double dpi;

    // 解析参数，设置字体大小（ptsize）和 dpi
    if (!PyArg_ParseTuple(args, "dd:set_size", &ptsize, &dpi)) {
        return NULL;
    }

    // 调用 C++ 函数 self->x->set_size(ptsize, dpi) 来设置字体大小和 dpi
    CALL_CPP("set_size", (self->x->set_size(ptsize, dpi)));

    // 返回 None
    Py_RETURN_NONE;
}

const char *PyFT2Font_set_charmap__doc__ =
    "set_charmap(self, i)\n"
    "--\n\n"
    "Make the i-th charmap current.\n";

static PyObject *PyFT2Font_set_charmap(PyFT2Font *self, PyObject *args)
{
    int i;

    // 解析参数，设置当前字符映射为第 i 个
    if (!PyArg_ParseTuple(args, "i:set_charmap", &i)) {
        return NULL;
    }

    // 调用 C++ 函数 self->x->set_charmap(i) 来设置当前字符映射
    CALL_CPP("set_charmap", (self->x->set_charmap(i)));

    // 返回 None
    Py_RETURN_NONE;
}

const char *PyFT2Font_select_charmap__doc__ =
    "select_charmap(self, i)\n"
    "--\n\n"
    "Select a charmap by its FT_Encoding number.\n";

static PyObject *PyFT2Font_select_charmap(PyFT2Font *self, PyObject *args)
{
    unsigned long i;

    // 解析参数，通过 FT_Encoding 数字 i 来选择字符映射
    if (!PyArg_ParseTuple(args, "k:select_charmap", &i)) {
        return NULL;
    }

    // 调用 C++ 函数 self->x->select_charmap(i) 来选择字符映射
    CALL_CPP("select_charmap", self->x->select_charmap(i));

    // 返回 None
    Py_RETURN_NONE;
}

const char *PyFT2Font_get_kerning__doc__ =
    "get_kerning(self, left, right, mode)\n"
    "--\n\n"
    "Get the kerning between *left* and *right* glyph indices.\n"
    "*mode* is a kerning mode constant:\n\n"
    "- KERNING_DEFAULT  - Return scaled and grid-fitted kerning distances\n"
    "- KERNING_UNFITTED - Return scaled but un-grid-fitted kerning distances\n"
    "- KERNING_UNSCALED - Return the kerning vector in original font units\n";

static PyObject *PyFT2Font_get_kerning(PyFT2Font *self, PyObject *args)
{
    FT_UInt left, right, mode;
    int result;
    int fallback = 1;

    // 解析参数，获取左右字形索引及 kerning 模式
    if (!PyArg_ParseTuple(args, "III:get_kerning", &left, &right, &mode)) {
        return NULL;
    }

    // 调用 C++ 函数 self->x->get_kerning(left, right, mode, (bool)fallback) 来获取字距
    CALL_CPP("get_kerning", (result = self->x->get_kerning(left, right, mode, (bool)fallback)));

    // 将结果转换为 Python 的长整型对象并返回
    return PyLong_FromLong(result);
}

const char *PyFT2Font_get_fontmap__doc__ =
    "_get_fontmap(self, string)\n"
    "--\n\n"
    "Get a mapping between characters and the font that includes them.\n"
    "A dictionary mapping unicode characters to PyFT2Font objects.";
static PyObject *PyFT2Font_get_fontmap(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    PyObject *textobj;
    // ...
    // 定义一个指向常量字符串数组的指针，最后一项为NULL，用于PyArg_ParseTupleAndKeywords函数的参数名
    const char *names[] = { "string", NULL };

    // 解析Python函数的参数，期望一个对象参数，并将其赋值给textobj
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O:_get_fontmap", (char **)names, &textobj)) {
        // 解析失败时返回空指针
        return NULL;
    }

    // 创建一个存储FT_ULong类型数据的集合，用于存储Unicode字符编码点
    std::set<FT_ULong> codepoints;
    size_t size;

    // 检查textobj是否为PyUnicode对象
    if (PyUnicode_Check(textobj)) {
        // 获取PyUnicode对象的长度
        size = PyUnicode_GET_LENGTH(textobj);
        // 遍历PyUnicode对象的每个字符，并插入到codepoints集合中
        for (size_t i = 0; i < size; ++i) {
            codepoints.insert(PyUnicode_ReadChar(textobj, i));
        }
    } else {
        // 如果textobj不是PyUnicode对象，则设置异常并返回空指针
        PyErr_SetString(PyExc_TypeError, "string must be str");
        return NULL;
    }

    // 创建一个空的Python字典对象char_to_font
    PyObject *char_to_font;
    if (!(char_to_font = PyDict_New())) {
        // 如果创建字典失败，则返回空指针
        return NULL;
    }

    // 遍历codepoints集合中的每个Unicode编码点
    for (auto it = codepoints.begin(); it != codepoints.end(); ++it) {
        auto x = *it;
        PyObject* target_font;
        int index;

        // 调用self对象的get_char_fallback_index方法，获取字符x的回退字体索引
        if (self->x->get_char_fallback_index(x, index)) {
            // 如果索引非负，则将target_font设置为self对象的fallbacks[index]
            if (index >= 0) {
                target_font = self->fallbacks[index];
            } else {
                // 否则将target_font设置为self对象本身的PyObject指针
                target_font = (PyObject *)self;
            }
        } else {
            // 如果get_char_fallback_index调用失败，暂时未处理递归情况，将target_font设置为self对象本身的PyObject指针
            // TODO 处理递归情况！
            target_font = (PyObject *)self;
        }

        // 创建一个Python字符串对象，表示字符x，并将其作为字典的键，将target_font作为值存入char_to_font字典中
        PyObject *key = NULL;
        bool error = (!(key = PyUnicode_FromFormat("%c", x))
                      || (PyDict_SetItem(char_to_font, key, target_font) == -1));
        Py_XDECREF(key);
        if (error) {
            // 如果在设置字典项时发生错误，释放char_to_font字典对象并设置异常信息，返回空指针
            Py_DECREF(char_to_font);
            PyErr_SetString(PyExc_ValueError, "Something went very wrong");
            return NULL;
        }
    }

    // 返回填充好的char_to_font字典对象
    return char_to_font;
/* 定义 Python 对象的文档字符串，描述 set_text 方法的作用和参数 */
const char *PyFT2Font_set_text__doc__ =
    "set_text(self, string, angle=0.0, flags=32)\n"
    "--\n\n"
    "Set the text *string* and *angle*.\n"
    "*flags* can be a bitwise-or of the LOAD_XXX constants;\n"
    "the default value is LOAD_FORCE_AUTOHINT.\n"
    "You must call this before `.draw_glyphs_to_bitmap`.\n"
    "A sequence of x,y positions is returned.\n";

/* 定义 PyFT2Font_set_text 函数，用于设置文本和角度 */
static PyObject *PyFT2Font_set_text(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    PyObject *textobj;
    double angle = 0.0;
    FT_Int32 flags = FT_LOAD_FORCE_AUTOHINT;
    std::vector<double> xys;
    const char *names[] = { "string", "angle", "flags", NULL };

    /* 解析 Python 函数的参数并设置文本和角度 */
    if (!PyArg_ParseTupleAndKeywords(
             args, kwds, "O|di:set_text", (char **)names, &textobj, &angle, &flags)) {
        return NULL;
    }

    std::vector<uint32_t> codepoints;
    size_t size;

    /* 检查参数是否为 Unicode 字符串，将其转换为 codepoints 数组 */
    if (PyUnicode_Check(textobj)) {
        size = PyUnicode_GET_LENGTH(textobj);
        codepoints.resize(size);
        for (size_t i = 0; i < size; ++i) {
            codepoints[i] = PyUnicode_ReadChar(textobj, i);
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "set_text requires str-input.");
        return NULL;
    }

    uint32_t* codepoints_array = NULL;
    if (size > 0) {
        codepoints_array = &codepoints[0];
    }

    /* 调用 C++ 的 set_text 方法来设置文本内容 */
    CALL_CPP("set_text", self->x->set_text(size, codepoints_array, angle, flags, xys));

    /* 将 xys 向量转换为 Python 数组并返回 */
    return convert_xys_to_array(xys);
}

/* 定义 Python 对象的文档字符串，描述 get_num_glyphs 方法的作用 */
const char *PyFT2Font_get_num_glyphs__doc__ =
    "get_num_glyphs(self)\n"
    "--\n\n"
    "Return the number of loaded glyphs.\n";

/* 定义 PyFT2Font_get_num_glyphs 函数，返回加载的字形数目 */
static PyObject *PyFT2Font_get_num_glyphs(PyFT2Font *self, PyObject *args)
{
    return PyLong_FromSize_t(self->x->get_num_glyphs());
}

/* 定义 Python 对象的文档字符串，描述 load_char 方法的作用和参数 */
const char *PyFT2Font_load_char__doc__ =
    "load_char(self, charcode, flags=32)\n"
    "--\n\n"
    "Load character with *charcode* in current fontfile and set glyph.\n"
    "*flags* can be a bitwise-or of the LOAD_XXX constants;\n"
    "the default value is LOAD_FORCE_AUTOHINT.\n"
    "Return value is a Glyph object, with attributes\n\n"
    "- width: glyph width\n"
    "- height: glyph height\n"
    "- bbox: the glyph bbox (xmin, ymin, xmax, ymax)\n"
    "- horiBearingX: left side bearing in horizontal layouts\n"
    "- horiBearingY: top side bearing in horizontal layouts\n"
    "- horiAdvance: advance width for horizontal layout\n"
    "- vertBearingX: left side bearing in vertical layouts\n"
    "- vertBearingY: top side bearing in vertical layouts\n"
    "- vertAdvance: advance height for vertical layout\n";

/* 定义 PyFT2Font_load_char 函数，加载指定字符并返回字形对象 */
static PyObject *PyFT2Font_load_char(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    long charcode;
    int fallback = 1;
    FT_Int32 flags = FT_LOAD_FORCE_AUTOHINT;
    const char *names[] = { "charcode", "flags", NULL };

    /* 继续定义该函数的具体实现 */

    /* 解析 Python 函数的参数并加载指定的字符 */
    if (!PyArg_ParseTupleAndKeywords(
             args, kwds, "li:load_char", (char **)names, &charcode, &flags)) {
        return NULL;
    }

    /* 调用 C++ 的 load_char 方法来加载指定字符并返回字形对象 */
    CALL_CPP("load_char", self->x->load_char(charcode, flags));

    /* 在此函数中没有明确的返回值要求，因此返回 NULL */
    return NULL;
}
    /* 使用 PyArg_ParseTupleAndKeywords 解析 Python 函数的参数和关键字参数。
       参数格式为 "l|i:load_char"，表示第一个参数是长整型，第二个参数是可选的整型。
       names 是参数的名称数组，charcode 和 flags 是解析后的输出参数。如果解析失败，
       返回 NULL。 */
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "l|i:load_char", (char **)names, &charcode,
                                     &flags)) {
        return NULL;
    }

    // 初始化一个 FT2Font 对象指针，初始值为 NULL
    FT2Font *ft_object = NULL;
    
    // 调用 C++ 函数 self->x->load_char，加载指定字符 charcode 的字形信息，
    // 结果存储在 ft_object 中，如果 fallback 为真，则进行降级处理。
    CALL_CPP("load_char", (self->x->load_char(charcode, flags, ft_object, (bool)fallback)));

    // 将 FT2Font 对象指针转换为 Python 的 PyGlyph 对象，并返回
    return PyGlyph_from_FT2Font(ft_object);
/* PyFT2Font_load_glyph__doc__ 字符串包含了 load_glyph 方法的文档字符串，描述了方法的参数和返回值 */
const char *PyFT2Font_load_glyph__doc__ =
    "load_glyph(self, glyphindex, flags=32)\n"
    "--\n\n"
    "Load character with *glyphindex* in current fontfile and set glyph.\n"
    "*flags* can be a bitwise-or of the LOAD_XXX constants;\n"
    "the default value is LOAD_FORCE_AUTOHINT.\n"
    "Return value is a Glyph object, with attributes\n\n"
    "- width: glyph width\n"
    "- height: glyph height\n"
    "- bbox: the glyph bbox (xmin, ymin, xmax, ymax)\n"
    "- horiBearingX: left side bearing in horizontal layouts\n"
    "- horiBearingY: top side bearing in horizontal layouts\n"
    "- horiAdvance: advance width for horizontal layout\n"
    "- vertBearingX: left side bearing in vertical layouts\n"
    "- vertBearingY: top side bearing in vertical layouts\n"
    "- vertAdvance: advance height for vertical layout\n";

/* PyFT2Font_load_glyph 函数实现，用于加载字形数据 */
static PyObject *PyFT2Font_load_glyph(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    FT_UInt glyph_index;
    FT_Int32 flags = FT_LOAD_FORCE_AUTOHINT;
    int fallback = 1;
    const char *names[] = { "glyph_index", "flags", NULL };

    /* 解析传入的参数，包括 glyph_index 和 flags */
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "I|i:load_glyph", (char **)names, &glyph_index,
                                     &flags)) {
        return NULL;
    }

    FT2Font *ft_object = NULL;
    /* 调用 C++ 函数 load_glyph 加载指定字形的数据 */
    CALL_CPP("load_glyph", (self->x->load_glyph(glyph_index, flags, ft_object, (bool)fallback)));

    /* 将加载得到的字形数据转换为 Python 对象并返回 */
    return PyGlyph_from_FT2Font(ft_object);
}

/* PyFT2Font_get_width_height__doc__ 字符串包含了 get_width_height 方法的文档字符串，描述了方法的返回值 */
const char *PyFT2Font_get_width_height__doc__ =
    "get_width_height(self)\n"
    "--\n\n"
    "Get the width and height in 26.6 subpixels of the current string set by `.set_text`.\n"
    "The rotation of the string is accounted for.  To get width and height\n"
    "in pixels, divide these values by 64.\n";

/* PyFT2Font_get_width_height 函数实现，用于获取当前文本字符串的宽度和高度 */
static PyObject *PyFT2Font_get_width_height(PyFT2Font *self, PyObject *args)
{
    long width, height;

    /* 调用 C++ 函数 get_width_height 获取当前文本字符串的宽度和高度 */
    CALL_CPP("get_width_height", (self->x->get_width_height(&width, &height)));

    /* 将获取到的宽度和高度转换为 Python 对象并返回 */
    return Py_BuildValue("ll", width, height);
}

/* PyFT2Font_get_bitmap_offset__doc__ 字符串包含了 get_bitmap_offset 方法的文档字符串，描述了方法的返回值 */
const char *PyFT2Font_get_bitmap_offset__doc__ =
    "get_bitmap_offset(self)\n"
    "--\n\n"
    "Get the (x, y) offset in 26.6 subpixels for the bitmap if ink hangs left or below (0, 0).\n"
    "Since Matplotlib only supports left-to-right text, y is always 0.\n";

/* PyFT2Font_get_bitmap_offset 函数实现，用于获取位图的偏移量 */
static PyObject *PyFT2Font_get_bitmap_offset(PyFT2Font *self, PyObject *args)
{
    long x, y;

    /* 调用 C++ 函数 get_bitmap_offset 获取位图的偏移量 */
    CALL_CPP("get_bitmap_offset", (self->x->get_bitmap_offset(&x, &y)));

    /* 将获取到的 x 和 y 偏移量转换为 Python 对象并返回 */
    return Py_BuildValue("ll", x, y);
}

/* PyFT2Font_get_descent__doc__ 字符串包含了 get_descent 方法的文档字符串，描述了方法的返回值 */
const char *PyFT2Font_get_descent__doc__ =
    "get_descent(self)\n"
    "--\n\n"
    "Get the descent in 26.6 subpixels of the current string set by `.set_text`.\n"
    "The rotation of the string is accounted for.  To get the descent\n"
    "in pixels, divide this value by 64.\n";
// 定义一个静态函数 PyFT2Font_get_descent，返回一个 Python 对象，表示当前字体对象的下行高度
static PyObject *PyFT2Font_get_descent(PyFT2Font *self, PyObject *args)
{
    long descent;  // 声明一个长整型变量 descent，用于存储下行高度

    // 调用 C++ 函数 get_descent() 获取下行高度，并将其赋值给 descent 变量
    CALL_CPP("get_descent", (descent = self->x->get_descent()));

    // 将 descent 转换为 Python 的长整型对象并返回
    return PyLong_FromLong(descent);
}

// 设置 PyFT2Font_draw_glyphs_to_bitmap__doc__ 文档字符串，描述 draw_glyphs_to_bitmap 方法的作用
const char *PyFT2Font_draw_glyphs_to_bitmap__doc__ =
    "draw_glyphs_to_bitmap(self, antialiased=True)\n"
    "--\n\n"
    "Draw the glyphs that were loaded by `.set_text` to the bitmap.\n"
    "The bitmap size will be automatically set to include the glyphs.\n";

// 定义 PyFT2Font 类的 draw_glyphs_to_bitmap 方法，用于将加载的字形绘制到位图上
static PyObject *PyFT2Font_draw_glyphs_to_bitmap(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    bool antialiased = true;  // 定义一个布尔变量 antialiased，默认为 true
    const char *names[] = { "antialiased", NULL };  // 定义一个参数名称数组，用于参数解析

    // 使用 PyArg_ParseTupleAndKeywords 解析传入的参数，设置 antialiased 参数
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&:draw_glyphs_to_bitmap",
                                     (char **)names, &convert_bool, &antialiased)) {
        return NULL;
    }

    // 调用 C++ 函数 draw_glyphs_to_bitmap(antialiased) 绘制字形到位图上
    CALL_CPP("draw_glyphs_to_bitmap", (self->x->draw_glyphs_to_bitmap(antialiased)));

    // 返回 None 对象
    Py_RETURN_NONE;
}

// 设置 PyFT2Font_get_xys__doc__ 文档字符串，描述 get_xys 方法的作用
const char *PyFT2Font_get_xys__doc__ =
    "get_xys(self, antialiased=True)\n"
    "--\n\n"
    "Get the xy locations of the current glyphs.\n"
    "\n"
    ".. deprecated:: 3.8\n";

// 定义 PyFT2Font 类的 get_xys 方法，用于获取当前字形的 xy 位置坐标
static PyObject *PyFT2Font_get_xys(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    // 发出警告，说明 get_xys 方法在 Matplotlib 3.8 中已弃用，并且将在 Matplotlib 3.10 中删除
    char const* msg =
        "FT2Font.get_xys is deprecated since Matplotlib 3.8 and will be removed in "
        "Matplotlib 3.10 as it is not used in the library. If you rely on it, "
        "please let us know.";
    if (PyErr_WarnEx(PyExc_DeprecationWarning, msg, 1)) {
        return NULL;
    }

    bool antialiased = true;  // 定义一个布尔变量 antialiased，默认为 true
    std::vector<double> xys;  // 声明一个双精度浮点数向量 xys，用于存储坐标

    const char *names[] = { "antialiased", NULL };  // 定义一个参数名称数组，用于参数解析

    // 使用 PyArg_ParseTupleAndKeywords 解析传入的参数，设置 antialiased 参数
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&:get_xys",
                                     (char **)names, &convert_bool, &antialiased)) {
        return NULL;
    }

    // 调用 C++ 函数 get_xys(antialiased, xys) 获取字形的 xy 位置坐标
    CALL_CPP("get_xys", (self->x->get_xys(antialiased, xys)));

    // 将获取到的坐标转换为 Python 数组对象并返回
    return convert_xys_to_array(xys);
}

// 设置 PyFT2Font_draw_glyph_to_bitmap__doc__ 文档字符串，描述 draw_glyph_to_bitmap 方法的作用
const char *PyFT2Font_draw_glyph_to_bitmap__doc__ =
    "draw_glyph_to_bitmap(self, image, x, y, glyph, antialiased=True)\n"
    "--\n\n"
    "Draw a single glyph to the bitmap at pixel locations x, y\n"
    "Note it is your responsibility to set up the bitmap manually\n"
    "with ``set_bitmap_size(w, h)`` before this call is made.\n"
    "\n"
    "If you want automatic layout, use `.set_text` in combinations with\n"
    "`.draw_glyphs_to_bitmap`.  This function is instead intended for people\n"
    "who want to render individual glyphs (e.g., returned by `.load_char`)\n"
    "at precise locations.\n";

// 定义 PyFT2Font 类的 draw_glyph_to_bitmap 方法，用于将单个字形绘制到位图的指定位置
static PyObject *PyFT2Font_draw_glyph_to_bitmap(PyFT2Font *self, PyObject *args, PyObject *kwds)
{
    PyFT2Image *image;  // 声明一个 PyFT2Image 对象指针 image
    double xd, yd;  // 声明两个双精度浮点数变量 xd 和 yd
    PyGlyph *glyph;  // 声明一个 PyGlyph 对象指针 glyph
    bool antialiased = true;  // 定义一个布尔变量 antialiased，默认为 true
    const char *names[] = { "image", "x", "y", "glyph", "antialiased", NULL };  // 定义一个参数名称数组，用于参数解析
    # 解析传入的参数列表和关键字参数，检查并提取参数值，格式为 "O!ddO!|O&:draw_glyph_to_bitmap"
    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwds,
                                     "O!ddO!|O&:draw_glyph_to_bitmap",
                                     (char **)names,
                                     &PyFT2ImageType,
                                     &image,
                                     &xd,
                                     &yd,
                                     &PyGlyphType,
                                     &glyph,
                                     &convert_bool,
                                     &antialiased)) {
        # 如果解析失败，则返回空指针
        return NULL;
    }

    # 调用 C++ 函数来绘制字形到位图，使用 self->x 指针调用 draw_glyph_to_bitmap 方法
    CALL_CPP("draw_glyph_to_bitmap",
             self->x->draw_glyph_to_bitmap(*(image->x), xd, yd, glyph->glyphInd, antialiased));

    # 返回 Python 中的 None 对象，表示成功执行而无需返回额外值
    Py_RETURN_NONE;
const char *PyFT2Font_get_glyph_name__doc__ =
    "get_glyph_name(self, index)\n"
    "--\n\n"
    "Retrieve the ASCII name of a given glyph *index* in a face.\n"
    "\n"
    "Due to Matplotlib's internal design, for fonts that do not contain glyph\n"
    "names (per FT_FACE_FLAG_GLYPH_NAMES), this returns a made-up name which\n"
    "does *not* roundtrip through `.get_name_index`.\n";



static PyObject *PyFT2Font_get_glyph_name(PyFT2Font *self, PyObject *args)
{
    unsigned int glyph_number;
    char buffer[128];
    int fallback = 1;

    // 解析输入参数，获取 glyph_number
    if (!PyArg_ParseTuple(args, "I:get_glyph_name", &glyph_number)) {
        return NULL;
    }
    
    // 调用底层 C++ 函数获取字形名称
    CALL_CPP("get_glyph_name", (self->x->get_glyph_name(glyph_number, buffer, (bool)fallback)));
    
    // 将 C 字符串转换为 Python Unicode 对象并返回
    return PyUnicode_FromString(buffer);
}



const char *PyFT2Font_get_charmap__doc__ =
    "get_charmap(self)\n"
    "--\n\n"
    "Return a dict that maps the character codes of the selected charmap\n"
    "(Unicode by default) to their corresponding glyph indices.\n";



static PyObject *PyFT2Font_get_charmap(PyFT2Font *self, PyObject *args)
{
    PyObject *charmap;
    
    // 创建一个空的 Python 字典 charmap
    if (!(charmap = PyDict_New())) {
        return NULL;
    }
    
    FT_UInt index;
    FT_ULong code = FT_Get_First_Char(self->x->get_face(), &index);
    
    // 遍历字符码表，将字符码和字形索引映射存入字典
    while (index != 0) {
        PyObject *key = NULL, *val = NULL;
        bool error = (!(key = PyLong_FromLong(code))
                      || !(val = PyLong_FromLong(index))
                      || (PyDict_SetItem(charmap, key, val) == -1));
        Py_XDECREF(key);
        Py_XDECREF(val);
        if (error) {
            Py_DECREF(charmap);
            return NULL;
        }
        code = FT_Get_Next_Char(self->x->get_face(), code, &index);
    }
    
    // 返回填充好的字符码表字典
    return charmap;
}



const char *PyFT2Font_get_char_index__doc__ =
    "get_char_index(self, codepoint)\n"
    "--\n\n"
    "Return the glyph index corresponding to a character *codepoint*.\n";



static PyObject *PyFT2Font_get_char_index(PyFT2Font *self, PyObject *args)
{
    FT_UInt index;
    FT_ULong ccode;
    int fallback = 1;

    // 解析输入参数，获取字符码 ccode
    if (!PyArg_ParseTuple(args, "k:get_char_index", &ccode)) {
        return NULL;
    }

    // 调用底层 C++ 函数获取字符码对应的字形索引
    CALL_CPP("get_char_index", index = self->x->get_char_index(ccode, (bool)fallback));

    // 将字形索引转换为 Python 长整型对象并返回
    return PyLong_FromLong(index);
}



const char *PyFT2Font_get_sfnt__doc__ =
    "get_sfnt(self)\n"
    "--\n\n"
    "Load the entire SFNT names table, as a dict whose keys are\n"
    "(platform-ID, ISO-encoding-scheme, language-code, and description)\n"
    "tuples.\n";



static PyObject *PyFT2Font_get_sfnt(PyFT2Font *self, PyObject *args)
{
    PyObject *names;

    // 检查字体是否包含 SFNT 名称表
    if (!(self->x->get_face()->face_flags & FT_FACE_FLAG_SFNT)) {
        PyErr_SetString(PyExc_ValueError, "No SFNT name table");
        return NULL;
    }

    // 获取 SFNT 名称表中的条目数量
    size_t count = FT_Get_Sfnt_Name_Count(self->x->get_face());

    // 创建一个空的 Python 字典 names
    names = PyDict_New();
    if (names == NULL) {
        return NULL;
    }

    // 返回创建的空字典 names
    return names;
}
    // 循环遍历 SFNT 名称列表
    for (FT_UInt j = 0; j < count; ++j) {
        // 用于存储单个 SFNT 名称的结构体
        FT_SfntName sfnt;
        // 获取指定索引 j 处的 SFNT 名称，并将错误状态保存在 error 中
        FT_Error error = FT_Get_Sfnt_Name(self->x->get_face(), j, &sfnt);

        // 如果发生错误，释放 names 字典对象并设置异常信息，然后返回空指针
        if (error) {
            Py_DECREF(names);
            PyErr_SetString(PyExc_ValueError, "Could not get SFNT name");
            return NULL;
        }

        // 创建代表 SFNT 名称的键对象 key
        PyObject *key = Py_BuildValue(
            "HHHH", sfnt.platform_id, sfnt.encoding_id, sfnt.language_id, sfnt.name_id);
        // 如果创建 key 失败，释放 names 和 key 对象，然后返回空指针
        if (key == NULL) {
            Py_DECREF(names);
            return NULL;
        }

        // 创建代表 SFNT 名称的值对象 val
        PyObject *val = PyBytes_FromStringAndSize((const char *)sfnt.string, sfnt.string_len);
        // 如果创建 val 失败，释放 key、names 和 val 对象，然后返回空指针
        if (val == NULL) {
            Py_DECREF(key);
            Py_DECREF(names);
            return NULL;
        }

        // 将 key-val 对添加到 names 字典中，如果添加失败则释放 key、val、names 对象，然后返回空指针
        if (PyDict_SetItem(names, key, val)) {
            Py_DECREF(key);
            Py_DECREF(val);
            Py_DECREF(names);
            return NULL;
        }

        // 释放 key 和 val 对象，准备处理下一个 SFNT 名称
        Py_DECREF(key);
        Py_DECREF(val);
    }

    // 成功处理完所有 SFNT 名称，返回填充好的 names 字典对象
    return names;
# 结束函数定义
}

# 定义函数 get_name_index，用于获取给定字形名称的字形索引
const char *PyFT2Font_get_name_index__doc__ =
    "get_name_index(self, name)\n"
    "--\n\n"
    "Return the glyph index of a given glyph *name*.\n"
    "The glyph index 0 means 'undefined character code'.\n";

# 实现函数 PyFT2Font_get_name_index
static PyObject *PyFT2Font_get_name_index(PyFT2Font *self, PyObject *args)
{
    char *glyphname;
    long name_index;
    # 解析参数，获取字形名称
    if (!PyArg_ParseTuple(args, "s:get_name_index", &glyphname)) {
        return NULL;
    }
    # 调用 C++ 函数获取字形名称对应的字形索引
    CALL_CPP("get_name_index", name_index = self->x->get_name_index(glyphname));
    # 返回字形索引
    return PyLong_FromLong(name_index);
}

# 定义函数 get_ps_font_info，用于获取 PS 字体信息
const char *PyFT2Font_get_ps_font_info__doc__ =
    "get_ps_font_info(self)\n"
    "--\n\n"
    "Return the information in the PS Font Info structure.\n";

# 实现函数 PyFT2Font_get_ps_font_info
static PyObject *PyFT2Font_get_ps_font_info(PyFT2Font *self, PyObject *args)
{
    PS_FontInfoRec fontinfo;

    # 获取 PS 字体信息
    FT_Error error = FT_Get_PS_Font_Info(self->x->get_face(), &fontinfo);
    if (error) {
        PyErr_SetString(PyExc_ValueError, "Could not get PS font info");
        return NULL;
    }

    # 返回 PS 字体信息
    return Py_BuildValue("ssssslbhH",
                         fontinfo.version ? fontinfo.version : "",
                         fontinfo.notice ? fontinfo.notice : "",
                         fontinfo.full_name ? fontinfo.full_name : "",
                         fontinfo.family_name ? fontinfo.family_name : "",
                         fontinfo.weight ? fontinfo.weight : "",
                         fontinfo.italic_angle,
                         fontinfo.is_fixed_pitch,
                         fontinfo.underline_position,
                         fontinfo.underline_thickness);
}

# 定义函数 get_sfnt_table，用于获取指定的 SFNT 表
const char *PyFT2Font_get_sfnt_table__doc__ =
    "get_sfnt_table(self, name)\n"
    "--\n\n"
    "Return one of the following SFNT tables: head, maxp, OS/2, hhea, "
    "vhea, post, or pclt.\n";

# 实现函数 PyFT2Font_get_sfnt_table
static PyObject *PyFT2Font_get_sfnt_table(PyFT2Font *self, PyObject *args)
{
    char *tagname;
    # 解析参数，获取 SFNT 表名称
    if (!PyArg_ParseTuple(args, "s:get_sfnt_table", &tagname)) {
        return NULL;
    }

    int tag;
    const char *tags[] = { "head", "maxp", "OS/2", "hhea", "vhea", "post", "pclt", NULL };

    # 遍历 SFNT 表名称列表，匹配输入的表名称
    for (tag = 0; tags[tag] != NULL; tag++) {
        if (strncmp(tagname, tags[tag], 5) == 0) {
            break;
        }
    }

    # 获取指定 SFNT 表的数据
    void *table = FT_Get_Sfnt_Table(self->x->get_face(), (FT_Sfnt_Tag)tag);
    if (!table) {
        Py_RETURN_NONE;
    }

    # 根据表类型进行不同的处理
    switch (tag) {
    # case 0分支：定义用于解析TrueType字体头部信息的字典格式字符串
    char head_dict[] =
        "{s:(h,H), s:(h,H), s:l, s:l, s:H, s:H,"
        "s:(l,l), s:(l,l), s:h, s:h, s:h, s:h, s:H, s:H, s:h, s:h, s:h}";
    # 将table强制转换为TT_Header指针类型
    TT_Header *t = (TT_Header *)table;
    # 使用Py_BuildValue函数构建Python对象，按照head_dict格式生成字典对象，包括TrueType头部的各个字段
    return Py_BuildValue(head_dict,
                         "version", FIXED_MAJOR(t->Table_Version), FIXED_MINOR(t->Table_Version),
                         "fontRevision", FIXED_MAJOR(t->Font_Revision), FIXED_MINOR(t->Font_Revision),
                         "checkSumAdjustment", t->CheckSum_Adjust,
                         "magicNumber", t->Magic_Number,
                         "flags", t->Flags,
                         "unitsPerEm", t->Units_Per_EM,
                         "created", t->Created[0], t->Created[1],
                         "modified", t->Modified[0], t->Modified[1],
                         "xMin", t->xMin,
                         "yMin", t->yMin,
                         "xMax", t->xMax,
                         "yMax", t->yMax,
                         "macStyle", t->Mac_Style,
                         "lowestRecPPEM", t->Lowest_Rec_PPEM,
                         "fontDirectionHint", t->Font_Direction,
                         "indexToLocFormat", t->Index_To_Loc_Format,
                         "glyphDataFormat", t->Glyph_Data_Format);



    # case 1分支：定义用于解析TrueType最大轮廓信息的字典格式字符串
    char maxp_dict[] =
        "{s:(h,H), s:H, s:H, s:H, s:H, s:H, s:H,"
        "s:H, s:H, s:H, s:H, s:H, s:H, s:H, s:H, s:H}";
    # 将table强制转换为TT_MaxProfile指针类型
    TT_MaxProfile *t = (TT_MaxProfile *)table;
    # 使用Py_BuildValue函数构建Python对象，按照maxp_dict格式生成字典对象，包括TrueType最大轮廓的各个字段
    return Py_BuildValue(maxp_dict,
                         "version", FIXED_MAJOR(t->version), FIXED_MINOR(t->version),
                         "numGlyphs", t->numGlyphs,
                         "maxPoints", t->maxPoints,
                         "maxContours", t->maxContours,
                         "maxComponentPoints", t->maxCompositePoints,
                         "maxComponentContours", t->maxCompositeContours,
                         "maxZones", t->maxZones,
                         "maxTwilightPoints", t->maxTwilightPoints,
                         "maxStorage", t->maxStorage,
                         "maxFunctionDefs", t->maxFunctionDefs,
                         "maxInstructionDefs", t->maxInstructionDefs,
                         "maxStackElements", t->maxStackElements,
                         "maxSizeOfInstructions", t->maxSizeOfInstructions,
                         "maxComponentElements", t->maxComponentElements,
                         "maxComponentDepth", t->maxComponentDepth);
    # 对于 case 2，根据给定的字典格式化输出 TrueType 的 OS/2 表数据
    case 2: {
        # 定义字符串，表示将要构建的字典的结构
        char os_2_dict[] =
            "{s:H, s:h, s:H, s:H, s:H, s:h, s:h, s:h,"
            "s:h, s:h, s:h, s:h, s:h, s:h, s:h, s:h, s:y#, s:(kkkk),"
            "s:y#, s:H, s:H, s:H}";
        # 将传入的表格数据强制转换为 TT_OS2 结构体指针
        TT_OS2 *t = (TT_OS2 *)table;
        # 使用 Py_BuildValue 函数根据 os_2_dict 格式化输出各个字段的数值
        return Py_BuildValue(os_2_dict,
                             "version", t->version,
                             "xAvgCharWidth", t->xAvgCharWidth,
                             "usWeightClass", t->usWeightClass,
                             "usWidthClass", t->usWidthClass,
                             "fsType", t->fsType,
                             "ySubscriptXSize", t->ySubscriptXSize,
                             "ySubscriptYSize", t->ySubscriptYSize,
                             "ySubscriptXOffset", t->ySubscriptXOffset,
                             "ySubscriptYOffset", t->ySubscriptYOffset,
                             "ySuperscriptXSize", t->ySuperscriptXSize,
                             "ySuperscriptYSize", t->ySuperscriptYSize,
                             "ySuperscriptXOffset", t->ySuperscriptXOffset,
                             "ySuperscriptYOffset", t->ySuperscriptYOffset,
                             "yStrikeoutSize", t->yStrikeoutSize,
                             "yStrikeoutPosition", t->yStrikeoutPosition,
                             "sFamilyClass", t->sFamilyClass,
                             "panose", t->panose, Py_ssize_t(10),
                             "ulCharRange", t->ulUnicodeRange1, t->ulUnicodeRange2, t->ulUnicodeRange3, t->ulUnicodeRange4,
                             "achVendID", t->achVendID, Py_ssize_t(4),
                             "fsSelection", t->fsSelection,
                             "fsFirstCharIndex", t->usFirstCharIndex,
                             "fsLastCharIndex", t->usLastCharIndex);
    }

    # 对于 case 3，根据给定的字典格式化输出 TrueType 的 Horizontal Header 表数据
    case 3: {
        # 定义字符串，表示将要构建的字典的结构
        char hhea_dict[] =
            "{s:(h,H), s:h, s:h, s:h, s:H, s:h, s:h, s:h,"
            "s:h, s:h, s:h, s:h, s:H}";
        # 将传入的表格数据强制转换为 TT_HoriHeader 结构体指针
        TT_HoriHeader *t = (TT_HoriHeader *)table;
        # 使用 Py_BuildValue 函数根据 hhea_dict 格式化输出各个字段的数值
        return Py_BuildValue(hhea_dict,
                             "version", FIXED_MAJOR(t->Version), FIXED_MINOR(t->Version),
                             "ascent", t->Ascender,
                             "descent", t->Descender,
                             "lineGap", t->Line_Gap,
                             "advanceWidthMax", t->advance_Width_Max,
                             "minLeftBearing", t->min_Left_Side_Bearing,
                             "minRightBearing", t->min_Right_Side_Bearing,
                             "xMaxExtent", t->xMax_Extent,
                             "caretSlopeRise", t->caret_Slope_Rise,
                             "caretSlopeRun", t->caret_Slope_Run,
                             "caretOffset", t->caret_Offset,
                             "metricDataFormat", t->metric_Data_Format,
                             "numOfLongHorMetrics", t->number_Of_HMetrics);
    }
    case 4: {
        // 定义垂直头部表的字典格式字符串
        char vhea_dict[] =
            "{s:(h,H), s:h, s:h, s:h, s:H, s:h, s:h, s:h,"
            "s:h, s:h, s:h, s:h, s:H}";
        // 将传入的表格指针转换为垂直头部表结构体指针
        TT_VertHeader *t = (TT_VertHeader *)table;
        // 使用字典格式字符串构建 Python 对象并返回
        return Py_BuildValue(vhea_dict,
                             "version", FIXED_MAJOR(t->Version), FIXED_MINOR(t->Version),
                             "vertTypoAscender", t->Ascender,
                             "vertTypoDescender", t->Descender,
                             "vertTypoLineGap", t->Line_Gap,
                             "advanceHeightMax", t->advance_Height_Max,
                             "minTopSideBearing", t->min_Top_Side_Bearing,
                             "minBottomSizeBearing", t->min_Bottom_Side_Bearing,
                             "yMaxExtent", t->yMax_Extent,
                             "caretSlopeRise", t->caret_Slope_Rise,
                             "caretSlopeRun", t->caret_Slope_Run,
                             "caretOffset", t->caret_Offset,
                             "metricDataFormat", t->metric_Data_Format,
                             "numOfLongVerMetrics", t->number_Of_VMetrics);
    }
    case 5: {
        // 定义字体后处理表的字典格式字符串
        char post_dict[] = "{s:(h,H), s:(h,H), s:h, s:h, s:k, s:k, s:k, s:k, s:k}";
        // 将传入的表格指针转换为字体后处理表结构体指针
        TT_Postscript *t = (TT_Postscript *)table;
        // 使用字典格式字符串构建 Python 对象并返回
        return Py_BuildValue(post_dict,
                             "format", FIXED_MAJOR(t->FormatType), FIXED_MINOR(t->FormatType),
                             "italicAngle", FIXED_MAJOR(t->italicAngle), FIXED_MINOR(t->italicAngle),
                             "underlinePosition", t->underlinePosition,
                             "underlineThickness", t->underlineThickness,
                             "isFixedPitch", t->isFixedPitch,
                             "minMemType42", t->minMemType42,
                             "maxMemType42", t->maxMemType42,
                             "minMemType1", t->minMemType1,
                             "maxMemType1", t->maxMemType1);
    }
    # 当 case 值为 6 时执行以下代码块，用于处理 TrueType 字体文件中的 pclt 表信息
    case 6: {
        # 定义一个字符数组，表示 pclt 表结构的 Python 字典格式
        char pclt_dict[] =
            "{s:(h,H), s:k, s:H, s:H, s:H, s:H, s:H, s:H, s:y#, s:y#, s:b, "
            "s:b, s:b}";
        # 将输入的表格指针转换为 TT_PCLT 结构体指针
        TT_PCLT *t = (TT_PCLT *)table;
        # 构建 Python 字典对象，映射 TrueType pclt 表中的字段到对应的值
        return Py_BuildValue(pclt_dict,
                             "version", FIXED_MAJOR(t->Version), FIXED_MINOR(t->Version),
                             "fontNumber", t->FontNumber,
                             "pitch", t->Pitch,
                             "xHeight", t->xHeight,
                             "style", t->Style,
                             "typeFamily", t->TypeFamily,
                             "capHeight", t->CapHeight,
                             "symbolSet", t->SymbolSet,
                             "typeFace", t->TypeFace, Py_ssize_t(16),
                             "characterComplement", t->CharacterComplement, Py_ssize_t(8),
                             "strokeWeight", t->StrokeWeight,
                             "widthType", t->WidthType,
                             "serifStyle", t->SerifStyle);
    }
    # 如果 case 值不是 6，则返回 None 对象
    default:
        Py_RETURN_NONE;
    }
// 定义 PyFT2Font_get_path__doc__ 字符串，用于描述 get_path 方法的文档字符串
const char *PyFT2Font_get_path__doc__ =
    "get_path(self)\n"
    "--\n\n"
    "Get the path data from the currently loaded glyph as a tuple of vertices, "
    "codes.\n";

// 定义 PyFT2Font_get_path 函数，获取当前加载的字形的路径数据
static PyObject *PyFT2Font_get_path(PyFT2Font *self, PyObject *args)
{
    // 调用 C++ 函数，返回当前字形的路径数据
    CALL_CPP("get_path", return self->x->get_path());
}

// 定义 PyFT2Font_get_image__doc__ 字符串，用于描述 get_image 方法的文档字符串
const char *PyFT2Font_get_image__doc__ =
    "get_image(self)\n"
    "--\n\n"
    "Return the underlying image buffer for this font object.\n";

// 定义 PyFT2Font_get_image 函数，返回字体对象的底层图像缓冲区
static PyObject *PyFT2Font_get_image(PyFT2Font *self, PyObject *args)
{
    // 获取字体对象的图像数据
    FT2Image &im = self->x->get_image();
    // 创建 NumPy 数组来保存图像数据
    npy_intp dims[] = {(npy_intp)im.get_height(), (npy_intp)im.get_width() };
    return PyArray_SimpleNewFromData(2, dims, NPY_UBYTE, im.get_buffer());
}

// 定义 PyFT2Font_postscript_name 函数，返回字体对象的 Postscript 名称
static PyObject *PyFT2Font_postscript_name(PyFT2Font *self, void *closure)
{
    // 获取字体对象的 Postscript 名称
    const char *ps_name = FT_Get_Postscript_Name(self->x->get_face());
    // 如果名称为 NULL，则返回 "UNAVAILABLE"
    if (ps_name == NULL) {
        ps_name = "UNAVAILABLE";
    }
    return PyUnicode_FromString(ps_name);
}

// 定义 PyFT2Font_num_faces 函数，返回字体对象的面数
static PyObject *PyFT2Font_num_faces(PyFT2Font *self, void *closure)
{
    return PyLong_FromLong(self->x->get_face()->num_faces);
}

// 定义 PyFT2Font_family_name 函数，返回字体对象的字族名称
static PyObject *PyFT2Font_family_name(PyFT2Font *self, void *closure)
{
    // 获取字体对象的字族名称
    const char *name = self->x->get_face()->family_name;
    // 如果名称为 NULL，则返回 "UNAVAILABLE"
    if (name == NULL) {
        name = "UNAVAILABLE";
    }
    return PyUnicode_FromString(name);
}

// 定义 PyFT2Font_style_name 函数，返回字体对象的风格名称
static PyObject *PyFT2Font_style_name(PyFT2Font *self, void *closure)
{
    // 获取字体对象的风格名称
    const char *name = self->x->get_face()->style_name;
    // 如果名称为 NULL，则返回 "UNAVAILABLE"
    if (name == NULL) {
        name = "UNAVAILABLE";
    }
    return PyUnicode_FromString(name);
}

// 定义 PyFT2Font_face_flags 函数，返回字体对象的面标志
static PyObject *PyFT2Font_face_flags(PyFT2Font *self, void *closure)
{
    return PyLong_FromLong(self->x->get_face()->face_flags);
}

// 定义 PyFT2Font_style_flags 函数，返回字体对象的风格标志
static PyObject *PyFT2Font_style_flags(PyFT2Font *self, void *closure)
{
    return PyLong_FromLong(self->x->get_face()->style_flags);
}

// 定义 PyFT2Font_num_glyphs 函数，返回字体对象的字形数
static PyObject *PyFT2Font_num_glyphs(PyFT2Font *self, void *closure)
{
    return PyLong_FromLong(self->x->get_face()->num_glyphs);
}

// 定义 PyFT2Font_num_fixed_sizes 函数，返回字体对象的固定大小数
static PyObject *PyFT2Font_num_fixed_sizes(PyFT2Font *self, void *closure)
{
    return PyLong_FromLong(self->x->get_face()->num_fixed_sizes);
}

// 定义 PyFT2Font_num_charmaps 函数，返回字体对象的字符映射数
static PyObject *PyFT2Font_num_charmaps(PyFT2Font *self, void *closure)
{
    return PyLong_FromLong(self->x->get_face()->num_charmaps);
}

// 定义 PyFT2Font_scalable 函数，检查字体对象是否可缩放
static PyObject *PyFT2Font_scalable(PyFT2Font *self, void *closure)
{
    // 如果字体对象是可缩放的，返回 Py_True，否则返回 Py_False
    if (FT_IS_SCALABLE(self->x->get_face())) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

// 定义 PyFT2Font_units_per_EM 函数，返回字体对象的 EM 单位数
static PyObject *PyFT2Font_units_per_EM(PyFT2Font *self, void *closure)
{
    return PyLong_FromLong(self->x->get_face()->units_per_EM);
}

// 定义 PyFT2Font_get_bbox 函数，返回字体对象的包围盒
static PyObject *PyFT2Font_get_bbox(PyFT2Font *self, void *closure)
{
    // 获取字体对象的包围盒
    FT_BBox *bbox = &(self->x->get_face()->bbox);
    // 返回包围盒的四个边界值
    return Py_BuildValue("llll",
                         bbox->xMin, bbox->yMin, bbox->xMax, bbox->yMax);
}

// 定义 PyFT2Font_ascender 函数，返回字体对象的上升高度
static PyObject *PyFT2Font_ascender(PyFT2Font *self, void *closure)
{
    return PyLong_FromLong(self->x->get_face()->ascender);
}
// 返回当前字体对象的 descender 属性作为 Python 的长整型对象
static PyObject *PyFT2Font_descender(PyFT2Font *self, void *closure)
{
    return PyLong_FromLong(self->x->get_face()->descender);
}

// 返回当前字体对象的 height 属性作为 Python 的长整型对象
static PyObject *PyFT2Font_height(PyFT2Font *self, void *closure)
{
    return PyLong_FromLong(self->x->get_face()->height);
}

// 返回当前字体对象的 max_advance_width 属性作为 Python 的长整型对象
static PyObject *PyFT2Font_max_advance_width(PyFT2Font *self, void *closure)
{
    return PyLong_FromLong(self->x->get_face()->max_advance_width);
}

// 返回当前字体对象的 max_advance_height 属性作为 Python 的长整型对象
static PyObject *PyFT2Font_max_advance_height(PyFT2Font *self, void *closure)
{
    return PyLong_FromLong(self->x->get_face()->max_advance_height);
}

// 返回当前字体对象的 underline_position 属性作为 Python 的长整型对象
static PyObject *PyFT2Font_underline_position(PyFT2Font *self, void *closure)
{
    return PyLong_FromLong(self->x->get_face()->underline_position);
}

// 返回当前字体对象的 underline_thickness 属性作为 Python 的长整型对象
static PyObject *PyFT2Font_underline_thickness(PyFT2Font *self, void *closure)
{
    return PyLong_FromLong(self->x->get_face()->underline_thickness);
}

// 根据字体对象的 stream.close 属性，返回相应的 Python 对象或文件名字符串
static PyObject *PyFT2Font_fname(PyFT2Font *self, void *closure)
{
    if (self->stream.close) {  // 如果 stream.close 为真，表示构造函数传入了文件名。
        // 返回通过调用 PyObject_GetAttrString 获取的文件名字符串对象
        return PyObject_GetAttrString(self->py_file, "name");
    } else {
        // 增加引用计数并返回字体对象关联的 Python 文件对象
        Py_INCREF(self->py_file);
        return self->py_file;
    }
}

// 将字体对象的图像数据填充到给定的 Py_buffer 结构中，并返回状态标志
static int PyFT2Font_get_buffer(PyFT2Font *self, Py_buffer *buf, int flags)
{
    // 获取字体对象的图像对象的引用
    FT2Image &im = self->x->get_image();

    // 增加字体对象的引用计数
    Py_INCREF(self);
    // 设置 Py_buffer 结构的对象指针为当前字体对象
    buf->obj = (PyObject *)self;
    // 设置 Py_buffer 结构的 buf 指向图像数据的首地址
    buf->buf = im.get_buffer();
    // 设置 Py_buffer 结构的 len 为图像数据的总长度（像素数）
    buf->len = im.get_width() * im.get_height();
    // 设置 Py_buffer 结构的 readonly 标志为可写
    buf->readonly = 0;
    // 设置 Py_buffer 结构的 format 为单字节字符 'B'
    buf->format = (char *)"B";
    // 设置 Py_buffer 结构的维度为 2（二维数组）
    buf->ndim = 2;
    // 设置 Py_buffer 结构的形状为图像的高度和宽度
    self->shape[0] = im.get_height();
    self->shape[1] = im.get_width();
    buf->shape = self->shape;
    // 设置 Py_buffer 结构的步长为图像宽度和 1
    self->strides[0] = im.get_width();
    self->strides[1] = 1;
    buf->strides = self->strides;
    // 设置 Py_buffer 结构的子偏移为空
    buf->suboffsets = NULL;
    // 设置 Py_buffer 结构的单元大小为 1 字节
    buf->itemsize = 1;
    // 设置 Py_buffer 结构的内部数据为 NULL
    buf->internal = NULL;

    // 返回 1 表示成功填充 Py_buffer 结构
    return 1;
}

// 初始化并返回 PyFT2Font 对象的类型对象
static PyTypeObject *PyFT2Font_init_type()
{
    // 定义一个静态的 PyGetSetDef 结构体数组，用于描述 Python 中的属性获取器和设置器
    static PyGetSetDef getset[] = {
        // 属性名为 "postscript_name"，获取器为 PyFT2Font_postscript_name
        {(char *)"postscript_name", (getter)PyFT2Font_postscript_name, NULL, NULL, NULL},
        // 属性名为 "num_faces"，获取器为 PyFT2Font_num_faces
        {(char *)"num_faces", (getter)PyFT2Font_num_faces, NULL, NULL, NULL},
        // 属性名为 "family_name"，获取器为 PyFT2Font_family_name
        {(char *)"family_name", (getter)PyFT2Font_family_name, NULL, NULL, NULL},
        // 属性名为 "style_name"，获取器为 PyFT2Font_style_name
        {(char *)"style_name", (getter)PyFT2Font_style_name, NULL, NULL, NULL},
        // 属性名为 "face_flags"，获取器为 PyFT2Font_face_flags
        {(char *)"face_flags", (getter)PyFT2Font_face_flags, NULL, NULL, NULL},
        // 属性名为 "style_flags"，获取器为 PyFT2Font_style_flags
        {(char *)"style_flags", (getter)PyFT2Font_style_flags, NULL, NULL, NULL},
        // 属性名为 "num_glyphs"，获取器为 PyFT2Font_num_glyphs
        {(char *)"num_glyphs", (getter)PyFT2Font_num_glyphs, NULL, NULL, NULL},
        // 属性名为 "num_fixed_sizes"，获取器为 PyFT2Font_num_fixed_sizes
        {(char *)"num_fixed_sizes", (getter)PyFT2Font_num_fixed_sizes, NULL, NULL, NULL},
        // 属性名为 "num_charmaps"，获取器为 PyFT2Font_num_charmaps
        {(char *)"num_charmaps", (getter)PyFT2Font_num_charmaps, NULL, NULL, NULL},
        // 属性名为 "scalable"，获取器为 PyFT2Font_scalable
        {(char *)"scalable", (getter)PyFT2Font_scalable, NULL, NULL, NULL},
        // 属性名为 "units_per_EM"，获取器为 PyFT2Font_units_per_EM
        {(char *)"units_per_EM", (getter)PyFT2Font_units_per_EM, NULL, NULL, NULL},
        // 属性名为 "bbox"，获取器为 PyFT2Font_get_bbox
        {(char *)"bbox", (getter)PyFT2Font_get_bbox, NULL, NULL, NULL},
        // 属性名为 "ascender"，获取器为 PyFT2Font_ascender
        {(char *)"ascender", (getter)PyFT2Font_ascender, NULL, NULL, NULL},
        // 属性名为 "descender"，获取器为 PyFT2Font_descender
        {(char *)"descender", (getter)PyFT2Font_descender, NULL, NULL, NULL},
        // 属性名为 "height"，获取器为 PyFT2Font_height
        {(char *)"height", (getter)PyFT2Font_height, NULL, NULL, NULL},
        // 属性名为 "max_advance_width"，获取器为 PyFT2Font_max_advance_width
        {(char *)"max_advance_width", (getter)PyFT2Font_max_advance_width, NULL, NULL, NULL},
        // 属性名为 "max_advance_height"，获取器为 PyFT2Font_max_advance_height
        {(char *)"max_advance_height", (getter)PyFT2Font_max_advance_height, NULL, NULL, NULL},
        // 属性名为 "underline_position"，获取器为 PyFT2Font_underline_position
        {(char *)"underline_position", (getter)PyFT2Font_underline_position, NULL, NULL, NULL},
        // 属性名为 "underline_thickness"，获取器为 PyFT2Font_underline_thickness
        {(char *)"underline_thickness", (getter)PyFT2Font_underline_thickness, NULL, NULL, NULL},
        // 属性名为 "fname"，获取器为 PyFT2Font_fname
        {(char *)"fname", (getter)PyFT2Font_fname, NULL, NULL, NULL},
        // 结束符，表示结构体数组的结束
        {NULL}
    };
    // 定义一个静态数组 methods，包含了多个 PyMethodDef 结构体
    static PyMethodDef methods[] = {
        // {"方法名", 函数指针, 参数标志, 方法文档字符串}
        {"clear", (PyCFunction)PyFT2Font_clear, METH_NOARGS, PyFT2Font_clear__doc__},
        {"set_size", (PyCFunction)PyFT2Font_set_size, METH_VARARGS, PyFT2Font_set_size__doc__},
        {"set_charmap", (PyCFunction)PyFT2Font_set_charmap, METH_VARARGS, PyFT2Font_set_charmap__doc__},
        {"select_charmap", (PyCFunction)PyFT2Font_select_charmap, METH_VARARGS, PyFT2Font_select_charmap__doc__},
        {"get_kerning", (PyCFunction)PyFT2Font_get_kerning, METH_VARARGS, PyFT2Font_get_kerning__doc__},
        {"set_text", (PyCFunction)PyFT2Font_set_text, METH_VARARGS|METH_KEYWORDS, PyFT2Font_set_text__doc__},
        {"_get_fontmap", (PyCFunction)PyFT2Font_get_fontmap, METH_VARARGS|METH_KEYWORDS, PyFT2Font_get_fontmap__doc__},
        {"get_num_glyphs", (PyCFunction)PyFT2Font_get_num_glyphs, METH_NOARGS, PyFT2Font_get_num_glyphs__doc__},
        {"load_char", (PyCFunction)PyFT2Font_load_char, METH_VARARGS|METH_KEYWORDS, PyFT2Font_load_char__doc__},
        {"load_glyph", (PyCFunction)PyFT2Font_load_glyph, METH_VARARGS|METH_KEYWORDS, PyFT2Font_load_glyph__doc__},
        {"get_width_height", (PyCFunction)PyFT2Font_get_width_height, METH_NOARGS, PyFT2Font_get_width_height__doc__},
        {"get_bitmap_offset", (PyCFunction)PyFT2Font_get_bitmap_offset, METH_NOARGS, PyFT2Font_get_bitmap_offset__doc__},
        {"get_descent", (PyCFunction)PyFT2Font_get_descent, METH_NOARGS, PyFT2Font_get_descent__doc__},
        {"draw_glyphs_to_bitmap", (PyCFunction)PyFT2Font_draw_glyphs_to_bitmap, METH_VARARGS|METH_KEYWORDS, PyFT2Font_draw_glyphs_to_bitmap__doc__},
        {"get_xys", (PyCFunction)PyFT2Font_get_xys, METH_VARARGS|METH_KEYWORDS, PyFT2Font_get_xys__doc__},
        {"draw_glyph_to_bitmap", (PyCFunction)PyFT2Font_draw_glyph_to_bitmap, METH_VARARGS|METH_KEYWORDS, PyFT2Font_draw_glyph_to_bitmap__doc__},
        {"get_glyph_name", (PyCFunction)PyFT2Font_get_glyph_name, METH_VARARGS, PyFT2Font_get_glyph_name__doc__},
        {"get_charmap", (PyCFunction)PyFT2Font_get_charmap, METH_NOARGS, PyFT2Font_get_charmap__doc__},
        {"get_char_index", (PyCFunction)PyFT2Font_get_char_index, METH_VARARGS, PyFT2Font_get_char_index__doc__},
        {"get_sfnt", (PyCFunction)PyFT2Font_get_sfnt, METH_NOARGS, PyFT2Font_get_sfnt__doc__},
        {"get_name_index", (PyCFunction)PyFT2Font_get_name_index, METH_VARARGS, PyFT2Font_get_name_index__doc__},
        {"get_ps_font_info", (PyCFunction)PyFT2Font_get_ps_font_info, METH_NOARGS, PyFT2Font_get_ps_font_info__doc__},
        {"get_sfnt_table", (PyCFunction)PyFT2Font_get_sfnt_table, METH_VARARGS, PyFT2Font_get_sfnt_table__doc__},
        {"get_path", (PyCFunction)PyFT2Font_get_path, METH_NOARGS, PyFT2Font_get_path__doc__},
        {"get_image", (PyCFunction)PyFT2Font_get_image, METH_NOARGS, PyFT2Font_get_image__doc__},
        // 最后一个元素为 NULL，表示 methods 数组的结束
        {NULL}
    };

    // 定义一个 PyBufferProcs 结构体 buffer_procs，并设置 bf_getbuffer 字段为 PyFT2Font_get_buffer 函数的指针
    static PyBufferProcs buffer_procs;
    buffer_procs.bf_getbuffer = (getbufferproc)PyFT2Font_get_buffer;
    # 设置 PyFT2FontType 的类型名称为 "matplotlib.ft2font.FT2Font"
    PyFT2FontType.tp_name = "matplotlib.ft2font.FT2Font";
    # 将 PyFT2FontType 的文档字符串设置为 PyFT2Font_init__doc__ 的值
    PyFT2FontType.tp_doc = PyFT2Font_init__doc__;
    # 设置 PyFT2FontType 的基本大小为 PyFT2Font 结构体的大小
    PyFT2FontType.tp_basicsize = sizeof(PyFT2Font);
    # 设置 PyFT2FontType 的析构函数为 PyFT2Font_dealloc 函数
    PyFT2FontType.tp_dealloc = (destructor)PyFT2Font_dealloc;
    # 设置 PyFT2FontType 的标志，包括默认标志和基础类型标志
    PyFT2FontType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    # 设置 PyFT2FontType 的方法为 methods 数组中定义的方法
    PyFT2FontType.tp_methods = methods;
    # 设置 PyFT2FontType 的属性获取和设置器为 getset 数组中定义的获取和设置器
    PyFT2FontType.tp_getset = getset;
    # 设置 PyFT2FontType 的新对象构造函数为 PyFT2Font_new 函数
    PyFT2FontType.tp_new = PyFT2Font_new;
    # 设置 PyFT2FontType 的初始化函数为 PyFT2Font_init 函数
    PyFT2FontType.tp_init = (initproc)PyFT2Font_init;
    # 设置 PyFT2FontType 的缓冲区协议为指向 buffer_procs 的指针
    PyFT2FontType.tp_as_buffer = &buffer_procs;

    # 返回指向 PyFT2FontType 结构体的指针
    return &PyFT2FontType;
// 定义一个静态的 PyModuleDef 结构体变量 moduledef，用于定义 Python 模块的基本信息，模块名为 "ft2font"
static struct PyModuleDef moduledef = { PyModuleDef_HEAD_INIT, "ft2font" };

// Python 模块初始化函数 PyInit_ft2font
PyMODINIT_FUNC PyInit_ft2font(void)
{
    // 导入 NumPy 数组对象的 C API
    import_array();

    // 初始化 FreeType 库，如果初始化失败则返回运行时错误异常
    if (FT_Init_FreeType(&_ft2Library)) {
        return PyErr_Format(
            PyExc_RuntimeError, "Could not initialize the freetype2 library");
    }
    
    FT_Int major, minor, patch;
    char version_string[64];
    
    // 获取 FreeType 库的版本信息
    FT_Library_Version(_ft2Library, &major, &minor, &patch);
    
    // 将版本信息格式化成字符串
    snprintf(version_string, sizeof(version_string), "%d.%d.%d", major, minor, patch);

    PyObject *m;
    // 返回 Python 模块对象 m
    return m;
}
```