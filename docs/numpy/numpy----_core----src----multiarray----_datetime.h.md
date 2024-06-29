# `.\numpy\numpy\_core\src\multiarray\_datetime.h`

```
#ifndef NUMPY_CORE_SRC_MULTIARRAY__DATETIME_H_
#define NUMPY_CORE_SRC_MULTIARRAY__DATETIME_H_

// 外部声明 datetime 字符串数组，用于存储日期时间单元字符串
extern NPY_NO_EXPORT char const *_datetime_strings[NPY_DATETIME_NUMUNITS];

// 外部声明 _days_per_month_table，用于存储每月天数的表格
extern NPY_NO_EXPORT int _days_per_month_table[2][12];

// 导入 Python 中的 datetime 模块
NPY_NO_EXPORT void
numpy_pydatetime_import(void);

/*
 * 如果给定的年份是闰年则返回 1，否则返回 0。
 */
NPY_NO_EXPORT int
is_leapyear(npy_int64 year);

/*
 * 计算从 1970 年纪元开始到指定日期时间结构的天数偏移量。
 */
NPY_NO_EXPORT npy_int64
get_datetimestruct_days(const npy_datetimestruct *dts);

/*
 * 使用提供的元数据创建 datetime 或 timedelta 类型的 dtype。
 */
NPY_NO_EXPORT PyArray_Descr *
create_datetime_dtype(int type_num, PyArray_DatetimeMetaData *meta);

/*
 * 使用给定的单元创建 datetime 或 timedelta 类型的 dtype。
 */
NPY_NO_EXPORT PyArray_Descr *
create_datetime_dtype_with_unit(int type_num, NPY_DATETIMEUNIT unit);

/*
 * 返回 datetime dtype 中包含的 DateTimeMetaData 指针。
 */
NPY_NO_EXPORT PyArray_DatetimeMetaData *
get_datetime_metadata_from_dtype(PyArray_Descr *dtype);

/*
 * 在数组中查找与 datetime64 类型匹配的字符串类型。
 */
NPY_NO_EXPORT int
find_string_array_datetime64_type(PyArrayObject *arr,
        PyArray_DatetimeMetaData *meta);

/*
 * type1 和 type2 必须是 NPY_DATETIME 或 NPY_TIMEDELTA。
 * 应用类型提升规则，返回提升后的类型。
 */
NPY_NO_EXPORT PyArray_Descr *
datetime_type_promotion(PyArray_Descr *type1, PyArray_Descr *type2);

/*
 * 从 'datetime64[D]' 值中提取当前年份内的月份编号。1 表示一月，以此类推。
 */
NPY_NO_EXPORT int
days_to_month_number(npy_datetime days);

/*
 * 解析元数据字符串到元数据 C 结构中。
 * 成功返回 0，失败返回 -1。
 */
NPY_NO_EXPORT int
parse_datetime_metadata_from_metastr(char const *metastr, Py_ssize_t len,
                                    PyArray_DatetimeMetaData *out_meta);

/*
 * 将日期时间类型字符串解析为 dtype 描述符对象。
 * 'type' 字符串应为以 NULL 结尾的字符串，len 应为其长度。
 */
NPY_NO_EXPORT PyArray_Descr *
parse_dtype_from_datetime_typestr(char const *typestr, Py_ssize_t len);

/*
 * 将由 'str' 和 'len' 组成的子字符串转换为日期时间单元的枚举值。
 * 'metastr' 用于错误消息，可能为 NULL。
 * 成功返回 0，失败返回 -1。
 */
NPY_NO_EXPORT NPY_DATETIMEUNIT
parse_datetime_unit_from_string(char const *str, Py_ssize_t len, char const *metastr);

/*
 * 将除数转换为较小单位的倍数，用于日期时间元数据。
 * 'metastr' 用于错误消息，如果除数不适用，则可以为 NULL。
 * 成功返回 0，失败返回 -1。
 */
NPY_NO_EXPORT int
convert_datetime_divisor_to_multiple(PyArray_DatetimeMetaData *meta,
                                    int den, char const *metastr);

#endif  // NUMPY_CORE_SRC_MULTIARRAY__DATETIME_H_
/*
 * Determines whether the 'divisor' metadata divides evenly into
 * the 'dividend' metadata.
 */
NPY_NO_EXPORT npy_bool
datetime_metadata_divides(
                        PyArray_DatetimeMetaData *dividend,
                        PyArray_DatetimeMetaData *divisor,
                        int strict_with_nonlinear_units);



/*
 * This provides the casting rules for the DATETIME data type units.
 *
 * Notably, there is a barrier between 'date units' and 'time units'
 * for all but 'unsafe' casting.
 */
NPY_NO_EXPORT npy_bool
can_cast_datetime64_units(NPY_DATETIMEUNIT src_unit,
                          NPY_DATETIMEUNIT dst_unit,
                          NPY_CASTING casting);



/*
 * This provides the casting rules for the DATETIME data type metadata.
 */
NPY_NO_EXPORT npy_bool
can_cast_datetime64_metadata(PyArray_DatetimeMetaData *src_meta,
                             PyArray_DatetimeMetaData *dst_meta,
                             NPY_CASTING casting);



/*
 * This provides the casting rules for the TIMEDELTA data type units.
 *
 * Notably, there is a barrier between the nonlinear years and
 * months units, and all the other units.
 */
NPY_NO_EXPORT npy_bool
can_cast_timedelta64_units(NPY_DATETIMEUNIT src_unit,
                          NPY_DATETIMEUNIT dst_unit,
                          NPY_CASTING casting);



/*
 * This provides the casting rules for the TIMEDELTA data type metadata.
 */
NPY_NO_EXPORT npy_bool
can_cast_timedelta64_metadata(PyArray_DatetimeMetaData *src_meta,
                             PyArray_DatetimeMetaData *dst_meta,
                             NPY_CASTING casting);



/*
 * Computes the conversion factor to convert data with 'src_meta' metadata
 * into data with 'dst_meta' metadata.
 *
 * If overflow occurs, both out_num and out_denom are set to 0, but
 * no error is set.
 */
NPY_NO_EXPORT void
get_datetime_conversion_factor(PyArray_DatetimeMetaData *src_meta,
                                PyArray_DatetimeMetaData *dst_meta,
                                npy_int64 *out_num, npy_int64 *out_denom);



/*
 * Given a pointer to datetime metadata,
 * returns a tuple for pickling and other purposes.
 */
NPY_NO_EXPORT PyObject *
convert_datetime_metadata_to_tuple(PyArray_DatetimeMetaData *meta);



/*
 * Converts a metadata tuple into a datetime metadata C struct.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
convert_datetime_metadata_tuple_to_datetime_metadata(PyObject *tuple,
                                        PyArray_DatetimeMetaData *out_meta,
                                        npy_bool from_pickle);



/*
 * Gets a tzoffset in minutes by calling the fromutc() function on
 * the Python datetime.tzinfo object.
 */
NPY_NO_EXPORT int
get_tzoffset_from_pytzinfo(PyObject *timezone, npy_datetimestruct *dts);



/*
 * Converts an input object into datetime metadata. The input
 * may be either a string or a tuple.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
/*
 * 将 PyObject 转换为 datetime 元数据。
 * 返回值是一个 Unicode 对象的新引用。
 * 在错误时返回 NULL。
 *
 * 如果 'skip_brackets' 为 true，则跳过 '[]'。
 */
NPY_NO_EXPORT PyObject *
metastr_to_unicode(PyArray_DatetimeMetaData *meta, int skip_brackets);

/*
 * 将 PyObject * 转换为 datetime，支持所有支持的形式。
 *
 * 如果单位的元数据事先不知道，将 meta->base 设置为 -1，
 * 此函数将使用默认值或输入对象中的值填充 meta。
 *
 * 'casting' 参数用于控制接受的输入类型以及处理方式。
 * 例如，使用 'unsafe' casting，无法识别的输入将被转换为 'NaT' 而不是抛出错误，
 * 而使用 'safe' casting，如果输入中有任何精度将被丢弃，则会抛出错误。
 *
 * 错误时返回 -1，成功时返回 0。
 */
NPY_NO_EXPORT int
convert_pyobject_to_datetime(PyArray_DatetimeMetaData *meta, PyObject *obj,
                             NPY_CASTING casting, npy_datetime *out);

/*
 * 将 PyObject * 转换为 timedelta，支持所有支持的形式。
 *
 * 如果单位的元数据事先不知道，将 meta->base 设置为 -1，
 * 此函数将使用默认值或输入对象中的值填充 meta。
 *
 * 'casting' 参数用于控制接受的输入类型以及处理方式。
 * 例如，使用 'unsafe' casting，无法识别的输入将被转换为 'NaT' 而不是抛出错误，
 * 而使用 'safe' casting，如果输入中有任何精度将被丢弃，则会抛出错误。
 *
 * 错误时返回 -1，成功时返回 0。
 */
NPY_NO_EXPORT int
convert_pyobject_to_timedelta(PyArray_DatetimeMetaData *meta, PyObject *obj,
                              NPY_CASTING casting, npy_timedelta *out);

/*
 * 将 datetime 转换为 PyObject *。
 *
 * 对于天或更粗的单位，返回一个 datetime.date 对象。
 * 对于微秒或更粗的单位，返回一个 datetime.datetime 对象。
 * 对于精度高于微秒的单位，返回一个整数。
 */
NPY_NO_EXPORT PyObject *
convert_datetime_to_pyobject(npy_datetime dt, PyArray_DatetimeMetaData *meta);

/*
 * 将 timedelta 转换为 PyObject *。
 *
 * 如果是 Not-a-time，则返回字符串 "NaT"。
 * 对于微秒或更粗的单位，返回一个 datetime.timedelta 对象。
 * 对于精度高于微秒的单位，返回一个整数。
 */
NPY_NO_EXPORT PyObject *
convert_timedelta_to_pyobject(npy_timedelta td, PyArray_DatetimeMetaData *meta);

/*
 * 根据秒偏移调整 datetimestruct。假设当前值是有效的。
 */
NPY_NO_EXPORT void
add_seconds_to_datetimestruct(npy_datetimestruct *dts, int seconds);

/*
 * 根据分钟偏移调整 datetimestruct。假设当前值是有效的。
 */
NPY_NO_EXPORT void
add_minutes_to_datetimestruct(npy_datetimestruct *dts, int minutes);
/*
 * 在给定的 npy_datetimestruct 结构体上增加指定的分钟数
 */
add_minutes_to_datetimestruct(npy_datetimestruct *dts, int minutes);

/*
 * 检查两个 datetime 类型的描述符是否具有等效的元数据
 * 如果元数据匹配，则返回 true
 */
NPY_NO_EXPORT npy_bool
has_equivalent_datetime_metadata(PyArray_Descr *type1, PyArray_Descr *type2);

/*
 * 将单个 datetime 从 src_meta 元数据转换为 dst_meta 元数据
 * 成功时返回 0，失败时返回 -1
 */
NPY_NO_EXPORT int
cast_datetime_to_datetime(PyArray_DatetimeMetaData *src_meta,
                          PyArray_DatetimeMetaData *dst_meta,
                          npy_datetime src_dt,
                          npy_datetime *dst_dt);

/*
 * 将单个 timedelta 从 src_meta 元数据转换为 dst_meta 元数据
 * 成功时返回 0，失败时返回 -1
 */
NPY_NO_EXPORT int
cast_timedelta_to_timedelta(PyArray_DatetimeMetaData *src_meta,
                            PyArray_DatetimeMetaData *dst_meta,
                            npy_timedelta src_dt,
                            npy_timedelta *dst_dt);

/*
 * 检查对象是否最适合视为 Datetime 或 Timedelta
 * 如果是，则返回 true；否则返回 false
 */
NPY_NO_EXPORT npy_bool
is_any_numpy_datetime_or_timedelta(PyObject *obj);

/*
 * 实现特定于 datetime 的 arange 函数
 */
NPY_NO_EXPORT PyArrayObject *
datetime_arange(PyObject *start, PyObject *stop, PyObject *step,
                PyArray_Descr *dtype);

/*
 * 通过递归下降序列结构来检查给定 Python 对象中的所有对象
 * 返回基于数据的 datetime 或 timedelta 类型的描述符
 */
NPY_NO_EXPORT PyArray_Descr *
find_object_datetime_type(PyObject *obj, int type_num);

/*
 * 初始化 datetime 类型的强制转换规则
 * 成功时返回 0，失败时返回 -1
 */
NPY_NO_EXPORT int
PyArray_InitializeDatetimeCasts(void);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY__DATETIME_H_ */
```