# `.\numpy\numpy\_core\src\multiarray\datetime_busday.h`

```py
/*
 * This header guards prevent multiple inclusions of the header file
 * NUMPY_CORE_SRC_MULTIARRAY_DATETIME_BUSDAY_H_ defines a unique identifier
 * to avoid redefinition of the content between the #ifndef and #endif directives.
 */
#ifndef NUMPY_CORE_SRC_MULTIARRAY_DATETIME_BUSDAY_H_
#define NUMPY_CORE_SRC_MULTIARRAY_DATETIME_BUSDAY_H_

/*
 * NPY_NO_EXPORT is a macro used to specify that the following function
 * should not be exposed when compiling as a shared library.
 * 
 * array_busday_offset is a function that computes business day offsets
 * for NumPy arrays. It is intended to be called from Python.
 */
NPY_NO_EXPORT PyObject *
array_busday_offset(PyObject *NPY_UNUSED(self),
                      PyObject *args, PyObject *kwds);

/*
 * NPY_NO_EXPORT is a macro used to specify that the following function
 * should not be exposed when compiling as a shared library.
 * 
 * array_busday_count is a function that counts business days
 * between dates for NumPy arrays. It is intended to be called from Python.
 */
NPY_NO_EXPORT PyObject *
array_busday_count(PyObject *NPY_UNUSED(self),
                      PyObject *args, PyObject *kwds);

/*
 * NPY_NO_EXPORT is a macro used to specify that the following function
 * should not be exposed when compiling as a shared library.
 * 
 * array_is_busday is a function that checks whether dates are business days
 * for NumPy arrays. It is intended to be called from Python.
 */
NPY_NO_EXPORT PyObject *
array_is_busday(PyObject *NPY_UNUSED(self),
                      PyObject *args, PyObject *kwds);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_DATETIME_BUSDAY_H_ */
```