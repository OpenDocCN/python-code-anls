# `.\numpy\numpy\_core\src\dummymodule.c`

```py
/*
 * -*- c -*-
 */

/*
 * This is a dummy module whose purpose is to get distutils to generate the
 * configuration files before the libraries are made.
 */

/* Define to prevent including deprecated API and use the current API version */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

/* Define to prevent importing array when not needed */
#define NO_IMPORT_ARRAY

/* Include Python.h header file for Python C API */
#include <Python.h>

/* Define the structure for methods (functions) in this module */
static struct PyMethodDef methods[] = {
    {NULL, NULL, 0, NULL}  /* Sentinel entry indicating end of method definitions */
};

/* Define the structure for the module itself */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,  /* Indicates initialization of PyModuleDef structure */
    "dummy",                /* Module name */
    NULL,                   /* Module documentation, unused in this case */
    -1,                     /* Size of per-interpreter state of the module, -1 indicates global variables */
    methods,                /* Array of module-level functions */
    NULL,                   /* Optional slot for module-level traverse function */
    NULL,                   /* Optional slot for module-level clear function */
    NULL,                   /* Optional slot for module-level free function */
    NULL                    /* Optional slot for module-level traversal context */
};

/* Initialization function for the module */
PyMODINIT_FUNC PyInit__dummy(void) {
    PyObject *m;

    /* Create the module object */
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;  /* Return NULL on failure to create module */
    }

    return m;  /* Return the created module object */
}
```