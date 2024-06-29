# `D:\src\scipysrc\matplotlib\extern\ttconv\pprdrv.h`

```
/*
 * -*- mode: c++; c-basic-offset: 4 -*-
 */

/*
 * Modified for use within matplotlib
 * 5 July 2007
 * Michael Droettboom
 */

/*
** ~ppr/src/include/pprdrv.h
** Copyright 1995, Trinity College Computing Center.
** Written by David Chappell.
**
** Permission to use, copy, modify, and distribute this software and its
** documentation for any purpose and without fee is hereby granted, provided
** that the above copyright notice appear in all copies and that both that
** copyright notice and this permission notice appear in supporting
** documentation.  This software is provided "as is" without express or
** implied warranty.
**
** This file last revised 5 December 1995.
*/

#include <vector>
#include <cassert>

/*
 * Encapsulates all of the output to write to an arbitrary output
 * function.  This both removes the hardcoding of output to go to stdout
 * and makes output thread-safe.  Michael Droettboom [06-07-07]
 */
class TTStreamWriter
{
 private:
    // Private copy and assignment
    TTStreamWriter& operator=(const TTStreamWriter& other);
    TTStreamWriter(const TTStreamWriter& other);

 public:
    TTStreamWriter() { }
    virtual ~TTStreamWriter() { }

    // Virtual function to write data (pure virtual function to be overridden)
    virtual void write(const char*) = 0;

    // Virtual functions for formatted output
    virtual void printf(const char* format, ...);
    virtual void put_char(int val);
    virtual void puts(const char* a);
    virtual void putline(const char* a);
};

// Function declaration for replacing newlines with spaces in a C string
void replace_newlines_with_spaces(char* a);

/*
 * A simple class for all ttconv exceptions.
 */
class TTException
{
    const char* message;
    TTException& operator=(const TTStreamWriter& other);
    TTException(const TTStreamWriter& other);

public:
    // Constructor initializing with an error message
    TTException(const char* message_) : message(message_) { }
    
    // Method to retrieve the error message
    const char* getMessage()
    {
        return message;
    }
};

/*
** No debug code will be included if this
** is not defined:
*/
/* #define DEBUG 1 */

/*
** Uncomment the defines for the debugging
** code you want to have included.
*/
#ifdef DEBUG
#define DEBUG_TRUETYPE          /* truetype fonts, conversion to Postscript */
#endif

// Conditional macro for debug output based on DEBUG_TRUETYPE
#if DEBUG_TRUETYPE
#define debug(...) printf(__VA_ARGS__)
#else
#define debug(...)
#endif

/* Do not change anything below this line. */

// Enumeration defining different types of font
enum font_type_enum
{
    PS_TYPE_3  = 3,            // Type 3 PostScript font
    PS_TYPE_42 = 42,           // Type 42 PostScript font
    PS_TYPE_42_3_HYBRID = 43,  // Hybrid Type 42/Type 3 PostScript font
};

/* Function prototype for inserting a TrueType font into output */
void insert_ttfont(const char *filename, TTStreamWriter& stream, font_type_enum target_type, std::vector<int>& glyph_ids);

/* end of file */
```