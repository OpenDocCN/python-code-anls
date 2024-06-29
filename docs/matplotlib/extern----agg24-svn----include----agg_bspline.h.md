# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_bspline.h`

```py
//----------------------------------------------------------------------------
// Anti-Grain Geometry - Version 2.4
// Copyright (C) 2002-2005 Maxim Shemanarev (http://www.antigrain.com)
//
// Permission to copy, use, modify, sell and distribute this software 
// is granted provided this copyright notice appears in all copies. 
// This software is provided "as is" without express or implied
// warranty, and with no claim as to its suitability for any purpose.
//
//----------------------------------------------------------------------------
// Contact: mcseem@antigrain.com
//          mcseemagg@yahoo.com
//          http://www.antigrain.com
//----------------------------------------------------------------------------
//
// class bspline
//
//----------------------------------------------------------------------------

#ifndef AGG_BSPLINE_INCLUDED
#define AGG_BSPLINE_INCLUDED

#include "agg_array.h"

namespace agg
{
    //----------------------------------------------------------------bspline
    // A very simple class of Bi-cubic Spline interpolation.
    // First call init(num, x[], y[]) where num - number of source points, 
    // x, y - arrays of X and Y values respectively. Here Y must be a function 
    // of X. It means that all the X-coordinates must be arranged in the ascending
    // order. 
    // Then call get(x) that calculates a value Y for the respective X. 
    // The class supports extrapolation, i.e. you can call get(x) where x is
    // outside the given with init() X-range. Extrapolation is a simple linear 
    // function.
    //
    //  See Implementation agg_bspline.cpp
    //------------------------------------------------------------------------
    class bspline 
    {
    public:
        // Default constructor
        bspline();
        
        // Constructor with number of points
        bspline(int num);
        
        // Constructor with arrays of x and y values
        bspline(int num, const double* x, const double* y);

        // Initialize with number of points
        void   init(int num);
        
        // Add a point with given x and y values
        void   add_point(double x, double y);
        
        // Prepare the spline for calculations
        void   prepare();

        // Initialize with arrays of x and y values
        void   init(int num, const double* x, const double* y);

        // Get interpolated y value for given x using linear extrapolation
        double get(double x) const;
        
        // Get interpolated y value for given x using stateful extrapolation
        double get_stateful(double x) const;
    
    private:
        // Copy constructor (private and not implemented)
        bspline(const bspline&);
        
        // Assignment operator (private and not implemented)
        const bspline& operator = (const bspline&);

        // Binary search for x0 in array x
        static void bsearch(int n, const double *x, double x0, int *i);
        
        // Extrapolate left of x using linear function
        double extrapolation_left(double x) const;
        
        // Extrapolate right of x using linear function
        double extrapolation_right(double x) const;
        
        // Interpolate y value for given x at index i
        double interpolation(double x, int i) const;

        int               m_max;        // Maximum number of points
        int               m_num;        // Current number of points
        double*           m_x;          // Array of x values
        double*           m_y;          // Array of y values
        pod_array<double> m_am;         // Array used for internal calculations
        mutable int       m_last_idx;   // Last index used in interpolation
    };
}

#endif
```