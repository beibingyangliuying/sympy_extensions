# -*- coding: utf-8 -*-
"""
Created on 2023/1/12 15:33

@author: chenjunhan
"""
import sympy as smp
from typing import Any, Tuple

n = smp.symbols('n', positive=True, integer=True)  # index, a positive integer


def fourier_series(function: Any, symbol: smp.Symbol, intervals: Tuple[Any, Any], finite: bool = True) -> \
        Tuple[Any, ...]:
    r"""
    Calculate the coefficient $a_0$, $a_n$ and $b_n$ of fourier series, then return them in analytic form.

    Explanation
    ===========

    The coefficient of the fourier series:

    .. math::
        L=b-a

    .. math::
        a_0=\frac{2}{L}\int_a^b{f\left( x \right) \mathrm{d}x}

    .. math::
        a_n=\frac{2}{L}\int_a^b{f\left( x \right) \cos \left( \frac{2\pi nx}{L} \right) \mathrm{d}x}

    .. math::
        b_n=\frac{2}{L}\int_a^b{f\left( x \right) \sin \left( \frac{2\pi nx}{L} \right) \mathrm{d}x}

    The complete formula:

    .. math::
        f\left( x \right) =a_0+\sum_{n=1}^{\infty}{a_n\cos \left( \frac{2\pi nx}{L} \right)}+\sum_{n=1}^{\infty}{b_n\sin \left( \frac{2\pi nx}{L} \right)}

    Examples
    ===========

    An easy example using piecewise function::

    >>> import sympy as smp
    >>> from sympy_extensions import *
    >>> t, p_0 = smp.symbols('t, p_0 ', real=True, positive=True)
    >>> p = smp.Piecewise((p_0 * smp.sin(t), t < 2 * smp.pi), (0, t < 3 * smp.pi))
    >>> s = series_extension.fourier_series(p, t, (0, 3 * smp.pi))
    >>> s[0]
        0
    >>> s[1]
        6*p_0*(cos(4*pi*n/3) - 1)/(pi*(4*n**2 - 9))
    >>> s[2]
        6*p_0*sin(4*pi*n/3)/(pi*(4*n**2 - 9))

    :param function: the target function
    :param symbol: the symbol which to be integrated
    :param intervals: the left and right ends
    :param finite: whether $\lim \left( b-a \right) =\infty $, (Not using yet)
    :return: the tuple of three coefficients: $a_0$, $a_n$, $b_n$
    """
    a = intervals[0]
    b = intervals[1]
    L = b - a

    a_0 = 2 / L * smp.Integral(function, (symbol, a, b))
    a_n = 2 / L * smp.Integral(function * smp.cos(2 * n * smp.pi * symbol / L), (symbol, a, b))
    b_n = 2 / L * smp.Integral(function * smp.sin(2 * n * smp.pi * symbol / L), (symbol, a, b))

    return tuple(map(lambda x: smp.simplify(x.doit()), (a_0, a_n, b_n)))
