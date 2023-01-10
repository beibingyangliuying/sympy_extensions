# -*- coding: utf-8 -*-
"""
Created on 2023/1/7 19:08

@author: chenjunhan
"""
import sympy as smp
from typing import Union


def generate_poly_by_zero_point(points: list, symbol: smp.Symbol, if_poly: bool = False) -> Union[smp.Mul, smp.Poly]:
    """
    Generate polynomial through zero points.
    :param points: a list that contains zero points
    :param symbol: sympy.Symbol
    :param if_poly: indicate whether to transfer the result into sympy.Poly or not
    :return: sympy.Mul or sympy.Poly
    """
    result = smp.Mul(*(symbol - i for i in points))
    if if_poly:
        return smp.Poly(result)
    else:
        return result
