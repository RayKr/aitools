import math
import cmath
from typing import Tuple

import sympy as sy
import sympy.core.add as add


def rect2polar(x, y):
    """直角坐标转为极坐标

    Args:
        x (float): x轴刻度
        y (float): y轴刻度

    Returns:
        Tuple(float, float): 半径，角度
    """
    cn = complex(x, y)
    p, t = cmath.polar(cn)
    dgr = math.degrees(t)
    dgr = round(dgr, 2) if dgr >= 0 else round(360 + dgr, 2)
    return round(p, 0), dgr if dgr >= 0 else 360 + dgr


def polar2rect(p: Tuple[int, float]):
    """极坐标转直角坐标

    Args:
        r (float): 半径
        float (float): 角度

    Returns:
        Tuple(float, float): 直角坐标x, y
    """
    r, theta = p
    cn1 = cmath.rect(r, math.radians(theta))
    return cn1.real, cn1.imag


def distance(p1: Tuple[int, float], p2: Tuple[int, float]) -> float:
    """计算极坐标系中两个点的距离

    Args:
        p1 (Tuple[int, float]): 第一个点(半径,角度)
        p2 (Tuple[float, float]): 第二个点(半径,角度)

    Returns:
        float: 两个点之间的欧式距离
    """
    r1, theta1 = p1
    r2, theta2 = p2
    return math.sqrt(
        r1**2
        + r2**2
        - 2 * r1 * r2 * math.cos(math.radians(theta1) - math.radians(theta2))
    )


def insec(p1, r1, p2, r2):
    """极系下计算两个圆的交点

    Args:
        p1 (array): 第一个圆心坐标(极坐标系)
        r1 (float): 第一个圆的半径
        p2 (array): 第二个圆心坐标(极坐标系)
        r2 (float): 第二个圆的半径

    Returns:
        array: 两个交点的极系坐标
    """
    x, y = polar2rect(p1)
    R = r1
    a, b = polar2rect(p2)
    S = r2
    d = math.sqrt((abs(a - x)) ** 2 + (abs(b - y)) ** 2)
    # print(x, y, a, b, R, S)
    assert d >= (abs(R - S)) and d <= (R + S), print("Two circles have no intersection")
    assert d != 0 or R != S, print("Two circles have same center!")

    A = (R**2 - S**2 + d**2) / (2 * d)
    h = math.sqrt(R**2 - A**2)
    x2 = x + A * (a - x) / d
    y2 = y + A * (b - y) / d
    x3 = x2 - h * (b - y) / d
    y3 = y2 + h * (a - x) / d
    x4 = x2 + h * (b - y) / d
    y4 = y2 - h * (a - x) / d
    return [rect2polar(x3, y3), rect2polar(x4, y4)]


def collinear(p1, p2, p3):
    """判断三点是否共线

    Args:
        p1 (_type_): _description_
        p2 (_type_): _description_
        p3 (_type_): _description_

    Returns:
        bool: True为共线，False为不共线
    """
    x1, y1 = polar2rect(p1)
    x2, y2 = polar2rect(p2)
    x3, y3 = polar2rect(p3)
    # 利用顶点坐标求三角形面积，如果面积不为0，则三点不共线
    return x1 * y2 - x2 * y1 + x2 * y3 - x3 * y2 + x3 * y1 - x1 * y3 == 0


def locate(
    p1: Tuple[int, float],
    d1: float,
    p2: Tuple[int, float],
    d2: float,
    p3: Tuple[int, float],
    d3: float,
):
    assert not collinear(p1, p2, p3), f"三点共线，无法定位"
    c1 = insec(p1, d1, p2, d2)
    c2 = insec(p1, d1, p3, d3)
    for r, t in c1:
        for r2, t2 in c2:
            if r == r2 and round(t, 0) == round(t2, 0):
                return r, t
    return None


def solve_nonlin_complete(*args):
    p1, p2, p3 = args[0], args[1], args[2]
    alpha12, alpha23, alpha13 = args[3], args[4], args[5]
    d1, d2, d3 = sy.symbols("d1 d2 d3")
    eq = [
        d1**2
        + d2**2
        - 2 * d1 * d2 * math.cos(math.radians(alpha12))
        - distance(p1, p2) ** 2,
        d2**2
        + d3**2
        - 2 * d2 * d3 * math.cos(math.radians(alpha23))
        - distance(p2, p3) ** 2,
        d1**2
        + d3**2
        - 2 * d1 * d3 * math.cos(math.radians(alpha13))
        - distance(p1, p3) ** 2,
        alpha12 + alpha23 - alpha13,
    ]
    return sy.nonlinsolve(eq, [d1, d2, d3])


if __name__ == "__main__":
    p1, p2, p3 = (100, 159.86), (0, 0), (100, 0)
    alpha12, alpha23, alpha13 = 29.16, 47.91, 77.06
    # 假设alpha12+alpha23=alpha13
    result = solve_nonlin_complete(p1, p2, p3, alpha12, alpha23, alpha13)
    fil = {}
    for r in result:
        x, y, z = r
        # print(r)
        if (
            not isinstance(x, add.Add)
            and not isinstance(y, add.Add)
            and not isinstance(z, add.Add)
            and x >= 0
            and y >= 0
            and z >= 0
        ):
            p = locate(p1, x, p2, y, p3, z)
            print(f'点位为：{p}')
