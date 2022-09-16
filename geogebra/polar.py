import math
import cmath
from typing import Tuple

import sympy as sy
import sympy.core.add as add


def in_error_range(a, b):
    """允许的误差范围

    Args:
        a (_type_): _description_
        b (_type_): _description_

    Returns:
        _type_: 在误差范围内则返回True，否则返回False
    """
    return round(a, 1) == round(b, 1)


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
    return round(p, 2), dgr if dgr >= 0 else 360 + dgr


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
    """通过三点定位

    Args:
        p1 (Tuple[int, float]): _description_
        d1 (float): _description_
        p2 (Tuple[int, float]): _description_
        d2 (float): _description_
        p3 (Tuple[int, float]): _description_
        d3 (float): _description_

    Returns:
        _type_: 所求的未知点位极坐标
    """
    assert not collinear(p1, p2, p3), f"三点共线，无法定位"
    c1 = insec(p1, d1, p2, d2)
    c2 = insec(p1, d1, p3, d3)
    for r, t in c1:
        for r2, t2 in c2:
            if r == r2 and round(t, 0) == round(t2, 0):
                return r, t
    return None


def cos_therom(x, y, d, theta):
    """余弦定理

    Args:
        x (_type_): 边x
        y (_type_): 边y
        d (_type_): 第三条边d，也是theta正对的边
        theta (_type_): x与y的夹角

    Returns:
        _type_: 余弦定理表达式
    """
    return x**2 + y**2 - 2 * x * y * math.cos(math.radians(theta)) - d**2


def angle_of_vector(v1, v2):
    """计算两个向量夹角

    Args:
        v1 (_type_): _description_
        v2 (_type_): _description_

    Returns:
        _type_: 夹角角度
    """
    vector_prod = v1[0] * v2[0] + v1[1] * v2[1]
    length_prod = math.sqrt(pow(v1[0], 2) + pow(v1[1], 2)) * math.sqrt(
        pow(v2[0], 2) + pow(v2[1], 2)
    )
    cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)
    return math.degrees(math.acos(cos))


def get_tri_angel(p0, i, p3):
    """通过三个点位获取角度

    Args:
        p0 (_type_): _description_
        i (_type_): _description_
        p3 (_type_): _description_

    Returns:
        _type_: 角度
    """
    A = p0
    I = i
    B = p3
    # 转为直角系坐标
    AR = polar2rect(A)
    BR = polar2rect(B)
    IR = polar2rect(I)
    # 变成向量
    AI_ = [IR[0], IR[1]]
    BI_ = [IR[0] - BR[0], IR[1] - BR[1]]
    agl = angle_of_vector(AI_, BI_)
    return round(agl, 2)


def solve_nonlineq(p0, p1, p2, p3, alpha01, alpha02, alpha12, alpha03):
    """核心算法，求解的整体过程

    Args:
        p0 (_type_): 中心点位
        p1 (_type_): 点位1
        p2 (_type_): 点位2
        p3 (_type_): 额外点位
        alpha01 (_type_): p0与p1的夹角
        alpha02 (_type_): p0与p2的夹角
        alpha12 (_type_): p1与p2的夹角
        alpha03 (_type_): p0与p3的夹角

    Returns:
        _type_: 最终定位的坐标
    """
    # 先计算出三个点两两距离
    d01, d02, d12 = (
        distance(p0, p1),
        distance(p0, p2),
        distance(p1, p2),
    )
    # 余弦定理方程组
    h0, h1, h2 = sy.symbols("h0 h1 h2")
    eq = [
        cos_therom(h0, h1, d01, alpha01),
        cos_therom(h0, h2, d02, alpha02),
        cos_therom(h1, h2, d12, alpha12),
    ]
    # 得到非线性方程组的解析解
    rst = sy.nonlinsolve(eq, [h0, h1, h2])
    # 过滤结果，筛选出合理的点位
    points = []
    for r in rst:
        x, y, z = r
        if (
            not isinstance(x, add.Add)
            and not isinstance(y, add.Add)
            and not isinstance(z, add.Add)
            and x >= 0
            and y >= 0
            and z >= 0
        ):
            p = locate(p0, x, p1, y, p2, z)
            # 一般会存在镜像点
            # 使用额外的点p3来验证角度是否与观测角度一致，不一致则为镜像点，需舍弃
            agl = get_tri_angel(p0, p, p3)
            # 如果在误差范围内，则视为同一个点位
            if in_error_range(agl, alpha03):
                points.append(p)
    assert len(points) == 1, f"定位点错误：{points}"
    return points[0]


if __name__ == "__main__":
    p0, p1, p2, p3 = (0, 0), (100, 0), (100, 159.86), (110, 190.89)
    alpha01, alpha02, alpha12, alpha03 = (47.91, 29.16, 77.06, 46.74)

    point = solve_nonlineq(p0, p1, p2, p3, alpha01, alpha02, alpha12, alpha03)
    print(point)
