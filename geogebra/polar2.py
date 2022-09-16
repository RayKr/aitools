import math
import cmath
from typing import Tuple

import sympy as sy
import sympy.core.add as add


def covert_rect_to_polar(x, y):
    cn = complex(x, y)
    p, t = cmath.polar(cn)
    dgr = math.degrees(t)
    dgr = round(dgr, 2) if dgr >= 0 else round(360 + dgr, 2)
    return round(p, 0), dgr if dgr >= 0 else 360 + dgr


def covert_polar_to_rect(p: Tuple[int, float]):
    r, theta = p
    cn1 = cmath.rect(r, math.radians(theta))
    return cn1.real, cn1.imag


def cal_dis(p1: Tuple[int, float], p2: Tuple[int, float]) -> float:
    r1, theta1 = p1
    r2, theta2 = p2
    return math.sqrt(
        r1**2
        + r2**2
        - 2 * r1 * r2 * math.cos(math.radians(theta1) - math.radians(theta2))
    )


def cal_two_circle_point(p1, r1, p2, r2):
    x, y = covert_polar_to_rect(p1)
    R = r1
    a, b = covert_polar_to_rect(p2)
    S = r2
    d = math.sqrt((abs(a - x)) ** 2 + (abs(b - y)) ** 2)
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
    return [covert_rect_to_polar(x3, y3), covert_rect_to_polar(x4, y4)]


def collinear(p1, p2, p3):
    x1, y1 = covert_polar_to_rect(p1)
    x2, y2 = covert_polar_to_rect(p2)
    x3, y3 = covert_polar_to_rect(p3)
    return x1 * y2 - x2 * y1 + x2 * y3 - x3 * y2 + x3 * y1 - x1 * y3 == 0


def locate(
    p1: Tuple[int, float],
    d1: float,
    p2: Tuple[int, float],
    d2: float,
    p3: Tuple[int, float],
    d3: float,
):
    assert not collinear(p1, p2, p3), f"error"
    c1 = cal_two_circle_point(p1, d1, p2, d2)
    c2 = cal_two_circle_point(p1, d1, p3, d3)
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
        - cal_dis(p1, p2) ** 2,
        d2**2
        + d3**2
        - 2 * d2 * d3 * math.cos(math.radians(alpha23))
        - cal_dis(p2, p3) ** 2,
        d1**2
        + d3**2
        - 2 * d1 * d3 * math.cos(math.radians(alpha13))
        - cal_dis(p1, p3) ** 2,
        alpha12 + alpha23 - alpha13,
    ]
    return sy.nonlinsolve(eq, [d1, d2, d3])


if __name__ == "__main__":
    p1, p2, p3 = (100, 159.86), (0, 0), (100, 0)
    alpha12, alpha23, alpha13 = 29.16, 47.91, 77.06
    result = solve_nonlin_complete(p1, p2, p3, alpha12, alpha23, alpha13)
    for r in result:
        x, y, z = r
        if (
            not isinstance(x, add.Add)
            and not isinstance(y, add.Add)
            and not isinstance(z, add.Add)
            and x >= 0
            and y >= 0
            and z >= 0
        ):
            p = locate(p1, x, p2, y, p3, z)
            print(f"{p}")
