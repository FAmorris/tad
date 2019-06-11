import math
import numpy as np
import pandas as pd

def calc_gisdistance(gis1, gis2):
    """
    函数根据两点的经纬度坐标计算两点之间的距离。

    Parameters:
        'gis1' - python list 或 tuple 对象，A 点 GIS 坐标，[经度, 纬度]。
        'gis2' - python list 或 tuple 对象，B 点 GIS 坐标，[经度, 纬度]。

    Returns:
        float 标量，两点之间距离，单位：km。

    Raises:
        AssertionError
    
    """
    assert (2 == len(gis1)) and (2 == len(gis2)), 'gis1 or gis2 length error.'
    for i in gis1 + gis2: assert i >= 0, 'gis1 or gis2 error'

    try:
        ra = 6378.140
        rb = 6356.755
        flatten = (ra - rb) / ra
        rad_lng_A, rad_lat_A, rad_lng_B, rad_lat_B = map(math.radians, gis1 + gis2)
        pA = math.atan(rb / ra * math.tan(rad_lat_A))
        pB = math.atan(rb / ra * math.tan(rad_lat_B))
        xx = math.acos(math.sin(pA) * math.sin(pB) + math.cos(pA) * math.cos(pB)
                * math.cos(rad_lng_A - rad_lng_B))
        c1 = (math.sin(xx) - xx) * (math.sin(pA) + math.sin(pB) ** 2 / math.cos(xx / 2)) ** 2
        c2 = (math.sin(xx) + xx) * (math.sin(pA) - math.sin(pB) ** 2 / math.cos(xx / 2)) ** 2
        dr = flatten / 8 * (c1 - c2)
        distance = ra * (xx + dr) * 1e3
    except ZeroDivisionError: distance = 0.0

    return distance


def area_gridding(gises, interval=100):
    """
    函数根据矩形区域 4 个角点的 GIS 经纬度，将矩形区域网格化。

    Parameters:
        'gises'    - python list 对象，矩形区域 4 个角点经纬度坐标集合，
                     每个角点坐标以 list 对象给出，[经度, 纬度]
        'interval' - 区域网格化两个网格点之间的距离，单位：m。

    Returns:
        python list 对象，包含网格化后每个网格点的经纬度，[[经度, 纬度], ...]

    Raises:
        AssertionError
    """
    assert 4 == len(gises), 'parameter "gises" length error.'
    for gis in gises: assert 2 == len(gis), 'gis length error.'
    border = pd.DataFrame(gises)
    desc = border.describe()

    xmin, ymin = desc.loc['min'].values
    assert (xmin >= 0) and (ymin >= 0), 'gis value error.'
    xmax, ymax = desc.loc['max'].values

    xsteps = round((xmax - xmin) / (0.00001 * interval / 1))
    ysteps = round((ymax - ymin) / (0.00001 * interval / 1.1))
    
    xborder, yborder = np.meshgrid(np.linspace(xmin, xmax, int(xsteps)),
            np.linspace(ymin, ymax, int(ysteps)))

    return np.concatenate([xborder.reshape((-1, 1)), yborder.reshape((-1, 1))], axis=1).tolist()
    

def module_test():
    from pprint import pprint
    print(calc_gisdistance([121.03538461538461, 30.6453125], [121.065, 30.575]))
    
    points = [[121.03, 30.5], [121.03, 30.65], [121.10, 30.5], [121.10, 30.65]]
    pprint(area_gridding(points))

if '__main__' == __name__: module_test()
