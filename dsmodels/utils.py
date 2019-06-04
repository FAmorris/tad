import math
import numpy as np
import pandas as pd

def calc_gisdistance(gis1, gis2):
    """
    函数根据两点的经纬度坐标计算两点之间的距离。

    Parameters:
        'gis1' - python list 或 tuple 对象，A 点 GIS 坐标，索引 0 应是经度，索引 1 应是维度。
        'gis2' - python list 或 tuple 对象，B 点 GIS 坐标，索引 0 应是经度，索引 1 应是维度。

    Returns:
        float 标量，两点之间距离，单位：km。

    Raises:
        AssertionError
    
    """
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
        distance = ra * (xx + dr)
    except ZeroDivisionError: distance = 0.0

    return distance


def area_gridding(gis1, gis2, gis3, gis4, interval=500):
    border = pd.DataFrame([gis1, gis2, gis3, gis4])
    desc = border.describe()
    
    xmin, ymin = desc.loc['min'].values
    xmax, ymax = desc.loc['max'].values
    xsteps = round((xmax - xmin) / (0.00001 * interval / 1))
    ysteps = round((ymax - ymin) / (0.00001 * interval / 1.1))
    
    xborder, yborder = np.meshgrid(np.linspace(xmin, xmax, int(xsteps)),
            np.linspace(ymin, ymax, int(ysteps)))

    return np.concatenate([xborder.reshape((-1, 1)), yborder.reshape((-1, 1))], axis=1)
    

def module_test():
    print(calc_gisdistance((120, 30), (120., 30)))


if '__main__' == __name__: module_test()
