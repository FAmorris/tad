from abc import ABC, abstractmethod
from datetime import datetime
import math
import pandas as pd
from scipy.interpolate import splev, splrep
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import utils

class SecurityModel(ABC):
    """
    安防模型抽象基类，定义物质名称、物质物理参数、环境参数。所有子类需要实现以下抽象方法
        'fit'  - 方法用于拟合某个区域的安全态势。
        'plot' - 方法用于绘制区域热力图。
    """
    _MAT_NE_PARAMS = pd.Series()
    _ENV_NE_PARAMS = pd.Series(['center_gis'])
    
    def __init__(self, material='', mat_params=pd.Series(), env_params=pd.Series()):
        """
        构造函数。
        
        Parameters:
            'material'   - 用于建模物质名称。
            'mat_params' - 用于建模物质相关物理参数。
            'env_params' - 用于建模物质所在环境参数。
            
        Returns:
            None
            
        Raises:
            None
        """
        self._results = pd.Series()
        self.set_material(material)
        self.set_material_params(mat_params)
        self.set_environment_params(env_params)
    
    def set_material(self, material): self._material = material
    
    def get_material(self): return self._material
    
    def set_material_params(self, mat_params): self._mat_params = mat_params.copy()
    
    def get_material_params(self): return self._mat_params
    
    def set_environment_params(self, env_params): self._env_params = env_params.copy()
    
    def get_environment_params(self): return self._env_params
    
    def _add_result(self, param, value): self._results[param] = value
    
    def _add_environment_param(self, param, value): self._env_params[param] = value
    
    def get_results(self): return self._results
    
    def assert_info(self, param): return 'parameter "{param}" loss or error.'.format(param=param)
    
    @abstractmethod
    def fit(self): pass
    
    @abstractmethod
    def plot(self): pass
    
    @staticmethod
    def get_necessary_mat_params():
        return SecurityModel._MAT_NE_PARAMS.copy()
    
    @staticmethod
    def get_necessary_env_params():
        return SecurityModel._ENV_NE_PARAMS.copy()
    
    def get_info(self, title='reports', width=80, v_width=40):
        p_width = width - v_width
        title_fmt = '{{title:^{width}}}'.format(width=width)
        item_fmt = '{{p:{p_width}}}{{v:>{v_width}}}'.format(p_width=p_width, v_width=v_width)
        
        info = title_fmt.format(title=title.title()) + '\n'
        info += '=' * width + '\n'
        info += item_fmt.format(p='Material', v=self.get_material()) + '\n'
        info += '=' * width + '\n'
        
        info += item_fmt.format(p='Material Parameter', v='Value') + '\n'
        info += '-' * width + '\n'
        mat_params = self.get_material_params()
        for i, v in mat_params.iteritems():
            info += item_fmt.format(p=i, v=str(v)) + '\n'
        info += '=' * width + '\n'
        
        info += item_fmt.format(p='Environment Parameter', v='Value') + '\n'
        info += '-' * width + '\n'
        env_params = self.get_environment_params()
        for i, v in env_params.iteritems():
            info += item_fmt.format(p=i, v=str(v)) + '\n'
        info += '=' * width + '\n'
        
        info += item_fmt.format(p='Result', v='Value') + '\n'
        info += '-' * width + '\n'
        results = self.get_results()
        for i, v in results.iteritems():
            info += item_fmt.format(p=i, v=str(v)) + '\n'
        info += '=' * width + '\n'
        
        return info
        
        
class ExplosionModel(SecurityModel):
    """
    物质爆炸模型抽象基类，'SecurityModel' 的子类，该抽象类实现了
    1000kg TNT 爆炸产生冲击波超压与爆炸中心之间距离的计算算法。
    """
    _DIST = (5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75)
    _PRES = (2.94, 2.06, 1.67, 1.27, 0.95, 0.76, 0.50, 0.33, 0.235, 0.17, 0.126, 0.079, 0.057,
             0.043, 0.033, 0.027, 0.0235, 0.0205, 0.018, 0.016, 0.0143, 0.013)
    _DATASET = pd.Series(_PRES, index=_DIST)
    _MAT_NE_PARAMS = pd.Series()
    _ENV_NE_PARAMS = pd.Series()
    
    def __init__(self, material='', mat_params=pd.Series(), env_params=pd.Series()):
        """
        构造函数。
        
        Parameters:
            'material'   - 用于建模物质名称。
            'mat_params' - 用于建模物质相关物理参数。
            'env_params' - 用于建模物质所在环境参数。
            
        Returns:
            None
            
        Raises:
        
        """
        
                      
        super().__init__(material=material, mat_params=mat_params, env_params=env_params)
        self._polyfit()
    
    def _get_pod_poly(self): return self._pod_poly
    
    def _get_dop_poly(self): return self._dop_poly
    
    def _polyfit(self):
        """
        方法采用三次样条插值算法拟合 1000kg TNT 产生的超压和距离之间的关系式。
        
        Parameters:
            None
            
        Returns:
            None
            
        Raises:
            
        """
        dataset = ExplosionModel._DATASET
        dataset_revers = dataset.sort_values()
        
        self._pod_poly = splrep(dataset.index.values, dataset.values)
        self._dop_poly = splrep(dataset_revers.values, dataset_revers.index.values)
        
    def tnt_overpressure_of(self, distance):
        """
        方法用于计算 1000kg TNT 爆炸时，指定点与爆炸中心之间距离对应的冲击波超压。
        
        Parameters:
            distance - 指定位置与爆炸中心之间的距离，单位：m。
            
        Returns:
            爆炸产生的冲击波超压，单位：MPa。
            
        Raises:
        
        """
        assert distance >= 0, self.assert_info('distance')

        if distance > 75: overpressure = 0
        elif distance < 5: overpressure = 3
        else: overpressure = float(splev(distance, self._get_pod_poly()))

        return overpressure
        
    def tnt_distance_of(self, overpressure): 
        """
        方法用于计算 1000kg TNT 爆炸时，某冲击波超压下，位置点与爆炸中心之间的距离。
        
        Parameters:
            overpressure - 爆炸产生的冲击波超压，单位：Mpa。
            
        Returns:
            位置点与爆炸中心之间的距离，单位：m。
            
        Raises:
        
        """
        assert overpressure >= 0, self.assert_info()

        if overpressure > 3: distance = 4
        elif overpressure < 0.01: distance = 80
        else: distance = float(splev(overpressure, self._get_dop_poly()))
        
        return distance

    @staticmethod
    def get_necessary_mat_params():
        tmp1 = SecurityModel.get_necessary_mat_params()
        tmp2 = ExplosionModel._MAT_NE_PARAMS.copy()
        return pd.concat([tmp1, tmp2], ignore_index=True)
    
    @staticmethod
    def get_necessary_env_params():
        tmp1 = SecurityModel.get_necessary_env_params()
        tmp2 = ExplosionModel._ENV_NE_PARAMS.copy()
        return pd.concat([tmp1, tmp2], ignore_index=True)
    
    def get_info(self, title='explosion reports', width=80, v_width=40):
        return super().get_info(title, width=width, v_width=v_width)
        
        
class FireModel(SecurityModel):
    """
    物质燃烧模型抽象基类，'SecurityModel' 的子类，
    """
    _MAT_NE_PARAMS = pd.Series()
    _ENV_NE_PARAMS = pd.Series()
    
    def __init__(self, material='', mat_params=pd.Series(), env_params=pd.Series()):
        """
        构造函数。
        
        Parameters:
            'material'   - 用于建立燃烧模型物质名称。
            'mat_params' - 用于建立燃烧模型物质相关物理参数。
            'env_params' - 用于建立燃烧模型物质所在环境参数。
            
        Returns:
            None
            
        Raises:
        
        """
        super().__init__(material=material, mat_params=mat_params, env_params=env_params)
        
    @staticmethod
    def get_necessary_mat_params():
        tmp1 = SecurityModel.get_necessary_mat_params()
        tmp2 = FireModel._MAT_NE_PARAMS.copy()
        return pd.concat([tmp1, tmp2], ignore_index=True)
    
    @staticmethod
    def get_necessary_env_params():
        tmp1 = SecurityModel.get_necessary_env_params()
        tmp2 = FireModel._ENV_NE_PARAMS.copy()
        return pd.concat([tmp1, tmp2], ignore_index=True)
        
    def get_info(self, title='fire reports', width=80, v_width=40):
        return super().get_info(title=title, width=width, v_width=v_width)
        
        
class GasDiffusionModel(SecurityModel):
    """
    气体扩散模型抽象子类，'SecurityModel' 的子类，该抽象类实现了
    赤纬计算、大气稳定度计算、扩散参数系数计算、扩散参数计算
    """
    
    # 太阳辐射等级表
    _SRLT = pd.DataFrame([[-2, -1, 1, 2, 3],
                          [-1, 0, 1, 2, 3],
                          [-1, 0, 0, 1, 1],
                          [0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0]])
                          
    # 大气稳定度
    _AST = pd.DataFrame([['A', 'A~B', 'B', 'D', 'E', 'F'],
                         ['A~B', 'B', 'C', 'D', 'E', 'F'],
                         ['B', 'B~C', 'C', 'D', 'D', 'E'],
                         ['C', 'C~D', 'D', 'D', 'D', 'D'],
                         ['D', 'D', 'D', 'D', 'D', 'D']], 
                         columns=['3', '2', '1', '0', '-1', '-2'])
    
    # 大气扩散参数系数表索引
    _DPCT_INDEX=[['A', 'A', 'A',
                  'A~B', 'A~B',
                  'B', 'B',
                  'B~C', 'B~C',
                  'C', 'C',
                  'C~D', 'C~D', 'C~D',
                  'D', 'D', 'D',
                  'D~E', 'D~E', 'D~E',
                  'E', 'E', 'E',
                  'E~F', 'E~F', 'E~F',
                  'F', 'F', 'F'],
                 [0, 1, 2,
                  1, 2,
                  1, 2,
                  1, 2,
                  1, 2,
                  0, 1, 2,
                  0, 1, 2,
                  0, 1, 2,
                  0, 1, 2,
                  0, 1, 2,
                  0, 1, 2]]
                          
    # 大气扩散参数系数表
    _DPCT = pd.DataFrame([[0.000000, 0.000000, 1.12154, 0.079990],  # A                垂直 0 ~ 300
                          [0.901074, 0.425809, 1.51360, 0.008548],  # A   水平 0~1000，垂直 300 ~ 500
                          [0.850934, 0.602052, 2.10881, 0.000212],  # A   水平 > 1000，垂直 > 500
                          [0.907722, 0.353828, 1.19986, 0.071909],  # A~B 水平 0~1000，垂直 0~500
                          [0.857974, 0.499203, 1.60119, 0.028618],  # A~B 水平 > 1000，垂直 > 500
                          [0.914370, 0.281846, 0.96444, 0.127190],  # B   水平 0~1000，垂直 0 ~ 500
                          [0.865014, 0.396353, 1.09356, 0.057025],  # B   水平 > 1000，垂直 > 500
                          [0.919325, 0.229500, 0.94102, 0.114682],  # B~C 水平 0~1000，垂直 0 ~ 500
                          [0.875086, 0.314238, 1.00770, 0.075718],  # B~C 水平 > 1000，垂直 > 500
                          [0.924279, 0.177154, 0.00000, 0.000000],  # C   水平 1~1000，垂直 0 ~ 500
                          [0.885157, 0.232123, 0.91760, 0.106803],  # C   水平 > 1000，垂直 > 0
                          [0.000000, 0.000000, 0.83863, 0.126152],  # C~D              垂直 0~ 2000
                          [0.926849, 0.143940, 0.75641, 0.235667],  # C~D 水平 1~1000，垂直 2000~10000
                          [0.886940, 0.189396, 0.81558, 0.136659],  # C~D 水平 > 1000，垂直 > 10000
                          [0.000000, 0.000000, 0.82621, 0.104634],  # D                垂直 1 ~ 1000
                          [0.929418, 0.110726, 0.63202, 0.400167],  # D   水平 1~1000，垂直 1000~10000
                          [0.888723, 0.146669, 0.55536, 0.810763],  # D   水平 > 1000，垂直 > 10000
                          [0.000000, 0.000000, 0.77686, 0.111771],  # D~E              垂直 0~2000
                          [0.925118, 0.098563, 0.57235, 0.528992],  # D~E 水平 1~1000，垂直 2000~10000
                          [0.892794, 0.124308, 0.49915, 1.037100],  # D~E 水平 >1000， 垂直 > 10000
                          [0.000000, 0.000000, 0.78837, 0.092753],  # E                垂直 0~1000
                          [0.920818, 0.086400, 0.56518, 0.433384],  # E   水平 1~1000，垂直 1000~10000
                          [0.896864, 0.101947, 0.41474, 1.732410],  # E   水平 > 1000，垂直 > 10000
                          [0.000000, 0.000000, 0.78639, 0.077415],  # E~F              垂直 0~1000
                          [0.925118, 0.070882, 0.54558, 0.401700],  # E~F 水平 0~1000，垂直 1000~10000
                          [0.892794, 0.087641, 0.36870, 2.069660],  # E~F 水平 > 1000，垂直 > 10000
                          [0.000000, 0.000000, 0.78440, 0.062077],  # F                垂直 0~1000
                          [0.929418, 0.055363, 0.52597, 0.370015],  # F   水平 0~1000，垂直 1000~10000
                          [0.888723, 0.073335, 0.32266, 2.406910]], # F   水平 > 1000，垂直 > 10000
                          index=_DPCT_INDEX,
                          columns=['alpha1', 'gama1', 'alpha2', 'gama2'])
    
    _MAT_NE_PARAMS = pd.Series()
    _ENV_NE_PARAMS = pd.Series(['center_longtitude',
                                'center_latitude',
                                'total_cloudiness',
                                'low_cloudiness',
                                'wind_speed',
                                'start_datetime'])
    
    def __init__(self, material='', mat_params=pd.Series(), env_params=pd.Series()):
        """
        构造方法。
        
        Parameters:
            'env_params' - pandas Series 对象，参数中必须包含以下 key - values
                'center_longtitude' - 区域经度，单位：°。
                'center_latitude'   - 区域纬度，单位：°。
                'total_cloudiness'  - 建模时环境总云量。
                'low_cloudiness'    - 建模时环境低云量。
                'wind_speed'        - 建模时环境风速，单位：m/s。
                'start_datetime'    - 事故起始时间戳，格式：'yyyy-m-d h24:mm:ss'。
                
        Returns:
            None
        
        Raises:
            KeyError
        """
        if (not mat_params.index.is_unique) or (not env_params.index.is_unique):
            raise KeyError('model parameter is not unique.')
            
        super().__init__(material=material, mat_params=mat_params, env_params=env_params)
        self._srlt = GasDiffusionModel._SRLT
        self._ast = GasDiffusionModel._AST
        self._dpct = GasDiffusionModel._DPCT
        
        self._nan_params = pd.concat([mat_params[mat_params.isnull()], 
                                      env_params[env_params.isnull()]], sort=False)
        
        if 'start_datetime' in self._nan_params:
            self._env_params['start_datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self._nan_params.drop('start_datetime', inplace=True)
    
    def _get_srlt(self): return self._srlt.copy()
    
    def _get_ast(self): return self._ast.copy()
    
    def _get_dpct(self): return self._dpct.copy()
        
    def calc_declination(self):
        """
        方法用于计算赤纬。根据国标标准 《GB/T13201-91》计算所给公式计算。
        δ = [0.006918 - 0.399912cosθ + 0.070257sinθ - 0.006758cos2θ + 0.000907sin2θ - 0.002697cos3θ + 0.00148sin3θ] * 180 / π
        
        Parameters:
            None
        
        Returns:
            赤纬，单位：°。
        
        Raises:
            None
        """
        env_params = self.get_environment_params()
        sdt = datetime.strptime(env_params['start_datetime'], '%Y-%m-%d %H:%M:%S')
        day = sdt.timetuple().tm_yday
        
        if 366 == day: day = 365
        theta = 360 * day / 365
        
        declination = (0.006918 - 0.399912 * math.cos(theta) + 0.070257 * math.sin(theta)\
                        - 0.006758 * math.cos(2 * theta) + 0.000907 * math.sin(2 * theta)\
                        - 0.002697 * math.cos(3 * theta) + 0.00148 * math.sin(3 * theta)) * (180 / math.pi)
        self._add_result('declination', declination)
        
        return declination
        
    def calc_solar_angle(self):
        """
        方法用于计算地方太阳高度角。根据国标标准 《GB/T13201-91》计算所给公式计算。        
        h = arcsin[sinφsinδ + cosφcosδ(15t + λ - 300)]
        φ：当地纬度。
        λ：当地经度。
        
        Parameters:
            None
        
        Returns:
            太阳高度角，单位：°
        
        Raises:
            AssertionError
        """
        env_params = self.get_environment_params()
        lgt = env_params['center_longtitude']
        lat = env_params['center_latitude']
        
        assert lat >= 0, self.assert_info('center_latitude')
        assert lgt >= 0, self.assert_info('center_longtitude')
        
        hour = datetime.strptime(env_params['start_datetime'], '%Y-%m-%d %H:%M:%S').hour
        
        declination = self.calc_declination()
        tmp = 15 * hour + lgt - 300
        solar_angle = math.asin(math.sin(lat) * math.sin(declination)\
                    + math.cos(lat) * math.cos(declination) * math.cos(tmp))
        self._add_result('solar_angle', solar_angle)
        
        return solar_angle
        
    def get_solar_radiation_level(self):
        """
        方法用于获取太阳辐射等级。
        
        Parameters:
            None
        
        Returns:
            太阳辐射等级。
        
        Raises:
            AssertionError
            KeyError
        """
        env_params = self.get_environment_params()
        tc = env_params['total_cloudiness']
        lc = env_params['low_cloudiness']
        
        assert tc >= 0, self.assert_info('total_cloudiness')
        assert lc >= 0, self.assert_info('low_cloudiness')
        assert tc >= lc, self.assert_info('total_cloudiness, low_cloudiness')
        
        hour = datetime.strptime(env_params['start_datetime'], '%Y-%m-%d %H:%M:%S').hour
        
        # 云量
        if (tc <= 4 and lc <= 4): row = 0
        elif (5 <= tc < 7) and (lc <= 4): row = 1
        elif (tc >= 8) and (lc <= 4): row = 2
        elif (tc >= 5) and (5 <= lc < 7): row = 3
        elif (tc >= 8) and (lc >= 8): row = 4
        
        # 日间
        if 7 <= hour < 19:
            solar_angle = self.calc_solar_angle()
            
            if solar_angle <= 15: col = 1
            elif 15 < solar_angle <= 35: col = 2
            elif 35 < solar_angle <= 65: col = 3
            elif solar_angle > 65: col = 4
        else: col = 0
        
        solar_radiation_level = self._get_srlt().iloc[row, col]
        self._add_result('solar_radiation_level', solar_radiation_level)
        
        return solar_radiation_level
        
    def get_atmospheric_stability(self):
        """
        方法用于获取大气稳定度。

        Parameters:
            None
        
        Returns:
            大气稳定程度。
        
        Raises:
            AssertionError
            KeyError
        """
        env_params = self.get_environment_params()
        wind_speed = env_params['wind_speed']
        
        assert wind_speed > 0, self.assert_info('wind_speed')
        
        srl = str(self.get_solar_radiation_level())
        
        # 风速
        if 0 < wind_speed <= 1.9: row = 0
        elif 1.9 < wind_speed <= 2.9: row = 1
        elif 2.9 < wind_speed <= 4.9: row = 2
        elif 4.9 < wind_speed <= 5.9: row = 3
        elif wind_speed >= 6.0: row = 4
        
        atmospheric_stability = self._get_ast().iloc[row][srl]
        self._add_result('atmospheric_stability', atmospheric_stability)
        
        return atmospheric_stability
        
    def get_diffusion_param_coeffs(self, pgis=None, hdis=-1):
        """
        方法用于获取气体扩散参数系数，包括 y 轴扩散参数系数和 z 轴扩散参数系数。
        
        Parameters:
            'pgis' - python array-like 对象，计算地地理坐标 [经度，纬度]。
            'hdis' - 下风向距离，单位：m。
            
        Returns:
            python tuple 对象，
            气体扩散参数系数，index=0。
            下风向距离，index=1，单位：m。
            
        Raises:
            AssertionError
            
        Note:
            必须给定 gis 坐标或下风向距离。两者都指定时优先使用 hdis
        """
        assert pgis or hdis >= 0, self.assert_info('pgis, hdis')
        if pgis: assert (2 == len(pgis)) and (pgis[0] >= 0) and (pgis[1] >= 0), self.assert_info('pgis')
        
        atmos_stat = self.get_atmospheric_stability()
        dpct = self._get_dpct()
        dpcs = dpct.loc[atmos_stat]
        vdpcgs = dpcs[['alpha1', 'gama1']]
        ddpcgs = dpcs[['alpha2', 'gama2']]
        
        # 计算下风向距离。
        x = hdis if hdis >= 0 else calc_gisdistance(pgis)
        
        if 'A' == atmos_stat:
            if 0 < x <= 300:
                vdpcg = vdpcgs.loc[1]
                ddpcg = ddpcgs.loc[0]
            elif 300 < x <= 500:
                vdpcg = vdpcgs.loc[1]
                ddpcg = ddpcgs.loc[1]
            elif 500 < x <= 1000:
                vdpcg = vdpcgs.loc[1]
                ddpcg = ddpcgs.loc[2]
            else:
                vdpcg = vdpcgs.loc[2]
                ddpcg = ddpcgs.loc[2]
        elif atmos_stat in ('A~B', 'B', 'B~C'):
            if 0 < x <= 500:
                vdpcg = vdpcgs.loc[0]
                ddpcg = ddpcgs.loc[0]
            elif 500 < x <= 1000:
                vdpcg = vdpcgs.loc[0]
                ddpcg = ddpcgs.loc[1]
            else:
                vdpcg = vdpcgs.loc[1]
                ddpcg = ddpcgs.loc[1]
        elif 'C' == atmos_stat:
            if 0 < x <= 1000:
                vdpcg = vdpcgs.loc[0]
                ddpcg = ddpcgs.loc[1]
            else:
                vdpcg = vdpcgs.loc[1]
                ddpcg = ddpcgs.loc[1]
        elif atmos_stat in ('D', 'E', 'E~F', 'F'):
            if 0 < x <= 1000:
                vdpcg = vdpcgs.loc[1]
                ddpcg = ddpcgs.loc[0]
            elif 1000 < x <= 10000:
                vdpcg = vdpcgs.loc[2]
                ddpcg = ddpcgs.loc[1]
            else:
                vdpcg = vdpcgs.loc[2]
                ddpcg = ddpcgs.loc[2]
        else:
            if 0 < x <= 1000:
                vdpcg = vdpcgs.loc[1]
                ddpcg = ddpcgs.loc[0]
            elif 1000 < x <= 2000:
                vdpcg = vdpcgs.loc[2]
                ddpcg = ddpcgs.loc[1]
            else:
                vdpcg = vdpcgs.loc[2]
                ddpcg = ddpcgs.loc[2]
    
        self._add_result('alpha1', vdpcg['alpha1'])
        self._add_result('gama1', vdpcg['gama1'])
        self._add_result('alpha2', ddpcg['alpha2'])
        self._add_result('gama2', ddpcg['gama2'])
        
        return vdpcg.tolist() + ddpcg.tolist(), x
    
    def calc_diffusion_parameters(self, pgis=None, hdis=-1, freq=30):
        """
        方法用于计算扩散参数。
        
        Parameters:
            'pgis' - 计算点 GIS 坐标。
            'hdis' - 垂直下风向轴的距离，单位：m。
            'freq' - 数据采样频率，会影响 sigma_y 取值。
            
        Returns:
            python tuple 对象，index=0：sigma_y，index=1：sigma_z
            
        Raises:
            AssertionError
            
        Note:
            必须给定 GIS 坐标或下风向距离。两者都指定时优先使用 hdis。
            30 <= freq < 6000
        """
        assert 30 <= freq < 6000, self.assert_info('freq')
        
        results = self.get_diffusion_param_coeffs(pgis, hdis)
        dpcgs = results[0]
        x = results[1]
        freq_h = freq / 60
        
        sigma_y = dpcgs[1] * math.pow(x, dpcgs[0])
        sigma_z = dpcgs[3] * math.pow(x, dpcgs[2])
        
        q = 0.2 if 0.5 <= freq_h < 1 else 0.3
        
        sigma_y *= math.pow(freq_h / 0.5, q)
        self._add_result('sigma_y(m)', sigma_y)
        self._add_result('sigma_z(m)', sigma_z)
        
        return sigma_y, sigma_z
    
    @staticmethod
    def get_necessary_mat_params():
        tmp1 = SecurityModel.get_necessary_mat_params()
        tmp2 = GasDiffusionModel._MAT_NE_PARAMS.copy()
        return pd.concat([tmp1, tmp2], ignore_index=True)
    
    @staticmethod
    def get_necessary_env_params():
        tmp1 = SecurityModel.get_necessary_env_params()
        tmp2 = GasDiffusionModel._ENV_NE_PARAMS.copy()
        return pd.concat([tmp1, tmp2], ignore_index=True)
    
    def get_info(self, title='gas diffusion report', width=80, v_width=40):
        return super().get_info(title=title, width=width, v_width=v_width)
            
            
def module_test():
    import pandas as pd
    
    env_params = pd.Series({'wind_speed': 1,
                            'center_longtitude': 120.0,
                            'center_latitude': 30.0,
                            'total_cloudiness': 5,
                            'low_cloudiness': 4})
    
    test = GasDiffusionModel('H3', env_params=env_params)
    test.get_diffusion_param_coeffs(hdis=100)
    print(test.get_info())
    
if '__main__' == __name__: module_test()


