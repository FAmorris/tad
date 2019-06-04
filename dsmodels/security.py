# coding: utf-8

import math
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base import ExplosionModel, FireModel, GasDiffusionModel
from utils import calc_gisdistance

class VaporCloudExplosion(ExplosionModel):
    """
    蒸汽云爆炸模型，可以用于计算蒸汽云质量、蒸汽云爆破能、蒸汽云爆炸 TNT 当量转换、
    蒸汽云冲击波超压计算、蒸汽云冲击波超压距离半径计算。
    """
    _MAT_NE_PARAMS = pd.Series(['material_density',
                                'combustion_heat'])
    
    _ENV_NE_PARAMS = pd.Series(['tnt_explosive_energy',
                                'material_volume',
                                'material_weight'])
    
    def __init__(self, material, mat_params, env_params):
        """
        构造函数。
        
        Parameters:
            material   - 燃烧物质名称。
            mat_params - pandas Series 对象，燃烧物质相关参数。包含以下 key - value
                'material_density' - 蒸汽云的密度，单位：kg/m^3。
                'combustion_heat'  - 蒸汽云对应物质的燃烧热，单位：kJ/m^3。        
                
            env_params - pandas Series 对象，环境参数。可包含以下键值对
                'material_weight'      - 泄露的质量，单位：kg。
                'material_volume'      - 泄露的体积，单位：m^3。
                'tnt_explosive_energy' - 1kg TNT 爆炸产生的爆破能，单位：kJ/kg。若指定该参数，
                                         则采用上述指定值计算，否则使用默认值 4500 kJ/kg。
                                         1kg TNT 爆破能为 4230 ~ 4836 kJ/kg。 
        Raises:
            KeyError
            
        Note:
            如果在构造函数的 'env_params' 参数中直接指定以下 key - value
                'material_weight' - 蒸汽云对应的物质质量，单位：kg。
                
            则可以省略以下 key
                'material_volume'  - 蒸汽云的体积，单位：m^3。
                'material_density' - 蒸汽云的密度，单位：kg/m^3。
                
            如果构造函数的参数 'env_params' 中指定以下 key - value
                'tnt_explosive_energy' - 1kg TNT 爆炸产生的爆破能，单位：kJ/kg。
                
            则采用上述指定值计算，否则使用默认值 4500 kJ/kg。1kg TNT 爆破能为 4230 ~ 4836 kJ/kg。
        """
        if (not mat_params.index.is_unique) or (not env_params.index.is_unique):
            raise KeyError('model parameter is not unique.')
            
        super().__init__(material=material, mat_params=mat_params, env_params=env_params)
        
        self._nan_params = pd.concat([mat_params[mat_params.isnull()], 
                                      env_params[env_params.isnull()]], sort=False)
    
    def calc_material_weight(self):
        """
        方法用于计算蒸汽云对应的质量。
            
        Parameters:
            None
        
        Returns:
            蒸汽云对应的质量，单位：kg。
        
        Raises:
            AssertionError。
        """
         
        env_params = self.get_environment_params()

        if env_params['material_weight'] > 0:
            return env_params['material_weight']

        mat_params = self.get_material_params()
        mat_volume = env_params['material_volume']
        mat_density = mat_params['material_density']
        assert mat_volume > 0, self.assert_info('material_volume')
        assert mat_density > 0, self.assert_info('material_density')
        
        mat_weight = mat_volume * mat_density
        self._env_params['material_weight'] = mat_weight
        
        return mat_weight
        
    def calc_explosive_energy(self, alpha=0.04, beta=1.8):
        """
        方法用于计算蒸汽云爆炸产生的爆破能。
        
        Parameters:
            alpha - TNT 当量系数，0.0002 ≤ alpha ≤ 0.149，默认 alpha = 0.04。
            beta  - 地面爆炸系数，默认 beta=1.8。
        
        Returns:
            蒸汽云爆炸产生的爆破能。单位：kJ。
        
        Raises:
            AssertionError。
            
        Notes:

        """
        assert (alpha > 0) and (beta > 0), self.assert_info('alpha, beta')
        mat_params = self.get_material_params()
        results = self.get_results()

        if 'explosive_energy' in results:
            return results['explosive_energy']

        combustion_heat = mat_params['combustion_heat']
        assert combustion_heat > 0, self.assert_info('combustion_heat')

        mat_weight = self.calc_material_weight()
        explosive_energy = alpha * beta * combustion_heat * mat_weight
        self._add_result('explosive_energy', explosive_energy)
        self._add_environment_param('alpha', alpha)
        self._add_environment_param('beta', beta)
        
        return explosive_energy
        
    def calc_turn_tnt(self, alpha=0.04, beta=1.8):
        """
        方法用于计算蒸汽云爆炸时的 TNT 当量。
        
        Parameters:
            alpha - TNT 当量系数，0.0002 ≤ alpha ≤ 0.149，默认 alpha = 0.04。
            beta  - 地面爆炸系数，默认 beta=1.8。
            
        Returns:
            蒸汽云爆炸时的 TNT 当量，单位：kg。
            
        Raise:
            AssertionError。
            
        Notes:
            
        """
        env_params = self.get_environment_params()
        results = self.get_results()

        if 'tnt_weight' in results:
            return results['tnt_weight']
        
        if 'tnt_explosive_energy' in self._nan_params:
            tnt_explosive_energy = 4500
        else:
            tnt_explosive_energy = env_params['tnt_explosive_energy']
            assert tnt_explosive_energy > 0.0, self.assert_info('tnt_explosive_energy')
        
        explosive_energy = self.calc_explosive_energy(alpha, beta)
        tnt_weight = explosive_energy / tnt_explosive_energy
        self._add_result('tnt_weight', tnt_weight)
        
        return tnt_weight
        
    def calc_wave_overpressure(self, x, alpha=0.04, beta=1.8):
        """
        方法用于计算泄漏物质发生蒸汽云爆炸时距离爆炸中心 x 米处的冲击波超压。
        
        Parameters:
            x     - 某位置到爆炸中心的距离，单位：m。
            alpha - TNT 当量系数，0.0002 ≤ alpha ≤ 0.149，默认 alpha = 0.04。
            beta  - 地面爆炸系数，默认 beta=1.8。
            
        Returns:
            距离爆炸中心 x 米处的冲击波超压，单位：m。
            
        Raises:
            'ZeroDivisionError' - 除数为 0 异常。
            'AssertionError'。
        """
        assert x > 0.0, self.assert_info('x')
        
        tnt_weight = self.calc_turn_tnt(alpha, beta)
        relative_dis = x / (0.1 * math.pow(tnt_weight, 1 / 3))
        wave_overpressure = self.tnt_overpressure_of(relative_dis)
        if wave_overpressure < 0: wave_overpressure = 0.0
        self._add_environment_param('relative_distance: {x}m'.format(x), relative_dis)
        self._add_result('distance: ({x})'.format(x=x), wave_overpressure)
        
        return wave_overpressure
        
    def calc_wave_radius(self, p, alpha=0.04, beta=1.8):
        """
        方法用于计算泄漏物质发生蒸汽云爆炸时 p Mpa 冲击波超压对应的冲击波半径。
        
        Parameters:
            p     - 冲击波超压，单位：Mpa。
            alpha - TNT 当量系数，0.0002 ≤ alpha ≤ 0.149，默认 alpha = 0.04。
            beta  - 地面爆炸系数，默认 beta=1.8。
            
        Returns:
            p Mpa 冲击波超压对应的冲击波半径。
        
        Raises:
            
        """
        assert p > 0.0, self.assert_info('p')
        
        tnt_weight = self.calc_turn_tnt(alpha, beta)
        relative_dis = self.tnt_distance_of(p)
        wave_radius = 0.1 * math.pow(tnt_weight, 1 / 3) * relative_dis
        if wave_radius < 0: wave_radius = 0.0
        self._add_result('overpressure: ({p})'.format(p=p), wave_radius)
        self._add_environment_param('relative_distance:{p}Mpa'.format(p=p), relative_dis)
        return wave_radius
        
    def fit(self): pass
    
    def plot(self): pass
    
    @staticmethod
    def get_necessary_mat_params():
        tmp1 = ExplosionModel.get_necessary_mat_params()
        tmp2 = VaporCloudExplosion._MAT_NE_PARAMS.copy()
        return pd.concat([tmp1, tmp2], ignore_index=True).drop_duplicates()
    
    @staticmethod
    def get_necessary_env_params():
        tmp1 = ExplosionModel.get_necessary_env_params()
        tmp2 = VaporCloudExplosion._ENV_NE_PARAMS.copy()
        return pd.concat([tmp1, tmp2], ignore_index=True).drop_duplicates()
    
    def get_info(self):
        return super().get_info('vapor cloud explosion model reports', width=80, v_width=40)
        
        
class PoolFire(FireModel):
    """
    液体池火模型，可以用于计算液体可燃物的燃烧速度、火焰高度、热辐射通量、
    入热辐射强度、热辐射半径。
    """
    _MAT_NE_PARAMS = pd.Series(['boiling_point',
                                'combustion_heat',
                                'specific_heat_capacity',
                                'gasification_heat',
                                'burning_speed'])
                               
    _ENV_NE_PARAMS = pd.Series(['pool_radius',
                                'env_temp',
                                'air_density'])
    
    def __init__(self, material, mat_params, env_params):
        """
        构造函数。
        
        Parameters:
            'material'   - 燃烧物质名称。
            'mat_params' - pandas Series 对象，燃烧物质相关参数。至少包含以下键值对
                'boiling_point'          - 沸点，单位：K。
                'combustion_heat'        - 燃烧热，单位：J/kg。
                'gasification_heat'      - 气化热，单位：J/kg。
                'specific_heat_capacity' - 定压比热容，单位：J/(kg·K)。
                'burning_speed'          - 燃烧速度，单位：kg/(m^2·s)。
                         
            'env_params' - pandas Series 对象，环境参数。至少包含以下键值对
                'pool_radius' - 液池半径。单位：m。
                'env_temp'    - 环境温度。单位：K。
                'air_density' - 事故点周围空气密度，单位：kg/m^3。
                
        Returns:
            None
        
        Raises:
            None
        """
        if (not mat_params.index.is_unique) or (not env_params.index.is_unique):
            raise KeyError('model parameter is not unique.')
            
        super().__init__(material=material, mat_params=mat_params, env_params=env_params)
        
        self._nan_params = pd.concat([mat_params[mat_params.isnull()], 
                                      env_params[env_params.isnull()]], sort=False)
        
    def calc_burning_speed(self):
        """
        方法用于计算池火事故中，液体燃烧物质的燃烧速度。
        
        Parameters:
            None
        Returns: 
            液体燃烧物质的燃烧速度，kg/(m^2·s)。
            
        Raises:
            ZeroDivisionError - 除数为 0 异常。
            KeyError          - 键不存在异常。
            AssertError       - 模型参数空值断言异常。
        """
        mat_params = self.get_material_params()

        if mat_params['burning_speed'] > 0: 
            return mat_params['burning_speed']

        env_params = self.get_environment_params()
        combustion_heat = mat_params['combustion_heat']
        specific_heat_capacity = mat_params['specific_heat_capacity']
        gasification_heat = mat_params['gasification_heat']
        
        assert combustion_heat > 0, self.assert_info('combustion_heat')
        assert specific_heat_capacity > 0, self.assert_info('specific_heat_capacity')
        assert gasification_heat > 0, self.assert_info('gasification_heat')
        assert not 'env_temp' in self._nan_params, self.assert_info('env_temp')
        
        env_temp = env_params['env_temp']
        
        delta_temp = boiling_point - env_temp
        
        if delta_temp > 0:
            burning_speed = (1e-3 * combustion_heat) / (specific_heat_capacity * delta_temp\
                    + gasification_heat)
        else:
            burning_speed = (1e-3 * combustion_heat) / gasification_heat
        
        self._env_params['burning_speed'] = burning_speed

        return burning_speed
        
    def calc_flame_height(self):
        """
        方法用于计算池火事故中，液体燃烧物质燃烧时火焰高度。
        
        Parameters:
            None
            
        Returns:
            液体燃烧物质燃烧时的火焰高度，m。
            
        Raises:
            ZeroDivisionError - 除数为 0 异常。
            AssertionError    - 模型参数空值断言异常。
        """
        mat_params = self.get_material_params()
        env_params = self.get_environment_params()
        results = self.get_results()
        
        if 'flame_height' in results:
            return results['flame_height']

        air_density = env_params['air_density']
        pool_radius = env_params['pool_radius']
        
        assert air_density > 0, self.assert_info('air_density')
        assert pool_radius > 0, self.assert_info('pool_radius')
        
        burning_speed = self.calc_burning_speed()
        tmp1 = burning_speed / (air_density * math.sqrt(19.6 * pool_radius))
        flame_height = 84 * pool_radius * math.pow(tmp1, 0.6)
        
        self._add_result('flame_height', flame_height)
        
        return flame_height

    def calc_heat_radiation(self, eta=0.24):
        """
        方法用于计算总热辐射通量。
        
        Parameters:
            eta - 燃烧效率因子，取值范围 0.13 ~ 0.35，默认 0.24。
        
        Returns:
            液体燃烧物质释放出的总热辐射通量, W。
        
        Raises:
            ZeroDivisionError - 除数为 0 异常。
            KeyError          - 键不存在异常。
            AssertError       - 模型参数空值断言异常。
        """
        mat_params = self.get_material_params()
        env_params = self.get_environment_params()
        results = self.get_results()

        if 'heat_radiation' in results:
            return results['heat_radiation']

        env_temp = env_params['env_temp']
        pool_radius = env_params['pool_radius']
        air_density = env_params['air_density']
        combustion_heat = mat_params['combustion_heat']
        
        assert not ('env_temp' in self._nan_params), self.assert_info('env_temp')
        assert combustion_heat > 0, self.assert_info('combustion_heat')
        assert eta > 0
        
        burning_speed = self.calc_burning_speed()
        flame_height = self.calc_flame_height()
        tmp1 = math.pi * pool_radius * ( pool_radius + 2 * flame_height)\
                       * burning_speed * eta * combustion_heat
        tmp2 = 72 * math.pow(burning_speed, 0.6) + 1
        heat_radiation = tmp1 / tmp2
        
        self._add_environment_param('eta', eta)
        self._add_result('heat_radiation', heat_radiation)
        
        return heat_radiation
        
    def calc_heat_radiation_strength(self, x, eta=0.24, theta=1.0):
        """
        方法用于计算距离池火事故点中心 x 米处的目标热辐射强度。
        
        Parameters:
            x     - 与池火事故点中心的距离，m。
            eta   - 燃烧效率因子，取值 0.13 ≤ eta ≤ 0.35 默认 eta = 0.24。
            theta - 热传导系数，默认 theta = 1.0。
        
        Returns:
            距离池火事故点中心 x 米处的目标热辐射强度。W/m^2。
        
        Raises:
            ZeroDivisionError - 除数为 0 异常。
            AssertionError       - 模型参数空值断言异常。
        """   
        assert x > 0, self.assert_info('x')
        assert theta > 0, self.assert_info('theta')
        
        heat_radiation = self.calc_heat_radiation(eta)
        heat_radiation_strength = (heat_radiation * theta) / (4 * math.pi * math.pow(x, 2))
        
        self._add_environment_param('theta', theta)
        self._add_result('d{}'.format(x), heat_radiation_strength)
        
        return heat_radiation_strength
        
    def calc_heat_radiation_radius(self, strength, eta=0.24, theta=1.0):
        """
        方法用于计算给定目标热辐射强度对应的半径。
        
        Parameters:
            strength - 目标入射热幅度强度，W/m^2。
            eta      - 燃烧效率因子（0.13 ≤ eta ≤ 0.35），默认 eta = 0.24。
            theta    - 热传导系数，默认 theta = 1.0。
        
        Returns:
            给定目标入射热辐射强度对应的半径，m。
        
        Raises:
            ZeroDivisionError - 除数为 0 异常。
            KeyError          - 键不存在异常。
        """     
        assert strength > 0, self.assert_info('strength')
        
        heat_radiation = self.calc_heat_radiation(eta)
        radius = math.sqrt((theta * heat_radiation) / (4 * math.pi * strength))
        
        self._add_environment_param('theta', theta)
        self._add_result('s{}'.format(strength), radius)
        
        return radius
        
    def fit(self): pass
    
    def plot(self): pass
    
    @staticmethod
    def get_necessary_mat_params():
        tmp1 = FireModel.get_necessary_mat_params()
        tmp2 = PoolFire._MAT_NE_PARAMS.copy()
        return pd.concat([tmp1, tmp2], ignore_index=True).drop_duplicates()
    
    @staticmethod
    def get_necessary_env_params():
        tmp1 = FireModel.get_necessary_env_params()
        tmp2 = PoolFire._ENV_NE_PARAMS.copy()
        return pd.concat([tmp1, tmp2], ignore_index=True).drop_duplicates()
    
    def get_info(self):
        return super().get_info(title='pool fire model reports', width=80, v_width=40)
        
        
class PointSourceGasDiffusion(GasDiffusionModel):
    _MAT_NE_PARAMS = pd.Series()
    _ENV_NE_PARAMS = pd.Series(['source_strength', 'wind_speed'])
    """
    该类实现了连续点源高斯气体扩散模型，该类继承 'GasDiffusionModel' 类。模型提供
    了气体扩散区域垂直下风向轴的距离计算、浓度计算、浓度分布计算。
    """
    def __init__(self, material, mat_params=pd.Series(), env_params=pd.Series()):
        """
        构造方法。
        
        Parameters:
            'mat_params' - pandas Series 对象，扩散气体固有属性集合，目前可不提供。
            'env_params' - pandas Series 对象，扩散气体所在环境属性集合，必须包含以下 key - value
                'wind_speed'        - 区域风速，单位：m/s。默认采样频率 0.5h。
                'center_longtitude' - 事故点经度。
                'center_latitude'   - 事故点纬度。
                'total_cloudiness'  - 建模时总云量，无量纲。
                'low_cloudiness'    - 建模时低云量，无量纲。
                'source_strength'   - 泄露点的源强，单位：g/s
        
        Raises:
            KeyError
        """
        if (not mat_params.index.is_unique) or (not env_params.index.is_unique):
            raise KeyError('model parameter is not unique.')
        super().__init__(material=material, mat_params=mat_params, env_params=env_params)

        self._nan_params = pd.concat([mat_params[mat_params.isnull()], 
                                      env_params[env_params.isnull()]], sort=False)
        
    def calc_source_strength(self):
        """
        g/s
        """
        env_params = self.get_environment_params()
        
        if env_params['source_strength'] > 0.0:
            return env_params['source_strength']
            
    def calc_vertical_distance(self, c, t, hdis, srch=0):
        """
        方法用于计算气体扩散区域垂直风向距离。
        
        Parameters:
            'c' - 目标浓度，单位：mg/m^3。
            't' - 时间发生后的时间，单位：s。
            'hdis' - 事故点水平方向距离的距离，单位：m。
            'srch' - 点源有效高度，单位：m。
        
        Returns:
            float 标量，气体扩散区域垂直风向距离，单位：m。
        
        Raises:
            AssertionError
            
        """
        assert c >= 0, self.assert_info('c')
        assert t > 0, self.assert_info('t')
        
        env_params = self.get_environment_params()
        wind_speed = env_params['wind_speed']
        source_strength = self.calc_source_strength()
        concentration = c + 1e-32
        weight = source_strength * t
        sigma_y, sigma_z = self.calc_diffusion_parameters(hdis=hdis)
        tmp1 = math.log(1e6 * weight / (wind_speed * c * math.pi * sigma_y * sigma_z))
        tmp2 = 0.5 * math.pow(srch / sigma_z, 2)
        b = math.sqrt(2 * math.pow(sigma_y, 2) * (tmp1 - tmp2))
        self._add_result('area({}mg/m^3) b:'.format(c), b)
        
        return b
    
    def calc_concentration(self, pgis=None, hdis=-1, vdis=0, ddis=0, srch=0, keep=True):
        """
        方法根据连续点源高斯扩散模型计算浓度，可根据 GIS 坐标（待扩展）或具体距离参数。
        
        Parameters:
            'pgis' - 目标位置的经纬度，pgis=[经度, 纬度]，目前不支持，待扩展。
            'hdis' - 下风向距离，单位：m。
            'vdis' - 垂直下风向距离，单位：m。
            'ddis' - 地面高度，单位：m。
            'srch' - 点源有效高度，单位：m。
            'keep' - 是否在内存中存储计算结果。
        
        Returns:
            float，目标位置的浓度，单位：mg/m^3。
        
        Raises:
            AssertionError
            
        Note:
        
        """
        assert ddis >= 0, self.assert_info('ddis')
        assert srch >= 0, self.assert_info('srch')
        assert vdis >= 0, self.assert_info('vdis')
        env_params = self.get_environment_params()
        wind_speed = env_params['wind_speed']
        assert wind_speed > 0, self.assert_info('wind_speed')

        hdis += 1e-32
        y = vdis
        
        sigma_y, sigma_z = self.calc_diffusion_parameters(pgis, hdis)
        source_strength = self.calc_source_strength()
        
        a1 = source_strength / (math.pi * wind_speed * sigma_y * sigma_z)
        a2 = -0.5 * math.pow(y / sigma_y, 2)
        a3 = -0.5 * math.pow((ddis - srch) / sigma_z, 2)
        a4 = -0.5 * math.pow((ddis + srch) / sigma_z, 2)
        
        if ((0 == srch) or (0 == ddis)) and (0 != vdis): 
            concentration = a1 * math.exp(a2 + a4)
        elif (0 == vdis) and (0 == ddis) and (0 != srch):
            concentration = a1 * math.exp(a4)
        elif (0 == vdis) and (0 == ddis) and (0 == srch):
            concentration = a1
        else:
            concentration = 0.5 * a1 * (math.exp(a2 + a3) + math.exp(a2 + a4))
        
        if concentration < 1e-6: concentration = 0.0
        if keep:self._add_result('concentration({}, {}, {}, {})'.format(hdis, vdis, ddis, srch), 
                                    concentration)
        
        return concentration
        
    def calc_distribution(self, cs, t, ddis=0, srch=0, step=10, hcd=False):
        """
        方法用于计算事故点目标浓度的分布范围。
        
        Parameters:
            'c'    - 目标浓度，单位：mg/m^3。
            't'    - 事故发生后的时间，单位：s。
            'ddis' - 地面高度，单位：m。
            'srch' - 点源有效高度，单位：m。
            'step' - 步长，减小该参数可提高精度，但会增加计算时间，0 < step < (t * 风速)， 单位：m。
            'hcd'   - 该参数用于指定是否返回下风向轴上的浓度分布。
        
        Returns:
            python list 对象。其中
                index = 0，python tuple 对象，表示椭圆分布区域的长短半轴，即 a 和 b，单位：m。
                index = 1，python tuple 对象，表示椭圆分布区域的横轴起始和终止点，单位：m。
                index = 2，python tuple 对象，表示最大值出现的横轴距离和最大浓度值，单位：m 和 mg/m^3
                如果 hcd = True，则 index = 4，pandas Series 对象，
                表示横轴的浓度分布，index 表示距离，values 表示浓度，单位：m 和 mg/m^3。
            
        Raises:
            AssertionError
            
        Note:
            如果给定的浓度值大于最大值，则椭圆分布区域和起始位置为 None，
            但返回最大值和最大值出现的位置。
            
        """
        assert srch >= 0, self.assert_info('srch')
        assert t > 0, self.assert_info('t')
        env_params = self.get_environment_params()
        wind_speed = env_params['wind_speed']
        assert wind_speed > 0.0, self.assert_info('wind_speed')
        hdis_max = math.ceil(wind_speed * t)
        assert hdis_max > step and step > 0, self.assert_info(step)
        
        num = hdis_max // step
        hdises = np.linspace(0, hdis_max, num + 1)
        points = pd.Series(hdises, index=hdises)
        concentrations = points.apply(lambda x: self.calc_concentration(hdis=x, ddis=ddis,\
                                                            srch=srch, keep=False))
        xm = concentrations.idxmax()
        cm = concentrations[xm]
        
        ab = []
        scopes = []
        for c in cs:
            assert c >= 0, self.assert_info('c')
            if c < cm:
                target_scope = concentrations[concentrations >= c]
                x1 = target_scope.index.values.min()
                x2 = target_scope.index.values.max()
                a = (x2 - x1) / 2
                self._add_result('area({}mg/m^3) a:'.format(c), a)
                b = self.calc_vertical_distance(c=c, t=t, hdis=a, srch=srch)
            else:
                a = b = x1 = x2 = None
        
            ab.append((a, b))
            scopes.append((x1, x2))
        results = [ab, scopes, (xm, cm)]
        
        if hcd:
            results.append(concentrations)
            
        return results
     
    def fit(self): pass
    
    def plot(self): pass
     
    @staticmethod
    def get_necessary_mat_params():
        tmp1 = GasDiffusionModel.get_necessary_mat_params()
        tmp2 = PointSourceGasDiffusion._MAT_NE_PARAMS.copy()
        return pd.concat([tmp1, tmp2], ignore_index=True).drop_duplicates() 
    
    @staticmethod
    def get_necessary_env_params():
        tmp1 = GasDiffusionModel.get_necessary_env_params()
        tmp2 = PointSourceGasDiffusion._ENV_NE_PARAMS.copy()
        return pd.concat([tmp1, tmp2], ignore_index=True).drop_duplicates() 
        
    def get_info(self):
        return super().get_info(title='point source gas diffusion model reports', width=80,\
                v_width=40)
        
        
def module_test():
    import pandas as pd
    gas_mat_params = pd.Series({'material_density': 0.79 * 1e3,
                                'combustion_heat': 45980})
    gas_env_params = pd.Series({'tnt_explosive_energy': 4675,
                                'material_volume': None,
                                'material_weight': 23700})
    
    gas = VaporCloudExplosion('gasline',mat_params=gas_mat_params, env_params=gas_env_params)
    gas.calc_wave_radius(0.1)
    print(gas.get_info())
    
    rawoil_mat_params = {'boiling_point': None,
                         'combustion_heat': 41030000,
                         'specific_heat_capacity': None,
                         'gasification_heat': None,
                         'burning_speed': 0.0781}
    rawoil_env_params = {'env_temp': 25,
                         'pool_radius': 24.7,
                         'air_density': 1.293}
    mat_params = pd.Series(rawoil_mat_params)
    env_params = pd.Series(rawoil_env_params)
    
    rawoil = PoolFire('rawoil', mat_params=mat_params, env_params=env_params)
    rawoil.calc_heat_radiation_strength(100, eta=0.35)
    rawoil.calc_heat_radiation_radius(strength=37500, eta=0.35)
    rawoil.calc_heat_radiation_radius(strength=25000, eta=0.35)
    rawoil.calc_heat_radiation_radius(strength=12500, eta=0.35)
    print(rawoil.get_info())
    
    env_params = pd.Series({'wind_speed': 1.5,
                            'center_longtitude': 121.0583333,
                            'center_latitude': 30.62083333,
                            'total_cloudiness': 5,
                            'low_cloudiness': 4,
                            'source_strength': 25000,
                            'start_datetime': '2019-01-01 00:00:00'})
    
    h2 = PointSourceGasDiffusion('H2', env_params=env_params)
    res = h2.calc_distribution([30], 360, srch=5, step=10)
    print(h2.get_info())
    print(res)
    
if '__main__' == __name__: 
    module_test()
