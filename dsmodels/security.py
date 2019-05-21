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
    _MAT_NE_PARAMS = pd.Series(['material_volume',
                                'material_density',
                                'combustion_heat',
                                'material_weight'])
    
    _ENV_NE_PARAMS = pd.Series(['tnt_explosive_energy'])
    
    def __init__(self, material, mat_params, env_params):
        """
        构造函数。
        
        Parameters:
            material   - 燃烧物质名称。
            mat_params - pandas Series 对象，燃烧物质相关参数。至少包含以下 key - value
            
                'material_volume'  - 蒸汽云的体积，单位：m^3。
                'material_density' - 蒸汽云的密度，单位：kg/m^3。
                         
            env_params - pandas Series 对象，环境参数。可包含以下键值对
            
                'tnt_explosive_energy' - 1kg TNT 爆炸产生的爆破能，单位：kJ/kg。若指定该参数，
                                         则采用上述指定值计算，否则使用默认值 4500 kJ/kg。
                                         1kg TNT 爆破能为 4230 ~ 4836 kJ/kg。 
        Returns:
            None
        
        Raises:
            None
        """
        if (not VaporCloudExplosion._MAT_NE_PARAMS.isin(mat_params.index).all())\
            or (not VaporCloudExplosion._ENV_NE_PARAMS.isin(env_params.index).all()):
            raise KeyError('model parameter loss.')
        if (not mat_params.index.is_unique) or (not env_params.index.is_unique):
            raise KeyError('model parameter is not unique.')
            
        super().__init__(material=material, mat_params=mat_params, env_params=env_params)
        
        self._nan_params = pd.concat([mat_params[mat_params.isnull()], 
                                      env_params[env_params.isnull()]], sort=False)
    
    def calc_material_weight(self):
        """
        方法用于计算蒸汽云对应的质量，构造函数中的 'mat_params' 
        参数需要包含以下键值对
            'material_volume'  - 蒸汽云的体积，单位：m^3。
            'material_density' - 蒸汽云的密度，单位：kg/m^3。
            
        Parameters:
            None
        
        Returns:
            蒸汽云对应的质量，单位：kg。
        
        Raises:
        
        """
        mat_params = self.get_material_params()
        mat_volume = mat_params['material_volume']
        mat_density = mat_params['material_density']
        
        assert mat_volume > 0.0, self.assert_info('material_volume')
        assert mat_density > 0.0, self.assert_info('material_density')
        
        mat_weight = mat_volume * mat_density
        self._add_result('material_weight(kg)', mat_weight)
        
        return mat_weight
        
    def calc_explosive_energy(self, alpha=0.04, beta=1.8):
        """
        方法用于计算蒸汽云爆炸产生的爆破能。构造函数中的 'mat_params' 参数需要包含以下 key - value
        
            'material_volume'  - 蒸汽云的体积，单位：m^3。
            'material_density' - 蒸汽云的密度，单位：kg/m^3。
            'combustion_heat'  - 蒸汽云对应物质的燃烧热，单位：kJ/m^3。
        
        Parameters:
            alpha - TNT 当量系数，0.0002 ≤ alpha ≤ 0.149，默认 alpha = 0.04。
            beta  - 地面爆炸系数，默认 beta=1.8。
        
        Returns:
            蒸汽云爆炸产生的爆破能。单位：kJ。
        
        Raises:
        
        Notes:
            如果在构造函数的 'env_params' 参数中直接指定以下 key - value
            
                'material_weight' - 蒸汽云对应的物质质量，单位：kg。
                
            则可以省略以下 key
            
                'material_volume'  - 蒸汽云的体积，单位：m^3。
                'material_density' - 蒸汽云的密度，单位：kg/m^3。
        """
        mat_params = self.get_material_params()
        combustion_heat = mat_params['combustion_heat']
        
        assert combustion_heat > 0.0, self.assert_info('combustion_heat')
        assert (alpha > 0.0) and (beta > 0.0), self.assert_info('alpha, beta')
        
        if 'material_weight' in self._nan_params:
            mat_weight = self.calc_material_weight()
        else:
            mat_weight = mat_params['material_weight']
        
        explosive_energy = alpha * beta * combustion_heat * mat_weight
        self._add_result('explosive_energy(kJ)', explosive_energy)
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
        
        Notes:
            如果构造函数的参数 'env_params' 中包含以下 key - value
                'tnt_explosive_energy' - 1kg TNT 爆炸产生的爆破能，单位：kJ/kg。
                
            则采用上述指定值计算，否则使用默认值 4500 kJ/kg。
            1kg TNT 爆破能为 4230 ~ 4836 kJ/kg。
        """
        env_params = self.get_environment_params()
        
        if 'tnt_explosive_energy' in self._nan_params:
            tnt_explosive_energy = 4500
        else:
            tnt_explosive_energy = env_params['tnt_explosive_energy']
            assert tnt_explosive_energy > 0.0, self.assert_info('tnt_explosive_energy')
        
        explosive_energy = self.calc_explosive_energy(alpha, beta)
        tnt_weight = explosive_energy / tnt_explosive_energy
        self._add_result('tnt_weight(kg)', tnt_weight)
        
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
        """
        import math
        
        assert x >= 0.0, self.assert_info('x')
        
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
        import math
        
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
        return pd.concat([tmp1, tmp2], ignore_index=True)
    
    @staticmethod
    def get_necessary_env_params():
        tmp1 = ExplosionModel.get_necessary_env_params()
        tmp2 = VaporCloudExplosion._ENV_NE_PARAMS.copy()
        return pd.concat([tmp1, tmp2], ignore_index=True)
    
    def get_info(self):
        return super().get_info('steam cloud explosion model reports', width=80, v_width=40)
        
        
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
            material   - 燃烧物质名称。
            mat_params - pandas Series 对象，燃烧物质相关参数。至少包含以下键值对
                
                'boiling_point'          - 沸点，单位：K。
                'combustion_heat'        - 燃烧热，单位：J/kg。
                'specific_heat_capacity' - 定压比热容，单位：J/(kg·K)。
                'gasification_heat'      - 气化热，单位：J/kg。
                         
            env_params - pandas Series 对象，环境参数。至少包含以下键值对
            
                'pool_radius' - 液池半径。单位：m。
                'env_temp'    - 环境温度。单位：K。
                'air_density' - 事故点周围空气密度，单位：kg/m^3。
                
        Returns:
            None
        
        Raises:
            None
        """
        if (not PoolFire._MAT_NE_PARAMS.isin(mat_params.index).all())\
            or (not PoolFire._ENV_NE_PARAMS.isin(env_params.index).all()):
            raise KeyError('model parameter loss.')
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
        env_params = self.get_environment_params()
        boiling_point = mat_params['boiling_point']
        combustion_heat = mat_params['combustion_heat']
        specific_heat_capacity = mat_params['specific_heat_capacity']
        gasification_heat = mat_params['gasification_heat']
        env_temp = env_params['env_temp']
        
        assert not (boiling_point is np.nan), self.assert_info('boiling_point')
        assert combustion_heat > 0.0, self.assert_info('combustion_heat')
        assert specific_heat_capacity > 0.0, self.assert_info('specific_heat_capacity')
        assert gasification_heat > 0.0, self.assert_info('gasification_heat')
        assert not (env_temp is np.nan), self.assert_info('env_temp')
        
        delta_temp = boiling_point - env_temp
        
        if delta_temp > 0:
            burning_speed = (1e-3 * combustion_heat) / (specific_heat_capacity * delta_temp + gasification_heat)
        else:
            burning_speed = (1e-3 * combustion_heat) / gasification_heat
        
        self._add_result('burning_speed', burning_speed)
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
            AssertError       - 模型参数空值断言异常。
        """
        import math
        
        mat_params = self.get_material_params()
        env_params = self.get_environment_params()
        air_density = env_params['air_density']
        pool_radius = env_params['pool_radius']
        
        assert air_density > 0.0, self.assert_info('air_density')
        assert pool_radius > 0.0, self.assert_info('pool_radius')
        
        if 'burning_speed' in self._nan_params:
            burning_speed = self.calc_burning_speed()
        else:
            burning_speed = mat_params['burning_speed']
            assert burning_speed > 0.0, self.assert_info('burning_speed')
        
        tmp1 = burning_speed / (air_density * math.sqrt(19.6 * pool_radius))
        flame_height = 84 * pool_radius * math.pow(tmp1, 0.6)
        
        self._add_result('flame_height(m)', flame_height)
        
        return flame_height

    def calc_heat_radiation(self, eta=0.24):
        """
        方法用于计算总热辐射通量。
        
        Parameters:
            eta - 燃烧效率，取值范围 0.13 ~ 0.35，默认 0.24。
        
        Returns:
            液体燃烧物质释放出的总热辐射通量, W。
        
        Raises:
            ZeroDivisionError - 除数为 0 异常。
            KeyError          - 键不存在异常。
            AssertError       - 模型参数空值断言异常。
        """
        import math
        
        mat_params = self.get_material_params()
        env_params = self.get_environment_params()
        env_temp = env_params['env_temp']
        pool_radius = env_params['pool_radius']
        air_density = env_params['air_density']
        combustion_heat = mat_params['combustion_heat']
        
        assert not('env_temp' in self._nan_params.index), self.assert_info('env_temp')
        assert combustion_heat > 0.0, self.assert_info('combustion_heat')
        assert eta > 0.0
        
        if 'burning_speed' in self._nan_params:
            burning_speed = self.calc_burning_speed()
        else:
            burning_speed = mat_params['burning_speed']
            assert burning_speed > 0.0, self.assert_info('burning_speed')
            
        flame_height = self.calc_flame_height()
        tmp1 = math.pi * pool_radius * ( pool_radius + 2 * flame_height)\
                       * burning_speed * eta * combustion_heat
        tmp2 = 72 * math.pow(burning_speed, 0.6) + 1
        heat_radiation = tmp1 / tmp2
        
        self._add_environment_param('eta', eta)
        self._add_result('heat_radiation(W)', heat_radiation)
        
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
            AssertError       - 模型参数空值断言异常。
        """
        import math
        
        assert x > 0.0, self.assert_info('x')
        assert theta > 0.0, self.assert_info('theta')
        
        heat_radiation = self.calc_heat_radiation(eta)
        heat_radiation_strength = (heat_radiation * theta) / (4 * math.pi * math.pow(x, 2))
        
        self._add_environment_param('theta', theta)
        self._add_result('distance: {}(m)'.format(x), heat_radiation_strength)
        
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
        import math
        
        assert strength > 0.0, self.assert_info('strength')
        
        heat_radiation = self.calc_heat_radiation(eta)
        radius = math.sqrt((theta * heat_radiation) / (4 * math.pi * strength))
        
        self._add_environment_param('theta', theta)
        self._add_result('strength: {}(W)'.format(strength), radius)
        
        return radius
        
    def fit(self): pass
    
    def plot(self): pass
    
    @staticmethod
    def get_necessary_mat_params():
        tmp1 = FireModel.get_necessary_mat_params()
        tmp2 = PoolFire._MAT_NE_PARAMS.copy()
        return pd.concat([tmp1, tmp2], ignore_index=True)
    
    @staticmethod
    def get_necessary_env_params():
        tmp1 = FireModel.get_necessary_env_params()
        tmp2 = PoolFire._ENV_NE_PARAMS.copy()
        return pd.concat([tmp1, tmp2], ignore_index=True)
    
    def get_info(self):
        return super().get_info(title='pool fire model reports', width=80, v_width=40)
        
        
class PointSourceGasDiffusion(GasDiffusionModel):
    _MAT_NE_PARAMS = pd.Series()
    _ENV_NE_PARAMS = pd.Series(['source_strength', 'wind_volicity'])
    
    def __init__(self, material, mat_params=pd.Series(), env_params=pd.Series()):
        if (not PointSourceGasDiffusion._ENV_NE_PARAMS.isin(env_params.index).all()):
            raise KeyError('model parameter loss.')
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
    
    def calc_concentration(self, pgis=None, hdis=-1, vdis=-1, ddis=0, srch=0):
        """
        mg/m^3
        """
        env_params = self.get_environment_params()
        wind_volicity = env_params['wind_volicity']
        
        assert wind_volicity > 0, self.assert_info('wind_volicity')
        assert pgis or (vdis >= 0), self.assert_info('pgis, vdis')
        assert ddis >= 0, self.assert_info('ddis')
        assert srch >= 0, self.assert_info('srch')
        
        if 0 == hdis: hdis += 1e-32
        
        sigma_y, sigma_z = self.calc_diffusion_parameters(pgis, hdis)
        y = vdis if vdis >= 0 else calc_gisdistance(pgis)
        source_strength = self.calc_source_strength()
        
        a1 = source_strength / (math.pi * wind_volicity * sigma_y * sigma_z)
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
        self._add_result('concentration({}, {}, {}, {})'.format(hdis, vdis, ddis, srch), concentration)
        
        return concentration
        
    def calc_distribution(self, c, srch=0, area=False):
        env_params = self.get_environment_params()
        wind_volicity = env_params['wind_volicity']
        assert srch >= 0, self.assert_info('srch')
        assert wind_volicity > 0.0, self.assert_info('wind_volicity')
        
        source_strength = self.calc_source_strength()
        
        if area:
            sigma_y, sigma_z = self.calc_diffusion_parameters(hdis=hdis)
            tmp1 = math.log(1e6 * source_strength / (wind_volicity * c * math.pi * sigma_y * sigma_z))
            tmp2 = 0.5 * math.pow(srch / sigma_z, 2)
            vdis = math.sqrt(2 * math.pow(sigma_y, 2) * (tmp1 - tmp2))
            self._add_result('area({}mg/m^3) h:'.format(c), hdis)
            self._add_result('area({}mg/m^3) v:'.format(c), vdis)
            return hdis, vdis
        self._add_result('radius({}mg/m^3):'.format(c), hdis)
        
        return hdis
     
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
        return super().get_info(title='point source gas diffusion model reports', width=80, v_width=40)
        
        
def module_test():
    import pandas as pd
    gas_mat_params = pd.Series({'material_volume': 3000 * 1e-2,
                                'material_density': 0.79 * 1e3,
                                'combustion_heat': 45980,
                                'material_weight': None})
    gas_env_params = pd.Series({'tnt_explosive_energy': 4675})
    
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
    
    env_params = pd.Series({'wind_volicity': 1.5,
                            'center_longtitude': 121.0212504359,
                            'center_latitude': 30.6650761821,
                            'total_cloudiness': 5,
                            'low_cloudiness': 4,
                            'source_strength': 25000})
    
    h2 = PointSourceGasDiffusion('H2', env_params=env_params)
    res = []
    for x in range(0, 100000, 10):
        res.append(h2.calc_concentration(hdis=x, vdis=0, srch=100))
    
    print(h2.get_info())
    
    import matplotlib.pyplot as plt
    
    plt.plot(range(0, 100000, 10), res)
    plt.show()
    
if '__main__' == __name__: 
    module_test()
    # print(PointSourceGasDiffusion.get_necessary_env_params())