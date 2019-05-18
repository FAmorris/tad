# coding: utf-8
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from abc import ABC, abstractmethod

class SecurityModel(ABC):
	"""
	安防模型抽象基类，定义物质名称、物质物理参数、环境参数。所有子类需要实现以下抽象方法
		'fit' - 方法用于拟合某个区域的安全态势。
		'plot' - 方法用于绘制区域热力图。
	"""

	def __init__(self, material=None, mat_params=None, env_params=None):
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

		import pandas as pd
		
		self._results = pd.Series()
		self.set_material(material)
		self.set_material_params(mat_params)
		self.set_environment_params(env_params)
    
	def set_material(self, material): self._material = material

	def get_material(self): return self._material

	def set_material_params(self, mat_params): self._mat_params = mat_params.copy()

	def get_material_params(self): return self._mat_params.copy()

	def set_environment_params(self, env_params): self._env_params = env_params.copy()

	def get_environment_params(self): return self._env_params.copy()

	def _add_result(self, param, value): self._results[param] = value

	def _add_environment_param(self, param, value): self._env_params[param] = value

	def get_results(self): return self._results.copy()

	def assert_info(self, param): 'Parameter "' + param + '" can not be None.'
    
	@abstractmethod
	def fit(self): pass

	@abstractmethod
	def plot(self): pass
	
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
			info += item_fmt.format(p=i, v=v) + '\n'
		info += '=' * width + '\n'
		
		info += item_fmt.format(p='Environment Parameter', v='Value') + '\n'
		info += '-' * width + '\n'
		env_params = self.get_environment_params()
		for i, v in env_params.iteritems():
			info += item_fmt.format(p=i, v=v) + '\n'
		info += '=' * width + '\n'
		
		info += item_fmt.format(p='Result', v='Value') + '\n'
		info += '-' * width + '\n'
		results = self.get_results()
		for i, v in results.iteritems():
			info += item_fmt.format(p=i, v=v) + '\n'
		info += '=' * width + '\n'
		
		return info


class ExplosionModel(SecurityModel):
	"""
	物质爆炸模型抽象基类，'SecurityModel' 的子类，该抽象类实现了
	1000kg TNT 爆炸产生冲击波超压与爆炸中心之间距离的计算算法。
	"""

	def __init__(self, material=None, mat_params=None, env_params=None):
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

		import pandas as pd

		self._dist = (5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75)
		self._pres = (2.94, 2.06, 1.67, 1.27, 0.95, 0.76, 0.50, 0.33, 0.235, 0.17, 0.126, 0.079, 0.057, 
					  0.043, 0.033, 0.027, 0.0235, 0.0205, 0.018, 0.016, 0.0143, 0.013)

		super().__init__(material=material, mat_params=mat_params, env_params=env_params)
		self._dataset = pd.Series(self._pres, index=self._dist)
		self.polyfit()

	def get_dataset(self): return self._dataset.copy()

	def get_pod_poly(self): return self._pod_poly

	def get_dop_poly(self): return self._dop_poly

	def polyfit(self):
		"""
		方法采用三次样条插值算法拟合 1000kg TNT 产生的超压和距离之间的关系式。

		Parameters:
			None

		Returns:
			None

		Raises:
			
		"""

		from scipy.interpolate import splrep

		dataset = self.get_dataset()
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

		from scipy.interpolate import splev

		return splev(distance, self.get_pod_poly())

	def tnt_distance_of(self, overpressure): 
		"""
		方法用于计算 1000kg TNT 爆炸时，某冲击波超压下，位置点与爆炸中心之间的距离。

		Parameters:
			overpressure - 爆炸产生的冲击波超压，单位：Mpa。

		Returns:
			位置点与爆炸中心之间的距离，单位：m。

		Raises:

		"""

		from scipy.interpolate import splev

		return splev(overpressure, self.get_dop_poly())

	def get_info(self, title='explosion reports', width=80, v_width=40):
		return super().get_info(title, width=width, v_width=v_width)


class FireModel(SecurityModel):
	"""
	物质燃烧模型抽象基类，'SecurityModel' 的子类，
	"""

	def __init__(self, material=None, mat_params=None, env_params=None):
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

	def get_info(self, title='fire reports', width=80, v_width=40):
		return super().get_info(title=title, width=width, v_width=v_width)
		

class GasDiffusion(SecurityModel):
	
	def __init__(self, material, mat_params, env_params):
		import pandas as pd
		
		super().__init__(material=material, mat_params=mat_params, env_params=env_params)
		
		self._srlt = pd.DataFrame([[-2, -1, 1, 2, 3],
								   [-1, 0, 1, 2, 3],
								   [-1, 0, 0, 1, 1],
								   [0, 0, 0, 0, 1],
								   [0, 0, 0, 0, 0]])
								   
		self._ast = pd.DataFrame([['A', 'A~B', 'B', 'D', 'E', 'F'],
								  ['A~B', 'B', 'C', 'D', 'E', 'F'],
								  ['B', 'B~C', 'C', 'D', 'D', 'E'],
								  ['C', 'C~D', 'D', 'D', 'D', 'D'],
								  ['D', 'D', 'D', 'D', 'D', 'D']], 
								  columns=['3', '2', '1', '0', '-1', '-2'])
		
		self._dpct_index=[[0, 1, 2, 
						   1, 2, 
						   1, 2, 
						   1, 2, 
						   1, 2, 
						   0, 1, 2, 
						   0, 1, 2, 
						   0, 1, 2, 
						   0, 1, 2, 
						   0, 1, 2, 
						   0, 1, 2],
						  ['A', 'A', 'A', 
						  'A~B', 'A~B', 
						  'B', 'B', 
						  'B~C', 'B~C', 
						  'C', 'C',
						  'C~D', 'C~D', 'C~D',
						  'D', 'D', 'D',
						  'D~E', 'D~E', 'D~E',
						  'E', 'E', 'E',
						  'E~F', 'E~F', 'E~F',
						  'F', 'F', 'F']]
						  
		self._dpct = pd.DataFrame([[0.000000, 0.000000, 1.12154, 0.079990],  # A                垂直 0 ~ 300
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
								   [0.000000, 0.000000, 0.78837, 0.092753],  # E 				垂直 0~1000
								   [0.920818, 0.086400, 0.56518, 0.433384],  # E   水平 1~1000，垂直 1000~10000
								   [0.896864, 0.101947, 0.41474, 1.732410],  # E   水平 > 1000，垂直 > 10000
								   [0.000000, 0.000000, 0.78639, 0.077415],  # E~F              垂直 0~1000
								   [0.925118, 0.070882, 0.54558, 0.401700],  # E~F 水平 0~1000，垂直 1000~10000
								   [0.892794, 0.087641, 0.36870, 2.069660],  # E~F 水平 > 1000，垂直 > 10000
								   [0.000000, 0.000000, 0.78440, 0.062077],  # F                垂直 0~1000
								   [0.929418, 0.055363, 0.52597, 0.370015],  # F   水平 0~1000，垂直 1000~10000
								   [0.888723, 0.073335, 0.32266, 2.406910]], # F   水平 > 1000，垂直 > 10000
								   index=self._dpct_index,
								   columns=['alpha1', 'gama1', 'alpha2', 'gama2'])
	
	def get_solar_radiation_level_table(self): return self._srlb.copy()
	
	def get_atmospheric_stability_table(self): return self._ast.copy()
	
	def _get_dpct(self): return self._dpct.copy()
		
	def calc_declination(self):
		from datetime import datetime
		import math
		
		day = datetime.now().timetuple().tm_yday

		if 366 == day: day = 365
		theta = 360 * day / 365
		
		declination = (0.006918 - 0.399912 * math.cos(theta) + 0.070257 * math.sin(theta)\
						- 0.006758 * math.cos(2 * theta) + 0.000907 * math.sin(2 * theta)\
						- 0.002697 * math.cos(3 * theta) + 0.00148 * math.sin(3 * theta)) * (180 / math.pi)
		
		return declination
		
	def calc_solar_angle(self):
		from datetime import datetime
		import math
		
		env_params = self.get_environment_params()
		
		lgt = env_params['longtitude']
		lat = env_params['latitude']
		
		declination = self.calc_declination()
		
		solar_angle = math.arcsin(math.sin(lat) * declination + math.cos(lat)\
						* math.cos(declination) * (15 * datetime.now().hour + lgt - 300))
		
		return solar_angle
		
	def get_solar_radiation_level(self):
		from datetime import datetime
		
		hour = datetime.now().hour
		env_params = self.get_environment_params()
		tcloudiness = env_params['total_cloudiness']
		lcloudiness = env_params['low_cloudiness']
		
		if (tcloudiness <= 4 and lcloudiness <= 4): row = 0
			elif (5 < tcloudiness < 7) and (lcloudiness <= 4): row = 1
			elif (tcloudiness >= 8) and (lcloudiness <= 4): row = 2
			elif (tcloudiness >= 5) and (5 < lcloudiness < 7): row = 3
			elif (tcloudiness >= 8) and (lcloudiness >= 8): row = 4
		
		if 7 <= hour <= 18:
			solar_angle = self.calc_solar_angle()
			
			if solar_angle <= 15: col = 1
			elif 15 < solar_angle <= 35: col = 2
			elif 35 < solar_angle <= 65: col = 3
			elif solar_angle > 65: col = 4
		else: col = 0
	
		solar_radiation_level = self.get_solar_radiation_level_table().iloc[row, col]
		
		return solar_radiation_level
		
	def get_atmospheric_stability(self):
		
		env_params = self.get_environment_params()
		wind_volicity = env_params['wind_volicity']
		srl = str(self.calc_solar_radiation_level())
		
		if 0 <= wind_volicity <= 1.9: row = 0
		elif 1.9 < wind_volicity <= 2.9: row = 1
		elif 2.9 < wind_volicity <= 4.9: row = 2
		elif 4.9 < wind_volicity <= 5.9: row = 3
		elif wind_volicity >= 6.0: row = 4
		
		atmospheric_stability = self.get_atmospheric_stability_table().iloc[row][srl]
		
		return atmospheric_stability
		
	def get_diffusion_param_coeffs(self, p_gis=None, p_dis=None):
		from utils import calc_gisdistance
		
		assert (p_gis and p_dis), self.assert_info('p_gis and p_dis') + 'both.'
		
		atmos_stat = self.get_atmospheric_stability()
		
		x = p_dis if p_dis else calc_gisdistance(p_gis)
		
		if 'A' == atmos_stat:
			row = 0
			if 0 < x < 300:
				alpha1 = 
			
			
		
		
		
		
		
			
		
		
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	