# coding: utf-8

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