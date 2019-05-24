#! /root/apps/python3/bin

from dsmodels.security import VaporCloudExplosion, PoolFire, PointSourceGasDiffusion

import flask
from flask import request
import flask_script
import pandas as pd
import json

app = flask.Flask(__name__)

@app.route('/')
def index():
    return 'hello, world!'


@app.route('/tad/v1.0.0/dsmodels/security/api/hurtScope', methods=['POST'])
def get_vce_radius():
    data = request.json['data']
    modeltype = request.json['modeltype']
    material = data['material']
    mat_params = pd.Series(data['mat_params'])
    env_params = pd.Series(data['env_params'])
    fparams = data['fparams']
    modelins = fparams['modelins']
    
    try:
        if 'VaporCloudExplosion' == modeltype:
            vce = VaporCloudExplosion(material, mat_params, env_params)
            modelouts = [vce.calc_wave_radius(modelin, fparams['alpha'], fparams['beta']) for modelin in modelins]
        elif 'PoolFire' == modeltype:
            pf = PoolFire(material, mat_params, env_params)
            modelouts = [pf.calc_heat_radiation_radius(modelin, fparams['eta'], fparams['theta']) for modelin in modelins]
        elif 'PointSourceGasDiffusion' == modeltype:
            psgd = PointSourceGasDiffusion(material, mat_params, env_params)
            modelouts = [psgd.calc_distribution(modelin, fparams['t'], fparams['ddis'], fparams['srch'], fparams['step']) for modelin in modelins]
            
        res_code = 0
        res_msg = 'success'
        res_data = {'material': material, 'modelouts': modelouts}
    except:
        res_code = 1
        res_msg = 'faild'
        res_data = None
        
    return json.dumps({'code': res_code, 'massege': res_msg, 'data': res_data})
    
    
if '__main__' == __name__:
    app_manager = flask_script.Manager(app)
    app_manager.run()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    