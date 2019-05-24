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


@app.route('/dsmodels/v1.0.0/security/vce/radius', methods=['POST'])
def get_vce_radius():
    data = request.json['data']
    material = data['material']
    mat_params = data['mat_params']
    env_params = data['env_params']
    ops = data['ops']
    
    try:
        vce = VaporCloudExplosion(material, mat_params, env_params)
        radiuses = [vce.calc_wave_radius(op, env_params['alpha'], env_params['beta']) for op in ops]
        res_code = 0
        res_msg = 'Success'
        res_data = radiuses
    except:
        res_code = 1
        res_msg = 'Faild'
        res_data = None
        
    res= {'code': res_code, 'massege': res_msg, 'data': res_data}
    return json.dumps(res)

if '__main__' == __name__:
    app_manager = flask_script.Manager(app)
    app_manager.run()