#! /root/apps/python3/bin

from dsmodels.security import VaporCloudExplosion, PoolFire, PointSourceGasDiffusion

import flask
import flask_script
import pandas as pd
import json

app = flask.Flask(__name__)

@app.route('/')
def index():
    return 'hello, world!'


@app.route('/dsmodels/v1.0.0/security/vce/radius', method=['POST'])
def get_vce_radius():
    data = request.json['data']
    
    return res

if '__main__' == __name__:
    app_manager = flask_sript.Manager(app)
    app_manager.run()