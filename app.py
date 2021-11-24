from flask import Flask,render_template,redirect,jsonify
from flask.globals import request
from services import api
import numpy as np

app = Flask(__name__)


@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/digit',methods=['POST'])
def digit():
    if request.method == 'POST':
        arr = request.json['array']
        arr = np.array(arr)

        cnn_pred = str(api.CNN(arr))
        # knn_pred = api.KNN(arr)
        lr_pred = str(api.lr(arr))
        mnb_pred = str(api.mnb(arr))
        # return str(pred)
        return jsonify(mnb_res=mnb_pred,l_r_res=lr_pred,cnn_res=cnn_pred)
        # return(str(cnn_pred))
    return render_template("index.html")


if __name__ == '__main__':
    # debug = True
    app.run(debug=True)