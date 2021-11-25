from flask import Flask,render_template,redirect,jsonify
from flask.globals import request
import numpy as np
import joblib   


class api:
    
    def l_r(self,arr):
        L_R = joblib.load('lr_digit_rec.pkl')
        pred = L_R.predict([arr])
        
        return pred
    
#     def mnb(self,arr):
#         mnb_ = joblib.load('mnb_digit_rec.pkl')
#         pred = mnb_.predict([arr])

#         return pred[0]

# def l_r(arr):

#     L_R = joblib.load('lr_digit_rec.pkl')
#     pred = L_R.predict([arr])
        
#     return pred



app = Flask(__name__)




@app.route('/')
def hello():
    return render_template("index.html")


@app.route('/digit',methods=['POST'])
def digit():
    if request.method == 'POST':
        arr = request.json['array']
        arr = np.array(arr)
        lr_pred = api().l_r(arr)
#         mnb_pred = api().mnb(arr)
        
        return jsonify(l_r_res=str(lr_pred[0]))

    return render_template("index.html")


if __name__ == '__main__':
    # debug = True
    app.run(debug=True)
