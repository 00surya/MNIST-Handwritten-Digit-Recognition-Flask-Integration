from flask import Flask,render_template,redirect,jsonify
from flask.globals import request
# from services import api
import numpy as np
import joblib
import pickle
from keras.models import load_model

# CNN_ = load_model('./services/CNN_digit_rec.h5')




# CNN_ = load_model('./services/CNN_digit_rec.h5')
# MNB_ = joblib.load('./services/mnb_digit_rec.pkl')

class api:
    
    def l_r(self,arr):
        L_R = joblib.load('lr_digit_rec.pkl')
        pred = L_R.predict([arr])
        
        return pred
    
    # def CNN(self,arr):
    #     arr = arr.reshape(1,28,28,1)
    #     c_pred = CNN_.predict([arr])
    #     c_pred = np.argmax(c_pred)
   
    #     return c_pred

    # def MNB(self,arr):
    #     pred = MNB_.predict([arr])
    #     return pred[0]


app = Flask(__name__)




@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/digit',methods=['POST'])
def digit():
    if request.method == 'POST':
        arr = request.json['array']
        arr = np.array(arr)

#         cnn_pred = api().CNN(arr)
        # knn_pred = api.KNN(arr)
        lr_pred = api().l_r(arr)
#         mnb_pred = api().MNB(arr)
        # print(str(lr_pred))
#         return jsonify(mnb_res=mnb_pred,l_r_res=lr_pred,cnn_res=cnn_pred)
        return jsonify(l_r_res=str(lr_pred[0]))
        # return(str(cnn_pred))
    return render_template("index.html")


if __name__ == '__main__':
    # debug = True
    app.run(debug=True)
