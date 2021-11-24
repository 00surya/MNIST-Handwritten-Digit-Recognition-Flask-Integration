from keras.models import load_model
import numpy as np
import joblib
import matplotlib.pylab as plt
import pickle

CNN_ = load_model('./services/CNN_digit_rec.h5')
LR_ = joblib.load('./services/lr_digit_rec.pkl')
MNB_ = joblib.load('./services/mnb_digit_rec.pkl')
# KNN_ = joblib.load('./services/knn_digit_rec.pkl')
# KNN_ = joblib.load('./services/knn_digit_rec.pkl')

def CNN(arr):

    arr = arr.reshape(1,28,28,1)
    c_pred = CNN_.predict([arr])
    c_pred = np.argmax(c_pred)
    return c_pred

# def KNN(arr):
#     pred = KNN_.predict([arr])
#     return pred

def lr(arr):
    pred = LR_.predict([arr])
    return pred[0]


def mnb(arr):
    pred = MNB_.predict([arr])
    return pred[0]