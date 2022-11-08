import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, url_for, request
import pickle

app = Flask(__name__)

def metric(features,jenis,y_test):
    
    if (jenis=='Ada'):
        pickled_model = pickle.load(open('modelAda.pkl', 'rb'))
        y_pred=pickled_model.predict(features)
        return metrics.accuracy_score(y_test.values,y_pred),metrics.precision_score(y_test.values,y_pred,average='weighted'),metrics.recall_score(y_test.values,y_pred,average='weighted')
    elif (jenis=='BC'):
        pickled_model = pickle.load(open('modelBc.pkl', 'rb'))
        y_pred=pickled_model.predict(features)
        return metrics.accuracy_score(y_test.values,y_pred),metrics.precision_score(y_test.values,y_pred,average='weighted'),metrics.recall_score(y_test.values,y_pred,average='weighted')
    elif (jenis=='HGBC'):
        pickled_model = pickle.load(open('modelHgbc.pkl', 'rb'))
        y_pred=pickled_model.predict(features)
        return metrics.accuracy_score(y_test.values,y_pred),metrics.precision_score(y_test.values,y_pred,average='weighted'),metrics.recall_score(y_test.values,y_pred,average='weighted')
    
def classification(features,jenis):
    if (jenis=='Ada'):
        pickled_model = pickle.load(open('modelAdaNS.pkl', 'rb'))
        features['Prediksi']=pickled_model.predict(features)
        acc_train=0.8
        recall_train=0.8
        precicion_train=0.8
        return features,acc_train,recall_train,precicion_train
    elif (jenis=='BC'):
        pickled_model = pickle.load(open('modelBcNS.pkl', 'rb'))
        features['Prediksi']=pickled_model.predict(features)
        acc_train=0.9875
        recall_train=0.9875
        precicion_train=0.9882
        return features,acc_train,recall_train,precicion_train
    elif (jenis=='HGBC'):
        pickled_model = pickle.load(open('modelHgbcNS.pkl', 'rb'))
        features['Prediksi']=pickled_model.predict(features)
        acc_train=0.9875
        recall_train=0.9875
        precicion_train=0.9882
        return features,acc_train,recall_train,precicion_train

    
@app.route('/', methods=['GET','POST'])
def index():
    if request.method=='GET':
        return render_template('index.html')
    elif request.method=='POST':
        fungsi=request.form["fungsi"]
        if fungsi=='performansi':
            jenis= request.form["jenis"]
            csv_file=request.files.get("file")
            df=pd.read_csv(csv_file)
            X_test=df.drop(columns='AQI')
            scaler=StandardScaler()
            X_test=scaler.fit_transform(X_test)
            y_test=df['AQI']


            akurasi,presisi,recall = metric(X_test,jenis,y_test)
            return render_template('Result.html', 
                                akurasi=akurasi,
                                presisi=presisi,
                                recall=recall)
        elif fungsi=='deteksi':
            jenis= request.form["jenis"]
            csv_file=request.files.get("file")
            X_test=pd.read_csv(csv_file)
            Data_pred,acc_train,recall_train,precicion_train=classification(X_test,jenis)
            data=Data_pred.to_numpy()
            #return Data_pred.to_html()
            return render_template('Table.html',
                                data=data,
                                akurasi=acc_train,
                                presisi=precicion_train,
                                recall=recall_train)


    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)