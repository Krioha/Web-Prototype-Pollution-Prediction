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
        return features
    elif (jenis=='BC'):
        pickled_model = pickle.load(open('modelBcNS.pkl', 'rb'))
        features['Prediksi']=pickled_model.predict(features)
        return features
    elif (jenis=='HGBC'):
        pickled_model = pickle.load(open('modelHgbcNS.pkl', 'rb'))
        features['Prediksi']=pickled_model.predict(features)
        return features

    
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
            Data_pred=classification(X_test,jenis)
            return Data_pred.to_html(classes="table table-striped")
            #return render_template('Table.html')


    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)