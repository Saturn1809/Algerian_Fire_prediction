import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask,request,jsonify,render_template

application = Flask(__name__)
app =application

## import ridge regressor and standard scaler pcikle
ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))


@app.route("/",methods=['GET','POST'])
def predict_datapoint():
        if request.method =="POST":
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))
            
            input_data = pd.DataFrame(
                [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]],
                columns=['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region']
)

            new_data_Scaled = standard_scaler.transform(input_data)
            result = ridge_model.predict(new_data_Scaled)
         
            return render_template('home.html',results=f"{result[0]:.2f}")

        else:
            return render_template('home.html', results="Error in prediction. Check inputs.")
           

if __name__ == "__main__":
    app.run(host="0.0.0.0")