import pandas as pd
from flask import Flask, request, render_template
import pickle

# Creating app
app = Flask(__name__)

# Home page
@app.route("/")
def index():
    result = None
    return render_template("index_house.html", result = result)

# Prediction page
@app.route("/predict", methods = ["POST"])
def predict():
    values = request.form.getlist('new_house')
    df_to_predict = pd.DataFrame({  'num_bed':[values[0]],
                                    'num_bath':[values[1]],
                                    'size_house':[values[2]],
                                    'size_lot':[values[3]],
                                    'num_floors':[values[4]],
                                    'is_waterfront':[values[5]],
                                    'condition':[values[6]],
                                    'size_basement':[values[7]],
                                    'year_built':[values[8]],
                                    'renovation_date':[values[9]],
                                    'zip':[values[10]],
                                    'latitude':[values[11]],
                                    'longitude':[values[12]],
                                    'avg_size_neighbor_houses':[values[13]],
                                    'avg_size_neighbor_lot':[values[14]]
                                })
    
    df_to_predict = df_to_predict.astype(dtype= {'num_bed':'int64',
                                    'num_bath':'float64',
                                    'size_house':'int64',
                                    'size_lot':'int64',
                                    'num_floors':'float64',
                                    'is_waterfront':'int64',
                                    'condition':'int64',
                                    'size_basement':'int64',
                                    'year_built':'int64',
                                    'renovation_date':'int64',
                                    'zip':'int64',
                                    'latitude':'float64',
                                    'longitude':'float64',
                                    'avg_size_neighbor_houses':'int64',
                                    'avg_size_neighbor_lot':'int64'})
    
    model = pickle.load(open('../../housing-sales-king-county/deploy-flask/models/model_lgbm_tuned.pkl', 'rb'))
    print('\ndf: ', df_to_predict)
    result = model.predict(df_to_predict)
    result = "%.2f" % result
    return render_template('index_house.html', result = result)

# Execute application
if __name__ == "__main__":
    app.run(host = 'localhost', port = 3000, debug = True)