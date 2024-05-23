import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Loading models
model_lgbm = pickle.load(open('../../housing-sales-king-county/deploy-flask/models/model_lgbm_tuned.pkl', 'rb'))
model_rf = pickle.load(open('../../housing-sales-king-county/deploy-flask/models/model_rfr_tuned.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index_house.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    def format_currency(value):
    # Format the value as dollar with dot as thousands separator and comma for decimals
        return "${:,.2f}".format(value).replace(',', ';').replace('.', ',').replace(';', '.')

    try:
        values = request.form.getlist('new_house')
        model_choice = request.form.get('model')

        df_to_predict = pd.DataFrame({
            'num_bed': [int(values[0])],
            'num_bath': [float(values[1])],
            'size_house': [int(values[2])],
            'size_lot': [int(values[3])],
            'num_floors': [float(values[4])],
            'is_waterfront': [int(values[5])],
            'condition': [int(values[6])],
            'size_basement': [int(values[7])],
            'year_built': [int(values[8])],
            'renovation_date': [int(values[9])],
            'zip': [int(values[10])],
            'latitude': [float(values[11])],
            'longitude': [float(values[12])],
            'avg_size_neighbor_houses': [int(values[13])],
            'avg_size_neighbor_lot': [int(values[14])]
        })

        if model_choice == 'compare_both_models':
            prediction_rf = model_rf.predict(df_to_predict)
            prediction_lgbm = model_lgbm.predict(df_to_predict)
            return jsonify({'prediction_rf': format_currency(prediction_rf[0]),
                            'prediction_lgbm': format_currency(prediction_lgbm[0])
                            })
        elif model_choice == 'random_forest':
            prediction = model_rf.predict(df_to_predict)
        else:
            prediction = model_lgbm.predict(df_to_predict)

        return jsonify({'prediction': format_currency(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='localhost', port=3000, debug=True)