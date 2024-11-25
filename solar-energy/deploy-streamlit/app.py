# Install and import dependencies
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pycaret.regression import *
import matplotlib.pyplot as plt
import seaborn as sns

# Web page tab configuration
st.set_page_config(page_title='Solar Energy Prediction',
                   page_icon='./solar-energy/deploy-streamlit/images/solar-panel-logo.png',
                   layout='wide',
                   initial_sidebar_state='expanded')

st.image('./solar-energy/deploy-streamlit/images/solar-panel-header.jpg')

with st.expander('About the project', expanded=False):
    st.write('The objective of the project is to deploy a predictive model that will predict the power output of horizontal PV panels, supporting the growth of sustainable energy solutions.')
    st.write('\n')
    st.write("Developed by https://www.linkedin.com/in/guilhermegpaschoalinoto/.")
    
# Sidebar with model version and choice of data input type
with st.sidebar:
    st.subheader('Solar Energy Output Predictor')

    database = st.radio('Source of input data:', ('Predictor','CSV'))

    if database == 'CSV':
        st.info("Use the button below to upload the CSV file directly from the GitHub repository.")
        file_git = st.button("Upload file from GitHub repository")
        if file_git:
            Xtest = pd.read_csv('https://raw.githubusercontent.com/guilhermegarcia-ai/ml-regression-models/refs/heads/main/solar-energy/deploy-streamlit/Xtest.csv', sep=',')
            st.session_state['Xtest'] = Xtest

if database == 'Predictor':
    # Header with application description
    st.title('Solar Energy Prediction')
    
    # Main tabs
    tab1, tab2 = st.tabs(["Forms", "Model Explanation"])
    with tab1:
        with st.form("features_form"):
            latitude = st.number_input(
                    'Latitude', min_value=20.0, max_value=60.0, value=25.0, step=0.5, format="%.2f")
            longitude = st.number_input(
                    'Longitude', min_value=-157.0, max_value=-81.0, value=-100.0, step=0.5, format="%.2f")
            altitude = st.number_input(
                    'Altitude', min_value=1, max_value=2000, value=1000, step=1)
            season = st.selectbox(
                    "Which Season are you in?",
                    ('Winter', 'Spring', 'Summer', 'Fall'))
            month = st.number_input(
                    'Number of Month', min_value=1, max_value=12, value=12, step=1)
            hour = st.number_input(
                    'Hour of the day', min_value=1, max_value=24, value=12, step=1)
            ambienttemp = st.slider(
                    "What's the Ambient Temperature? (°F)", min_value=-20.0, max_value=70.0, value=25.0, step=0.5, format="%.2f")
            humidity = st.slider(
                    "What's the Humidity?", min_value=0.0, max_value=100.0, value=50.0, step=0.5, format="%.2f")
            windspeed = st.slider(
                    "What's the Wind's Speed?", min_value=0.0, max_value=50.0, value=25.0, step=0.5, format="%.2f")
            visibility = st.number_input(
                    "What's the visibility?", min_value=0, max_value=10, value=5, step=1)
            pressure = st.slider(
                    "What's the pressure?", min_value=780.0, max_value=1030.0, value=905.0, step=0.5, format="%.2f")
            cloudceiling = st.slider(
                    "What's the Cloud's Ceiling?", min_value=0, max_value=730, value=365, step=1)
            submitted = st.form_submit_button("Predict")

        # Creating a DataFrame with the values entered by user
        user_data = pd.DataFrame({
                                'Latitude': [latitude],
                                'Longitude': [longitude],
                                'Altitude': [altitude],
                                'Month': [month],
                                'Hour': [hour],
                                'Season': [season],
                                'Humidity': [humidity],
                                'AmbientTemp': [ambienttemp],
                                'Wind.Speed': [windspeed],
                                'Visibility': [visibility],
                                'Pressure': [pressure],
                                'Cloud.Ceiling': [cloudceiling]
        })
        
        # Running prediction model with values entered by user
        if submitted:
            mdl_lgbm = load_model('./solar-energy/deploy-streamlit/solar_energy_pycaret_lgbm')
            ypred = mdl_lgbm.predict(user_data)
            st.subheader('Prediction result')
            st.success(':white_check_mark: Prediction runned succesfully.')
            st.info(':memo: To understand the decision-making process of the algorithm, please see the "Model Explanation" tab.')

    with tab2:
        if submitted != True:
            st.warning('To understand the decision-making process of the algorithm, please fill out the form.')
        else:
            # Data provided
            st.subheader('Data provided')
            user_data['PolyPwr'] = ypred

            def color_pred(prediction):
                color = 'lightgreen'
                return f'background-color: {color}'

            st.dataframe(user_data.style.applymap(color_pred, subset=['PolyPwr']), width=1200)
            
            # Feature importance
            st.subheader('Feature importance')

            # Feature importances
            feat_importances = pd.Series(mdl_lgbm.feature_importances_, index=mdl_lgbm.feature_name_)

            # Sort feature importances in descending order
            feat_importances = feat_importances.sort_values(ascending=False)

            # Create a bar plot using Plotly
            fig = px.bar(
                    x=feat_importances.values,
                    y=feat_importances.index,
                    orientation='h',  # Horizontal bar chart
                    labels={'x': 'Importance Value', 'y': 'Feature'},
                    text=np.round(feat_importances.values, 2),
                    title='Feature Importance',
            )
            fig.update_traces(marker_color='cornflowerblue', textposition='outside')
            fig.update_layout(
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    xaxis=dict(showgrid=True),
                    yaxis=dict(showgrid=False),
            )

            # Display the chart in Streamlit
            st.plotly_chart(fig, use_container_width=True)

elif database == 'CSV':
    # Header with application description
    st.title('Solar Energy Prediction')
    if 'Xtest' in st.session_state:
        Xtest = st.session_state['Xtest']
        
        mdl_lgbm = load_model('./solar-energy/deploy-streamlit/solar_energy_pycaret_lgbm')
        ypred = mdl_lgbm.predict(Xtest.drop(columns=['PolyPwr']))
        
        # Raw dataset
        st.subheader('Raw Dataset')
        num_rows_raw = st.slider('Choose how many rows for raw dataframe:',
                                        min_value = 1, 
                                        max_value = Xtest.shape[0], 
                                        step = 10,
                                        value = 5)
        st.dataframe(Xtest.head(num_rows_raw), width=1200)

        # Predictions dataset
        st.subheader('Dataset with predictions')

        Xtest_pred = Xtest.copy()
        Xtest_pred['PolyPwr_PRED'] = ypred
        
        def color_pred(prediction):
                color = 'lightgreen'
                return f'background-color: {color}'

        num_rows_pred = st.slider('Choose how many rows for predictions dataframe:',
                                        min_value = 1,
                                        max_value = Xtest_pred.shape[0],
                                        step = 4,
                                        value = 5)
        
        st.dataframe(Xtest_pred.head(num_rows_pred).style.applymap(color_pred, subset=['PolyPwr_PRED']), width=1200)
        
        st.markdown(f'Total number of rows filtered: {Xtest_pred.head(num_rows_pred).shape[0]}')

        csv = Xtest_pred.head(num_rows_pred).to_csv(sep = ';', decimal = ',', index = False)
        
        st.download_button(label = 'Download dataset (.csv)',
                                data = csv,
                                file_name = 'Solar_Energy_Predictions.csv',
                                mime = 'text/csv')

        st.subheader('Residual plot')

        Xtest_pred["Residuals"] = Xtest_pred["PolyPwr"] - Xtest_pred["PolyPwr_PRED"]

        def plot_residuals(df):
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=df["PolyPwr_PRED"], y=df["Residuals"], color="blue", alpha=0.7)
            plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
            plt.xlabel("Predicted PolyPwr", fontsize=10)
            plt.ylabel("Residuals (Actual - Predicted)", fontsize=10)
            plt.grid(False)
            return plt

        residual_plot = plot_residuals(Xtest_pred)
        st.pyplot(residual_plot)

        # Feature importance
        st.subheader('Feature importance')

        # Feature importances
        feat_importances = pd.Series(mdl_lgbm.feature_importances_, index=mdl_lgbm.feature_name_)

        # Sort feature importances in descending order
        feat_importances = feat_importances.sort_values(ascending=False)

        # Create a bar plot using Plotly
        fig = px.bar(
                    x=feat_importances.values,
                    y=feat_importances.index,
                    orientation='h',  # Horizontal bar chart
                    labels={'x': 'Importance Value', 'y': 'Feature'},
                    text=np.round(feat_importances.values, 2),
                    title='Feature Importance',
        )
        fig.update_traces(marker_color='cornflowerblue', textposition='outside')
        fig.update_layout(
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    xaxis=dict(showgrid=True),
                    yaxis=dict(showgrid=False),
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning('To view this page, please upload a CSV file.')