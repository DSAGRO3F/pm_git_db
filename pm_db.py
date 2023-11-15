import streamlit as st
import traceback
import pandas as pd
import numpy as np
import sklearn
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.dates import date2num
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
from dateutil.parser import parse

# Initialisation page streamlit
st.set_page_config(page_title="Visualisation état capteurs ", layout='wide')

# Set connection
session = requests.Session()

# Définition base url local
# base_url = 'http://localhost:5000/'

# Définition base url Heroku
base_url = 'https://predictive-maintenance-api-6ea7f441053d.herokuapp.com/'

# Construction requête pour récupérer dataframe df
end_point_df = 'df'
url_df = base_url + end_point_df

def get_df(session, url_df):
    try:
        result = session.get(url_df)
        result = result.json()
        print(result)
        return result

    except Exception as e:
        print("An exception occurred:", e)
        traceback.print_exc()
        return {}

data_origine = get_df(session, url_df)
df_origine = pd.DataFrame(data_origine)

# st.info('df_origin: ')
# st.dataframe(df_origine[0:2])


# Construction requête récupération des ID équipements.
end_point_id = 'id'
url_id = base_url + end_point_id
# print(url_id)

def get_id(session, url_id):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36'
        }
        results_id = session.get(url_id, headers = headers)
        retry = Retry(connect=3, backoff_factor=1)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        results_id = results_id.json()
        return results_id

    except Exception as e:
        print("An exception occurred:", e)
        traceback.print_exc()
        return {}


liste_id = get_id(session, url_id)
make_choice = st.sidebar.selectbox('Sélection Id équipement:', liste_id)


# Construction fonction pour récupérer les données relatives à un Id équipement
id = str(make_choice)
end_point_sensors = 'sensors_data/' + id
url_sensors = base_url + end_point_sensors


def load_sensors_data(session, url_sensors):
    try:
        sensors_data = session.get(url_sensors)
        json_sensors_data = sensors_data.json()
        return json_sensors_data

    except Exception as e:
        print('An exception occured:', e)
        traceback.print_exc()
        return {}


data = load_sensors_data(session, url_sensors)
#print(data)
df = pd.DataFrame.from_dict(data)
print('----------')
print('df columns: {}'.format(df.columns))

print('type_1: {}'.format(df['DATE'].dtypes))
print("df['DATE'][0:2]_1: {}".format(df['DATE'][0:2]))
print('-----------')


df['DATE'] = df['DATE'].astype('str')
df['DATE'] = pd.to_datetime(df['DATE'])
print('type_2: {}'.format(df['DATE'].dtypes))
print("df['DATE'][0:2]_2: {}".format(df['DATE'][0:2]))
print('-----------')


df_date = df[['DATE']]
# df_date.style.format({"DATE": lambda t: t.strftime("%d-%m-%Y")})
print('type_3: {}'.format(df_date.dtypes))
print("df_date[0:2]_3: {}".format(df_date[0:2]))
print('-----------')



# Récupération dataset X_test:
end_point_X_test = '/X_test/' + id
url_X_test = base_url + end_point_X_test


def get_X_test(session, url_X_test):
    try:
        X_test_data = session.get(url_X_test)
        json_X_test_data = X_test_data.json()
        return json_X_test_data

    except Exception as e:
        print('An exception occured:', e)
        traceback.print_exc()
        return {}


X_test_data = get_X_test(session, url_X_test)
X_test = pd.DataFrame.from_dict(X_test_data)
print('----------')
print('X_Test columns: {}'.format(X_test.columns))
print('----------')
print("X_test['DATE].dtypes_1: {}".format(X_test['DATE'].dtypes))
print('----------')
print("X_test['DATE][0:3]_1: {}".format(X_test['DATE'][0:3]))
print('----------')


X_test['DATE'] = X_test['DATE'].astype('str')
X_test['DATE'] = pd.to_datetime(X_test['DATE'])
print('----------')
print("X_test['DATE].dtypes_2: {}".format(X_test['DATE'].dtypes))
print('----------')
print("X_test['DATE][0:3]_2: {}".format(X_test['DATE'][0:3]))


# 1. Exploitation data pour visualisation data capteurs
l_sensors = ['S13', 'S15', 'S16', 'S17', 'S18', 'S19', 'S5', 'S8']
df_sensors = df[l_sensors]
df_sensors['DATE'] = df['DATE'].tolist()
print('----------')
print('df_sensors.columns: {}'.format(df_sensors.columns))
print('df_sensors[0:2]: {}'.format(df_sensors['DATE'][0:2]))


# 2. Exploitation data pour visualisation data peak values
l_peak = [(name + '_peak') for name in l_sensors]
# print('l_peak: {}'.format(l_peak))
df_peak = df[l_peak]
# print('peak: {}'.format(df_peak[0:2]))

# 3. Exploitation data rolling
l_roll = [(name + '_roll') for name in l_sensors]
df_roll = df[l_roll]

# print('df_sensors: {}'.format(df_sensors[0:2]))


# Initialisation des onglets streamlit
tab1, tab2, tab3 = st.tabs([":file_folder: Data sensors", ":bar_chart: Sensor graphs", ":bar_chart: Preds & Graph"])
with tab1:
    # Page initialization
    st.title('Data sensors')
    st.info('Grab data sensors per equipment...in process')

    with st.container():
        st.title(':black[1. Data sensors]')
        st.write('Getting data...')
        st.dataframe(df)

with tab2:
    with st.container():
        st.title(':red[2. Data visualisation]')
        st.info('Data sensors graphs: data sensors')
        n_rows = 4
        n_cols = len(l_sensors)//n_rows
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8,10))
        plt.tight_layout(pad=4.0)
        cols = l_sensors
        roll_cols = l_roll

        for i, ax in enumerate(axes.flatten()):
            col = cols[i]
            roll_col = roll_cols[i]
            print('col: {}'.format(col))
            pd_serie = df_sensors[col]
            df_temp = pd.concat([df_sensors['DATE'], pd_serie], axis=1)
            print("df_temp: {}".format(df_temp[0:2]))

            pd_serie_roll = df_roll[roll_col]
            df_temp_roll = pd.concat([df_sensors['DATE'], pd_serie_roll], axis=1)

            # print('type_4: {}'.format(df_temp_2['DATE'].dtypes))
            # df_temp_2['DATE'] = pd.to_datetime(df_temp_2["DATE"], format="%Y-%m-%d")
            # print('type_5: {}'.format(df_temp_2['DATE'].dtypes))

            # df_temp_2_roll['DATE'] = pd.to_datetime(df_temp_2_roll["DATE"], format="%Y-%m-%d")

            # Calcul val.moy., min, max
            avg_val = df_temp[col].mean(axis=0)
            # print('avg_val: {}'.format(avg_val))
            min_val = df_temp[col].min(axis=0)
            max_val = df_temp[col].max(axis=0)

            # Définition du segment des dates sur axe des 'x'
            x_avg_seg = [date2num(df_temp['DATE'][0]), date2num(df_temp['DATE'][len(pd_serie) - 1])]
            # print('x_avg_seg_0: {}'.format(date2num(df_temp['DATE'][0])))

            # Plot constantes + data sensors
            df_temp.plot.line(x='DATE', y=col, ax=ax)

            line_1 = ax.hlines(y=avg_val, xmin=x_avg_seg[0], xmax=x_avg_seg[1], color='r', linestyle='dashed')
            line_2 = ax.hlines(y=min_val, xmin=x_avg_seg[0], xmax=x_avg_seg[1], color='g', linestyle='dashed')
            line_3 = ax.hlines(y=max_val, xmin=x_avg_seg[0], xmax=x_avg_seg[1], color='b', linestyle='-.')
            line_4 = df_temp_roll.plot.line(x='DATE', y=roll_col, ax=ax, color='yellow', linestyle='dashed')

            ax.legend([line_1, line_2, line_3, line_4],['avg', 'min', 'max', 'rolling'], fontsize=4)
            ax.set_title(col, fontsize=6)

            ax.set_ylabel('sensor_data', fontsize=5)
            ax.set_xlabel('DATE', fontsize=3)
            ax.tick_params(axis='x', labelrotation=45, labelsize=5)
            ax.tick_params(axis='y', labelsize=5)
        st.pyplot(fig)



    with st.container():
        st.title(':red[3. Visualisation peak values]')
        st.info('Data sensors graphs: peak values')
        n_rows = 4
        n_cols = len(l_peak)//n_rows
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6,8))
        plt.tight_layout(pad=3.0)
        peak_cols = l_peak

        for i, ax in enumerate(axes.flatten()):
            peak_col = peak_cols[i]
            pd_serie_peak = df_peak[peak_col]
            df_temp_peak = pd.concat([df_sensors['DATE'], pd_serie_peak], axis=1)

            line_1 = df_temp_peak.plot.line(x='DATE', y=peak_col, ax=ax)
            ax.legend([line_1], ['peak_values'], fontsize=4)
            ax.set_title(peak_col, fontsize=6)

            ax.set_ylabel('Peak values', fontsize=5)
            ax.set_xlabel('DATE', fontsize=3)
            ax.tick_params(axis='x', labelrotation=45, labelsize=5)
            ax.tick_params(axis='y', labelsize=5)
        st.pyplot(fig)

with tab3:
    with st.container():
        st.title(':red[4. Predictions data]')
        st.info('Predictions data')
        st.dataframe(X_test)

    with st.container():
        st.title(':red[5. Predictions scatter]')
        st.info('Predictions data graph: display predictions data. For some equipement, '
                'prediction values are above cutoff value of 0.5. In that case maintenance activities must be handled.')

        fig, ax = plt.subplots(figsize=(10,4))
        #X_test['DATE'] = mdates.num2date(mdates.datestr2num(X_test['DATE']))
        sns.scatterplot(data=X_test, x='DATE', y='y_pred', hue='y_pred_cutoff', ax=ax)
        ax.set_title('Distribution valeurs prédites')

        fig.autofmt_xdate()
        ax.set_ylabel('preds', fontsize=5)
        ax.set_xlabel('DATE', fontsize=5)
        st.pyplot(fig)
