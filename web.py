from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import scipy.stats
import streamlit as st
from PIL import Image

st.write("""
# Xác định tuổi thai bằng trí tuệ nhân tạo
""")

image = Image.open(r"C:\Users\tthlo\Downloads\tuoithai2\Robot-AI-machine-learning-hero.jpg")
st.image(image, use_column_width=True, caption = 'AI')

def computeTuoithai(weeks):
    ngay_lst = []
    for i in weeks:
        if pd.isnull(i):
            ngay_lst.append('NaN')
        else:
            if '/' in i:
                tuan = i[0:2]
                ngay = i[3]
                ngay = int(ngay) + int(tuan) * 7
            else:
                ngay = int(i[0:2])*7
            ngay_lst.append(ngay)
    return(ngay_lst)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

df = pd.read_csv(r"C:\Users\tthlo\Downloads\tuoithai2\param_SA4D_2020.csv")
df = df.dropna(subset=['tuoithai'])
df['tuoithai_ngay'] = computeTuoithai(df['tuoithai'])
df = df[['mabn', 'HC', 'AC', 'FL', 'tuoithai_ngay', 'chandoan']]
df = df.dropna()
df = df[df['chandoan'].str.contains('TTTON', 'IUI')]
df = df[['HC', 'AC', 'FL', 'tuoithai_ngay']].reset_index(drop=True)
id = df[df['tuoithai_ngay'] == 'NaN'].index.tolist()
df = df.drop(df.index[[id]])
df = df[df['tuoithai_ngay'] >= 140]
df = df[df['tuoithai_ngay'] <= 280]
df['tuoithai_ngay'] = df['tuoithai_ngay'].astype(str).astype(float)
df['HC'] = df['HC'].astype(str).astype(float)
df['AC'] = df['AC'].astype(str).astype(float)
df['FL'] = df['FL'].astype(str).astype(float)
df = df.replace(2511.0, 251.0)
df = df[df['FL'] < 100]
df = df[df['FL'] > 10]

x_train, x_test, y_train, y_test = train_test_split(df[['HC', 'AC', 'FL']], 
                                                    df['tuoithai_ngay'], test_size = 0.2, random_state = 28)

def get_user_input():
    HC = st.number_input('Chu vi vòng đầu')
    AC = st.number_input('Chu vi vòng bụng')
    FL = st.number_input('Chieu dai xương đùi')
    
    user_data = {'HC': HC,
                 'AC': AC,
                 'FL': FL
                 }
    
    features = pd.DataFrame(user_data, index = [0])
    return(features)

user_input = get_user_input()

st.subheader('User Input:')
st.write(user_input)
    
model = RandomForestRegressor(n_estimators = 200, random_state = 40, n_jobs = -1, 
                              max_depth = 8, max_features = 'sqrt')
model.fit(x_train, y_train)

model.fit(x_train, y_train)

st.subheader('Model performance:')
st.write(f'MAE: {mean_absolute_error(y_test, model.predict(x_test))}')

prediction = model.predict(user_input)

st.subheader('Regression result:')
st.write(f'Tuoi thai: {prediction}')