import pandas as pd
import streamlit as st
from PIL import Image
import pickle

st.write("""
# Xác định tuổi thai bằng trí tuệ nhân tạo
""")

image = Image.open(r"Robot-AI-machine-learning-hero.jpg")
st.image(image, use_column_width=True, caption = 'AI')

model = pickle.load(open('RF.pkl', 'rb'))

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
prediction = model.predict(user_input)
st.subheader('Kết quả')
st.write(f'Tuổi thai: {prediction}')