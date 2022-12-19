# Libraries
import pandas as pd
import streamlit as st
from fbprophet import Prophet

# Configurations
st.set_page_config( 
  page_title = '2022.exit()',
  page_icon = '☃️',
  layout="wide")


st.markdown("<h1 style='text-align:center;'>🎄🎅🏼 Predict How Good Is Your Next Year ☃️❄️ </h1>", unsafe_allow_html=True)

st.sidebar.header('Rate your years between 1 to 10')

# Take user inputs and create a dataframe
def user_input_features():  
    rate_2018 = st.sidebar.slider('Year 2018', 0,10)
    rate_2019 = st.sidebar.slider('Year 2019', 0,10)
    rate_2020 = st.sidebar.slider('Year 2020', 0,10)
    rate_2021 = st.sidebar.slider('Year 2021', 0,10)
    rate_2022 = st.sidebar.slider('Year 2022', 0,10)

    data = {'year': [2018,2019,2020,2021,2022,2023],
            'rate': [rate_2018,rate_2019,rate_2020,rate_2021,rate_2022,None]}


    features = pd.DataFrame(data)
    features['year'] = pd.to_datetime(features['year'], format='%Y')  # convert int to datetime (year only)
    features['rate'] = features['rate'].astype('int', errors='ignore')  # convert float to int
    features = features.rename(columns={'year': 'ds', 'rate':'y'})  # rename as ds and y for prediction
    
    return features

df = user_input_features()


# Train Model
def train_model(df):
    m = Prophet(interval_width=0.95, daily_seasonality=False)
    model = m.fit(df)

    future = m.make_future_dataframe(periods=1, freq='Y')
    forecast = m.predict(future)
    pred = forecast['yhat'].iat[-1]  # Just predict 2023 rate
    return pred 

result = train_model(df)
result = int(result)


def plot_graph():
    chart_data = df['y']
    return chart_data

chart_data = plot_graph()
   

# Buttons
if st.sidebar.button("Make it Snow!❄️"):  # Easter Egg :3
    st.snow()


if st.button("Predict 2023!"):
    if result > 5:
        st.balloons()
        st.markdown(f"<h1 style='text-align:center;'><font color='green'>{result}</font></h1>",unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;'><font color= 'green'>Congratulations! A good year awaits you.🥳</font></h2>", unsafe_allow_html=True)  
        st.line_chart(chart_data)  
        
    elif result == 5:
        st.markdown(f"<h1 style='text-align:center;'><font color='orange'>{result}</font></h1>",unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;'><font color='orange'>Neither good nor bad. Then it's up to you to make this year a good one!😉</font></h2>", unsafe_allow_html=True)
        st.line_chart(chart_data)    

    elif 0 < result < 5:
        st.markdown(f"<h1 style='text-align:center;'><font color='red'>{result}</font></h1>",unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;'><font color='red'>Which of our years went well so that this year will be good, right? Nevermind..😒</font></h2>", unsafe_allow_html=True)  
        st.line_chart(chart_data) 

    elif result < 0:
        result = 0
        st.markdown(f"<h1 style='text-align:center;'><font color='red'>{result}</font></h1>",unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center;'><font color='red'>My prediction is probably wrong.(I hope..)😔</font></h2>", unsafe_allow_html=True)
        st.line_chart(chart_data) 