import base64
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model using joblib (correct file path)
pipe = joblib.load('IPL-Winner-Predictor-main/pipe.pkl')

# Function to load image as base64
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load background image
img = get_img_as_base64("IPL-Winner-Predictor-main/background.jpg")

# Set background image and styling in the app
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img}");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}
[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}
[data-testid="stHeader"], [data-testid="stToolbar"] {{
background: rgba(0,0,0,0);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# App Title
st.markdown("""
    <h1 style='text-align: center; color: orange;'>IPL Victory Predictor</h1>
""", unsafe_allow_html=True)

# Adding a logo (Optional)
st.markdown("<h3 style='text-align: center; color: orange;'>Predict your team’s winning probability!</h3>", unsafe_allow_html=True)

# Dropdowns for selecting teams
st.markdown("<h3 style='color: orange;'>Select Teams:</h3>", unsafe_allow_html=True)

teams = ['--- select ---', 'Sunrisers Hyderabad', 'Mumbai Indians', 'Kolkata Knight Riders', 
         'Royal Challengers Bangalore', 'Kings XI Punjab', 'Chennai Super Kings', 
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Bangalore', 'Hyderabad', 'Kolkata', 'Mumbai', 'Visakhapatnam', 'Indore', 'Durban', 
          'Chandigarh', 'Delhi', 'Dharamsala', 'Ahmedabad', 'Chennai', 'Ranchi', 'Nagpur', 
          'Mohali', 'Pune', 'Bengaluru', 'Jaipur', 'Port Elizabeth', 'Centurion', 'Raipur', 
          'Sharjah', 'Cuttack', 'Johannesburg', 'Cape Town', 'East London', 'Abu Dhabi', 
          'Kimberley', 'Bloemfontein']

col1, col2 = st.columns(2)

with col1:
   batting_team = st.selectbox('Select Batting Team', teams)

with col2:
    if batting_team == '--- select ---':
        bowling_team = st.selectbox('Select Bowling Team', teams)
    else:
        filtered_teams = [team for team in teams if team != batting_team]
        bowling_team = st.selectbox('Select Bowling Team', filtered_teams)

# Venue selection
seleted_city = st.selectbox('Select Venue', cities)

# Numeric inputs with slider and input boxes for target, score, overs, and wickets
st.markdown("<h3 style='color: orange;'>Match Details:</h3>", unsafe_allow_html=True)
target = st.slider('Target', min_value=50, max_value=300, step=1)

col1, col2, col3 = st.columns(3)

# Change the label font color to orange
with col1:
    st.markdown("<h4 style='color: orange;'>Score</h4>", unsafe_allow_html=True)
    score = st.number_input(' ', min_value=0, max_value=target, step=1)
with col2:
    st.markdown("<h4 style='color: orange;'>Overs Completed</h4>", unsafe_allow_html=True)
    overs = st.slider(' ', min_value=0.0, max_value=20.0, step=0.1)
with col3:
    st.markdown("<h4 style='color: orange;'>Wickets Down</h4>", unsafe_allow_html=True)
    wickets = st.number_input(' ', min_value=0, max_value=10, step=1)

# Predict button with style and action
if st.button('Predict Winning Probability'):
    try:
        # Calculate inputs for the model
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets_remaining = 10 - wickets
        crr = score / overs if overs > 0 else 0
        rrr = runs_left / (balls_left / 6) if balls_left > 0 else 0

        # Prepare input data for model
        input_data = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [seleted_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets_remaining': [wickets_remaining],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Model prediction
        result = pipe.predict_proba(input_data)
        loss = result[0][0]
        win = result[0][1]

        # Display results with a progress bar
        st.markdown("<h3 style='color: orange;'>Winning Probability</h3>", unsafe_allow_html=True)
        
        st.markdown(f"<h4 style='color: orange;'>{batting_team} Winning Probability: {round(win * 100)}%</h4>", unsafe_allow_html=True)
        st.progress(int(win * 100))

        st.markdown(f"<h4 style='color: orange;'>{bowling_team} Winning Probability: {round(loss * 100)}%</h4>", unsafe_allow_html=True)
        st.progress(int(loss * 100))

    except Exception as e:
        st.error("An error occurred. Please check your inputs.")
        st.write(e)