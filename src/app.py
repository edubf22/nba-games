"""
Basic idea for a Streamlit application to be used for ML model deployment
"""

import numpy as np
import dill as pickle
import streamlit as st

# Loading model 
model = pickle.load(open('src/nba_game_prediction.pkl', 'rb'))

# Loading scaler
scaler = pickle.load(open('src/scaler.pkl', 'rb'))

# Create a function for prediction

def nba_match_prediction(input_data):

    # Input data should contain the same features that we trained the model on
    # Ideally, we would have a dropdown list where uses chooses teams - team stats are stored in the model
    # App would then pair the team stats for selected teams and predict based on that data
    # For now, let's pretend the user can enter the team stats manually

    # Change the feature to a numpy array - if data is not as an array yet
    input_as_array = np.asarray(input_data)

    # Reshape the array to predict for only one instance
    input_reshaped = input_as_array.reshape(1, -1)

    # Scale the input data
    input_scaled = scaler.transform(input_reshaped)

    prediction = model.predict(input_scaled)
    print(prediction)

    if prediction[0] == 0:
        return 'And the visitors win!'
    else:
        return 'And the home team wins!'

def main():
    # Page config
    st.set_page_config(page_title='NBA Predictions', page_icon=':basketball:')

    # Page title
    st.title('NBA Game Outcome Prediction')
    st.subheader('By [Eduardo (Edu) Ferreira](https://github.com/edubf22)')
    st.write('Enter the stats for the visitor and home teams and get a prediction of who will win the game!')
    st.write('At the current stage, the stats need to be entered manually. You can find the stats in the folloing page: [Basketball-Reference](https://www.basketball-reference.com/leagues/NBA_2023.html)')

    # Input fields - get from user
    ORtg_V = st.sidebar.number_input('Visitor Offensive Rating (e.g. 115)')
    DRtg_V = st.sidebar.number_input('Visitor Defensive Rating (e.g. 110)')
    TS_V = st.sidebar.number_input('Visitor True Shooting % (e.g. 0.60)')
    TOV_V = st.sidebar.number_input('Visitor Turnovers per 100 Possessions')
    ORB_V = st.sidebar.number_input('Visitor Offensive Rebound % (e.g. 20)')
    DRB_V = st.sidebar.number_input('Visitor Defensive Rebound % (e.g. 79)')
    ORtg_H = st.sidebar.number_input('Home Offensive Rating (e.g. 115)')
    DRtg_H = st.sidebar.number_input('Home Defensive Rating (e.g. 110)')
    TS_H = st.sidebar.number_input('Home True Shooting % (e.g. 0.60)')
    TOV_H = st.sidebar.number_input('Home Turnovers per 100 Possessions')
    ORB_H = st.sidebar.number_input('Home Offensive Rebound % (e.g. 20)')
    DRB_H = st.sidebar.number_input('Home Defensive Rebound % (e.g. 79)')

    # Prediction block
    outcome = ''

    # Creating a button for prediction
    if st.button('Tell me who wins'):
        outcome = nba_match_prediction([ORtg_V, DRtg_V, TS_V, TOV_V, ORB_V, DRB_V, ORtg_V, DRtg_V, TS_H, TOV_H, ORB_H, DRB_H])

    st.success(outcome)

if __name__ == '__main__':
    main()
