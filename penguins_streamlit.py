# import streamlit as st
# import pickle
# rf_pickle = open('random_forest_penguin.pickle', 'rb')
# map_pickle = open('output_penguin.pickle', 'rb')
# rfc = pickle.load(rf_pickle)
# unique_penguin_mapping = pickle.load(map_pickle)
# st.write(rfc)
# st.write(unique_penguin_mapping)
# As with our previous Streamlit apps, we run the following code in the terminal

# import streamlit as st
# import pickle
# rf_pickle = open('random_forest_penguin.pickle', 'rb')
# map_pickle = open('output_penguin.pickle', 'rb')
# rfc = pickle.load(rf_pickle)
# unique_penguin_mapping = pickle.load(map_pickle)
# rf_pickle.close()
# map_pickle.close()
# island = st.selectbox('Penguin Island', options=['Biscoe', 'Dream', 'Torgerson'])
# sex = st.selectbox('Sex', options=['Female', 'Male'])
# bill_length = st.number_input('Bill Length (mm)', min_value=0)
# bill_depth = st.number_input('Bill Depth (mm)', min_value=0)
# flipper_length = st.number_input('Flipper Length (mm)', min_value=0)
# body_mass = st.number_input('Body Mass (g)', min_value=0)
# st.write('the user inputs are {}'.format([island, sex, bill_length,bill_depth, flipper_length, body_mass]))


# import streamlit as st
# import pickle
# st.title('Penguin Classifier')
# st.write("This app uses 6 inputs to predict the species of penguin using""a model built on the Palmer's Penguin's dataset. Use the form below"" to get started!")
# rf_pickle = open('random_forest_penguin.pickle', 'rb')
# map_pickle = open('output_penguin.pickle', 'rb')
# rfc = pickle.load(rf_pickle)
# unique_penguin_mapping = pickle.load(map_pickle)
# rf_pickle.close()
# map_pickle.close()
# island = st.selectbox('Penguin Island', options=['Biscoe', 'Dream', 'Torgerson'])
# sex = st.selectbox('Sex', options=['Female', 'Male'])
# bill_length = st.number_input('Bill Length (mm)', min_value=0)
# bill_depth = st.number_input('Bill Depth (mm)', min_value=0)
# flipper_length = st.number_input('Flipper Length (mm)', min_value=0)
# body_mass = st.number_input('Body Mass (g)', min_value=0)
# island_biscoe, island_dream, island_torgerson = 0, 0, 0
# if island == 'Biscoe':
#     island_biscoe = 1
# elif island == 'Dream':
#     island_dream = 1
# elif island == 'Torgerson':
#     island_torgerson = 1
# sex_female, sex_male = 0, 0
# if sex == 'Female':
#     sex_female = 1
# elif sex == 'Male':
#     sex_male = 1

# new_prediction = rfc.predict([[bill_length, bill_depth, flipper_length,body_mass, island_biscoe, island_dream,island_torgerson, sex_female,sex_male]])
# prediction_species = unique_penguin_mapping[new_prediction][0]
# st.write('We predict your penguin is of the {}species'.format(prediction_species))


import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
st.title('Penguin Classifier')
st.write("This app uses 6 inputs to predict the species of penguin using""a model built on the Palmer's Penguin's dataset. Use the form below"" to get started!")

# password_guess = st.text_input('What is the Password?')
# if password_guess != st.secrets["password"]:
#     st.stop()

penguin_file = st.file_uploader('Upload your own penguin data')
# This first section imports the libraries that we need, adds the title – as we have
# used before, and adds the file_uploader() function. What happens, however,
# when the user has yet to upload a file? We can set the default to load our random
# forest model if there is no penguin file, as shown in the next section of code:
if penguin_file is None:
    rf_pickle = open('random_forest_penguin.pickle', 'rb')
    map_pickle = open('output_penguin.pickle', 'rb')
    rfc = pickle.load(rf_pickle)
    unique_penguin_mapping = pickle.load(map_pickle)
    rf_pickle.close()
    map_pickle.close()
# The next problem we need to solve is how to take in the user's data, clean it, and
# train a model based on it. Luckily, we can reuse the model training code that we
# have already created and put it within our else statement in the next code block:
else:
    penguin_df = pd.read_csv(penguin_file)
    penguin_df = penguin_df.dropna()
    output = penguin_df['species']
    features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm','flipper_length_mm', 'body_mass_g', 'sex']]
    features = pd.get_dummies(features)
    output, unique_penguin_mapping = pd.factorize(output)
    x_train, x_test, y_train, y_test = train_test_split(
    features, output, test_size=.8)
    rfc = RandomForestClassifier(random_state=15)
    rfc.fit(x_train, y_train)
    y_pred = rfc.predict(x_test)
    score = round(accuracy_score(y_pred, y_test), 2)
    st.write('We trained a Random Forest model on these data,'' it has a score of {}! Use the ''inputs below to try out the model.'.format(score))

with st.form('user_inputs'):
    island = st.selectbox('Penguin Island', options=['Biscoe', 'Dream', 'Torgerson'])
    sex = st.selectbox('Sex', options=['Female', 'Male'])
    bill_length = st.number_input('Bill Length (mm)', min_value=0)
    bill_depth = st.number_input('Bill Depth (mm)', min_value=0)
    flipper_length = st.number_input('Flipper Length (mm)', min_value=0)
    body_mass = st.number_input('Body Mass (g)', min_value=0)
    st.form_submit_button()

island_biscoe, island_dream, island_torgerson = 0, 0, 0
if island == 'Biscoe':
    island_biscoe = 1
elif island == 'Dream':
    island_dream = 1
elif island == 'Torgerson':
    island_torgerson = 1
sex_female, sex_male = 0, 0
if sex == 'Female':
    sex_female = 1
elif sex == 'Male':
    sex_male = 1

new_prediction = rfc.predict([[bill_length, bill_depth, flipper_length,body_mass, island_biscoe, island_dream,island_torgerson, sex_female,sex_male]])
prediction_species = unique_penguin_mapping[new_prediction][0]
#st.write('We predict your penguin is of the {} species'.format(prediction_species))

st.subheader("Predicting Your Penguin's Species:")
st.write('We predict your penguin is of the {} species'.format(prediction_species))
st.write('We used a machine learning (Random Forest) model to ''predict the species, the features used in this prediction '' are ranked by relative importance below.')
st.image('feature_importance.png')






