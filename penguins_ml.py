# import pandas as pd
# penguin_df = pd.read_csv('penguins.csv')
# print(penguin_df.head())
# import pandas as pd
# penguin_df = pd.read_csv('penguins.csv')
# penguin_df.dropna(inplace=True)
# output = penguin_df['species']
# features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm','flipper_length_mm', 'body_mass_g', 'sex']]
# features = pd.get_dummies(features)
# print('Here are our output variables')
# print(output.head())
# print('Here are our feature variables')
# print(features.head())

# import pandas as pd
# from sklearn.metrics import accuracy_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# penguin_df = pd.read_csv('penguins.csv')
# penguin_df.dropna(inplace=True)
# output = penguin_df['species']
# features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm','flipper_length_mm', 'body_mass_g', 'sex']]
# features = pd.get_dummies(features)
# output, uniques = pd.factorize(output)
# x_train, x_test, y_train, y_test = train_test_split(
# features, output, test_size=.8)
# rfc = RandomForestClassifier(random_state=15)
# rfc.fit(x_train, y_train)
# y_pred = rfc.predict(x_test)
# score = accuracy_score(y_pred, y_test)
# print('Our accuracy score for this model is {}'.format(score))

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
penguin_df = pd.read_csv('penguins.csv')
penguin_df.dropna(inplace=True)
output = penguin_df['species']
features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm','flipper_length_mm', 'body_mass_g', 'sex']]
features = pd.get_dummies(features)
output, uniques = pd.factorize(output)
# x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=.8)
# rfc = RandomForestClassifier(random_state=15)
# rfc.fit(x_train, y_train)
# y_pred = rfc.predict(x_test)
# score = accuracy_score(y_pred, y_test)
# print('Our accuracy score for this model is {}'.format(score))
# rf_pickle = open('random_forest_penguin.pickle', 'wb')
# pickle.dump(rfc, rf_pickle)
# rf_pickle.close()
# output_pickle = open('output_penguin.pickle', 'wb')
# pickle.dump(uniques, output_pickle)
# output_pickle.close()



x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=.8)
rfc = RandomForestClassifier(random_state=15)
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)
score = accuracy_score(y_pred, y_test)
print('Our accuracy score for this model is {}'.format(score))
rf_pickle = open('random_forest_penguin.pickle', 'wb')
pickle.dump(rfc, rf_pickle)
rf_pickle.close()
output_pickle = open('output_penguin.pickle', 'wb')
pickle.dump(uniques, output_pickle)
output_pickle.close()

fig, ax = plt.subplots()
ax = sns.barplot(x=rfc.feature_importances_, y=features.columns)
plt.title('Which features are the most important for species prediction?')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
fig.savefig('feature_importance.png')
# Now when we rerun pengiuns_ml.py, we should see a file called
# feature_importance.png, which we can call from our Streamlit app. Let's do
# that now! We can use the st.image() function to load an image from our png and
# print it to our penguin app. The following code adds our image to the Streamlit
# app and also improves our explanations around the prediction we made. Because
# of the length of this code block, we will just show the new code from the point
# where we start to predict using the user's data:
# new_prediction = rfc.predict([[bill_length, bill_depth, flipper_length,body_mass, island_biscoe, island_dream,island_torgerson, sex_female, sex_male]])
# prediction_species = unique_penguin_mapping[new_prediction][0]
# st.subheader("Predicting Your Penguin's Species:")
# st.write('We predict your penguin is of the {} species'.format(prediction_species))
# st.write('We used a machine learning (Random Forest) model to ''predict the species, the features used in this prediction '
# ' are ranked by relative importance below.')
# st.image('feature_importance.png')