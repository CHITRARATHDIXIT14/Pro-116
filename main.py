import pandas as pd
import csv
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 

df = pd.read_csv('data.csv')

toefl_score = df['TOEFL'].tolist()
admit_chances = df['Chance'].tolist()
university_rating = df['University'].tolist()

colors = []
for i in admit_chances:
    if i == 1:
        colors.append("green")
    else:
        colors.append("red")

fig = go.Figure(data=go.Scatter(
    x=toefl_score,
    y=university_rating,
    mode='markers',
    marker=dict(color=colors)
))

factors = df[['TOEFL' , 'University']]
admit = df['Chance']

toefl_train , toefl_test , admit_train , admit_test = train_test_split(factors , admit , test_size = 0.25 , random_state = 0)
print(toefl_train[0:10])

sc_x = StandardScaler()

toefl_train = sc_x.fit_transform(toefl_train)
toefl_test = sc_x.transform(toefl_test)

admit_pred = classifier.predict(toefl_test)

print("Accuracy:" , accuracy_score(admit_test, admit_pred))