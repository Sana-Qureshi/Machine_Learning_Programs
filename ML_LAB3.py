import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression 
print("*\n*Predicitng if a person would buy a life insurance based on his age using logistic regression**")

df=pd.read_csv('insurance.csv')
print("**Insurance data**")
print(df)

x=df[['age']]
y=df.bought_insurance

from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test=train_test_split(x,y)
model=LogisticRegression()

print("\nLength of train data: ",len(x_train))
print("\nLength of test data: ",len(x_test))

model.fit(x_train, y_train)
plt.scatter(df.age, df.bought_insurance, marker='+', color='red')

print("\nTest data :")
print(x_test)

print("\nPredicted output is: ",model.predict(x_test))
print("\nAccuracy score of model: ",model.score(x_test, y_test))