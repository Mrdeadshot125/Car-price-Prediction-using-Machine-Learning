import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("carprice.csv")
print(df)

plt.scatter(df['milage'],df['price'])
plt.xlabel("milage")
plt.ylabel("price")
plt.show()
plt.scatter(df['age'],df['price'])
plt.xlabel("age")
plt.ylabel("price")
plt.show()
X = df[['milage','age']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
print(X_train)
print(X_test)

clf = LinearRegression()
clf.fit(X_train, y_train)

clf.predict(X_test)
print(y_test)

print(clf.score(X_test, y_test))
