import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import os
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
# pip install tensorflowjs
# import tensorflowjs as tfjs




data = pd.read_csv("world_internet_user.csv")

os.system("md save")
# sys.stdout = open("code.tet","w")


print(data.shape)
print(data.dtypes)
print(data.isna().sum())
print(data.columns.values)
print(data.head(10))

print(data[data.columns.values[0]].apply(len).values)

print(data.groupby("Region")["Population"].sum().sort_values(ascending=False))
x = data.groupby("Region")["Population"].sum().sort_values(ascending=False).index
y = data.groupby("Region")["Population"].sum().sort_values(ascending=False).values
plt.figure(figsize=(10,5))
sns.barplot(x, y)
plt.title("groupby Region sum Population")
plt.tight_layout()
plt.savefig("save/"+"groupby Region sum Population"+".png")
plt.show()


print(data.groupby("Region")["Internet Users"].sum().sort_values(ascending=False))
x = data.groupby("Region")["Internet Users"].sum().sort_values(ascending=False).index
y = data.groupby("Region")["Internet Users"].sum().sort_values(ascending=False).values
plt.figure(figsize=(10,5))
sns.barplot(x, y)
plt.title("groupby Region sum Internet_Users")
plt.tight_layout()
plt.savefig("save/"+"groupby Region sum Internet_Users"+".png")
plt.show()


print(data.groupby("Region")["% of Population"].sum().sort_values(ascending=False))
x = data.groupby("Region")["% of Population"].sum().sort_values(ascending=False).index
y = data.groupby("Region")["% of Population"].sum().sort_values(ascending=False).values
plt.figure(figsize=(10,5))
sns.barplot(x, y)
plt.title("groupby Region sum % of Population")
plt.tight_layout()
plt.savefig("save/"+"groupby Region sum % of Population"+".png")
plt.show()


for x in data.columns.values:
    print(x.upper())
    print("_"*30)
    print(data[x].value_counts())
    print("_"*50)
    print(data[x].describe())
    print("_"*100)

data["class"] = 0
print(data.head())

data.loc[data["Population"] >= 5000000, "class"] = 1
data.loc[data["Internet Users"] >= 7000000, "class"] = 1
data.loc[data["% of Population"] >= 50, "class"] = 1

print(data["class"].value_counts())

plt.figure(figsize=(10,5))
# sns.heatmap(data.corr(), annot= True,cmap="hot")
sns.heatmap(data.corr(),center=True)
plt.title("heatmap")
plt.tight_layout()
plt.show()


La = LabelEncoder()

for x in range(2):
    data[data.columns.values[x]] = La.fit_transform(data[data.columns.values[x]])

print(data.head())


x = data.iloc[:,:-1]
y = data.iloc[:,-1]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=44, shuffle =True)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(x_test.shape)

# LogisticRegression
print("LogisticRegression")
Lo = LogisticRegression()
Lo.fit(x_train, y_train)
print(Lo.score(x_train, y_train))
print(Lo.score(x_test, y_test))

print("_"*30)

# DecisionTreeClassifier
print("DecisionTreeClassifier")
Dt = DecisionTreeClassifier(max_depth=5)
Dt.fit(x_train, y_train)
print(Dt.score(x_train, y_train))
print(Dt.score(x_test, y_test))

y_pred = Dt.predict(x_test)
print("_"*30)
print(y_test[:10].values)
print(y_pred[:10])
print("_"*30)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# save model
# import sklearn.externals.joblib as jb
import pickle as pk
pk.dump(Dt , open('Model_Dt.sav','wb'))

model = pk.load(open('Model_Dt.sav','rb'))

print("_"*30)
print("predict_value")
print(model.predict([[1,1,1,1,1]]))
print(model.predict([[1,1,100000,1000000,51]]))
print("_"*30)


# tfjs.converters.save_keras_model(model, 'models')

import sklearn_json as skjson 

skjson.to_json(Dt,"model.json")