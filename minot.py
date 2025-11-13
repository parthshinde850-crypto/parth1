import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
cols = ['age','workclass','fnlwgt','education','education-num','marital-status',
        'occupation','relationship','race','sex','capital-gain','capital-loss',
        'hours-per-week','native-country','income']

data = pd.read_csv(url, names=cols, delimiter=',', skipinitialspace=True)
data = data.replace('?', pd.NA).dropna()

label = LabelEncoder()
for i in data.columns:
    if data[i].dtype == 'object':
        data[i] = label.fit_transform(data[i])

X = data.drop('income', axis=1)
y = data['income']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=0)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

yp = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, yp))
print(confusion_matrix(y_test, yp))
print(classification_report(y_test, yp))