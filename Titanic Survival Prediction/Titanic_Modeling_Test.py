import joblib
import pandas as pd
from sklearn.model_selection import train_test_split


model = joblib.load("titanic_model.pkl")
df = pd.read_csv("Cleaned_Data.csv")

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = joblib.load("titanic_model.pkl")
predictions = model.predict(X_test)

sample['Sex'] = sample['Sex'].map({'male': 0, 'female': 1})
sample['Embarked'] = sample['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

sample = pd.DataFrame([{
    "Pclass": 3,
    "Sex": 0,
    "Age": 22,
    "Fare": 7.25,
    "Embarked": 0,
    "SibSp": 0,
    "Parch": 0,
    "FamilySize": 1,
    "IsAlone": 1,
    "Title_Mr": 1,
    "Title_Mrs": 0,
    "Title_Miss": 0,
    "Title_Rare": 0
}])

pred = model.predict(sample)
print(pred)

sample = pd.DataFrame([{
    "Pclass": 1,
    "Sex": 1,
    "Age": 28,
    "Fare": 100,
    "Embarked": 1,
    "SibSp": 0,
    "Parch": 0,
    "FamilySize": 1,
    "IsAlone": 1,
    "Title_Mr": 0,
    "Title_Mrs": 1,
    "Title_Miss": 0,
    "Title_Rare": 0
}])
pred = model.predict(sample)
print(pred)
