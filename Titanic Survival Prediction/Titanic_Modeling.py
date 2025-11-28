import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix



df = pd.read_csv("Cleaned_Data.csv")
print(df.shape)
print(df['Survived'].value_counts())
print(df.dtypes)


X = df.drop(columns=['Survived','Name','Ticket','PassengerId'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object','bool','category']).columns.tolist()

num_cols = [c for c in num_cols if c != 'Survived']

num_pipe = Pipeline([
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

pipe = Pipeline([('pre', preprocessor),
                 ('clf', LogisticRegression(max_iter=1000))])

scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
print("Logistic CV accuracy:", scores.mean())

pipe_dt = Pipeline([('pre', preprocessor),
                    ('clf', DecisionTreeClassifier(random_state=42))])
print("DecisionTree CV accuracy:", cross_val_score(pipe_dt, X_train, y_train, cv=5).mean())


pipe_rf = Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier(random_state=42))])
print("RF CV:", cross_val_score(pipe_rf, X_train, y_train, cv=5, scoring='roc_auc').mean())


pipe_rf.fit(X_train, y_train)
y_pred = pipe_rf.predict(X_test)
y_proba = pipe_rf.predict_proba(X_test)[:,1]

print("Accuracy", accuracy_score(y_test, y_pred))
print("Precision", precision_score(y_test, y_pred))
print("Recall", recall_score(y_test, y_pred))
print("F1", f1_score(y_test, y_pred))
print("ROC AUC", roc_auc_score(y_test, y_proba))
print("Confusion matrix:\\n", confusion_matrix(y_test, y_pred))

param_grid = {
  'clf__n_estimators':[100,200],
  'clf__max_depth':[None,5,10]
}
pipe_rf = Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier(random_state=42))])
gs = GridSearchCV(pipe_rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
gs.fit(X_train, y_train)
print("Best params:", gs.best_params_)
print("Best CV score:", gs.best_score_)


best_model = gs.best_estimator_
joblib.dump(best_model, "titanic_model.pkl")


# ROC
RocCurveDisplay.from_estimator(best_model, X_test, y_test)
plt.show()

# Confusion matrix
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
plt.show()
