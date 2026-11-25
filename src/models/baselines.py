from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def models():
    return {'Dummy':DummyClassifier(strategy='prior'),'Logistic':LogisticRegression(max_iter=400),'Tree':DecisionTreeClassifier(max_depth=7,random_state=42),'RandomForest':RandomForestClassifier(n_estimators=80,max_depth=12,n_jobs=2,random_state=42)}
