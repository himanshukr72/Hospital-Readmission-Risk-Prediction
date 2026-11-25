from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

def models():
    return {'AdaBoost':AdaBoostClassifier(n_estimators=60,learning_rate=.5,random_state=42),'GradientBoosting':GradientBoostingClassifier(n_estimators=60,max_depth=2,random_state=42),'XGBoost':XGBClassifier(n_estimators=100,max_depth=4,learning_rate=.08,tree_method='hist',n_jobs=2,random_state=42,eval_metric='logloss')}
