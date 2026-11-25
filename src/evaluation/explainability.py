import numpy as np,pandas as pd
from sklearn.inspection import permutation_importance

def explain(pipe,train,test,y,out,tree_pipe):
    sample=test.sample(min(500,len(test)),random_state=42); labels=y.loc[sample.index]
    r=permutation_importance(pipe,sample,labels,scoring='average_precision',n_repeats=3,random_state=42,n_jobs=1)
    pd.DataFrame({'feature':sample.columns,'importance':r.importances_mean,'std':r.importances_std}).sort_values('importance',ascending=False).to_csv(out/'permutation_importance.csv',index=False)
    import xgboost as xgb
    z=tree_pipe.named_steps['encode'].transform(tree_pipe.named_steps['ordinal'].transform(sample)); model=tree_pipe.named_steps['model']; vals=model.get_booster().predict(xgb.DMatrix(z),pred_contribs=True)
    o=tree_pipe.named_steps['ordinal']; num=o.transformers_[0][2]; cats=o.transformers_[1][2]; encoder=tree_pipe.named_steps['encode'].named_transformers_['cat']; names=list(num)+list(encoder.get_feature_names_out(cats));
    pd.DataFrame({'encoded_feature':names,'mean_absolute_shap':np.abs(vals[:,:-1]).mean(axis=0)}).sort_values('mean_absolute_shap',ascending=False).to_csv(out/'xgboost_shap.csv',index=False)
