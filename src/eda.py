from pathlib import Path
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from .features import NUM

def run(x,y,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for c in x:
        if c in NUM:
            a=x.loc[y==0,c].dropna(); b=x.loc[y==1,c].dropna()
            if len(a) and len(b):
                u,p=mannwhitneyu(b,a); rows.append((c,'Mann-Whitney',p,2*u/(len(a)*len(b))-1))
        else:
            v=x[c].fillna('Unknown'); tab=pd.crosstab(v,y)
            if min(tab.shape)>1:
                chi,p,_,_=chi2_contingency(tab); rows.append((c,'Chi-square',p,(chi/(tab.values.sum()*min(tab.shape[0]-1,tab.shape[1]-1)))**.5))
            pd.crosstab(v,y).assign(total=lambda a:a.sum(axis=1)).to_csv(out/f'{c}_counts.csv')
    r=pd.DataFrame(rows,columns=['feature','test','p_value','effect_size']); r['p_bonferroni']=(r.p_value*len(r)).clip(upper=1); r.to_csv(out/'associations.csv',index=False)
    x[NUM].describe().to_csv(out/'numeric_summary.csv'); x[NUM].corr(method='spearman').to_csv(out/'spearman.csv')
