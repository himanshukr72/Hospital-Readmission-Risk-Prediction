import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve
from sklearn.calibration import calibration_curve

def plots(y,preds,out):
    for name in ['roc','pr','calibration']:
        fig,ax=plt.subplots(figsize=(8,6))
        for k,p in preds.items():
            if name=='roc': x,z,_=roc_curve(y,p)
            elif name=='pr': z,x,_=precision_recall_curve(y,p)
            else: z,x=calibration_curve(y,p,n_bins=10,strategy='quantile')
            ax.plot(x,z,label=k)
        if name in ['roc','calibration']: ax.plot([0,1],[0,1],'--',color='gray')
        ax.set(xlabel={'roc':'False positive rate','pr':'Recall','calibration':'Mean predicted probability'}[name],ylabel={'roc':'True positive rate','pr':'Precision','calibration':'Observed fraction'}[name]); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out/f'{name}.png',dpi=140); plt.close(fig)
