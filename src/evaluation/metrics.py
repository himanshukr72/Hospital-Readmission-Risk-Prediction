from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,brier_score_loss,confusion_matrix)
def scores(y,p,t=.5):
    z=p>=t
    tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel()
    return dict(accuracy=accuracy_score(y,z),precision=precision_score(y,z,zero_division=0),recall=recall_score(y,z),f1=f1_score(y,z,zero_division=0),roc_auc=roc_auc_score(y,p),pr_auc=average_precision_score(y,p),brier=brier_score_loss(y,p),specificity=tn/max(1,tn+fp),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))
