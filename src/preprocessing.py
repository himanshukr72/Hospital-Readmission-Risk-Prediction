from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from .features import NUM

def build(x):
    num=[c for c in x if c in NUM]; cat=[c for c in x if c not in NUM]
    ordinal=ColumnTransformer([('num',Pipeline([('fill',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('fill',SimpleImputer(strategy='constant',fill_value='Unknown')),('ordinal',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))]),cat)],verbose_feature_names_out=False)
    # Ordinal encoding is used only as an intermediate representation for mixed-type resampling.
    encode=ColumnTransformer([('num','passthrough',list(range(len(num)))),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),list(range(len(num),len(num)+len(cat))))],verbose_feature_names_out=False)
    return ordinal,encode,list(range(len(num),len(num)+len(cat)))
