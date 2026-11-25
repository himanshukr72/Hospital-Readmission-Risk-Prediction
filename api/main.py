import os
from functools import lru_cache
from fastapi import FastAPI,HTTPException
from src.predict import load,predict
app=FastAPI(title='Diabetic readmission research API')
@lru_cache
def bundle(): return load(os.getenv('MODEL_PATH','models/best.joblib'))
@app.get('/health')
def health():
    try: return {'ready':bool(bundle())}
    except FileNotFoundError: raise HTTPException(503,'Train model first')
@app.get('/schema')
def schema(): return {'required_fields':bundle()['raw_fields']}
@app.post('/predict')
def infer(record:dict):
    try: return predict(bundle(),record)
    except FileNotFoundError: raise HTTPException(503,'Train model first')
    except (ValueError,KeyError,TypeError) as e: raise HTTPException(422,str(e))
