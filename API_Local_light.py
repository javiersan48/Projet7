from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import StreamingResponse
from PIL import Image
from io import BytesIO
import pandas as pd
import pickle
import uvicorn
from fastapi.responses import JSONResponse
import shap
import numpy as np
from fastapi.responses import HTMLResponse

app = FastAPI()
security = HTTPBasic()

# Vérification des informations d'identification
def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = "Openclassroom"
    correct_password = "Jerome_S"
    if (
        credentials.username != correct_username
        or credentials.password != correct_password
    ):
        raise HTTPException(
            status_code=401,
            detail="Identifiants invalides",
            headers={"WWW-Authenticate": "Basic"},
        )
def convert_columns_to_numeric(df):
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns
    
    for column in non_numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    
    return df
# def json_handler(x):
#     if isinstance(x, float) and x != x:
#         return None
#     raise TypeError(f"Type {type(x)} not serializable")



# @app.get("/data_explore")
# def charger_donnees_csv():
#     try:
#         with open(r'P7_Data_Dashboard_explore.csv') as f:
#             reader = csv.DictReader(f)
#             data = list(reader)
#             json_data = json.dumps(data, default=json_handler)
#             return JSONResponse(content=json_data)
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)



@app.get("/data_explore")
def read_explore_csv(credentials: HTTPBasicCredentials = Depends(security)):
    verify_credentials(credentials)
    df = pd.read_csv(r'./Données Dashboard/P7_Data_Dashboard_explore.csv')
    df = df.fillna('') 
    data = df.to_dict(orient="records")
    return data

@app.get("/data_predict")
def read_predict_csv(credentials: HTTPBasicCredentials = Depends(security)):
    verify_credentials(credentials)
    df = pd.read_csv(r'./Données Dashboard/P7_Data_Dashboard_predict.csv')
    df = df.fillna('') 
    data = df.to_dict(orient="records")
    return data





model_path = r'./Données Dashboard/LightGBM_with_threshold.pkl'
with open(model_path, 'rb') as f:
    model_with_threshold = pickle.load(f)
    model = model_with_threshold['model']

df2 = pd.read_csv(r'./Données Dashboard/P7_Data_Dashboard_predict.csv')
df2.drop(columns=['Unnamed: 0'], inplace=True)
df2 = df2.fillna('') 
df2.set_index('SKIDCURR', inplace=True)
df2 = convert_columns_to_numeric(df2)
# Créer un explainer SHAP avec le modèle entraîné
explainer = shap.Explainer(model)

@app.get("/global_shap_values")
async def get_shap_values(credentials: HTTPBasicCredentials = Depends(security)):
    verify_credentials(credentials)
    
    # Calculer les valeurs SHAP pour les données d'entrée
    shap_values = explainer.shap_values(df2)
    shap_values_list = [values.tolist() for values in shap_values]
    return JSONResponse(content={"shap_values": shap_values_list}, media_type="application/json")

def get_local_shap_values(num_client):
    # Obtenir les données utilisateur correspondant au numéro client
    user = df2[df2.index == int(num_client)]
    
    # Calculer les valeurs SHAP pour les données utilisateur
    shap_values = explainer.shap_values(user)
    
    # Obtenir la valeur attendue (expected value)
    expected_value = explainer.expected_value[0]
    
    return shap_values, expected_value

@app.get("/local_shap_values/{num_client}")
async def get_shap_values_by_client(num_client: str, credentials: HTTPBasicCredentials = Depends(security)):
    verify_credentials(credentials)
    
    # Obtenir les valeurs SHAP pour le numéro client spécifié
    shap_values, expected_value = get_local_shap_values(num_client)
    shap_values_list = [values.tolist() for values in shap_values]
    return JSONResponse(content={"shap_values": shap_values_list, "expected_value": expected_value}, media_type="application/json")

def get_probabilities(num_client):
    # Obtenir les données utilisateur correspondant au numéro client
    user = df2[df2.index == int(num_client)]
    
    # Calculer les probabilités de prédiction pour les données utilisateur
    probas_user = model.predict_proba(user)
    
    # Créer un dictionnaire des probabilités arrondies
    probabilities = dict(zip(model.classes_, np.round(probas_user[0], 3)))
    
    return probabilities

@app.get("/probabilities/{num_client}")
async def get_probabilities_by_client(num_client: str, credentials: HTTPBasicCredentials = Depends(security)):
    verify_credentials(credentials)
    
    # Obtenir les probabilités de prédiction pour le numéro client spécifié
    probabilities = get_probabilities(num_client)
    
    return JSONResponse(content={"probabilities": probabilities}, media_type="application/json")

    
if __name__ == '__main__':
   main()
