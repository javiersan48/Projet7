import json
from fastapi.testclient import TestClient
from API_Local_light import app
from starlette.status import HTTP_200_OK

def test_get_probabilities_by_client():
    client = TestClient(app)
    num_client = "100001"  # Numéro client à tester
    credentials = ("Openclassroom", "Jerome_S")  # Informations d'identification correctes

    # Effectuer une requête GET à la route avec le numéro client spécifié
    response = client.get(f"/probabilities/{num_client}", auth=credentials)

    assert response.status_code == HTTP_200_OK  # Vérifier que la réponse a un code 200 (OK)
    assert response.headers["content-type"] == "application/json"  # Vérifier le type de contenu de la réponse

    # Convertir le contenu JSON de la réponse en dictionnaire Python
    content = json.loads(response.content)

    # Vérifier que le dictionnaire contient la clé "probabilities"
    assert "probabilities" in content

    # Vérifier que la valeur associée à la clé "probabilities" est un dictionnaire
    assert isinstance(content["probabilities"], dict)

    # Vérifier le chiffre renvoyé par probabilities['1.0']
    assert "1.0" in content["probabilities"]
    expected_probability = 0.044  # Remplacer par le chiffre attendu pour le numéro client spécifié
    assert round(content["probabilities"]["1.0"], 4) == expected_probability

