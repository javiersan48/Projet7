# Projet 7 Openclassrooms parcours Data Science : Implémentez un modèle de scoring
## Scenario 
L’entreprise « Prêt à dépenser » nous emploie pour développer un modèle de scoring évaluant la probabilité de défaut de paiement d’un client. Un dashboard doit être fourni pour permettre aux conseillers bancaires de demander l’avis de l’algorithme et d’en expliquer les prédictions.

## Les données
Le jeux de données originaux sont disponibles à cette adresse : https://www.kaggle.com/c/home-credit-default-risk/data   
Le dossier Données Dashboard contient les données pour le dashboard.

## Modélisation
LightGBM_with_threshold.pkl est le modèle de classification retenu pour répondre à la problématique avec son seuil optimal.  
Modélisation.ipynb est le notebook démontrant la préparation des données et la modélisation.

## Déploiement
Streamlit_Local_light.py est le code du tableau de bord déployé. Le lien du dashboard est :  https://p7dash.streamlit.app/
API_Local_light.py est le code de l’API FastAPI transmettant les prédictions. Le déploiement est fait avec Heroku sur la branche API de ce projet.

## Documentation
Note méthodologique explique la démarche utilisée  
Implémentez un modèle de scoring.pp est le powerpoint utilisé pour la soutenance.
