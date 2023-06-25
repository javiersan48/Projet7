import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from PIL import Image
from lime import lime_tabular
import shap
import plotly.graph_objects as go
import requests
import json
from streamlit.components import v1 as components
from io import BytesIO
import warnings

logo_image = Image.open("logo_pret_a_depenser.PNG")
st.set_page_config(
    page_title="EMPRUNT - AIDE A LA DECISION",
    page_icon= logo_image,
    layout="wide",
)
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

def convert_columns_to_numeric(df):
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns
    
    for column in non_numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    
    return df

def normalize_score(probability):

    if probability <= 0.11:

        score = 50 + (probability / 0.11) * 50

    else:

        score =  ((probability - 0.11) / (1 - 0.11)) * 50

    return score
    
url = "https://projet7api.streamlit.app/data_explore"
response = requests.get(url, auth=("Openclassroom", "Jerome_S"))

try:
    raw_data = pd.DataFrame(response.json())
    # Further processing of the DataFrame
except ValueError as e:
    print("Error decoding JSON:", e)

data_url = "https://projet7api.streamlit.app/data_predict"
response = requests.get(data_url, auth=("Openclassroom", "Jerome_S"))

try:
    data = pd.DataFrame.from_records(response.json())
    data.set_index('SKIDCURR', inplace=True)

    # Check if 'Unnamed: 0' column exists before dropping
    if 'Unnamed: 0' in data.columns:
        data.drop(columns=['Unnamed: 0'], inplace=True)

    # Further processing of the DataFrame
    data = convert_columns_to_numeric(data)

    # Drop 'Unnamed: 0' column in raw_data
    if 'Unnamed: 0' in raw_data.columns:
        raw_data.drop(columns=['Unnamed: 0'], inplace=True)
    raw_data = raw_data.reset_index()

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(data),
        feature_names=data.columns,
        mode="classification",
    )
except ValueError as e:
    print("Error decoding JSON:", e)


    
def jauge(valeur):
    if valeur < 0.50:
        color = 'red'
    elif valeur < 0.60:
        color = 'orange'
    else:
        color = 'green'
    
    st.plotly_chart(go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(valeur, 3) * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probabilité de remboursement"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color}}  # Regroupement de toutes les propriétés de la jauge sous la clé 'gauge'
    )))
 

def global_explaination():
    url = "https://projet7api.streamlit.app/global_shap_values"
    response = requests.get(url, auth=("Openclassroom", "Jerome_S"))

    shap_values = json.loads(response.content)["shap_values"]
    shap_values = np.array(shap_values)



    # Générer le graphique d'importance globale des features
    shap.summary_plot(shap_values[0], data, plot_type='bar', class_names=["Crédit remboursé", "Défaut de paiement"])

    # Afficher la figure dans Streamlit
    st.pyplot(plt.gcf())

def explication_locale(numéro_client):
    url = f"https://projet7api.streamlit.app/local_shap_values/{numéro_client}"
    response = requests.get(url, auth=("Openclassroom", "Jerome_S"))

    data_temp = json.loads(response.content)
    shap_values = np.array(data_temp["shap_values"])
    expected_value = data_temp["expected_value"]
    
    fig, ax = plt.subplots()
    shap.waterfall_plot(
        shap.Explanation(values=shap_values[0][0], base_values=expected_value, feature_names=data.columns),
        max_display=20,
        show=False
    )
    # Afficher le graphique dans Streamlit
    st.pyplot(fig)

    

def plot_preds_proba(customer_id):
    """
    This functions aims plot income, annuities and credit vizuals
    """
    user_infos = {
        "Income": raw_data[raw_data["SK_ID_CURR"] == customer_id][
            "AMT_INCOME_TOTAL"
        ].values[0],
        "Credit": raw_data[raw_data["SK_ID_CURR"] == customer_id]["AMT_CREDIT"].values[
            0
        ],
        "Annuity": raw_data[raw_data["SK_ID_CURR"] == customer_id][
            "AMT_ANNUITY"
        ].values[0],
    }
    pred_proba_df = pd.DataFrame(
        {"Amount": user_infos.values(), "Operation": user_infos.keys()}
    )
    c = (
        alt.Chart(pred_proba_df)
        .mark_bar()
        .encode(x="Operation", y="Amount", color="Operation")
        .properties(width=330, height=310)
    )
    st.altair_chart(c)

def display_data_exploration():
    st.subheader("Exploration des données")

    feature1 = st.selectbox("Sélectionner la feature 1", raw_data.columns)
    feature2 = st.selectbox("Sélectionner la feature 2", raw_data.columns)

    # Distribution de la feature 1 avec différenciation de couleur selon la classe 'TARGET'
    fig1 = px.histogram(raw_data, x=feature1, color='TARGET', nbins=30)
    st.plotly_chart(fig1)

    # Distribution de la feature 2 avec différenciation de couleur selon la classe 'TARGET'
    fig2 = px.histogram(raw_data, x=feature2, color='TARGET', nbins=30)
    st.plotly_chart(fig2)

    # Coefficient de corrélation entre les 2 features
    correlation = raw_data[[feature1, feature2]].corr().iloc[0, 1]
    correlation_percentage = round(correlation * 100, 2)  # Convertir en pourcentage
    st.subheader("Coefficient de corrélation")
    st.write(f"Corrélation entre {feature1} et {feature2} : {correlation_percentage}%")

    # Nuage de points avec différenciation de couleur selon la classe 'TARGET'
    fig3 = px.scatter(raw_data.sample(n=1000, random_state=42), x=feature1, y=feature2, color='TARGET', opacity=0.3)
    st.plotly_chart(fig3)



    
def main():
    
    with st.sidebar:
        col1, col2, col3 = st.columns(3)
        col2.image(logo_image, use_column_width=True)
        st.markdown("""---""")
        st.markdown(
            """
                        <h4 style='text-align: center; color: black;'> Qui sommes-nous? </h4>
                        """,
            unsafe_allow_html=True,
        )
        st.info("""Nous finançons vos rêves. Voiture, maison, mariage... Nous vous offrons à travers des emprunts bancaires la possibilité de financer vos projets. Nos conseillers seront toujours à l'écoute et vous aideront au mieux dans votre développement.""")

        st.markdown("""---""")
        cover_image = Image.open("PhotoJ.PNG")
        
        st.sidebar.image(cover_image, use_column_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🏠 Présentation du tableau de bord", "📈 Prédictions et analyse", "🗃 Exploration des données", "Etat du Data Drift"]
    )
    tab1.markdown("""---""")
    tab1.subheader("Aide à la décision à l'octroi de crédit")
    tab1.markdown(
        "Cet outil est mis à la disposition de nos collaborateurs afin de les aider dans leur prise de décision quand à l'octroi ou non de crédit. Les informations disponibles dans ce tableau ne contiennent pas d'informations confidentielles. Il est donc possible de les montrer au client dans un acte de transparence. Nous rappelons à nos clients qui ne sont pas présentement admissibles à l'emprunt qu'ils le seront probablement demain et que nous les aideront au mieux dans ce moment difficile de leur vie."
    )
    tab1.markdown("""---""")

    with tab1.markdown("**Project Lifecycle**"):
        col1, col2 = st.columns(2)
        col1.info("**Comment fonctionne l'algorithme**")
        LightGBM_image = Image.open("LightGBM.png")
        
        col1.image(LightGBM_image, use_column_width=True)
        col1.markdown(
            "<span style='font-size: larger;'>L'outil d'aide à la décision utilise un modèle LightGBM. Le modèle prend en entrée des données prétraitées et renvoie une probabilité de remboursement ou non du crédit. Un client qui ne rembourse pas son crédit coûtant beaucoup plus cher que ce que rapporte un client qui le rembourse pour le même montant, le seuil de tolérance a été modifié afin de préserver la rentabilité de l'entreprise et la bonne poursuite de son activité. Une explication des raisons de la décision du modèle est fournie ainsi qu'une explication de la manière de fonctionner de l'algorithme en général. Enfin, afin que les conseillers puissent se faire leur idée des données générales, les aidant ainsi à mieux prendre leur décision, la possibilité d'explorer les données leur a été offerte.</span>",
            unsafe_allow_html=True,
            )


        dreamcredit_image = Image.open("dreamcredit.PNG")
        col2.info("**Réalisez vos rêves**")
        col2.image(dreamcredit_image, use_column_width=True)

    with tab2.subheader("Loan Scoring Model"):
        with st.form(key="myform"):

            user_liste = data.index.tolist()
            user_id_value = st.selectbox('Sélectionnez un numéro client', user_liste)

            submit_button = st.form_submit_button(label="Show")

            if submit_button:
                if isinstance(user_id_value, int) and user_id_value in data.index:
                    st.write("Client choisi : ", user_id_value)
                    col1, col2 = st.columns(2)
            
                    user = data[data.index == int(user_id_value)]

                    url = f"https://projet7api.streamlit.app/probabilities/{user_id_value}"
                    response = requests.get(url, auth=("Openclassroom", "Jerome_S"))

                    probabilities = json.loads(response.content)["probabilities"]

       
                    with col1:
                        st.info("Information client")
               
                     
                        dict_infos = {
                            "Age": int(user["DAYSBIRTH"] / -365),
                            "Sexe": user["CODEGENDER"]
                            .replace([1, 0], ["Femme", "Homme"])
                            .item(),
                  
                            "Années d'emploi": int( user["DAYSEMPLOYED"].values / -365),
                     
                            "Montant du revenu": user["AMTINCOMETOTAL"].item(),
                        }
                        st.write(dict_infos)

                        st.markdown("""---""")
                        st.info("Le prêt demandé")
                        dict_infos = {
     
                            "Montant_credit": user["AMTCREDIT"].item(),
                            "Annuites": user["AMTANNUITY"].item(),
                        }
                        user = data[data.index == int(user_id_value)]
                        st.write(dict_infos)
               
                        st.markdown("""---""")
                        st.info("Probabilité")
                        # threshold = 0.11
                        # prediction_value = probabilities['1.0'] if probabilities['1.0'] > threshold else 0
                        # normalized_value = 1 - ((prediction_value - threshold) / (1 - threshold)) if prediction_value > 0 else 0.5
                        normalized_value = normalize_score(probabilities['1.0'])/100
                       
                        if round(normalized_value * 100, 2) >= 60:
                            st.metric(
                                "Score élevé",
                                value=round(normalized_value * 100, 2),
                                delta=f"{round((normalized_value-0.6)*100,2)}",
                            )

                            st.success(
                                "Le client a de bonnes probabilités de rembourser son crédit", icon="✅"
                            )
                        elif 50 < round(normalized_value * 100, 2) < 60:
                            st.metric(
                                "Score moyen",
                                value=round(normalized_value * 100, 2),
                                delta=f"{round((normalized_value-0.6)*100,2)}",
                            )

                            st.warning(
                                "Le client pourrait avoir des difficultés à rembourser son emprunt",
                                icon="⚠️",
                            )
                        else:
                            st.metric(
                                "Score faible",
                                value=round(normalized_value * 100, 2),
                                delta=f"{round((normalized_value-0.6)*100,2)}",
                            )
                            st.error("Le client ne pourra pas rembourser", icon="🚨")
                    with col2:
                        st.info("Explication de la décision")
                        explication_locale(user_id_value)
                        st.markdown("""---""")
              
                        jauge(normalized_value)
                else:
                    st.error("Veuillez, s'il vous plait, entrer un numéro de client valide.", icon="🚨")

            else:
                st.markdown("""---""")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Comment fonctionne le tableau de bord**")
                    st.info(
                        """
                        1. Sélectionner un **numéro client**.
                        2. Cliquez sur le bouton **Show** .
                        3. Analysez les informations du client et la prédiction du modèle.
                        """,
                    )
                    st.markdown("**Comment fonctionne l'explication du modèle**")
                    st.success(
                        """
                        L'explication du modèle démontre par ordre croissant les variables ayant le plus
                        d'impact sur la décision finale. On y voit donc la contribution de chaque variable.
                        """,
                    )

                with col2:
                    st.markdown("**Explication du modèle**")
                    global_explaination()

    with tab3.subheader("Data Drift Report"):
        st.markdown("""---""")
        
        col1, col2 = st.columns(2)
        sample_data = raw_data

        with col1:
            st.subheader("Exploration des données")

            feature1 = st.selectbox("Sélectionner la feature 1", sample_data.columns)
            feature2 = st.selectbox("Sélectionner la feature 2", sample_data.columns)

            fig1 = go.Figure()
            fig2 = go.Figure()

            for target_value, color in zip(sample_data['TARGET'].unique(), ['red', 'green']):
                fig1.add_trace(go.Histogram(
                    x=sample_data[sample_data['TARGET'] == target_value][feature1],
                    nbinsx=30,
                    name=str(target_value),
                    marker_color=color,
                    opacity=0.7
                ))

                fig2.add_trace(go.Histogram(
                    x=sample_data[sample_data['TARGET'] == target_value][feature2],
                    nbinsx=30,
                    name=str(target_value),
                    marker_color=color,
                    opacity=0.7
                    ))

            fig1.update_layout(
                xaxis_title=feature1,
                yaxis_title='Count',
                barmode='overlay',
                bargap=0.1
                )
            st.plotly_chart(fig1)

            fig2.update_layout(
                xaxis_title=feature2,
                yaxis_title='Count',
                barmode='overlay',
                bargap=0.1
                )
            st.plotly_chart(fig2)

        with col2:
            st.subheader("Analyse bivariée")

            fig3 = go.Figure()
            colors = ['red', 'green']

            for target_value, color in zip(sample_data['TARGET'].unique(), colors):
                df_explore_sample = sample_data[sample_data['TARGET'] == target_value].sample(n=min(99, len(sample_data[sample_data['TARGET'] == target_value])), random_state=42)

                fig3.add_trace(go.Scatter(
                    x=df_explore_sample[feature1],
                    y=df_explore_sample[feature2],
                    mode='markers',
                    name=str(target_value),
                    marker=dict(color=color, opacity=0.3)
                    ))

            fig3.update_layout(
                xaxis_title=feature1,
                yaxis_title=feature2,
                showlegend=True
                )
            st.plotly_chart(fig3)


    with tab4.subheader("Etat du Data Drift"):
        with open("DataDrift.html", "r", encoding="utf-8") as report:
            html_content = report.read()
        components.html(html_content, height=1500)



# if __name__ == "__main__":
    # main()
