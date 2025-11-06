import joblib
import numpy as np
import pandas as pd 
import seaborn as sn
import matplotlib.pyplot as plt
import csv
from django.utils.timezone import now
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import RocCurveDisplay, classification_report, auc, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from django.utils.timezone import now
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from reportlab.lib.pagesizes import A4
from django.contrib.auth.hashers import check_password
import base64
import os
from django.conf import settings
from io import BytesIO
from django.db import models
from django.db.models import Count, Avg, Sum
from django.db.models.functions import TruncDate
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from django.http import JsonResponse
import plotly.graph_objects as go
from django.shortcuts import render, redirect, get_object_or_404
from .forms import CreditRequestForm, OrganisationRegisterForm, OrganisationLoginForm
from .models import Client, Organisation, OptionVariable, Modele
from django.core.paginator import Paginator
import plotly.express as px
import plotly.offline as opy
import plotly.io as pio






def home(request):
    if "organisation_id" not in request.session:
        messages.error(request, "Veuillez vous connecter pour accéder à l'accueil.")
        return redirect("connexion")

    # Récupérer les données depuis la session
    organisation = {
        "nom": request.session.get("organisation_nom", ""),
        "email": request.session.get("organisation_email", ""),
        "telephone": request.session.get("organisation_tel", ""),
        "responsable": request.session.get("organisation_responsable", ""),
    }

    return render(request, 'prediction_defaut/home.html', {"organisation": organisation})

def Accueil(request):
    return render(request, 'prediction_defaut/Accueil.html')

def inscription(request):
    if request.method == "POST":
        form = OrganisationRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Inscription avec succès")
            return redirect("connexion")
    else:
        form = OrganisationRegisterForm()
    return render(request, 'prediction_defaut/inscription.html', {"form": form})

def connexion(request):
    if "organisation_id" in request.session:
        request.session.flush()  # supprime toutes les données de session
        messages.success(request, "Vous avez été déconnecté.")
        return redirect("connexion")

    if request.method == "POST":
        form = OrganisationLoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data["email"]
            mot_de_passe = form.cleaned_data["mot_de_passe"]

            try:
                org = Organisation.objects.get(email=email)
                if mot_de_passe == org.mot_de_passe:
                    # connexion réussie
                    request.session["organisation_id"] = org.id_organisation
                    request.session["organisation_nom"] = org.nom_org
                    request.session["organisation_email"] = org.email
                    request.session["organisation_tel"] = org.telephone
                    request.session["organisation_responsable"] = org.responsable

                    messages.success(request, f"Bienvenue {org.nom_org}")
                    return redirect("home")
                else:
                    messages.error(request, "Mot de passe incorrect.")
            except Organisation.DoesNotExist:
                messages.error(request, "Organisation introuvable.")
    else:
        form = OrganisationLoginForm()

    return render(request, 'prediction_defaut/connexion.html', {"form": form})



# Fonction pour récupérer les variables importantes
# --- Fonction pour récupérer les noms de features ---
def _get_feature_names_from_column_transformer(column_transformer):
    feature_names = []
    for name, transformer, cols in column_transformer.transformers_:
        if transformer == 'drop' or transformer is None:
            continue
        if transformer == 'passthrough':
            if isinstance(cols, (list, tuple, np.ndarray)):
                feature_names.extend(list(cols))
            else:
                feature_names.append(cols)
            continue
        est = transformer
        if hasattr(transformer, 'named_steps'):
            last_name = list(transformer.named_steps.keys())[-1]
            est = transformer.named_steps[last_name]
        try:
            names = est.get_feature_names_out(cols)
            feature_names.extend(list(names))
        except Exception:
            if isinstance(cols, (list, tuple, np.ndarray)):
                feature_names.extend(list(cols))
            else:
                feature_names.append(cols)
    return feature_names

# --- Fonction pour extraire les importances ---
def get_feature_importances_from_pipeline(pipeline, top_n=8):
    try:
        fitted_pre = pipeline.named_steps.get('preprocessor')
        if fitted_pre is None:
            return None
        feature_names = _get_feature_names_from_column_transformer(fitted_pre)
        clf = pipeline.named_steps.get('classifier')
        if clf is None:
            return None

        if hasattr(clf, "feature_importances_"):
            importances = np.array(clf.feature_importances_)
        elif hasattr(clf, "coef_"):
            coef = clf.coef_
            if coef.ndim == 1:
                importances = np.abs(coef)
            else:
                importances = np.mean(np.abs(coef), axis=0)
        else:
            return None

        if len(feature_names) != importances.shape[0]:
            return None

        total = importances.sum()
        perc = (importances / total) * 100 if total > 0 else np.zeros_like(importances)
        feat_imp = list(zip(feature_names, perc))
        feat_imp_sorted = sorted(feat_imp, key=lambda x: x[1], reverse=True)
        return feat_imp_sorted[:top_n]
    except Exception as e:
        return None

    
def modele(request):
    # 🔹 Vérifier organisation connectée
    organisation_id = request.session.get("organisation_id")
    if not organisation_id:
        messages.error(request, "Veuillez vous connecter.")
        return redirect("connexion")

    organisation = get_object_or_404(Organisation, id_organisation=organisation_id)

    # 🔹 Liste des algorithmes pour le template
    algorithmes = [
        {"id": "lr", "name": "Logistic Regression", "desc": "Rapide et interprétable"},
        {"id": "deg", "name": "Decision Tree", "desc": "Arbre de décision - Simple à comprendre"},
        {"id": "rf", "name": "Random Forest", "desc": "Ensemble d'arbres - Robuste et précis"},
        {"id": "svm", "name": "SVM", "desc": "Efficace sur petits datasets"},
    ]

    # 🔹 Récupérer le fichier uploadé pour cette organisation
    fs = FileSystemStorage(location=os.path.join(os.getcwd(), "media"))
    filename = request.session.get(f'uploaded_file_org_{organisation_id}')
    if not filename or not fs.exists(filename):
        messages.error(request, "Aucun fichier trouvé pour votre organisation. Veuillez ré-uploader.")
        return redirect('upload')

    file_path = fs.path(filename)

    try:
        df = _read_file_with_pandas(file_path, filename)
    except Exception as e:
        messages.error(request, f"Impossible de lire le fichier : {e}")
        return redirect('upload')

    # Colonnes numériques et catégorielles
    quant_cols = df.select_dtypes(exclude='object').columns.to_list()
    quant_cols = [c for c in quant_cols if c not in ['id_client', 'defaut']]

    cat_cols = df.select_dtypes(include='object').columns.to_list()

    # Pipelines de preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('numeric', numeric_transformer, quant_cols),
        ('categorical', categorical_transformer, cat_cols)
    ])

    # Train/Test split
    train, test = train_test_split(df, test_size=0.3, random_state=123, stratify=df["defaut"])
    X_train, X_test, Y_train, Y_test = (
        train.drop(['defaut', 'id_client'], axis=1),
        test.drop(['defaut', 'id_client'], axis=1),
        train['defaut'],
        test['defaut']
    )

    results = []
    best_score = -np.inf
    best_model = None
    best_algo = None
    best_name = None

    if request.method == "POST":
        selected_models = request.POST.getlist("modele")
        for mod in selected_models:
            if mod == 'lr':
                model = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('classifier', LogisticRegression(random_state=42, max_iter=2000))])
                algo_name = "Logistic Regression"
            elif mod == 'deg':
                model = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('classifier', DecisionTreeClassifier(random_state=42, max_depth=5))])
                algo_name = "Decision Tree"
            elif mod == 'rf':
                model = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('classifier', RandomForestClassifier(random_state=42))])
                algo_name = "Random Forest"
            elif mod == 'svm':
                model = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('classifier', SVC(probability=True, random_state=42))])
                algo_name = "SVM"
            else:
                continue

            # 🔹 Entraînement
            model.fit(X_train, Y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            # Scores
            pre = round(precision_score(Y_test, preds), 3)*100
            rap = round(recall_score(Y_test, preds), 3)*100
            f1 = round(f1_score(Y_test, preds), 3)*100
            auc = round(roc_auc_score(Y_test, probs), 3)*100 if probs is not None else None

            results.append({
                "algo": mod,
                "name": algo_name,
                "precision": pre,
                "f1": f1,
                "rappel": rap,
                "auc": auc
            })

            # 🔹 Meilleur modèle
            if auc and auc > best_score:
                best_score = auc
                best_model = model
                best_algo = mod
                best_name = algo_name

        if best_model:
            importances = get_feature_importances_from_pipeline(best_model, top_n=8)

            # 🔹 Sauvegarde du modèle dans un dossier dédié à l'organisation
            model_folder = os.path.join(fs.location, "models", str(organisation_id))
            os.makedirs(model_folder, exist_ok=True)
            model_filename = f"best_model_{best_algo}.pkl"
            model_path = os.path.join(model_folder, model_filename)
            joblib.dump(best_model, model_path)

            # 🔹 Enregistrement du chemin dans la session
            org_models_key = f"trained_models_org_{organisation_id}"
            trained_models = request.session.get(org_models_key, {})
            trained_models[best_name] = model_path
            request.session[org_models_key] = trained_models

            # 🔹 Enregistrement en base
            Modele.objects.create(
                nom_modele=best_name,
                algorithme=best_algo,
                metrique="auc",
                fichier_modele=f"models/{organisation_id}/{model_filename}",
                date_creation=now(),
                organisation=organisation
            )

            # 🔹 Sauvegarde des résultats en session pour affichage
            org_key = f"org_{organisation_id}"
            request.session[org_key] = {
                'trained_models': {best_name: model_path},  # chemins des modèles
                'results': results,
                'best_name': best_name,
                'best_algo': best_algo,
                'importances': importances
            }

            messages.success(request, f"Modèle entraîné avec succès : {best_name}")
            return render(request, "prediction_defaut/modele.html", {
                "algorithmes": algorithmes,
                "results": results,
                "best_algo": best_algo,
                "best_name": best_name,
            })

    return render(request, "prediction_defaut/modele.html", {
        "algorithmes": algorithmes
    })

        

           
    


REQUIRED_COLUMNS = {
    "id_client","age","sexe","raison_pret","statut_matrimonial","niveau_education",
    "epargne_disponible","revenu_mensuel","montant_demande","anciennete_bancaire",
    "statut_logement","nbre_personne_charge","type_activite",
    "historique_remboursement","garanties","defaut"
}
MAX_EXTRA_COLS = 2

def _read_file_with_pandas(path, filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".xls", ".xlsx"):
        return pd.read_excel(path)
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError("Format non supporté (utilisez .csv, .xls ou .xlsx).")

def upload(request):
    organisation_id = request.session.get("organisation_id")
    if not organisation_id:
        return redirect("connexion")
    if request.method == "POST":
        uploaded_file = request.FILES.get("file")
        if not uploaded_file:
            messages.error(request, "Aucun fichier reçu.")
            return render(request, 'prediction_defaut/upload.html')

        fs = FileSystemStorage(location=settings.MEDIA_ROOT)  # utilise MEDIA_ROOT par défaut
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)

        # Vérifier que le fichier existe physiquement
        if not fs.exists(filename):
            messages.error(request, "Échec de sauvegarde du fichier sur le serveur.")
            return render(request, 'prediction_defaut/upload.html')

        # (Facultatif) log pour debug
        print("DEBUG: fichier sauvegardé:", file_path)

        try:
            df = _read_file_with_pandas(file_path, filename)
        except Exception as e:
            messages.error(request, f"Erreur lecture fichier : {e}")
            fs.delete(filename)
            return render(request, 'prediction_defaut/upload.html')

        # Validation colonnes
        cols = set(df.columns.astype(str))
        missing = REQUIRED_COLUMNS - cols
        extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]

        if missing:
            messages.error(request, f"Colonnes manquantes : {', '.join(missing)}")
            fs.delete(filename)
            return render(request, 'prediction_defaut/upload.html')

        if len(extra) > MAX_EXTRA_COLS:
            messages.error(request, f"Trop de colonnes supplémentaires ({len(extra)} > {MAX_EXTRA_COLS}).")
            fs.delete(filename)
            return render(request, 'prediction_defaut/upload.html')
        else: 
            # Exemple : récupération des noms de colonnes importantes
            variable1_label = extra[0] if len(extra) >= 1 else None  # ou une logique adaptée
            variable2_label = extra[1] if len(extra) == 2 else None

            # Sauvegarde en session
            request.session['variable1_label'] = variable1_label
            request.session['variable2_label'] = variable2_label

        # Tout OK -> sauver filename en session et rediriger vers resume
        request.session[f'uploaded_file_org_{organisation_id}'] = filename
        messages.success(request, f"Fichier '{uploaded_file.name}' importé avec succès.")

    return render(request, 'prediction_defaut/upload.html')


def resume(request):
    # 🔹 Récupérer l'organisation connectée
    organisation_id = request.session.get("organisation_id")
    if not organisation_id:
        messages.error(request, "Veuillez vous connecter.")
        return redirect("connexion")

    # 🔹 Récupérer le fichier uploadé pour cette organisation
    fs = FileSystemStorage(location=os.path.join(os.getcwd(), "media"))
    filename = request.session.get(f'uploaded_file_org_{organisation_id}')
    if not filename or not fs.exists(filename):
        messages.error(request, "Aucun fichier trouvé pour votre organisation. Veuillez ré-uploader.")
        return redirect('upload')

    file_path = fs.path(filename)

    try:
        df = _read_file_with_pandas(file_path, filename)
    except Exception as e:
        messages.error(request, f"Impossible de lire le fichier : {e}")
        return redirect('upload')

    # 🔹 Calcul des résumés statistiques
    quant_cols = df.select_dtypes(exclude='object').columns.to_list()
    quant_cols = [c for c in quant_cols if c != 'id_client']

    cat_cols = df.select_dtypes(include='object').columns.to_list()

    nombre_client = df.shape[0]
    age_moyen = round(df['age'].mean(), 1)
    revenu_moyen = round(df['revenu_mensuel'].mean(), 2)
    montant_moyen = round(df['montant_demande'].mean(), 2)
    score_moyen = round(df['defaut'].mean(), 2)
    taux_defaut = score_moyen * 100

    # 🔹 Graphiques univariés
    univarie = []
    for col in quant_cols:
        fig, ax = plt.subplots(figsize=(4,2))
        df[col].hist(ax=ax)
        ax.set_title(f"{col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Effectifs")
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        univarie.append(base64.b64encode(buf.read()).decode('utf-8'))

    for col in cat_cols:
        fig, ax = plt.subplots(figsize=(6,4))
        df[col].value_counts().plot(kind="bar", ax=ax)
        ax.set_title(f"{col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Effectifs")
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        univarie.append(base64.b64encode(buf.read()).decode('utf-8'))

    # 🔹 Graphiques crosstab vs défaut
    charts = []
    for col in cat_cols:
        fig, ax = plt.subplots(figsize=(6,4))
        pd.crosstab(df[col], df["defaut"]).plot(kind="bar", ax=ax)
        ax.set_title(f"{col} vs défaut")
        ax.set_xlabel(col)
        ax.set_ylabel("Effectifs")
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        charts.append(base64.b64encode(buf.read()).decode('utf-8'))

    return render(request, 'prediction_defaut/resume.html', {
        "univarie": univarie,
        "charts": charts,
        "filename": filename,
        "nombre_client": nombre_client,
        "age_moyen": age_moyen,
        "revenu_moyen": revenu_moyen,
        "montant_moyen": montant_moyen,
        "taux_defaut": taux_defaut
    })
def resultat(request):
    # 🔹 Vérifier organisation connectée
    organisation_id = request.session.get("organisation_id")
    if not organisation_id:
        messages.error(request, "Veuillez vous connecter.")
        return redirect("connexion")

    # 🔹 Récupérer les résultats pour cette organisation depuis la session
    org_data = request.session.get(f"org_{organisation_id}", {})

    results = org_data.get("results", [])
    best_name = org_data.get("best_name", None)
    best_algo = org_data.get("best_algo", None)
    importances = org_data.get("importances", None)

    return render(request, "prediction_defaut/resultat.html", {
        "results": results,
        "best_name": best_name,
        "best_algo": best_algo,
        "importances": importances
    })

from collections import defaultdict

def get_top_factors(model, preprocessor, top_n=10):
    """Extrait l'importance des variables depuis un pipeline"""
    try:
        if hasattr(model.named_steps['classifier'], "feature_importances_"):
            importances = model.named_steps['classifier'].feature_importances_
            features = preprocessor.get_feature_names_out()
        elif hasattr(model.named_steps['classifier'], "coef_"):
            importances = abs(model.named_steps['classifier'].coef_[0])
            features = preprocessor.get_feature_names_out()
        else:
            return []
        factors = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        return [(f, round(imp * 100, 2)) for f, imp in factors[:top_n] if imp > 0.05]
    except Exception:
        return []

def prediction(request):
    prediction = None
    probability = None
    top_factors = []
    pdf_ready = False

    # 🔹 Organisation connectée
    organisation_id = request.session.get("organisation_id")
    if not organisation_id:
        return redirect("connexion")
    organisation = get_object_or_404(Organisation, id_organisation=organisation_id)

    # 🔹 Récupérer les modèles en session pour l'organisation
    org_models_key = f"trained_models_org_{organisation_id}"
    trained_models = request.session.get(org_models_key, {})

    # 🔹 Ajouter le modèle par défaut
    default_model_path = os.path.join(settings.BASE_DIR, "ml_model", "best_model.pkl")
    if os.path.exists(default_model_path):
        trained_models["Modèle par défaut"] = default_model_path

    if not trained_models:
        messages.error(request, "Aucun modèle disponible pour cette organisation. Veuillez en entraîner un.")
        return redirect("modele")

    # 🔹 Formulaire avec choix de modèle
    form = CreditRequestForm(request.POST or None, request=request)

    if request.method == "POST" and "download_pdf" in request.POST:
        # PDF
        result_data = request.session.get(f"last_prediction_org_{organisation_id}")
        if not result_data:
            return HttpResponse("Aucune prédiction enregistrée.", status=400)
        response = HttpResponse(content_type="application/pdf")
        response['Content-Disposition'] = 'attachment; filename="evaluation_client.pdf"'
        generate_pdf(response, organisation.nom_org, result_data)
        return response

    elif request.method == "POST" and form.is_valid():
        cleaned_data = form.cleaned_data
        cols = [
            'age', 'sexe', 'raison_pret', 'statut_matrimonial', 'niveau_education',
            'epargne_disponible', 'revenu_mensuel', 'montant_demande',
            'anciennete_bancaire', 'statut_logement', 'nbre_personne_charge',
            'type_activite', 'historique_remboursement', 'garanties'
        ]

        # 🔹 Charger le modèle sélectionné dans le formulaire
        selected_model_name = cleaned_data.get("modele_choice", "Modèle par défaut")
        model_path = trained_models.get(selected_model_name, trained_models.get("Modèle par défaut"))

        try:
            model = joblib.load(model_path)
        except Exception as e:
            messages.error(request, f"Erreur de chargement du modèle : {e}")
            return redirect("modele")

        # 🔹 Créer le client
        client = Client.objects.create(
            nom_et_prenom = cleaned_data['nom_et_prenom'],
            age = cleaned_data['age'],
            sexe = cleaned_data['sexe'],
            raison_pret = cleaned_data['raison_pret'],
            statut_matrimonial = cleaned_data['statut_matrimonial'],
            niveau_education = cleaned_data['niveau_education'],
            epargne_disponible = cleaned_data['epargne_disponible'],
            revenu_mensuel = cleaned_data['revenu_mensuel'],
            montant_demande = cleaned_data['montant_demande'],
            anciennete_bancaire = cleaned_data['anciennete_bancaire'],
            statut_logement = cleaned_data['statut_logement'],
            nbre_personne_charge = cleaned_data['nbre_personne_charge'],
            type_activite = cleaned_data['type_activite'],
            historique_remboursement = cleaned_data['historique_remboursement'],
            garanties = cleaned_data['garanties'],
            organisation = organisation
        )

        # 🔹 Prédiction
        data2 = pd.DataFrame([cleaned_data])[cols]
        prediction = model.predict(data2)[0]
        try:
            probability = model.predict_proba(data2)[0][1]
        except (IndexError, AttributeError):
            probability = 0.0

        client.score_de_risque = probability
        client.niveau = (
            'Faible' if probability < 0.4 
            else 'Modéré' if probability < 0.5
            else 'Élevé'
        )

        # 🔹 Variables explicatives uniquement si défaut
        if prediction == 1:
            top_factors = get_top_factors(model, model.named_steps['preprocessor'])
        else:
            top_factors = []

        # 🔹 Sauvegarde en session par organisation
        request.session[f"last_prediction_org_{organisation_id}"] = {
            "prediction": int(prediction),
            "probability": round(probability * 100, 2),
            "top_factors": top_factors,
            "client_data": cleaned_data,
        }
        pdf_ready = True
        client.save()

    return render(request, 'prediction_defaut/prediction.html', {
        "form": form,
        "prediction": prediction,
        "probability": probability,
        "top_factors": top_factors,
        "pdf_ready": pdf_ready,
    })

def generate_pdf(response, org_name, result_data):
    c = canvas.Canvas(response, pagesize=A4)
    width, height = A4

    # Entête
    c.setFillColorRGB(0.1, 0.45, 0.2)
    c.rect(0, height-80, width, 80, fill=True, stroke=False)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height-50, f"{org_name} - FICHE D'EVALUATION DU RISQUE")

    # Résultat
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 14)
    y = height - 120
    status = '🚨 Défaut de paiement' if result_data['prediction']==1 else '✅ Client fiable'
    c.drawString(50, y, f"Résultat : {status}")
    y -= 20
    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Probabilité de défaut : {result_data['probability']}%")
    y -= 30

    # Variables explicatives
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Variables explicatives principales :")
    y -= 20
    for f, imp in result_data.get("top_factors", []):
        c.drawString(70, y, f"- {f}: {imp}%")
        y -= 20
    y -= 20

    # Tableau des données client
    data = [["Variable", "Valeur"]]
    for k, v in result_data["client_data"].items():
        data.append([k, str(v)])

    table = Table(data, colWidths=[200, 300])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0B3D91")),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('ALIGN',(0,0),(-1,-1),'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('BACKGROUND',(0,1),(-1,-1),colors.whitesmoke),
        ('GRID',(0,0),(-1,-1),0.5,colors.grey)
    ]))
    table.wrapOn(c, width, height)
    table.drawOn(c, 50, y - len(data)*20)

    c.showPage()
    c.save()

def historique(request):
    # Récupérer l'organisation
    organisation = request.session.get("organisation_id")
    if not organisation:
        return redirect("connexion")  # si non connecté
    
    # Suppression d'un client si POST
    if request.method == "POST":
        client_id = request.POST.get("client_id")
        client = get_object_or_404(Client, id_client=client_id, organisation=organisation)
        client.delete()

    # Recherche
    query = request.GET.get('search', '') 
    clients_qs = Client.objects.filter(organisation=organisation)
    if query:
        clients_qs = clients_qs.filter(
            nom_et_prenom__icontains=query
        ) | clients_qs.filter(
            id_client__icontains=query
        )
    
    # Pagination
    clients_qs = clients_qs.order_by('-date')
    paginator = Paginator(clients_qs, 10) 
    page_number = request.GET.get('page')  # paramètre de la page
    page_obj = paginator.get_page(page_number)

    return render(request, 'prediction_defaut/historique.html', {
        'page_obj': page_obj,
        'query': query
    })


def analyse(request): 
    organisation_id = request.session.get("organisation_id")
    if not organisation_id:
        return redirect("connexion")

    clients = Client.objects.filter(organisation_id=organisation_id)

    total = clients.count()
    stats = clients.aggregate(
        somme=Sum('montant_demande'),
        moyenne=Avg('score_de_risque')
    )
    niveau = clients.filter(niveau='Élevé').count()

    # Clients groupés par niveau de risque
    data = clients.values("niveau").order_by("niveau").annotate(total_count=Count("organisation_id"))

    # Camembert
    niveaux = [item["niveau"] for item in data]
    counts = [item["total_count"] for item in data]

    fig = px.pie(names=niveaux, values=counts)
    fig.update_layout(
        height=350, width=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    graph = opy.plot(fig, auto_open=False, output_type="div")

    # Graphique Tranche d'âge
    df = pd.DataFrame(clients.values("age", "score_de_risque"))
    bins = [19, 29, 39, 49, 59]
    labels = ["20-29", "30-39", "40-49", "50-59"]
    df["Tranche_Age"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)
    grouped = df.groupby("Tranche_Age").agg(
        Score_Moyen=("score_de_risque", "mean"),
        Nb_Clients=("score_de_risque", "count")
    ).reset_index()
    grouped["Score_Moyen"] = grouped["Score_Moyen"] * 100
    melted = grouped.melt(
        id_vars="Tranche_Age",
        value_vars=["Score_Moyen", "Nb_Clients"],
        var_name="Type",
        value_name="Valeur"
    )

    fig1 = px.bar(
        melted, x="Tranche_Age", y="Valeur", color="Type",
        barmode="group",
        color_discrete_map={"Score_Moyen": "#1f7bb4", "Nb_Clients": "#ff7f0e"}
    )
    fig1.update_layout(
        height=300, width=480,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="", yaxis_title="",
        legend_title_text="", autosize=True,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray", griddash="dot")
    fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray", griddash="dot")
    graph_score = fig1.to_html(full_html=False)

    # Graphique Montant vs Score
    df = pd.DataFrame(clients.values("montant_demande", "score_de_risque"))
    df["montant_demande"] = df["montant_demande"] / 1000
    df["score_de_risque"] = df["score_de_risque"] * 100
    df = df.sort_values("montant_demande")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["montant_demande"], y=df["score_de_risque"],
        mode="lines+markers", name="Score du risque (x100)",
        fill="tozeroy",
        line=dict(width=2, color="purple"),
        fillcolor="rgba(128,0,128,0.3)"
    ))
    fig2.update_layout(
        autosize=True, margin=dict(l=40,r=40,t=60,b=60),
        height=300, width=480,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Montant du prêt (en milliers de FCFA)", yaxis_title="Score du risque",
        legend_title_text="", legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )
    fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray", griddash="dot")
    fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray", griddash="dot")
    graph_aire = fig2.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})

    # Graphique Score dans le temps
    df = pd.DataFrame(clients.values("date", "score_de_risque"))
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%m-%d-%Y")
    df["score_de_risque"] = df["score_de_risque"] * 100
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df["date"], y=df["score_de_risque"],
        mode="lines+markers",
        name="Score du risque (x100)",
        line=dict(width=2, color="blue")
    ))
    fig3.update_layout(
        autosize=True, margin=dict(l=40,r=40,t=60,b=60),
        height=300, width=480,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Date", yaxis_title="Score du risque",
        legend_title_text="", legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )
    fig3.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray", griddash="dot")
    fig3.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray", griddash="dot")
    graph_line = fig3.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})

    return render(request, 'prediction_defaut/analyse.html', {
        "graph": graph,
        "graph_score": graph_score,
        "total": total,
        "stats": stats,
        "niveau": niveau,
        "clients": clients,
        "graph_aire": graph_aire,
        "graph_line": graph_line
    })

    
# Create your views here.
