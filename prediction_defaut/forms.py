from django import forms
from .models import Client, Organisation, Modele


class CreditRequestForm(forms.ModelForm):
    modele_choice = forms.ChoiceField(label="Modèle à utiliser", required=True)

    class Meta:
        model = Client
        exclude = ['id_client', 'date', 'score_de_risque', 'niveau', 'organisation', ' variable1_value', ' variable2_value']

    def __init__(self, *args, **kwargs):
        request = kwargs.pop("request", None)  # 🔹 Récupérer la requête pour accéder à la session
        organisation_id = None
        if request:
            organisation_id = request.session.get("organisation_id")

        super().__init__(*args, **kwargs)

        for field_name, field in self.fields.items():
            field.required = True

        for f in ("organisation", "variable1_value", "variable2_value"):
            if f in self.fields:
                self.fields[f].required = False
        # 🔹 Construire la liste des choix de modèles
        choices = [("default", "Modèle par défaut")]

        if organisation_id and request:
            org_models_key = f"trained_models_org_{organisation_id}"
            trained_models = request.session.get(org_models_key, {})

            # Ajouter les modèles en session (par organisation)
            for name in trained_models.keys():
                if name != "Modèle par défaut":
                    choices.append((name, name))

        self.fields['modele_choice'].choices = choices

            


class OrganisationRegisterForm(forms.ModelForm):
    class Meta:
        model = Organisation
        fields = ["nom_org", "telephone", "email", "responsable", "mot_de_passe"]
        labels = {
            'nom_org': "Nom de l'organisation",
            'email': "Adresse email",
            'telephone': "Numéro de téléphone",
            'responsable': "Responsable",
            'mot_de_passe': "Mot de passe",
        }
        widgets = {
            'nom_org': forms.TextInput(attrs={'class': 'form-input', 'placeholder': ' ', 'required': True}),
            'email': forms.EmailInput(attrs={'class': 'form-input', 'placeholder': ' ', 'required': True}),
            'telephone': forms.TextInput(attrs={'class': 'form-input', 'placeholder': ' ', 'required': True}),
            'responsable': forms.TextInput(attrs={'class': 'form-input', 'placeholder': ' ', 'required': True}),
            'mot_de_passe': forms.PasswordInput(attrs={'class': 'form-input', 'placeholder': ' ', 'required': True}),
        }

class OrganisationLoginForm(forms.Form):
    email = forms.EmailField(label="Email de l'organisation")
    mot_de_passe = forms.CharField(widget=forms.PasswordInput, label="Mot de passe")