
from django.db import models

SEXE_CHOICES = [
("", "Sélectionnez le sexe"),
   ("H", "Homme"), 
   ("F", "Femme"),
   ]

raison_pret = [
    ("", "Sélectionnez la raison du prêt"),
    ("développement d'activité", "Développement d'activité"), 
    ("achat d'équipement", "Achat d'équipement"), 
    ("agriculture/Elevages", "Agriculture/Elevages"), 
    ("eduction/Formation", "Eduction/Formation"),
    ("santé/Urgence médicale", "Santé/Urgence médicale"),
    ("amélioration habitat", "Amélioration habitat"),
    ("autres besoins", "Autres besoins"),
    ]

statut_matrimonial_choices = [
    ("", "Sélectionnez le statut matrimonial"),
    ("célibataire", "Célibataire"), 
    ("marié(e)", "Marié(e)"), 
    ("divorcé(e)", "Divorcé(e)"), 
    ("veuf(ve)", "Veuf(ve)"),
    ]
education_choices = [
    ("", "Sélectionnez le niveau d'éducation"),
    ("primaire", "Primaire"), 
    ("secondaire", "Secondaire"), 
    ("etude supérieure", "Etude supérieure"), 
    ("Formation professionnelle", "Formation professionnelle"),
    ]
logement_choices= [
    ("", "Sélectionnez le statut du logement"),
    ("propriétaire", "Propriétaire"), 
    ("locataire", "Locataire"), 
    ("hébergé en famille","Hébergé en famille"), 
    ("autre situation", "Autre situation"),
    ]
activite_choices = [
    ("", "Sélectionnez le type d'activité"),
    ("indépendant/Commerçant", "Indépendant/Commerçant"), 
    ("agriculteur/Eleveur", "Agriculteur/Eleveur"), 
    ("artisan","Artisan"), 
    ("salarié","Salarié"), 
    ('secteur informel', 'Secteur informel'), 
    ('sans activité','Sans activité'),
    ]
historique_choices= [
    ("", "Sélectionnez l'historique"),
    ('excellent(jamais de retard)','Excellent(jamais de retard)'), 
     ('bon(quelques retards mineurs)', 'Bon(quelques retards mineurs)'), 
     ('moyen(retards occasionnels)', 'Moyen(retards occasionnels)'), 
     ('défaillant(retards fréquents)','Défaillant(retards fréquents)'),
     ]
garanties_choices = [
    ("", "Sélectionnez une garantie"),
    ('bien immobilier(maison, terrain)','Bien immobilier(maison, terrain)'), 
    ('véhicule(voiture, moto, camion)', 'Véhicule(voiture, moto, camion)'), 
    ('equipement professionnels', 'Equipement professionnels'), 
    ('bijoux/Objets de valeur', 'Bijoux/Objets de valeur'), 
    ('caution personnels','Caution personnels'), 
    ('aucune garantie matérielle','Aucune garantie matérielle'),
    ]

class AuthGroup(models.Model):
    name = models.CharField(unique=True, max_length=150)

    class Meta:
        managed = False
        db_table = 'auth_group'


class AuthGroupPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)
    permission = models.ForeignKey('AuthPermission', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_group_permissions'
        unique_together = (('group', 'permission'),)


class AuthPermission(models.Model):
    name = models.CharField(max_length=255)
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING)
    codename = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'auth_permission'
        unique_together = (('content_type', 'codename'),)


class AuthUser(models.Model):
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField(blank=True, null=True)
    is_superuser = models.IntegerField()
    username = models.CharField(unique=True, max_length=150)
    first_name = models.CharField(max_length=150)
    last_name = models.CharField(max_length=150)
    email = models.CharField(max_length=254)
    is_staff = models.IntegerField()
    is_active = models.IntegerField()
    date_joined = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'auth_user'


class AuthUserGroups(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_groups'
        unique_together = (('user', 'group'),)


class AuthUserUserPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    permission = models.ForeignKey(AuthPermission, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_user_permissions'
        unique_together = (('user', 'permission'),)


class Client(models.Model):
    id_client = models.AutoField(db_column='id_client', primary_key=True)
    nom_et_prenom = models.CharField(db_column='nom_et_prenom',max_length=200, blank=True, null=True)
    age = models.IntegerField(db_column='age', blank=True, null=True)  # Field name made lowercase.
    sexe = models.CharField(db_column='sexe', max_length=200, choices=SEXE_CHOICES,  blank=True, null=True)  # Field name made lowercase.
    raison_pret = models.CharField(db_column='raison_pret', max_length=200, choices=raison_pret,  blank=True, null=True)
    statut_matrimonial = models.CharField(db_column='statut_matrimonial', choices=statut_matrimonial_choices, max_length=200, blank=True, null=True)  # Field name made lowercase.
    niveau_education = models.CharField(db_column='niveau_education', choices=education_choices, max_length=200, blank=True, null=True)  # Field name made lowercase.
    epargne_disponible = models.IntegerField(db_column='epargne_disponible',blank=True, null=True)
    revenu_mensuel = models.IntegerField(db_column='revenu_mensuel',blank=True, null=True)
    montant_demande = models.IntegerField(db_column='montant_demande',blank=True, null=True)
    anciennete_bancaire = models.IntegerField(db_column='anciennete_bancaire', blank=True, null=True)
    statut_logement = models.CharField(db_column='statut_logement', choices=logement_choices, max_length=200, blank=True, null=True)
    nbre_personne_charge = models.IntegerField(db_column='nbre_personne_charge',blank=True, null=True)
    type_activite = models.CharField(db_column='type_activite', choices=activite_choices, max_length=200, blank=True, null=True)
    historique_remboursement = models.CharField(db_column='historique_remboursement', choices=historique_choices, max_length=200, blank=True, null=True)
    garanties = models.CharField(db_column='garanties', choices=garanties_choices, max_length=200, blank=True, null=True)
    date = models.DateTimeField(db_column='date', auto_now_add=True)
    score_de_risque = models.DecimalField(db_column='score_de_risque', max_digits=12, decimal_places=2)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    niveau = models.CharField(db_column='Niveau', max_length=45, blank=True, null=True)#Field name made lowercase.
    organisation = models.ForeignKey('Organisation', models.DO_NOTHING, db_column='organisation', blank=True, null=True)
    variable1_value = models.CharField(max_length=500, blank=True, null=True)
    variable2_value = models.CharField(max_length=500, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'client'


class DjangoAdminLog(models.Model):
    action_time = models.DateTimeField()
    object_id = models.TextField(blank=True, null=True)
    object_repr = models.CharField(max_length=200)
    action_flag = models.PositiveSmallIntegerField()
    change_message = models.TextField()
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING, blank=True, null=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'django_admin_log'


class DjangoContentType(models.Model):
    app_label = models.CharField(max_length=100)
    model = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'django_content_type'
        unique_together = (('app_label', 'model'),)


class DjangoMigrations(models.Model):
    id = models.BigAutoField(primary_key=True)
    app = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    applied = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_migrations'


class DjangoSession(models.Model):
    session_key = models.CharField(primary_key=True, max_length=40)
    session_data = models.TextField()
    expire_date = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_session'


class Modele(models.Model):
    id_modele = models.AutoField(primary_key=True)
    nom_modele = models.CharField(max_length=500, blank=True, null=True)
    algorithme = models.CharField(max_length=500, blank=True, null=True)
    metrique = models.CharField(max_length=45, blank=True, null=True)
    fichier_modele = models.CharField(max_length=200, blank=True, null=True)
    date_creation = models.DateTimeField(blank=True, null=True)
    organisation = models.ForeignKey('Organisation', models.DO_NOTHING, db_column='organisation', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'modele'


class OptionVariable(models.Model):
    id_option_variable = models.AutoField(primary_key=True)
    organisation = models.ForeignKey('Organisation', models.DO_NOTHING, db_column='organisation', blank=True, null=True)
    nom_variable = models.CharField(max_length=500, blank=True, null=True)
    valeur_option = models.CharField(max_length=500, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'option_variable'


class Organisation(models.Model):
    id_organisation = models.AutoField(primary_key=True)
    nom_org = models.CharField(db_column='Nom_org', max_length=500, blank=True, null=True)  # Field name made lowercase.
    telephone = models.CharField(db_column='Telephone', max_length=500, blank=True, null=True)  # Field name made lowercase.
    email = models.CharField(db_column='Email', max_length=500, blank=True, null=True)  # Field name made lowercase.
    responsable = models.CharField(db_column='Responsable', max_length=500, blank=True, null=True) 
    mot_de_passe = models.CharField(db_column='Mot_de_passe', max_length=500, blank=True, null=True)  # Field name made lowercase.
    variable1_label = models.CharField(max_length=500, blank=True, null=True)
    variable2_label = models.CharField(max_length=500, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'organisation'
