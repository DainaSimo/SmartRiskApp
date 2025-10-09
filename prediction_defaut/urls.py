from django.urls import path,include
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.Accueil, name='Accueil'),
    path('home/', views.home, name='home'),
    path('inscription/', views.inscription, name='inscription'),
    path('connexion/', views.connexion, name='connexion'),
    path('modele/', views.modele, name='modele'),
    path('upload/', views.upload, name='upload'),
    path('resume/', views.resume, name='resume'),
    path('resultat/', views.resultat, name='resultat'),
    path('prediction/', views.prediction, name='prediction'),
    path('historique/', views.historique, name='historique'),
    path('analyse/', views.analyse, name='analyse')
    
    ]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)