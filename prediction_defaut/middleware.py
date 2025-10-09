from .models import Organisation  # ton modèle utilisateur personnalisé

class CustomAuthMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        user_id = request.session.get('organisation_id')  # ID stocké lors du login
        if user_id:
            try:
                request.user = Organisation.objects.get(id_organisation=user_id)
            except Organisation.DoesNotExist:
                request.user = None
        else:
            request.user = None
        return self.get_response(request)