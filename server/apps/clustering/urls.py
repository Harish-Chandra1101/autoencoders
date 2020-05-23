from apps.clustering.views import AutoEncoderViewset
from django.urls import path

urlpatterns = [
    path('cluster/', AutoEncoderViewset.as_view(actions={
        'post': 'clusters',
    }))
]
