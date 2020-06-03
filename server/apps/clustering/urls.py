from apps.clustering.viewsets.clustering_views import (
    ClusteringViewsets,
    CsvUploadView,
)
from django.urls import path

urlpatterns = [
    path('cluster/', ClusteringViewsets.as_view(actions={
        'post': 'cluster',
    })),
    path('upload_dataset/', CsvUploadView.as_view(actions={
        'post': 'upload_csv',
    }))
]
