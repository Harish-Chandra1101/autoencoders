import logging
import json
import numpy as np
from rest_framework import viewsets
from rest_framework.response import Response
from apps.clustering.auto_encoder.autoencoder import Autoencoder
from apps.clustering.cluster import Cluster
from apps.clustering.utils import read_csv
from apps.clustering.models import CsvData

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ClusteringViewsets(viewsets.ViewSet):
    def cluster(self, request):
        cluster_config = request.data['data']['clustering']
        feature_config = request.data['data']['feature']
        csv_obj = CsvData.objects.get(id=request.data['data']['csv_id'])
        dataset = np.array(json.loads(csv_obj.csv_data))
        auto_enc_obj = Autoencoder(config=feature_config, data=dataset[1:])
        logger.info('Training autoencoders')
        try:
            auto_enc_obj.train_ae()
            features = auto_enc_obj.get_features()
        except Exception as e:
            logger.error('Error in training autoencoder: ', e)
        logger.info('Clustering...')
        try:
            cluster_obj = Cluster(data=features, cluster_config=cluster_config)
            labels = cluster_obj.kmeans()
            return Response({
                'success': True,
                'labels': labels
            })
        except Exception as e:
            raise e


class CsvUploadView(viewsets.ViewSet):
    def upload_csv(self, request):
        _name = request.FILES.keys()
        upload_name = list(_name)[0]
        try:
            file_data = read_csv(request, upload_name)
            file_data_list = file_data.tolist()
            csv_model_obj = CsvData()
            csv_model_obj.name = upload_name
            csv_model_obj.csv_data = json.dumps(file_data_list)
            csv_model_obj.save()
            return Response({
                'success': True,
                'csv_file_id': csv_model_obj.id,
            })
        except Exception as e:
            return Response({
                'success': False,
                'errors': [e],
            })
