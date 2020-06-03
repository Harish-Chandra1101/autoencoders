import uuid
from django.db import models
from django.contrib.postgres.fields import JSONField


class CsvData(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4(), editable=False)
    name = models.CharField(max_length=200)
    csv_data = JSONField()
