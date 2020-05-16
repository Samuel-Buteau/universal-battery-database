from django.core.management.base import BaseCommand
from django.db import transaction

from filename_database.models import *
from cycling.models import *

wanted_cell_ids = [
    83220, 83083, 82012, 82993, 82410, 82311, 82306, 81625, 57706,
]


class Command(BaseCommand):

    def handle(self, *args, **options):
        # remove files that are not:
        # valid
        # non-deprecated
        # cycling
        # files with wanted cell_id
        with transaction.atomic():
            DatabaseFile.objects.filter(valid_metadata = None).delete()
            DatabaseFile.objects.exclude(
                valid_metadata__experiment_type = ExperimentType.objects.get(
                    category = Category.objects.get(name = "cycling"),
                    subcategory = SubCategory.objects.get(
                        name = "neware"
                    )
                )
            ).delete()
            DatabaseFile.objects.filter(deprecated = True).delete()
            DatabaseFile.objects.filter(is_valid = False).delete()
            DatabaseFile.objects.exclude(
                valid_metadata__cell_id__in = wanted_cell_ids
            ).delete()

            # Remove ValidMetadata which is not attached to anything.
            ValidMetadata.objects.filter(databasefile = None).delete()
