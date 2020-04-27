from django.core.management.base import BaseCommand

from neware_parser.models import *


def create_cycle_tags():
    my_tags = [
        ("rate_maps", i)

        for i in range(7)
    ]
    for tag_name, i in my_tags:
        _ = CycleTag.objects.get_or_create(namespace = tag_name, index = i)


def debug_crop_cycles(crop):
    for _ in range(100):
        print("DEBUG CROP CYCLES!!!!!")
    Cycle.objects.filter(cycle_number__gte = crop).delete()
    CyclingFile.objects.all().update(
        import_time = datetime.datetime(1970, 1, 1))
    CyclingFile.objects.all().update(
        process_step_time = datetime.datetime(1970, 1, 1))
    CyclingFile.objects.all().update(
        process_cycle_time = datetime.datetime(1970, 1, 1))


class Command(BaseCommand):

    def handle(self, *args, **options):
        create_cycle_tags()
        debug_crop_cycles(3000)
