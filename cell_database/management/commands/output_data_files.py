from django.core.management.base import BaseCommand
from cell_database.dataset_visualization import *

def output_files(options):
    for dataset in Dataset.objects.all():
        # output dataset to csv
        data = compute_dataset(dataset, field_request_default)
        dataset_name, wet_names, filt_names, filt_colors, filt_pos = get_dataset_labels(dataset)
        output_dataset_to_csv(data, dataset_name, wet_names, filt_names, csv_format_default,
                              options['output_dir'])
        if options['visuals']:
            output_dataset_to_plot(data, dataset_name, wet_names, filt_names, filt_colors, filt_pos,
                                   options['output_dir'])

class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('--output_dir', default='')

        parser.add_argument('--visuals', dest='visuals', action='store_true')
        parser.add_argument('--no-visuals', dest='visuals', action='store_false')
        parser.set_defaults(visuals=False)

    def handle(self, *args, **options):
        output_files(options)
