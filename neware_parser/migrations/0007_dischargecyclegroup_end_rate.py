# Generated by Django 2.2.6 on 2020-02-24 18:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('neware_parser', '0006_chargecyclegroup_end_rate'),
    ]

    operations = [
        migrations.AddField(
            model_name='dischargecyclegroup',
            name='end_rate',
            field=models.FloatField(default=0.0),
            preserve_default=False,
        ),
    ]
