# Generated by Django 2.2.6 on 2019-11-05 22:46

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('WetCellDatabase', '0009_auto_20191104_1642'),
    ]

    operations = [
        migrations.RenameField(
            model_name='composite',
            old_name='proprietary_name',
            new_name='name',
        ),
    ]
