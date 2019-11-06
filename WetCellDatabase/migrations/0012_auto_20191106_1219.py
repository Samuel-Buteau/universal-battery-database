# Generated by Django 2.2.6 on 2019-11-06 16:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('WetCellDatabase', '0011_auto_20191106_1218'),
    ]

    operations = [
        migrations.AddField(
            model_name='drycell',
            name='proprietary',
            field=models.BooleanField(blank=True, default=False),
        ),
        migrations.AlterField(
            model_name='drycell',
            name='name',
            field=models.CharField(blank=True, max_length=300, null=True),
        ),
    ]
