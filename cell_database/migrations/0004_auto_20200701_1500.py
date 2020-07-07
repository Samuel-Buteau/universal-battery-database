# Generated by Django 2.2.11 on 2020-07-01 18:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('cell_database', '0003_ratiocomponent_overridden_component_type'),
    ]

    operations = [
        migrations.AlterField(
            model_name='ratiocomponent',
            name='overridden_component_type',
            field=models.CharField(blank=True, choices=[('sa', 'salt'), ('ad', 'additive'), ('so', 'solvent'), ('am', 'active_material'), ('co', 'conductive_additive'), ('bi', 'binder'), ('se', 'separator_material')], max_length=2, null=True),
        ),
    ]