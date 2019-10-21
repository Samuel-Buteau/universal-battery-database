from django.contrib import admin
from django.urls import path
from ElectrolyteDatabase import views as ElectrolyteDatabase
from . import views

from django.conf.urls import url


app_name = 'ElectrolyteDatabase'

urlpatterns = [
    path('', views.main_page, name='main_page'),

    path('register_new_molecule',views.register_new_molecule, name='register_new_molecule'),
    path('register_new_electrolyte',views.register_new_electrolyte, name='register_new_electrolyte'),
    path('register_or_modify_dry_cell_models',views.register_or_modify_dry_cell_models, name='register_or_modify_dry_cell_models'),
    path('register_new_cathode_family',views.register_new_cathode_family, name='register_new_cathode_family'),
    path('register_new_anode_family',views.register_new_anode_family, name='register_new_anode_family'),
    path('register_new_cathode_specific_material_ratio',views.register_new_cathode_specific_material_ratio, name='register_new_cathode_specific_material_ratio'),
    path('register_new_anode_specific_material_ratio',views.register_new_anode_specific_material_ratio, name='register_new_anode_specific_material_ratio'),
    path('register_new_cathode_coating',views.register_new_cathode_coating, name='register_new_cathode_coating'),
    path('specify_and_edit_wet_cells/',views.specify_and_edit_wet_cells, name='specify_and_edit_wet_cells'),
    path('register_new_cell_attribute/',views.register_new_cell_attribute, name='register_new_cell_attribute'),
    path('link_box_to_cell/',views.link_box_to_cell, name='link_box_to_cell'),
    path('register_cathode_anode_info/',views.register_cathode_anode_info, name='register_cathode_anode_info'),

]

