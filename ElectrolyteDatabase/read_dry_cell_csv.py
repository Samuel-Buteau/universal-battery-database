def dry_cell_csv_to_db(file):

    from pandas import DataFrame
    import pandas as pd
    from ElectrolyteDatabase.models import DryCell, AnodeBinder, CathodeBinder, Cathode, CathodeConductiveAdditive,AnodeConductiveAdditive, MechanicalPouch, BuildInfo, Box, OtherInfo,\
        Separator, Anode, CathodeActiveMaterials, AnodeActiveMaterials, CathodeCoating, VendorInfo
    import numpy
    import math
    import re
    from django.db.models import Q

    df = pd.read_csv(file)

    cell_models = []

    for model in df:
        if model != 'BASICS' and model != 'Cell Model':
            cell_models.append(model)
    count = 0
    for model in cell_models:

        if len(DryCell.objects.filter(cell_model=model)) != 0:

            continue

        else:

            cathode_notes = False

            anode_notes = False

            cell = DryCell()
            cell.save()

            print(count)

            cell.cell_model = model

            print(model)

            mechanical_pouch = MechanicalPouch()

            mechanical_pouch.dry_cell = cell

            mechanical_pouch.save()

            build_info = BuildInfo()

            build_info.save()

            build_info.dry_cell = cell

            cathode = Cathode(dry_cell=cell)

            cathode.save()

            vendor_info = VendorInfo(dry_cell=cell)
            vendor_info.save()

            separator = Separator()
            separator.save()
            separator.dry_cell = cell
            anode = Anode(dry_cell = cell)
            anode.save()

            cathode_binder = CathodeBinder(cathode=cathode)
            cathode_binder.save()
            anode_binder = AnodeBinder(anode=anode)
            anode_binder.save()

            cathode_active_materials = CathodeActiveMaterials()

            cathode_active_materials.save()

            anode_active_materials = AnodeActiveMaterials()

            anode_active_materials.save()

            other_info = OtherInfo(other_info_cell=cell)
            other_info.save()




            for ind, row in df.iterrows():



                if not (not isinstance(row[model], str) and math.isnan(row[model])):

        ## Electrolyte Info

                    #TODO: Reading in electrolyte information needs more thought


        ## BASICS

                    if row['Cell Model'] == 'Family / Group':
                        cell.family = row[model]

                    if row['Cell Model'] == 'Version Number':
                        cell.version_number = row[model]

                    if row['Cell Model'] == 'Short Description / Sample Purpose':
                        cell.description = row[model]

                    if row['Cell Model'] == 'Quantity':
                        cell.quantity = row[model]

                    if row['Cell Model'] == 'Packing Date':
                        cell.packing_date = row[model]

                    if row['Cell Model'] == 'Ship Date':
                        cell.ship_date = row[model]

                    if row['Cell Model'] == 'Marking on box':
                        cell.marking_on_box = row[model]

                    if row['Cell Model'] == 'Shipping SOC':
                        cell.shipping_soc = row[model]

                    if row['Cell Model'] == 'Energy Estimate (Wh)':
                        cell.energy_estimate_wh = row[model]


                    #TODO: Unsure how to handle of these in the database. For the most part the spreadsheet simply has a float using the unit 'Ah' for the Capacity estimate however
                    # sometimes the units change and rather than there being something like '2.3' there is instead '200 mAh/g'

                    if row['Cell Model'] == 'Capacity Estimate (Ah)':

                        if not re.match("^\d+?\.\d+?$|^\d+$", row[model]) is None:
                            cell.capacity_estimate_ah = row[model]

                        else:
                            continue

                    if row['Cell Model'] == 'Mass estimate (g)':
                        cell.mass_esitmate_g = row[model]

                    if row['Cell Model'] == 'Max Charge Voltage':
                        voltage = row[model]

                        voltage_value = re.sub(r'\s|v|V','',voltage)

                        cell.max_charge_voltage_v = voltage_value

                    if row['Cell Model'] == 'DCR Estimate':
                        cell.dcr_estimate = row[model]

                    if row['Cell Model'] == 'Chemisty Freeze Date Requested':
                        cell.chemistry_freeze_date_requested = row[model]



        ## Build Info




                    if row['Cell Model'] == 'cathode active lot #':
                        build_info.cathode_active_lot = row[model]


                    if row['Cell Model'] == 'anode active lot #':
                        build_info.anode_active_lot = row[model]

                    if row['Cell Model'] == 'separator lot #':
                        build_info.separator_lot = row[model]

                    if row['Cell Model'] == 'Cathode mix lot #':
                        build_info.cathode_mix_lot = row[model]

                    if row['Cell Model'] == 'Anode active lot #':
                        build_info.anode_mix_lot = row[model]

                    if row['Cell Model'] == 'Cell Assembly lot #':
                        build_info.cell_assembly_lot = row[model]

                    if row['Cell Model'] == 'Mix/Coat Location':
                        build_info.mix_coat_location = row[model]

                    if row['Cell Model'] == 'Winding Location':
                        build_info.winding_location = row[model]

                    if row['Cell Model'] == 'Assembly Location':
                        build_info.assembly_location = row[model]

                    if row['Cell Model'] == 'Other Mechanical Notes':
                        build_info.other_mechanical_notes = row[model]

                    if row['Cell Model'] == 'Other Electrode Notes':
                        build_info.other_electrode_notes = row[model]

                    if row['Cell Model'] == 'Other Processs Notes':
                        build_info.other_process_notes = row[model]

                    if row['Cell Model'] == 'Other Notes':
                        build_info.other_notes = row[model]




        ## MECHANICAL POUCH



                    if row['Cell Model'] == ' Outer Taping Notes':
                        mechanical_pouch.outer_taping = row[model]


                    if row['Cell Model'] == 'Cell Width(mm)':
                        mechanical_pouch.cell_width_mm = row[model]

                    if row['Cell Model'] == 'Cell Length(mm)':
                        mechanical_pouch.cell_length_mm = row[model]

                    if row['Cell Model'] == 'Cell Thickness(mm)':
                        mechanical_pouch.cell_thickness_mm = row[model]

                    if row['Cell Model'] == 'Seal Width Side (mm)':
                        mechanical_pouch.seal_width_side_mm = row[model]

                    if row['Cell Model'] == 'Seal Width Top(mm)':
                        mechanical_pouch.seal_width_top_mm = row[model]

                    if row['Cell Model'] == 'Cathode Tab polymer material':
                        mechanical_pouch.cathode_tab_polymer_material = row[model]

                    if row['Cell Model'] == 'Anode Tab polymer material':
                        mechanical_pouch.anode_tab_polymer_material = row[model]

                    if row['Cell Model'] == 'Metal Bag sheet thickness (mm)':
                        mechanical_pouch.metal_bag_sheet_thickness_mm = row[model]

        ## POSITIVE


                    if row['Cell Model'] == 'Metal Bag sheet structure':
                        cathode.metal_bag_sheet_structure = row[model]



                    #TODO: Cathode active 1 ... 3 need more thought (parsing)
                    # Q: Should this be handled manually? Each case is very different and in many cases
                    # would be inefficient to build a parser that is specific enough
                    ## Ex: RockNMC532 (BDA5000) vs. LiNi0.5Mn0.3Co0.2O2 + Coating A



                    if row['Cell Model'] == 'Cathode conductive additive Notes':

                        string = row[model]
                        re.sub(r'\s$|^\s', '', string)

                        if not re.search(r'\+',string):
                            cathode_conductive_additive = CathodeConductiveAdditive(notes=string,cathode=cathode)
                            cathode_conductive_additive.save()


                        else:
                            additive_list = re.split(r'\s\+\s|\+\s|\s\+',string)

                            for additive in additive_list:
                                cathode_conductive_additive = CathodeConductiveAdditive(notes=additive,cathode=cathode)
                                cathode_conductive_additive.save()



                    if row['Cell Model'] == 'Positive Electrode Compostion Notes':
                        cathode.positive_electrode_composition_notes = row[model]

                    if row['Cell Model'] == 'Positive Electrode Loading(mg/cm^2)':
                        cathode.positive_electrode_loading_mg_cm2 = row[model]


                    #TODO: The value '3.2-3.3' was used for positive electrode density, how should this be handled?

                    if row['Cell Model'] == 'Positive Electrode Density (g/cm^3)':

                        if not re.match("^\d+?\.\d+?$|^\d+$", row[model]) is None:
                            cathode.positive_electrode_density_g_cm3 = row[model]

                        else:
                            continue

                    if row['Cell Model'] == 'Positive Electrode Porosity':
                        cathode.positive_electrode_porosity = row[model]

                    if row['Cell Model'] == 'Positive Electrode Thickness(um)':
                        cathode.positive_electrode_thickness_um = row[model]

                    if row['Cell Model'] == 'Positive Electrode Length single side':
                        cathode.positive_electrode_length_single_side = row[model]

                    if row['Cell Model'] == 'Positive Electrode Length double side':
                        cathode.positive_electrode_length_double_side = row[model]

                    if row['Cell Model'] == 'Positive Electrode Width':
                        cathode.positive_electrode_width = row[model]

                    if row['Cell Model'] == 'Positive Tab position from core':
                        cathode.positive_tab_position_from_core = row[model]

                    if row['Cell Model'] == 'Positive Foil thickness (um)':
                        cathode.positive_foil_thickness_um = row[model]

                    if row['Cell Model'] == 'Positive Functional Layer Notes':
                        cathode.positive_functional_layer_notes = row[model]

                    if row['Cell Model'] == 'Positive Functional Thickness':
                        cathode.positive_functional_thickness = row[model]


                    if re.match(r'Cathode active \d',str(row['Cell Model'])) and (not cathode_notes):


                        for i, r in df.iterrows():

                            if (r['Cell Model'] == 'Cathode active 1 Notes') and (not (not isinstance(r[model], str) and math.isnan(r[model]))):
                                cathode_active_materials.cathode_active_1_notes = str(r[model])

                            if (r['Cell Model'] == 'Cathode active 2 Notes') and (not (not isinstance(r[model], str) and math.isnan(r[model]))):
                                cathode_active_materials.cathode_active_2_notes = str(r[model])

                            if (r['Cell Model'] == 'Cathode active 3 Notes') and (not (not isinstance(r[model], str) and math.isnan(r[model]))):
                                cathode_active_materials.cathode_active_3_notes = str(r[model])

                        cathode_notes = True



        ## SEPARATOR



                    if row['Cell Model'] == 'Separator Notes':
                        separator.separator_notes = row[model]

                    if row['Cell Model'] == 'Separator Base Thickness':
                        separator.separator_base_thickness = row[model]

                    if row['Cell Model'] == 'Separator Width(mm)':
                        separator.separator_width_mm = row[model]

                    if row['Cell Model'] == 'Separator Functional Layer Notes':
                        separator.separator_functional_layer = row[model]

                    if row['Cell Model'] == 'Separator Functional Thickness':
                        separator.separator_functional_thickness = row[model]

                    if row['Cell Model'] == 'Separator Overhang in Core (mm)':
                        separator.separator_overhang_in_core_mm = row[model]

        ## NEGATIVE




                    #TODO: Anode active 1 ... 3 incomplete (same as the case for Cathode active 1...3)

                    if row['Cell Model'] == 'Anode conductive additive Notes':

                        string = row[model]
                        re.sub(r'\s$|^\s', '', string)

                        if not re.search(r'\+',string):
                            anode_conductive_additive = AnodeConductiveAdditive(notes=string,anode=anode)
                            anode_conductive_additive.save()


                        else:
                            additive_list = re.split(r'\s\+\s|\+\s|\s\+',string)

                            for additive in additive_list:
                                anode_conductive_additive = AnodeConductiveAdditive(notes = additive,anode=anode)
                                anode_conductive_additive.save()


                    if row['Cell Model'] == 'Negative Electrode Compostion Notes':
                        anode.negative_electrode_composition_notes = row[model]

                    if row['Cell Model'] == 'Negative Electrode Loading(mg/cm^2)':
                        anode.negative_electrode_loading_mg_cm2 = row[model]

                    if row['Cell Model'] == 'Negative Electrode Density':
                        anode.negative_electrode_density_g_cm3 = row[model]

                    if row['Cell Model'] == 'Negative Electrode Porosity':
                        anode.negative_electrode_porosity = row[model]

                    if row['Cell Model'] == 'Negative Electrode Thickness (um)':
                        anode.negative_electrode_thickness_um = row[model]

                    if row['Cell Model'] == 'Negative Electrode Length single side':
                        anode.negative_electrode_length_single_side = row[model]

                    if row['Cell Model'] == 'Negative Electrode Length double side':
                        anode.negative_electrode_length_double_side = row[model]

                    if row['Cell Model'] == 'Negative Electrode Width':
                        anode.negative_electrode_width = row[model]

                    if row['Cell Model'] == 'Negative Tab position from core':
                        anode.negative_tab_position_from_core = row[model]

                    if row['Cell Model'] == 'Negative Foil thickness (um)':
                        anode.negative_foil_thickness_um = row[model]

                    if row['Cell Model'] == 'Negative Tab Notes':
                        anode.negative_tab_notes = row[model]

                    if row['Cell Model'] == '2 Tab Notes':
                        anode.tab_2_notes = row[model]

                    if row['Cell Model'] == 'Negative Functional Layer Notes':
                        anode.negative_functional_layer = row[model]

                    if row['Cell Model'] == 'Negative Functional Thickness':
                        anode.negative_functional_thickness = row[model]


                    if re.match(r'Anode active \d', str(row['Cell Model'])) and (not anode_notes):

                        for i, r in df.iterrows():

                            if (r['Cell Model'] == 'Anode active 1 Notes') and (
                            not (not isinstance(r[model], str) and math.isnan(r[model]))):
                                anode_active_materials.anode_active_1_notes = str(r[model])

                            if (r['Cell Model'] == 'Anode active 2 Notes') and (
                            not (not isinstance(r[model], str) and math.isnan(r[model]))):
                                anode_active_materials.anode_active_2_notes = str(r[model])

                            if (r['Cell Model'] == 'Anode active 3 Notes') and (
                            not (not isinstance(r[model], str) and math.isnan(r[model]))):
                                anode_active_materials.anode_active_3_notes = str(r[model])

                            if (r['Cell Model'] == 'Anode active 4 Notes') and (
                            not (not isinstance(r[model], str) and math.isnan(r[model]))):
                                anode_active_materials.anode_active_4_notes = str(r[model])

                        anode_notes = True



        ## Cathode and Anode Binder

                    if row['Cell Model'] == 'Cathode Binder 1 Notes':
                        cathode_binder.cathode_binder_1_notes = row[model]

                    if row['Cell Model'] == 'Cathode Binder 3 Notes':
                        cathode_binder.cathode_binder_2_notes = row[model]

                    if row['Cell Model'] == 'Anode Binder 1 Notes':
                        anode_binder.anode_binder_1_notes = row[model]

                    if row['Cell Model'] == 'Anode Binder 2 Notes':
                        anode_binder.anode_binder_2_notes = row[model]

                    if row['Cell Model'] == 'Anode Binder 3 Notes':
                        anode_binder.anode_binder_3_notes = row[model]

                    cathode_binder.save()
                    anode_binder.save()

        ## VENDOR INFO


                    if row['Cell Model'] == 'Cathode active 1 Vendor':
                        vendor_info.cathode_active_1_vendor = row[model]

                    if row['Cell Model'] == 'Cathode active 2 Vendor':
                        vendor_info.cathode_active_2_vendor = row[model]

                    if row['Cell Model'] == 'Cathode active 3 Vendor':
                        vendor_info.cathode_active_3_vendor = row[model]

                    if row['Cell Model'] == 'Cathode additive Vendor':
                        vendor_info.cathode_additive_vendor = row[model]

                    if row['Cell Model'] == 'Cathode Binder 1 Vendor':
                        vendor_info.cathode_binder_1_vendor = row[model]

                    if row['Cell Model'] == 'Cathode Binder 2 Vendor':
                        vendor_info.cathode_binder_2_vendor = row[model]

                    if row['Cell Model'] == 'Cathode Binder 3 Vendor':
                        vendor_info.cathode_binder_3_vendor = row[model]

                    if row['Cell Model'] == 'Anode active 1 Vendor':
                        vendor_info.anode_active_1_vendor = row[model]

                    if row['Cell Model'] == 'Anode active 2 Vendor':
                        vendor_info.anode_active_2_vendor = row[model]

                    if row['Cell Model'] == 'Anode active 3 Vendor':
                        vendor_info.anode_active_3_vendor = row[model]

                    if row['Cell Model'] == 'Anode active 4 Vendor':
                        vendor_info.anode_active_4_vendor = row[model]

                    if row['Cell Model'] == 'Anode Binder 1 Vendor':
                        vendor_info.anode_binder_1_vendor = row[model]

                    if row['Cell Model'] == 'Anode Binder 2 Vendor':
                        vendor_info.anode_binder_1_vendor = row[model]

                    if row['Cell Model'] == 'Anode Binder 3 Vendor':
                        vendor_info.anode_binder_1_vendor = row[model]

                    if row['Cell Model'] == 'Negative foil Vendor':
                        vendor_info.negative_foil_vendor = row[model]

                    if row['Cell Model'] == 'Separator Vendor':
                        vendor_info.separator_vendor = row[model]

                    if row['Cell Model'] == 'Separator Coat Vendor':
                        vendor_info.separator_coat_vendor = row[model]

                    if row['Cell Model'] == 'Gasket Vendor':
                        vendor_info.gasket_vendor = row[model]

                    if row['Cell Model'] == 'Can Vendor':
                        vendor_info.can_vendor = row[model]

                    if row['Cell Model'] == 'Metal Bag sheet vendor':
                        vendor_info.metal_bag_sheet_vendor = row[model]

                    if row['Cell Model'] == 'Top Cap Vendor':
                        vendor_info.top_cap_vendor = row[model]

                    if row['Cell Model'] == 'Outer Tape Vendor':
                        vendor_info.outer_tape_vendor = row[model]

                    vendor_info.save()






        ## OTHER


                    if row['Cell Model'] == 'Jellyroll Centering':
                        other_info.jellyroll_centering = row[model]

                    if row['Cell Model'] == 'Ni Tab Rear Tape  (material)':
                        other_info.ni_tab_rear_tape_material = row[model]

                    if row['Cell Model'] == 'Ni Tab Rear tape  Width(mm) ':
                        other_info.ni_tab_rear_tape_width_mm = row[model]

                    if row['Cell Model'] == 'Anoode Front Substrate Length':
                        other_info.anode_front_substrate_length = row[model]

                    if row['Cell Model'] == 'Anode End Substrate Length':
                        other_info.anode_end_substrate_length = row[model]

                    if row['Cell Model'] == '(-) Tab Ultra Sonic Welding Spots ':
                        other_info.negative_tab_ultra_sonic_welding_spots = row[model]

                    if row['Cell Model'] == 'Starting Can Height(mm)':
                        other_info.starting_can_height_mm = row[model]

                    if row['Cell Model'] == '(+) Tab Laser Welding Spots':
                        other_info.positive_tab_laser_welding_spots = row[model]

                    if row['Cell Model'] == 'Ni Tab Rear tape  Width(mm) ':
                        other_info.ni_tab_rear_tape_width_mm = row[model]

                    if row['Cell Model'] == 'alpha':
                        other_info.alpha = row[model]

                    if row['Cell Model'] == 'beta':
                        other_info.beta = row[model]

                    if row['Cell Model'] == 'gamma':
                        other_info.gamma = row[model]

                    if row['Cell Model'] == 'gamma':
                        other_info.gamma = row[model]


        ## Box

                    if row['Cell Model'] == 'Box_ID' or (row['Cell Model'] == '' and re.match(r'\d+', row[model])):

                        if len(Box.objects.filter(box_id_number=row[model])) == 0:

                            box = Box(cell_model=cell,box_id_number=row[model])
                            box.save()



                    cathode.cathode_active_materials = cathode_active_materials
                    anode.anode_active_materials = anode_active_materials
                    cathode.save()
                    cathode_active_materials.save()
                    cell.save()
                    build_info.save()
                    mechanical_pouch.save()
                    anode_active_materials.save()
                    separator.save()
                    anode.save()
                    other_info.save()


            count += 1