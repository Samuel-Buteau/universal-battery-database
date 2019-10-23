def read_sql_cell_table(box_model,model_cell_electrolyte):

    from pandas import DataFrame
    import pandas as pd
    from WetCellDatabase.models import DryCell, Box, Electrolyte, Alias, WetCell
    import numpy
    import math
    import re
    from django.db.models import Q

    box_model_df = pd.read_csv(box_model,error_bad_lines=False)

    model_cell_electrolyte_df = pd.read_csv(model_cell_electrolyte,error_bad_lines=False)


    ## Box - model  CSV

    print("MODEL-BOX CSV")

    count1 = 0

    for ind, row in box_model_df.iterrows():

        count1 += 1
        print(count1)

        box_already_saved = False

        existing_models = []
        existing_box_ids = []

        box_id = row['box_id']

        model_name = row['model_name']

        for box in Box.objects.all():
            existing_box_ids.append(box.box_id_number)

        for cell in DryCell.objects.all():
            existing_models.append(cell.cell_model)

        if not (not isinstance(model_name, str) and math.isnan(model_name)):

            if model_name in existing_models:

                if not (not isinstance(box_id, str) and math.isnan(box_id)):

                    if box_id in existing_box_ids:

                        if (Box.objects.get(box_id_number=box_id)).cell_model is None:

                            box = Box.objects.get(box_id_number=box_id)
                            box.cell_model = DryCell.objects.get(cell_model=model_name)
                            box.save()

                    else:

                        box_instance = Box(cell_model=DryCell.objects.get(cell_model=model_name), box_id_number=box_id)
                        box_instance.save()



    ## Model-Cell id-Electrolyte

    print('CELL-BOX-ELECTROLYTE CSV')

    count2 = 0

    previous_cell_ids = []

    for ind, row in model_cell_electrolyte_df.iterrows():


        count2 += 1

        print(count2)

        cell_id = row['cellid']

        if cell_id in previous_cell_ids:

            continue

        previous_cell_ids.append(cell_id)

        box_id = row['box_id']

        electrolyte_alias = row['electrolyte']

        existing_box_ids = []

        existing_electrolyte_aliases = []

        existing_cell_ids = []

        for box in Box.objects.all():
            existing_box_ids.append(box.box_id_number)

        for alias in Alias.objects.all():

            if not alias.electrolyte is None:
                existing_electrolyte_aliases.append(alias.name)

        for cell in WetCell.objects.all():

            if not cell.cell_id in existing_cell_ids:
                existing_electrolyte_aliases.append(cell.cell_id)


        box_id_present = False
        cell_id_present = False
        electrolyte_present = False
        box_exists = False

        box = Box()
        box.save()
        electrolyte = Electrolyte()
        electrolyte.save()


        if not (not isinstance(box_id, str) and math.isnan(box_id)) and box_id != 0:

            box_id_present = True

            if box_id in existing_box_ids:

                box.delete()
                box = Box.objects.get(box_id_number=box_id)

                box_exists = True

        else:
            box.delete()

        if not (not isinstance(electrolyte_alias, str) and math.isnan(electrolyte_alias)):

            electrolyte_present = True

            if electrolyte_alias in existing_electrolyte_aliases:

                electrolyte.delete()
                electrolyte = (Alias.objects.get(name=electrolyte_alias)).electrolyte
        else:
            electrolyte.delete()

        if not (not isinstance(cell_id, str) and math.isnan(cell_id)):

            cell_id_present = True


        save_wet_cell = False

        if cell_id_present:

            save_wet_cell = True

            if cell_id in existing_cell_ids:

                wet_cell = WetCell.objects.get(cell_id=cell_id)

            else:

                wet_cell = WetCell()
                wet_cell.save()

                wet_cell.cell_id = cell_id

        else:

            wet_cell = WetCell()
            wet_cell.save()


        if box_id_present:

            wet_cell.box = box

            save_wet_cell = True

            if box_exists:
                wet_cell.dry_cell = (Box.objects.get(box_id_number=box_id)).cell_model

            if not box_exists:
                wet_cell.dry_cell = None

        if electrolyte_present:

            wet_cell.electrolyte = electrolyte

            save_wet_cell = True


        if save_wet_cell:
            wet_cell.save()

        else:
            wet_cell.delete()
































