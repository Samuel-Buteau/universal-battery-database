
def electrolyte_csv_to_db(file):

    from WetCellDatabase.models import Electrolyte, ElectrolyteComponent, Alias, ElectrolyteMolecule
    from pandas import DataFrame
    import pandas as pd
    import numpy
    import math
    import re
    from WetCellDatabase.models import test_similarity

    df = pd.read_csv(file)

    for ind, row in df.iterrows():

        if row['Past name'] != 'Present electrolyte description' and \
            row['Past name'] != 'If there is no indication of solvent' and \
            row['Past name'] != 'IF there is no indication of salt concentration' and \
            row['Past name'] != 'If additive loading normalization not indicated' and \
            row['Past name'] != 'if VC211' and \
            row['Past name'] != 'if it is PES211':

            e = Electrolyte()
            e.save()

            if not(not isinstance(row['Past name'],str) and math.isnan(row['Past name'])):
                e.alias_set.create(name = row['Past name'])

            if not(not isinstance(row['Electrolyte ID'],str) and math.isnan(row['Electrolyte ID'])):
                e.alias_set.create(name = row['Electrolyte ID'])


            if not(not isinstance(row['LiPF6 concentration /molal'],str) and math.isnan(row['LiPF6 concentration /molal'])):

                if len(ElectrolyteMolecule.objects.filter(name='LiPF6')) == 0:
                    ElectrolyteMolecule.objects.create(name='LiPF6', can_be_salt=True)

                if re.match(r'\d',str(row['LiPF6 concentration /molal'])) and float(row['LiPF6 concentration /molal']) != 0:

                    e.component_set.create(molecule=ElectrolyteMolecule.objects.get(name='LiPF6'), molal=float(row['LiPF6 concentration /molal']))


            if not(not isinstance(row['Salt NO.2 name'],str) and math.isnan(row['Salt NO.2 name'])):

                if len(ElectrolyteMolecule.objects.filter(name=row['Salt NO.2 name'])) == 0:
                    ElectrolyteMolecule.objects.create(name=row['Salt NO.2 name'], can_be_salt=True)

                if not(not isinstance(row['Salt NO.2 concentration /molal'],str) and math.isnan(row['Salt NO.2 concentration /molal'])):
                    if re.match(r'\d',str(row['Salt NO.2 concentration /molal'])) and float(row['Salt NO.2 concentration /molal']) != 0:
                        e.component_set.create(molecule=ElectrolyteMolecule.objects.get(name=row['Salt NO.2 name']),
                                               molal=float(row['Salt NO.2 concentration /molal']))
                else:
                    e.component_set.create(molecule=ElectrolyteMolecule.objects.get(name=row['Salt NO.2 name']),
                                           molal=1.0)

            if not (not isinstance(row['Salt NO.3 name'], str) and math.isnan(row['Salt NO.3 name'])):

                if len(ElectrolyteMolecule.objects.filter(name=row['Salt NO.3 name'])) == 0:
                    ElectrolyteMolecule.objects.create(name=row['Salt NO.3 name'], can_be_salt=True)

                if not(not isinstance(row['Salt NO.3 concentration /molal'],str) and math.isnan(row['Salt NO.3 concentration /molal'])):
                    if re.match(r'\d',str(row['Salt NO.3 concentration /molal'])) and float(row['Salt NO.3 /molal']) != 0:
                        e.component_set.create(molecule=ElectrolyteMolecule.objects.get(name=row['Salt NO.3 name']),
                                               molal=float(row['Salt NO.3 concentration /molal']))
                else:
                    e.component_set.create(molecule=ElectrolyteMolecule.objects.get(name=row['Salt NO.3 name']),
                                           molal=1.0)


            if not(not isinstance(row['EC /wt%'],str) and math.isnan(row['EC /wt%'])):

                if len(ElectrolyteMolecule.objects.filter(name='EC')) == 0:
                    ElectrolyteMolecule.objects.create(name='EC', can_be_additive=True, can_be_solvent=True)

                if re.match(r'\d',str(row['EC /wt%'])) and float(row['EC /wt%']) != 0:
                    e.component_set.create(molecule=ElectrolyteMolecule.objects.get(name='EC'),
                                           weight_percent=float(row['EC /wt%']))

            if not(not isinstance(row['EMC /wt%'],str) and math.isnan(row['EMC /wt%'])):

                if len(ElectrolyteMolecule.objects.filter(name='EMC')) == 0:
                    ElectrolyteMolecule.objects.create(name='EMC', can_be_additive=True, can_be_solvent=True)
                if re.match(r'\d',str(row['EMC /wt%'])) and float(row['EMC /wt%']) != 0:
                    e.component_set.create(molecule=ElectrolyteMolecule.objects.get(name='EMC'),
                                           weight_percent=float(row['EMC /wt%']))

            if not(not isinstance(row['DMC /wt%'],str) and math.isnan(row['DMC /wt%'])):

                if len(ElectrolyteMolecule.objects.filter(name='DMC')) == 0:
                    ElectrolyteMolecule.objects.create(name='DMC', can_be_additive=True, can_be_solvent=True)

                if re.match(r'\d',str(row['DMC /wt%'])) and float(row['DMC /wt%']) != 0:
                    e.component_set.create(molecule=ElectrolyteMolecule.objects.get(name='DMC'),
                                           weight_percent=float(row['DMC /wt%']))

            if not(not isinstance(row['MA /wt%'],str) and math.isnan(row['MA /wt%'])):

                if len(ElectrolyteMolecule.objects.filter(name='MA')) == 0:
                    ElectrolyteMolecule.objects.create(name='MA', can_be_additive=True, can_be_solvent=True)


                if re.match(r'\d',str(row['MA /wt%'])) and float(row['MA /wt%']) != 0:
                    e.component_set.create(molecule=ElectrolyteMolecule.objects.get(name='MA'),
                                           weight_percent=float(row['MA /wt%']))


            if (not(not isinstance(row['Compound No.1 name'],str) and math.isnan(row['Compound No.1 name']))) and (not(not isinstance(row['Compound No.1 /wt%'],str) and math.isnan(row['Compound No.1 /wt%']))):

                if len(ElectrolyteMolecule.objects.filter(name=row['Compound No.1 name'])) == 0:
                    ElectrolyteMolecule.objects.create(name=row['Compound No.1 name'], can_be_additive=True, can_be_solvent=True)


                if re.match(r'\d',str(row['Compound No.1 /wt%'])) and float(row['Compound No.1 /wt%']) != 0:
                    e.component_set.create(molecule=ElectrolyteMolecule.objects.get(name=row['Compound No.1 name']),
                                           weight_percent=float(row['Compound No.1 /wt%']))

                else:
                    e.component_set.create(molecule=ElectrolyteMolecule.objects.get(name=row['Compound No.1 name']))

            if (not (not isinstance(row['Compound No.2 name2'], str) and math.isnan(row['Compound No.2 name2']))) and (
            not (not isinstance(row['Compound No.2 /wt%'], str) and math.isnan(
                    row['Compound No.2 /wt%']))):

                if len(ElectrolyteMolecule.objects.filter(name=row['Compound No.2 name2'])) == 0:
                    ElectrolyteMolecule.objects.create(name=row['Compound No.2 name2'], can_be_additive=True, can_be_solvent=True)

                if re.match(r'\d',str(row['Compound No.2 /wt%'])) and float(row['Compound No.2 /wt%']) != 0:
                    e.component_set.create(molecule=ElectrolyteMolecule.objects.get(name=row['Compound No.2 name2']),
                                           weight_percent=float(row['Compound No.2 /wt%']))

                else:
                    e.component_set.create(molecule=ElectrolyteMolecule.objects.get(name=row['Compound No.2 name2']))

            if (not(not isinstance(row['Compound No.3 name'],str) and math.isnan(row['Compound No.3 name']))) and (not(not isinstance(row['Compound No.3 /wt%'],str) and math.isnan(row['Compound No.3 /wt%']))):

                if len(ElectrolyteMolecule.objects.filter(name=row['Compound No.3 name'])) == 0:
                    ElectrolyteMolecule.objects.create(name=row['Compound No.3 name'], can_be_additive=True, can_be_solvent=True)


                if re.match(r'\d',str(row['Compound No.3 /wt%'])) and float(row['Compound No.3 /wt%']) != 0:
                    e.component_set.create(molecule=ElectrolyteMolecule.objects.get(name=row['Compound No.3 name']),
                                           weight_percent=float(row['Compound No.3 /wt%']))

                else:
                    e.component_set.create(molecule=ElectrolyteMolecule.objects.get(name=row['Compound No.3 name']))
            if (not(not isinstance(row['Compound No.4 name'],str) and math.isnan(row['Compound No.4 name']))) and (not(not isinstance(row['Compound No.4 /wt%'],str) and math.isnan(row['Compound No.4 /wt%']))):

                if len(ElectrolyteMolecule.objects.filter(name=row['Compound No.4 name'])) == 0:
                    ElectrolyteMolecule.objects.create(name=row['Compound No.4 name'], can_be_additive=True, can_be_solvent=True)


                if re.match(r'\d',str(row['Compound No.4 /wt%'])) and float(row['Compound No.4 /wt%']) != 0:
                    e.component_set.create(molecule=ElectrolyteMolecule.objects.get(name=row['Compound No.4 name']),
                                           weight_percent=float(row['Compound No.4 /wt%']))

                else:
                    e.component_set.create(molecule=ElectrolyteMolecule.objects.get(name=row['Compound No.4 name']))
            if (not(not isinstance(row['Compound No.5 name'],str) and math.isnan(row['Compound No.5 name']))) and (not(not isinstance(row['Compound No.5 /wt%'],str) and math.isnan(row['Compound No.5 /wt%']))):

                if len(ElectrolyteMolecule.objects.filter(name=row['Compound No.5 name'])) == 0:
                    ElectrolyteMolecule.objects.create(name=row['Compound No.5 name'], can_be_additive=True, can_be_solvent=True)


                if re.match(r'\d',str(row['Compound No.5 /wt%'])) and float(row['Compound No.5 /wt%']) != 0:
                    e.component_set.create(molecule=ElectrolyteMolecule.objects.get(name=row['Compound No.5 name']),
                                           weight_percent=float(row['Compound No.5 /wt%']))

                else:
                    e.component_set.create(molecule=ElectrolyteMolecule.objects.get(name=row['Compound No.5 name']))
            if (not(not isinstance(row['Compound No.6 name'],str) and math.isnan(row['Compound No.6 name']))) and (not(not isinstance(row['Compound No.6 /wt%'],str) and math.isnan(row['Compound No.6 /wt%']))):

                if len(ElectrolyteMolecule.objects.filter(name=row['Compound No.6 name'])) == 0:
                    ElectrolyteMolecule.objects.create(name=row['Compound No.6 name'], can_be_additive=True, can_be_solvent=True)


                if re.match(r'\d',str(row['Compound No.6 /wt%'])) and float(row['Compound No.6 /wt%']) != 0:
                    e.component_set.create(molecule=ElectrolyteMolecule.objects.get(name=row['Compound No.6 name']),
                                           weight_percent=float(row['Compound No.6 /wt%']))

                else:
                    e.component_set.create(molecule=ElectrolyteMolecule.objects.get(name=row['Compound No.6 name']))

            print(e.component_set.all())


            if len(e.component_set.all()) == 0:
                print('PROPRIETARY DEFINED')
                e.proprietary = True
                e.shortstring = 'PROPRIETARY - {}'.format(re.sub(' ', '_', row['Past name']))

                e.alias_set.create(name=e.shortstring)

            else:
                e.shortstring = e.generate_shortstring
                e.proprietary = False

                e.alias_set.create(name=e.shortstring)

            print(ind)

            if not e.proprietary:

                elec_dict1 = {'salts':{},'solvents':{}}

                for component in e.component_set.all():
                    if component.molecule.can_be_salt:
                        elec_dict1['salts'][component.molecule.name] = component.molal

                    if component.molecule.can_be_additive or component.molecule.can_be_solvent:
                        elec_dict1['solvents'][component.molecule.name] = component.weight_percent


                def loop_through_electrolytes():


                    for electrolyte in Electrolyte.objects.all():

                        if electrolyte.proprietary:

                            continue

                        else:
                            elec_dict2 = {'salts':{},'solvents':{}}

                            for component in electrolyte.component_set.all():

                                if component.molecule.can_be_salt:
                                    elec_dict2['salts'][component.molecule.name] = component.molal

                                if component.molecule.can_be_additive or component.molecule.can_be_solvent:
                                    elec_dict2['solvents'][component.molecule.name] = component.weight_percent

                            if test_similarity(elec_dict1,elec_dict2) and (e.id != electrolyte.id):

                                print('SIMILAR')
                                return True



                    print('NOT SIMILAR')
                    return False

                similar = loop_through_electrolytes()

                if not similar:
                    e.save()
                else:
                    if not (not isinstance(row['Past name'], str) and math.isnan(row['Past name'])):
                        if len(Alias.objects.filter(name=row['Past name'])) == 0:
                            Alias.objects.create(electrolyte=e,name=row['Past name'])

                    if not (not isinstance(row['Electrolyte ID'], str) and math.isnan(row['Electrolyte ID'])):
                        if len(Alias.objects.filter(name=row['Electrolyte ID'])) == 0:
                            Alias.objects.create(electrolyte=e,name=row['Electrolyte ID'])

                    e.delete()

            else:
                if not (not isinstance(row['Past name'], str) and math.isnan(row['Past name'])):
                    if len(Alias.objects.filter(name=row['Past name'])) == 0:
                        e.save()
                    else:
                        e.delete()




