import time

import matplotlib.pyplot as plt

import numpy
import tensorflow as tf
from django.core.management.base import BaseCommand

from neware_parser.DegradationModel import DegradationModel
from neware_parser.models import *
from neware_parser.plot import *
from neware_parser.Key import *
import csv
from scipy.interpolate import InterpolatedUnivariateSpline






def v_curves_harvest(fit_args):
    if not os.path.exists(fit_args[Key.PATH_PLOTS]):
        os.makedirs(fit_args[Key.PATH_PLOTS])

    v_curves_folder = fit_args[Key.PATH_V_CURVES]
    v_curves_meta = fit_args[Key.PATH_V_CURVES_META]


    if not os.path.exists(v_curves_folder):
        print("Path \"" + v_curves_folder + "\" does not exist.")
        return

    if not os.path.exists(v_curves_meta):
        print("Metadata file \"" + v_curves_meta + "\" does not exist.")
        return

    def process_row(row):
        if len(row) < 3:
            return None,None,None
        key = row[0]
        electrode_string = row[1]
        rate_string = row[2]

        if not (electrode_string == 'cathode' or electrode_string == 'anode'):
            return None,None,None

        try:
            rate = float(rate_string)
        except:
            return None,None,None

        return key, electrode_string, rate

    metadatas = []
    with open(v_curves_meta) as csvfile:
        my_reader = csv.reader(csvfile)
        for row in my_reader:
            key, electrode, rate = process_row(row)
            if key is not None:
                metadatas.append((key, electrode, rate))

    print(metadatas)

    def process_v_curve_row(row):
        if len(row) < 2:
            return None,None
        q_string = row[0]
        v_string = row[1]

        try:
            q = float(q_string)
            v = float(v_string)
            return q, v

        except:
            return None,None



    datas = {}
    for key, electrode, rate in metadatas:
        if not electrode in datas.keys():
            datas[electrode] = {}

        if not rate in datas[electrode].keys():
            datas[electrode][rate] = []

        with open(os.path.join(v_curves_folder, key)) as csvfile:
            my_reader = csv.reader(csvfile)
            for row in my_reader:
                q, v = process_v_curve_row(row)
                if q is not None:
                    datas[electrode][rate].append((q,v))

        datas[electrode][rate] = sorted(datas[electrode][rate], key=lambda x: x[1])

        datas[electrode][rate] = numpy.array(datas[electrode][rate])

    # Plot
    fig = plt.figure(figsize=[11, 10])
    ax = fig.add_subplot(111)
    for electrode in datas.keys():
        for rate in datas[electrode].keys():
            shift = numpy.min([s[0] for s in datas[electrode][rate]])
            datas[electrode][rate][:, 0] = datas[electrode][rate][:, 0] - shift
            ax.plot(
                [s[0] for s in datas[electrode][rate]],
                [s[1] for s in datas[electrode][rate]],
                label='{}, rate={}'.format(electrode, rate)
            )

    ax.legend()

    savefig("v_curves_ground_truth.png", fit_args)
    plt.close(fig)


    with open(os.path.join(fit_args[Key.PATH_V_CURVES], 'v_curves.file'), 'wb') as file:
        pickle.dump(datas, file, protocol= pickle.HIGHEST_PROTOCOL)






def v_curves_complex(fit_args):
    v_curves_folder = fit_args[Key.PATH_V_CURVES]
    v_curves_meta = fit_args[Key.PATH_V_CURVES_META]

    if not os.path.exists(v_curves_folder):
        print("Path \"" + v_curves_folder + "\" does not exist.")
        return

    if not os.path.exists(v_curves_meta):
        print("Metadata file \"" + v_curves_meta + "\" does not exist.")
        return

    def process_metadata_row(row):
        if len(row) < 5:
            return None,None,None,None
        key = row[0]
        electrode_string = row[1]
        c_rate_string = row[2]
        resistance_string = row[3]
        name = row[4]

        if not (electrode_string == 'cathode' or electrode_string == 'anode'):
            return None,None,None,None,None

        try:
            c_rate = float(c_rate_string)
            resistance = float(resistance_string)
            return key, electrode_string, c_rate, resistance, name
        except:
            return None,None,None,None,None



    metadatas = []
    with open(v_curves_meta) as csvfile:
        my_reader = csv.reader(csvfile)
        for row in my_reader:
            key, electrode, c_rate, resistance, name = process_metadata_row(row)
            if key is not None:
                metadatas.append((key, electrode, c_rate, resistance, name))



    def process_row(row):
        if len(row) < 4:
            return None, None, None, None
        current_string = row[0]
        voltage_string = row[1]
        capacity_string = row[2]
        step_count_string = row[3]


        try:
            current = float(current_string)
            voltage = float(voltage_string)
            capacity = float(capacity_string)
            step_count = int(step_count_string)
        except:
            return None, None, None, None

        return current, voltage, capacity, step_count


    datas = {}
    for key, electrode_string, c_rate, resistance, name in metadatas:
        datas[name] = {}
        with open(os.path.join(v_curves_folder, key)) as csvfile:
            my_reader = csv.reader(csvfile)
            for row in my_reader:
                current,voltage, capacity, step_count = process_row(row)
                if not step_count in datas[name].keys():
                    datas[name][step_count] = []

                datas[name][step_count].append((current, voltage, capacity))

        already_seen_c_over_20 = False
        new_data = {}
        for dat in datas[name].values():
            avg_current = numpy.average([d[0] for d in dat])
            if avg_current < -0.0001:
                idealized_current =round(20.*avg_current/c_rate)/20.
                if abs(idealized_current - (-.05)) < 0.001 and not already_seen_c_over_20:
                    already_seen_c_over_20 = True
                elif abs(idealized_current - (-.05)) < 0.001 and already_seen_c_over_20:
                    continue

                new_data[abs(idealized_current)] = numpy.array([[s[2], s[1]] for s in dat if abs(s[0] - avg_current) < 0.001])



        already_seen_c_over_20 = 0
        new_data_charge = {}
        for dat in datas[name].values():
            avg_current = numpy.average([d[0] for d in dat])
            if avg_current > 0.0001:
                idealized_current = round(20. * avg_current / c_rate) / 20.
                if abs(idealized_current - (.05)) < 0.001 and already_seen_c_over_20 == 1:
                    already_seen_c_over_20 +=1
                elif abs(idealized_current - (.05)) < 0.001 and already_seen_c_over_20 == 0:
                    already_seen_c_over_20 +=1
                    continue
                elif abs(idealized_current - (.05)) < 0.001 and already_seen_c_over_20 > 1:
                    continue
                # if (idealized_current - (-1.)) < 0.05:
                #     continue

                new_data_charge[abs(idealized_current)] = numpy.array(
                    [[s[2], s[1]] for s in dat if abs(s[0] - avg_current) < 0.001])

        shift = 0.
        for rate in new_data.keys():
            shift = max(shift, max([s[0] for s in new_data[rate]]))
        for rate in new_data_charge.keys():
            shift = max(shift, max([s[0] for s in new_data_charge[rate]]))
        colors = {}
        template_colors = ['k', 'r', 'g', 'b']
        for i, my_rate in enumerate(sorted(list(new_data_charge.keys()))):
            colors[my_rate] = template_colors[i]
        # Plot orig
        fig = plt.figure(figsize=[11, 10])
        ax = fig.add_subplot(111)
        for rate in new_data.keys():
            ax.plot(
                [s[0] for s in new_data[rate]],
                [s[1] for s in new_data[rate]],
                label='rate={}'.format(rate),
                c=colors[rate]
            )
            ax.plot(
                [s[0] for s in new_data_charge[rate]],
                [s[1] for s in new_data_charge[rate]],
                c=colors[rate]
            )

        ax.legend()

        savefig("v_curves_original_{}.png".format(name), fit_args)
        plt.close(fig)

        # Plot mod
        fig = plt.figure(figsize=[11, 10])
        ax = fig.add_subplot(111)


        for rate in new_data.keys():
            # delta_v = rate* .2
            delta_v = rate * resistance
            # mult = mult + 0.05
            ax.plot(
                [(s[0] - shift) / shift for s in new_data[rate]],
                [delta_v + s[1] for s in new_data[rate]],
                label='rate={}'.format(rate),
                c=colors[rate]
            )
            ax.plot(
                [(s[0] - shift) / shift for s in new_data_charge[rate]],
                [-delta_v + s[1] for s in new_data_charge[rate]],
                c=colors[rate]
            )

        ax.legend()

        savefig("v_curves_{}.png".format(name), fit_args)
        plt.close(fig)





class Command(BaseCommand):
    def add_arguments(self, parser):

        required_args = [
            "--" + Key.PATH_PLOTS,
            "--" + Key.PATH_V_CURVES,
            "--" + Key.PATH_V_CURVES_META,
        ]

        float_args = {
        }

        int_args = {
        }

        for arg in required_args:
            parser.add_argument(arg, required = True)
        for arg in float_args:
            parser.add_argument(arg, type = float, default = float_args[arg])
        for arg in int_args:
            parser.add_argument(arg, type = int, default = int_args[arg])


    def handle(self, *args, **options):
        v_curves_harvest(options)
