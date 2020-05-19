import matplotlib.pyplot as plt

import numpy
from django.core.management.base import BaseCommand

from cycling.models import *
from plot import *
from Key import *
import csv


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
            return None, None, None
        key = row[0]
        electrode_string = row[1]
        rate_string = row[2]

        if not (electrode_string == "cathode" or electrode_string == "anode"):
            return None, None, None

        try:
            rate = float(rate_string)
        except:
            return None, None, None

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
            return None, None
        q_string = row[0]
        v_string = row[1]

        try:
            q = float(q_string)
            v = float(v_string)
            return q, v

        except:
            return None, None

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
                    datas[electrode][rate].append((q, v))

        datas[electrode][rate] = sorted(
            datas[electrode][rate], key = lambda x: x[1],
        )

        datas[electrode][rate] = numpy.array(datas[electrode][rate])

    # Plot
    fig = plt.figure(figsize = [11, 10])
    ax = fig.add_subplot(111)
    for electrode in datas.keys():
        for rate in datas[electrode].keys():
            shift = numpy.min([s[0] for s in datas[electrode][rate]])
            datas[electrode][rate][:, 0] = datas[electrode][rate][:, 0] - shift
            ax.plot(
                [s[0] for s in datas[electrode][rate]],
                [s[1] for s in datas[electrode][rate]],
                label = "{}, rate={}".format(electrode, rate)
            )

    ax.legend()

    savefig("v_curves_ground_truth.png", fit_args)
    plt.close(fig)

    with open(
        os.path.join(fit_args[Key.PATH_V_CURVES], "v_curves.file"), "wb",
    ) as file:
        pickle.dump(datas, file, protocol = pickle.HIGHEST_PROTOCOL)


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
        if len(row) < 4:
            return None, None, None, None
        key = row[0]
        cap_string = row[1]
        c_rate_string = row[2]
        name = row[3]

        try:
            c_rate = float(c_rate_string)
            cap = float(cap_string)
            return key, cap, c_rate, name
        except:
            return None, None, None, None

    metadatas = []
    with open(v_curves_meta) as csvfile:
        my_reader = csv.reader(csvfile)
        for row in my_reader:
            key, cap, c_rate, name = process_metadata_row(row)
            if key is not None:
                metadatas.append((key, cap, c_rate, name))

    print(metadatas)

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
    for key, cap, c_rate, name in metadatas:
        datas[name] = {}
        with open(os.path.join(v_curves_folder, key)) as csvfile:
            my_reader = csv.reader(csvfile)
            for row in my_reader:
                current, voltage, capacity, step_count = process_row(row)
                if not step_count in datas[name].keys():
                    datas[name][step_count] = []

                datas[name][step_count].append((current, voltage, capacity))

        for step_count in datas[name].keys():
            avg_current = numpy.average([d[0] for d in datas[name][step_count]])
            idealized_current = avg_current / c_rate
            new_data = [[s[2] / cap, s[1]] for s in datas[name][step_count]]

            with open(
                os.path.join(
                    v_curves_folder,
                    "HALF_CELL_V_CURVE_ID={}_RATE={:.2f}_STEP={}.csv".format(
                        name, idealized_current, step_count,
                    ),
                ),
                "w", newline = "",
            ) as outfile:
                my_writer = csv.writer(outfile)
                for row in new_data:
                    my_writer.writerow([str(x) for x in row])


class Command(BaseCommand):
    def add_arguments(self, parser):

        required_args = [
            Key.PATH_V_CURVES,
            Key.PATH_V_CURVES_META,
        ]

        float_args = {}

        int_args = {}

        for arg in required_args:
            parser.add_argument("--" + arg, required = True)
        for arg in float_args:
            parser.add_argument(arg, type = float, default = float_args[arg])
        for arg in int_args:
            parser.add_argument(arg, type = int, default = int_args[arg])

        parser.add_argument(
            "--" + "mode",
            required = True,
            choices = ["split_csv", "register_csv", "legacy_compile"],
        )
        parser.add_argument("--" + Key.PATH_PLOTS)

    def handle(self, *args, **options):
        if options["mode"] == "split_csv":
            v_curves_complex(options)
        elif options["mode"] == "register_csv":
            print("not implemented yet:", options["mode"])
            # v_curves_harvest(options)
        else:
            print("not implemented yet:", options["mode"])
