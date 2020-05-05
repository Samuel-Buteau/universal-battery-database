from django.db import models
import base64
import numpy
import pickle
import datetime
from django.utils import timezone
from django.db.models import Q, Max,Min
import matplotlib.pyplot as plt
import filename_database.models
import os
from io import BytesIO


def id_dict_from_id_list(id_list):
    n = len(id_list)
    id_dict = {}
    for i in range(n):
        id_dict[id_list[i]] = i

    return id_dict



def get_files_for_barcode(barcode):
    return CyclingFile.objects.filter(database_file__deprecated=False).filter(database_file__valid_metadata__barcode=barcode)

def clamp(a, x, b):
    x = min(x, b)
    x = max(x, a)
    return x


def make_voltage_grid(min_v, max_v, n_samples, my_barcodes):
    if n_samples < 2:
        n_samples = 2
    all_cycs = Cycle.objects.filter(
        discharge_group__barcode__in = my_barcodes,
        valid_cycle = True
    )
    my_max = max(
        all_cycs.aggregate(Max("chg_maximum_voltage"))[
            "chg_maximum_voltage__max"
        ],
        all_cycs.aggregate(Max("dchg_maximum_voltage"))[
            "dchg_maximum_voltage__max"
        ]
    )
    my_min = min(
        all_cycs.aggregate(Min("chg_minimum_voltage"))[
            "chg_minimum_voltage__min"
        ],
        all_cycs.aggregate(Min("dchg_minimum_voltage"))[
            "dchg_minimum_voltage__min"
        ]
    )

    my_max = clamp(min_v, my_max, max_v)
    my_min = clamp(min_v, my_min, max_v)

    delta = (my_max - my_min) / float(n_samples - 1)
    return numpy.array([my_min + delta * float(i) for i in range(n_samples)])


def make_current_grid(min_c, max_c, n_samples, my_barcodes):
    if n_samples < 2:
        n_samples = 2
    all_cycs = Cycle.objects.filter(
        discharge_group__barcode__in = my_barcodes,
        valid_cycle = True
    )
    my_max = max(abs(
        all_cycs.aggregate(Max("chg_maximum_current"))[
            "chg_maximum_current__max"
        ]),
        abs(all_cycs.aggregate(Max("dchg_maximum_current"))[
            "dchg_maximum_current__max"
        ])
    )

    my_min = min(
        abs(all_cycs.aggregate(Min("chg_minimum_current"))[
            "chg_minimum_current__min"
        ]),

        abs(all_cycs.aggregate(Min("dchg_minimum_current"))[
            "dchg_minimum_current__min"
        ])
    )

    my_max = clamp(min_c, my_max, max_c)
    my_min = clamp(min_c, my_min, max_c)

    my_max = current_to_log_current(my_max)
    my_min = current_to_log_current(my_min)
    delta = (my_max - my_min) / float(n_samples - 1.)

    return numpy.array([my_min + delta * float(i) for i in range(n_samples)])


def current_to_log_current(current):
    return numpy.log(abs(current) + 1e-5)



#def temperature_to_arrhenius(temp):
#   return numpy.exp(-1. / (temp + 273))

def make_sign_grid():
    return numpy.array([1., -1.])

def make_temperature_grid(min_t, max_t, n_samples, my_barcodes):
    if n_samples < 2:
        n_samples = 2

    my_files = CyclingFile.objects.filter(database_file__deprecated=False).filter(database_file__valid_metadata__barcode__in=my_barcodes)
    my_max = my_files.aggregate(Max("database_file__valid_metadata__temperature"))[
            "database_file__valid_metadata__temperature__max"
        ]
    my_min = my_files.aggregate(Min("database_file__valid_metadata__temperature"))[
        "database_file__valid_metadata__temperature__min"
    ]

    my_max = clamp(min_t, my_max, max_t)
    if my_max < 55.:
        my_max = 55.

    my_min = clamp(min_t, my_min, max_t)
    if my_min > 20.:
        my_min = 20.


    delta = (my_max - my_min) / float(n_samples - 1)

    return numpy.array([my_min + delta * float(i) for i in range(n_samples)])


def plot_barcode(barcode, path_to_plots = None, lower_cycle=None, upper_cycle=None, show_invalid=False, vertical_barriers=None, list_all_options=None, figsize = None):
    files_barcode = CyclingFile.objects.filter(
        database_file__deprecated=False,
        database_file__valid_metadata__barcode=barcode).order_by("database_file__last_modified")
    if figsize is None:
        figsize = [5., 5.]

    colors = ["k", "r", "b", "g", "c", "m", "y"]
    rates = {0.05:"C/20", 0.5:"C/2", 1.:"1C", 2.:"2C", 3.:"3C"}

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(3.)
    #TODO: unify this with the overview plot
    for counter, cycle_group in enumerate(get_discharge_groups_from_barcode(barcode)):
        if show_invalid:
            dat = [[True, ".", 100],
                   [False, "x", 5]]
        else:
            dat = [[True, ".", 100]]

        for d in dat:

            all_cycles = cycle_group.cycle_set.filter(valid_cycle=d[0])
            q_curves= []
            for f in files_barcode:
                offset_cycle = f.database_file.valid_metadata.start_cycle
                if lower_cycle is None and upper_cycle is None:
                    cycles = all_cycles.filter(cycling_file=f)
                else:
                    cycles = all_cycles.filter(cycling_file=f).filter(cycle_number__range=(lower_cycle-offset_cycle, upper_cycle-offset_cycle))
                if cycles.exists():
                    q_curves.append( numpy.array([
                        [cyc.cycle_number + offset_cycle, cyc.dchg_total_capacity]
                        for cyc in cycles.order_by("cycle_number")])
                    )

            if not any([len(q_curve) != 0 for q_curve in q_curves]):
                continue
            q_curve = numpy.concatenate(q_curves, axis=0)

            if d[0]:
                chg = cycle_group.end_rate_prev
                found = False
                for k in rates.keys():
                    if abs(k - chg)/k < 0.2:
                        chg = rates[k]
                        found = True
                        break
                if not found:
                    chg = "{:1.2f}".format(chg)
                dchg= cycle_group.constant_rate
                found = False
                for k in rates.keys():
                    if abs(k - dchg)/k < 0.2:
                        dchg = rates[k]
                        found = True
                        break
                if not found:
                    dchg = "{:1.2f}".format(dchg)

                ax.scatter(q_curve[:, 0], q_curve[:, 1], c=colors[counter],
                           marker=d[1],
                           s=d[2],
                           label="{}:{}".format(
                               # counter,
                               chg, dchg))
            else:
                ax.scatter(q_curve[:, 0], q_curve[:, 1], c=colors[counter],
                           marker=d[1],
                           s=d[2]
                           )

    file_colors = ["k", "c", "b", "g", "r", "k"]

    file_leg = []
    if len(files_barcode) >= 1:
        for f_i, f in enumerate(files_barcode):
            if show_invalid:
                min_cycle = Cycle.objects.filter(cycling_file=f).aggregate(Min("cycle_number"))[
                                "cycle_number__min"] + f.database_file.valid_metadata.start_cycle
                max_cycle = Cycle.objects.filter(cycling_file=f).aggregate(Max("cycle_number"))[
                                "cycle_number__max"] + f.database_file.valid_metadata.start_cycle

            else:
                min_cycle = Cycle.objects.filter(cycling_file=f, valid_cycle=True).aggregate(Min("cycle_number"))[
                                "cycle_number__min"] + f.database_file.valid_metadata.start_cycle
                max_cycle = Cycle.objects.filter(cycling_file=f, valid_cycle=True).aggregate(Max("cycle_number"))[
                                "cycle_number__max"] + f.database_file.valid_metadata.start_cycle

            if lower_cycle is not None:
                if min_cycle < lower_cycle:
                    min_cycle = lower_cycle - .5
                if min_cycle > upper_cycle:
                    continue

            if upper_cycle is not None:
                if max_cycle > upper_cycle:
                    max_cycle = upper_cycle + .5
                if max_cycle < lower_cycle:
                    continue

            bla = plt.axvspan(min_cycle, max_cycle, ymin=.05 * (1+f_i), ymax=.05 * (2 + f_i),
                              facecolor=file_colors[f_i],
                              alpha=0.1
                              )
            file_leg.append( (bla,"File {} Last Modif: {}-{}-{}. Size: {}KB".format(
                f_i,
                f.database_file.last_modified.year,
                f.database_file.last_modified.month,
                f.database_file.last_modified.day,
                int(f.database_file.filesize/1024)) ))

    if vertical_barriers is not None:
        for index_set_i in range(len(vertical_barriers) + 1):
            col = ["1.", ".1"][index_set_i % 2]
            if index_set_i == 0 and len(vertical_barriers) > 0:
                min_x, max_x = (lower_cycle - 0.5, vertical_barriers[0])
            elif index_set_i == 0 and len(vertical_barriers) == 0:
                min_x, max_x = (lower_cycle - 0.5, upper_cycle + 0.5)
            elif index_set_i == len(vertical_barriers):
                min_x, max_x = (vertical_barriers[-1], upper_cycle + 0.5)
            else:
                min_x, max_x = (vertical_barriers[index_set_i - 1], vertical_barriers[index_set_i])
            print(min_x, max_x)
            ax.axvspan(min_x, max_x, facecolor=col,
                       alpha=0.1)
            plt.text(0.9 * min_x + .1 * max_x, .99 * ax.get_ylim()[0] + .01 * ax.get_ylim()[1],
                     list_all_options[index_set_i], size=18)

        for index_set_i in range(len(list_all_options) - 1):
            plt.axvline(x=vertical_barriers[index_set_i], color="k", linestyle="--")

    ax.tick_params(direction="in", length=7, width=2, labelsize=11, bottom=True, top=True, left=True,
                   right=True)
    leg1 = ax.legend(loc= "upper right")

    if len(file_leg) > 0:
        if list_all_options is None:
             loc="lower left"
        else:
             loc="upper left"
        leg2 = ax.legend([x[0] for x in file_leg], [x[1] for x in file_leg], loc=loc)
        ax.add_artist(leg1)
    plt.tight_layout(pad=0.)

    if path_to_plots is not None:
        plt.savefig(
            os.path.join(path_to_plots, "Initial_{}.png".format(barcode)))
        plt.close(fig)
        return None
    else:
        if vertical_barriers is None:
            dpi = 50
        else:
            dpi = 300
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=dpi)
        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8").replace("\n", "")
        buf.close()
        return image_base64


#TODO(sam): use common mechanism as in compile_dataset/ml_smoothing for ordering
#TODO(sam): set default color rules in the UI.
def get_discharge_groups_from_barcode(barcode):
    return list(DischargeCycleGroup.objects.filter(barcode=barcode).order_by("constant_rate"))




class CyclingFile(models.Model):
    database_file = models.OneToOneField(filename_database.models.DatabaseFile, on_delete=models.CASCADE)
    import_time = models.DateTimeField(default=datetime.datetime(1970, 1, 1))
    process_time = models.DateTimeField(default=datetime.datetime(1970, 1, 1))


    def get_cycles_array(self, fil=Q()):
        return numpy.array(
            [
                (
                    cyc.id,
                    cyc.cycle_number,
                    cyc.chg_total_capacity,
                    cyc.chg_average_voltage,
                    cyc.chg_minimum_voltage,
                    cyc.chg_maximum_voltage,
                    cyc.chg_average_current_by_capacity,
                    cyc.chg_average_current_by_voltage,
                    cyc.chg_minimum_current,
                    cyc.chg_maximum_current,
                    cyc.chg_duration,


                    cyc.dchg_total_capacity,
                    cyc.dchg_average_voltage,
                    cyc.dchg_minimum_voltage,
                    cyc.dchg_maximum_voltage,
                    cyc.dchg_average_current_by_capacity,
                    cyc.dchg_average_current_by_voltage,
                    cyc.dchg_minimum_current,
                    cyc.dchg_maximum_current,
                    cyc.dchg_duration,



                )


                for cyc in self.cycle_set.filter(fil).order_by("cycle_number")
            ],
            dtype = [
                ("id", int),
                ("cycle_number", int),
                ("chg_total_capacity", float),
                ("chg_average_voltage", float),
                ("chg_minimum_voltage", float),
                ("chg_maximum_voltage", float),
                ("chg_average_current_by_capacity", float),
                ("chg_average_current_by_voltage", float),
                ("chg_minimum_current", float),
                ("chg_maximum_current", float),
                ("chg_duration", float),

                ("dchg_total_capacity", float),
                ("dchg_average_voltage", float),
                ("dchg_minimum_voltage", float),
                ("dchg_maximum_voltage", float),
                ("dchg_average_current_by_capacity", float),
                ("dchg_average_current_by_voltage", float),
                ("dchg_minimum_current", float),
                ("dchg_maximum_current", float),
                ("dchg_duration", float),

            ]
        )



class ChargeCycleGroup(models.Model):
    barcode = models.IntegerField()
    constant_rate = models.FloatField()
    end_rate = models.FloatField()
    end_rate_prev = models.FloatField()
    end_voltage = models.FloatField()
    end_voltage_prev = models.FloatField()

class DischargeCycleGroup(models.Model):
    barcode = models.IntegerField()
    constant_rate = models.FloatField()
    end_rate = models.FloatField()
    end_rate_prev = models.FloatField()
    end_voltage = models.FloatField()
    end_voltage_prev = models.FloatField()

class Cycle(models.Model):
    cycling_file = models.ForeignKey(CyclingFile, on_delete=models.CASCADE)
    cycle_number = models.IntegerField()
    def get_offset_cycle(self):
        """
        Really important that this only be called when the file is known to be valid!!!
        """
        return float(self.cycling_file.database_file.valid_metadata.start_cycle) + self.cycle_number

    def get_temperature(self):
        """
        Really important that this only be called when the file is known to be valid!!!
        """
        return float(self.cycling_file.database_file.valid_metadata.temperature)


    charge_group = models.ForeignKey(ChargeCycleGroup, null=True, on_delete=models.SET_NULL)
    discharge_group = models.ForeignKey(DischargeCycleGroup, null=True, on_delete=models.SET_NULL)

    valid_cycle = models.BooleanField(default=True)

    processed = models.BooleanField(default=False)

    chg_total_capacity = models.FloatField(null=True)
    chg_average_voltage = models.FloatField(null=True)
    chg_minimum_voltage = models.FloatField(null=True)
    chg_maximum_voltage = models.FloatField(null=True)
    chg_average_current_by_capacity = models.FloatField(null=True)
    chg_average_current_by_voltage = models.FloatField(null=True)
    chg_minimum_current = models.FloatField(null=True)
    chg_maximum_current = models.FloatField(null=True)
    chg_duration = models.FloatField(null=True)

    dchg_total_capacity = models.FloatField(null=True)
    dchg_average_voltage = models.FloatField(null=True)
    dchg_minimum_voltage = models.FloatField(null=True)
    dchg_maximum_voltage = models.FloatField(null=True)
    dchg_average_current_by_capacity = models.FloatField(null=True)
    dchg_average_current_by_voltage = models.FloatField(null=True)
    dchg_minimum_current = models.FloatField(null=True)
    dchg_maximum_current = models.FloatField(null=True)
    dchg_duration = models.FloatField(null=True)


    def get_first_discharge_step(self):

        steps = self.step_set.filter(step_type__contains="CC_DChg").order_by("cycle__cycle_number", "step_number")

        if len(steps) == 0:
            return None
        else:
            return steps[0]


    def get_first_charge_step(self):
        steps = self.step_set.filter(step_type__contains="CC_Chg").order_by(
            "cycle__cycle_number",
            "step_number"
        )
        if len(steps) == 0:
            steps = self.step_set.filter(step_type__contains="CCCV_Chg").order_by(
                "cycle__cycle_number",
                "step_number"
            )
        if len(steps) == 0:
            return None
        else:
            return steps[0]

class Step(models.Model):
    cycle = models.ForeignKey(Cycle, on_delete=models.CASCADE)
    step_number = models.IntegerField()
    step_type = models.CharField(max_length=200)
    start_time = models.DateTimeField()
    second_accuracy = models.BooleanField()

    total_capacity = models.FloatField(null=True)
    average_voltage = models.FloatField(null=True)
    minimum_voltage = models.FloatField(null=True)
    maximum_voltage = models.FloatField(null=True)
    average_current_by_capacity = models.FloatField(null=True)
    average_current_by_voltage = models.FloatField(null=True)
    minimum_current = models.FloatField(null=True)
    maximum_current = models.FloatField(null=True)
    duration = models.FloatField(null=True)

    constant_voltage = models.FloatField(null=True)
    end_voltage = models.FloatField(null=True)
    end_voltage_prev = models.FloatField(null=True)
    constant_current = models.FloatField(null=True)
    end_current = models.FloatField(null=True)
    end_current_prev = models.FloatField(null=True)

    """
     numpy list, float, voltages (V)
     numpy list, float, currents (mA)
     numpy list, float, capacities (mAh)
     numpy list, float, absolute times (h), delta t between now and the first cycle.
    """
    v_c_q_t_data = models.BinaryField(null=True)

    def get_v_c_q_t_data(self):
        return pickle.loads(base64.decodebytes(self.v_c_q_t_data))

    def set_v_c_q_t_data(self, v_c_q_t_data):
        np_bytes = pickle.dumps(v_c_q_t_data)
        np_base64 = base64.b64encode(np_bytes)
        self.v_c_q_t_data = np_base64





