from machine_learning.DegradationModelBlackbox import DegradationModel


class DegradationModelStudent(DegradationModel):

    def __init__(
        self, depth: int, width: int, bottleneck: int, n_sample: int,
        options: dict, cell_dict: dict, random_matrix_q, n_channels = 16,
    ):
        super(DegradationModel, self).__init__(
            depth, width, bottleneck, n_sample, options, cell_dict,
            random_matrix_q, n_channels,
        )
       