import numpy as np

from pointcloud.config import DATA_PATH

LABELS = {
    0: "Ground",
    1: "High Vegetation",
    2: "Buildings",
    3: "Walls",
    4: "Bridge",
    5: "Parking",
    6: "Rail",
    7: "traffic Roads",
    8: "Street Furniture",
    9: "Cars",
    10: "Footpath",
    11: "Bikes",
    12: "Water",
}


class SensatDataset:
    def __init__(self) -> None:
        self.data_path = DATA_PATH / "sensat_urban"
        self.label_to_names = LABELS
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k in self.label_to_names.keys()])
        self.label_to_index = {v: k for k, v in enumerate(self.label_values)}
