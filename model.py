import htm.bindings.algorithms as algos
import torch.nn as nn
from htm.bindings.sdr import SDR

import utils


class Model:
    def __init__(self):
        pass


class HTMLayer(nn.Module):
    def __init__(self, shape,
                 columnDimensions=1000,
                 potentialPct=0.85,
                 potentialRadius=1000,
                 globalInhibition=True,
                 localAreaDensity=0.04395604395604396,
                 synPermInactiveDec=0.006,
                 synPermActiveInc=0.04,
                 synPermConnected=0.14,
                 boostStrength=3.0,
                 wrapAround=True
                 ):
        super().__init__()
        self.sp = algos.SpatialPooler(
            inputDimensions=shape,
            columnDimensions=columnDimensions,
            potentialPct=potentialPct,
            potentialRadius=potentialRadius,
            globalInhibition=globalInhibition,
            localAreaDensity=localAreaDensity,
            synPermInactiveDec=synPermInactiveDec,
            synPermActiveInc=synPermActiveInc,
            synPermConnected=synPermConnected,
            boostStrength=boostStrength,
            wrapAround=wrapAround
        )
        self.tm = algos.TemporalMemory()

        self.sdr = SDR(self.sp.getColumnDimensions())

    def forward(self, x):
        encoded_sdr = utils.tensor_to_sdr(x)
        # Run the spatial pooler
        self.sp.compute(encoded_sdr, None, self.sdr)
        # Run the temporal memory
        self.tm.compute(self.sdr, learn=True)
        # Extract the predicted SDR and convert it to a tensor
        predicted = utils.sdr_to_tensor(self.tm.getActiveCells())
        # Extract the anomaly score
        anomaly = self.tm.anomaly

        return predicted, anomaly
