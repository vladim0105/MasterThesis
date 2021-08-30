import htm.bindings.algorithms as algos
import torch.nn as nn
import torchvision.models.resnet
from htm.bindings.sdr import SDR

import utils


class Model:
    def __init__(self):
        pass


class CNNLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet.resnet18(pretrained=True)
        # Replace the final layer with an identity layer
        self.model.fc = nn.Identity()

    def forward(self, x):
        x = self.model(x)
        return x


class HTMLayer(nn.Module):
    def __init__(self, shape,
                 columnDimensions=(1000,),
                 potentialPct=0.1,
                 potentialRadius=7,
                 globalInhibition=True,
                 localAreaDensity=0.1,
                 synPermInactiveDec=0.02,
                 synPermActiveInc=0.1,
                 synPermConnected=0.5,
                 boostStrength=100.0,
                 wrapAround=False
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
            wrapAround=wrapAround,
            seed=0
        )
        self.tm = algos.TemporalMemory(columnDimensions=columnDimensions, seed=0, cellsPerColumn=5)


    def forward(self, x):
        encoded_sdr = utils.tensor_to_sdr(x)
        active_sdr = SDR(self.sp.getColumnDimensions())
        # Run the spatial pooler
        self.sp.compute(encoded_sdr, True, active_sdr)
        print(active_sdr.size)
        # Run the temporal memory
        self.tm.compute(active_sdr, learn=True)
        # Extract the predicted SDR and convert it to a tensor
        predicted = utils.sdr_to_tensor(self.tm.getActiveCells())
        print(predicted.size())
        # Extract the anomaly score
        anomaly = self.tm.anomaly

        return predicted, anomaly
