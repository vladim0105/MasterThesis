import htm.bindings.algorithms
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
                 potentialPct=1,
                 potentialRadius=7,
                 globalInhibition=True,
                 localAreaDensity=0.1, # Seems to affect the model sensitivity
                 synPermInactiveDec=0.08,
                 synPermActiveInc=0.14,
                 synPermConnected=0.5,
                 boostStrength=0.0,
                 wrapAround=False,
                 seed=0
                 ):
        super().__init__()
        self.sp = algos.SpatialPooler(
            inputDimensions=(shape[1], shape[0]),
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
            seed=seed
        )
        self.tm = algos.TemporalMemory(columnDimensions=columnDimensions, seed=seed, cellsPerColumn=40, maxNewSynapseCount=40)


    def forward(self, x, learn):
        encoded_sdr = utils.tensor_to_sdr(x)
        active_sdr = SDR(self.sp.getColumnDimensions())
        # Run the spatial pooler
        self.sp.compute(input=encoded_sdr, learn=learn, output=active_sdr)
        # Run the temporal memory
        self.tm.compute(activeColumns=active_sdr, learn=learn)
        # Extract the predicted SDR and convert it to a tensor
        predicted = self.tm.getActiveCells()
        # Extract the anomaly score
        anomaly = self.tm.anomaly

        return predicted, anomaly
