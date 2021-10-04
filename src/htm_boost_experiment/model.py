from typing import TypedDict, NamedTuple

import htm.bindings.algorithms
import htm.bindings.algorithms as algos
import torch.nn as nn
import torchvision.models.resnet
from htm.bindings.sdr import SDR
from htm.algorithms.anomaly_likelihood import AnomalyLikelihood
import utils


class SpatialPoolerArgs:
    def __init__(self):
        self.inputDimensions = (0,)
        self.columnDimensions = (0,)
        self.potentialPct = 0.01
        self.potentialRadius = 100
        self.globalInhibition = True
        self.localAreaDensity = 0.01  # Seems to affect the model sensitivity
        self.synPermInactiveDec = 0.08
        self.synPermActiveInc = 0.14
        self.synPermConnected = 0.5
        self.boostStrength = 0.0
        self.stimulusThreshold = 1
        self.wrapAround = True
        self.seed = 0



class TemporalMemoryArgs:
    def __init__(self):
        self.columnDimensions = (0,)
        self.seed = 0
        self.initialPermanence=0.21
        self.predictedSegmentDecrement=0.01
        self.connectedPermanence=0.7
        self.permanenceIncrement = 0.01
        self.permanenceDecrement = 0.01

class Model:
    def __init__(self):
        pass


class CNNLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet.resnet18(pretrained=True).cuda()
        # Replace the final layer with an identity layer
        self.model.fc = nn.Identity().cuda()

    def forward(self, x):
        x = self.model(x)
        return x


class HTMLayer(nn.Module):
    def __init__(self, sp_args: SpatialPoolerArgs, tm_args: TemporalMemoryArgs):
        super().__init__()
        self.sp = algos.SpatialPooler(**sp_args.__dict__)
        self.tm = algos.TemporalMemory(**tm_args.__dict__)
        self.anom = AnomalyLikelihood(learningPeriod=100)

    def forward(self, encoded_sdr, learn):
        active_sdr = SDR(self.sp.getColumnDimensions())
        # Run the spatial pooler
        self.sp.compute(input=encoded_sdr, learn=learn, output=active_sdr)
        # Run the temporal memory
        self.tm.compute(activeColumns=active_sdr, learn=learn)
        # Extract the predicted SDR and convert it to a tensor
        predicted = self.tm.getActiveCells()
        # Extract the anomaly score
        anomaly = self.tm.anomaly
        if learn:
            anomaly_likelihood = 0
        else:
            anomaly_likelihood = self.anom.anomalyProbability(0, anomaly)
        return predicted, anomaly, anomaly_likelihood
