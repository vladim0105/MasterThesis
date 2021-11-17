from typing import TypedDict, NamedTuple

import htm.bindings.algorithms
import htm.bindings.algorithms as algos
from htm.bindings.sdr import SDR


class SpatialPoolerArgs:
    def __init__(self):
        self.inputDimensions = (0,)
        self.columnDimensions = (0,)
        self.potentialPct = 0.01
        """The percent of the inputs, within a column's
            potential radius, that a column can be connected to. If set to
            1, the column will be connected to every input within its
            potential radius. This parameter is used to give each column a
            unique potential pool when a large potentialRadius causes
            overlap between the columns. At initialization time we choose
            ((2*potentialRadius + 1)^(# inputDimensions) * potentialPct)
            input bits to comprise the column's potential pool."""
        self.potentialRadius = 100
        """This parameter determines the extent of the
            input that each column can potentially be connected to. This
            can be thought of as the input bits that are visible to each
            column, or a 'receptive field' of the field of vision. A large
            enough value will result in global coverage, meaning
            that each column can potentially be connected to every input
            bit. This parameter defines a square (or hyper square) area: a
            column will have a max square potential pool with sides of
            length (2 * potentialRadius + 1)."""
        self.globalInhibition = True
        """If true, then during inhibition phase the
            winning columns are selected as the most active columns from the
            region as a whole. Otherwise, the winning columns are selected
            with respect to their local neighborhoods. Global inhibition
            boosts performance significantly but there is no topology at the
            output."""

        self.localAreaDensity = 0.01
        """The desired density of active columns within
            a local inhibition area (the size of which is set by the
            internally calculated inhibitionRadius, which is in turn
            determined from the average size of the connected potential
            pools of all columns). The inhibition logic will insure that at
            most N columns remain ON within a local inhibition area, where
            N = localAreaDensity * (total number of columns in inhibition
            area). """
        self.synPermInactiveDec = 0.08
        """The amount by which the permanence of an
            inactive synapse is decremented in each learning step."""
        self.synPermActiveInc = 0.14
        """The amount by which the permanence of an
            active synapse is incremented in each round."""
        self.synPermConnected = 0.5
        self.boostStrength = 0.0
        """A number greater or equal than 0, used to
            control boosting strength. No boosting is applied if it is set to 0.
            The strength of boosting increases as a function of boostStrength.
            Boosting encourages columns to have similar activeDutyCycles as their
            neighbors, which will lead to more efficient use of columns. However,
            too much boosting may also lead to instability of SP outputs."""
        self.stimulusThreshold = 1
        self.wrapAround = True
        self.dutyCyclePeriod = 1000
        """The period used to calculate duty cycles.
            Higher values make it take longer to respond to changes in
            boost. Shorter values make it potentially more unstable and
            likely to oscillate."""
        self.seed = 0


class TemporalMemoryArgs:
    def __init__(self):
        self.columnDimensions = (0,)
        self.seed = 0

        self.initialPermanence = 0.21
        """Initial permanence of a new synapse."""

        self.predictedSegmentDecrement = 0.01
        self.connectedPermanence = 0.7

        self.permanenceIncrement = 0.01
        """Amount by which permanences of synapses are incremented during learning."""

        self.permanenceDecrement = 0.01


class HTMLayer:
    def __init__(self, sp_args: SpatialPoolerArgs, tm_args: TemporalMemoryArgs):
        super().__init__()
        self.sp = algos.SpatialPooler(**sp_args.__dict__)
        self.tm = algos.TemporalMemory(**tm_args.__dict__)

    def __call__(self, encoded_sdr, learn):
        active_sdr = SDR(self.sp.getColumnDimensions())
        # Run the spatial pooler
        self.sp.compute(input=encoded_sdr, learn=learn, output=active_sdr)
        self.sp.getSynPermConnected()
        # Run the temporal memory
        self.tm.compute(activeColumns=active_sdr, learn=learn)
        # Extract the predicted SDR and convert it to a tensor
        predicted = self.tm.getActiveCells()
        # Extract the anomaly score
        anomaly = self.tm.anomaly
        return predicted, anomaly, active_sdr
