'''
Extended stimulation for neural fields with subcortical regions
'''
import numpy

from tvb.datatypes import surfaces, volumes, connectivity, equations
from tvb.basic.neotraits.api import HasTraits, NArray, Attr
from tvb.datatypes.patterns import SpatioTemporalPattern, SpatialPattern

class StimuliSurface_new(SpatioTemporalPattern):
    """
    A spatio-temporal pattern defined in a Surface DataType.
    It includes the list of focal points.
    """

    # surface = Attr(field_type=surfaces.CorticalSurface, label="Surface")
    spatial = Attr(field_type=equations.DiscreteEquation,
                 label="Spatial Equation",
                 default=equations.DiscreteEquation())

    weight = NArray(label="scaling")

    @property
    def weight_array(self):
        """
        Wrap weight List into a Numpy array, as it is requested by the simulator.
        """
        return numpy.array(self.weight)[:, numpy.newaxis]

    def configure_space(self, region_mapping=None):
        """
        Do necessary preparations in order to use this stimulus.
        NOTE: this was previously done in simulator configure_stimuli() method.
        It no needs to be used in stimulus viewer also.
        """
        # dis_shp = (self.surface.number_of_vertices, numpy.size(self.focal_points_surface))
        # TODO: When this was in Simulator it was number of nodes, using surface vertices
        # breaks surface simulations which include non-cortical regions.

        distance = self.weight_array
        super(StimuliSurface_new, self).configure_space(distance)


class TI_SpatioTemporalPattern(SpatioTemporalPattern):
    """
    Combine space and time equations direcly into pattern
    of shape [spatial indices, temporal indices].
    This is only done for TI when computing temporal equations of tDCS.
    """

    temporal = Attr(field_type=equations.TemporalApplicableEquation, label="Temporal Equation")
    # space must be shape (x, 1); time must be shape (1, t)
    # time = None
    _temporal_pattern = None
    spatial = Attr(field_type=equations.DiscreteEquation,
                 label="Spatial Equation",
                 default=equations.DiscreteEquation())

    weight = NArray(label="scaling")
    _pattern = NArray(label="spatio temporal pattern")

    @property
    def spatiotemporal_pattern(self):
        """
        Return a discrete representation of the spatio-temporal pattern.
        """
        return numpy.array(self._pattern)

    #
    # def summary_info(self):
    #     """ Extend the base class's summary dictionary. """
    #     summary = super(SpatioTemporalPatternTI, self).summary_info()
    #     summary["Temporal equation"] = self.temporal.__class__.__name__
    #     summary["Temporal parameters"] = self.temporal.parameters
    #     return summary

    # @property
    # def temporal_pattern(self):
    #     """
    #     Return a discrete representation of the temporal pattern.
    #     """
    #     return self._temporal_pattern
    #
    # def configure_time(self, time):
    #     """
    #     Stores the time vector, physical units (ms), as an attribute of the
    #     spatio-temporal pattern and uses it to generate the temporal pattern
    #     vector.
    #     """
    #     self.time = time
    #     # Generate a discrete representation of the temporal pattern.
    #     self._temporal_pattern = numpy.reshape(self.temporal.evaluate(self.time), (1, -1))
    #
    @property
    def weight_array(self):
        """
        Wrap weight List into a Numpy array, as it is requested by the simulator.
        """
        return numpy.array(self.weight)[:, numpy.newaxis]

    def configure_space(self, region_mapping=None):
        """
        Do necessary preparations in order to use this stimulus.
        NOTE: this was previously done in simulator configure_stimuli() method.
        It no needs to be used in stimulus viewer also.
        """
        distance = self.weight_array
        super(TI_SpatioTemporalPattern, self).configure_space(distance)

    def __call__(self, temporal_indices=None, spatial_indices=None):
        """
        The temporal pattern vector, set by the configure_time method, is
        combined with the spatial pattern vector, set by the configure_space
        method, to form a spatiotemporal pattern.

        Called with a single time index as an argument, the spatial pattern at
        that point in time is returned. This is the standard usage within a
        simulation where the current simulation time point is retrieved.

        Called without any arguments, by default a big array representing the
        entire spatio-temporal pattern is returned. While this may be useful for
        visualisation, say of region level spatio-temporal patterns, care should
        be taken as when surfaces are considered the returned array can be
        potentially quite large.
        """
        pattern = None#self.spatiotemporal_pattern
        if temporal_indices is not None and spatial_indices is None:
            pattern = self.spatiotemporal_pattern[:, temporal_indices]
            # pattern = self._spatial_pattern * self._temporal_pattern[0, temporal_indices]
        # elif temporal_indices is None and spatial_indices is None:
        #     pattern = self._spatial_pattern * self._temporal_pattern
        # elif temporal_indices is not None and spatial_indices is not None:
        #     pattern = self._spatial_pattern[spatial_indices, 0] * self._temporal_pattern[0, temporal_indices]
        # elif temporal_indices is None and spatial_indices is not None:
        #     pattern = self._spatial_pattern[spatial_indices, 0] * self._temporal_pattern
        else:
            print('ENTERED HERE ERROR')
            self.log.error("%s: Well, that shouldn't be possible..." % repr(self))
        return pattern
