# -*- coding: utf-8 -*-
#

"""
Spatially extended Epileptor model.

"""
import numpy
from tvb.simulator.models.base import ModelNumbaDfun
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, List, Range, Final, HasTraits, Attr
from tvb.datatypes.equations import SpatialApplicableEquation, FiniteSupportEquation
from tvb.simulator.lab import *
import scipy
from scipy.optimize import fsolve



def get_equilibrium(model, init):
    nvars = len(model.state_variables)
    cvars = len(model.cvar)
    def func(x):
        fx = model.dfun(x.reshape((nvars, 1, 1)),
                        np.zeros((cvars, 1, 1)))
        return fx.flatten()
    x = fsolve(func, init)
    return x

def zero_rows(M, rows):
    diag = scipy.sparse.eye(M.shape[0]).tolil()
    for r in rows:
        diag[r, r] = 0
    return diag.dot(M)

def zero_columns(M, columns):
    diag = scipy.sparse.eye(M.shape[1]).tolil()
    for c in columns:
        diag[c, c] = 0
    return M.dot(diag)

class LaplaceKernel(SpatialApplicableEquation, FiniteSupportEquation):
    """
    A Laplace kernel equation.
    offset: parameter to extend the behaviour of this function
    when spatializing model parameters.
    """

    equation = Final(
        label="Laplace kernel",
        default="amp * (1./(2.*b)) * (exp(-abs(var)/b)) + offset",
        )

    parameters = Attr(
        field_type=dict,
        label="Laplace parameters",
        default=lambda: {"amp": 1.0, "b": 1.0, "offset": 0.0})

# overwrite function of local connectivity to take into account the vertex area
class LocalConnectivity_new(local_connectivity.LocalConnectivity):
    def compute(self) :
        self.log.info("Mapping geodesic distance through the LocalConnectivity.")

        # Start with data being geodesic_distance_matrix, then map it through equation
        # Then replace original data with result...
        self.matrix_gdist.data = self.equation.evaluate(self.matrix_gdist.data)

        # scale by vertex areas and skip homogenization part
        area_mtx = scipy.sparse.diags(self.surface.vertex_areas)
        self.matrix_gdist = self.matrix_gdist * area_mtx
        self.matrix = self.matrix_gdist.tocsr()

# add functionality to compute vertex area to surface
class CorticalSurface_new(surfaces.CorticalSurface) :
    _vertex_areas = None
    _triangle_areas = None
    @property
    def vertex_areas(self):
        """An array specifying the area belonging to the vertices of a surface."""
        if self._vertex_areas is None:
            self._vertex_areas  = self._find_vertex_areas()
        return self._vertex_areas

    def _find_vertex_areas(self):
        """Calculates the area belonging to the vertices of a surface."""
        vertex_areas = numpy.zeros(self.number_of_vertices)
        for i, triangle in enumerate(self.triangles):
            nverts = len(triangle) # This should always be 3 - it is a triangle afterall.
            for j in triangle:
                vertex_areas[j] += self.triangle_areas[i]/nverts
        return vertex_areas


class SpatEpi(ModelNumbaDfun):
    _ui_name = "SpatEpi"
    ui_configurable_parameters = []

    y0 = NArray(
        label="y0",
        default=numpy.array([1]),
        doc="Additive coefficient for the second state variable")

    tau0 = NArray(
        label="tau0",
        default=numpy.array([2857.0]),
        doc="Temporal scaling in the third state variable")

    tau2 = NArray(
        label="tau2",
        default=numpy.array([10.0]),
        doc="Temporal scaling in the fifth state variable")

    x0 = NArray(
        label="x0",
        domain=Range(lo=-3.0, hi=-1.0, step=0.1),
        default=numpy.array([-1.6]),
        doc="Epileptogenicity parameter")

    Iext = NArray(
        label="Iext",
        domain=Range(lo=1.5, hi=5.0, step=0.1),
        default=numpy.array([3.1]),
        doc="External input current to the first population")

    Iext2 = NArray(
        label="Iext2",
        domain=Range(lo=0.0, hi=1.0, step=0.05),
        default=numpy.array([0.45]),
        doc="External input current to the second population")

    gamma = NArray(
        label="gamma",
        default=numpy.array([0.01]),
        doc="Temporal integration scaling"
    )

    gamma11 = NArray(
        label="gamma11",
        default=numpy.array([1.0]),
        doc="Scaling of local connections 1-1"
    )

    gamma22 = NArray(
        label="gamma22",
        default=numpy.array([1.0]),
        doc="Scaling of local connections 2-2"
    )

    gamma12 = NArray(
        label="gamma12",
        default=numpy.array([1.0]),
        doc="Scaling of local connections 1-2"
    )

    gamma_glob = NArray(
        label="gamma_glob",
        default=numpy.array([1.0]),
        doc="Scaling of the global connections"
    )

    theta11 = NArray(
        label="theta11",
        default=numpy.array([-1.1]),
        doc="Firing threshold 1-1"
    )

    theta22 = NArray(
        label="theta22",
        default=numpy.array([-0.5]),
        doc="Firing threshold 2-2"
    )

    theta12 = NArray(
        label="theta12",
        default=numpy.array([-1.1]),
        doc="Firing threshold 1-2"
    )

    tt = NArray(
        label="tt",
        default=numpy.array([1.0]),
        domain=Range(lo=0.001, hi=10.0, step=0.001),
        doc="Time scaling of the whole system")

    state_variable_range = Final(
        label="State variable ranges [lo, hi]",
        default={"u1": numpy.array([-2., 1.]),
                 "u2": numpy.array([-20., 2.]),
                 "s": numpy.array([2.0, 5.0]),
                 "q1": numpy.array([-2., 0.]),
                 "q2": numpy.array([0., 2.]),
                 "g": numpy.array([-1., 1.])},
        doc="Typical bounds on state variables in the Epileptor model."
        )

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=['u1', 'u2', 's', 'q1', 'q2', 'g', 'q1 - u1'],
        default=['q1 - u1', 's'],
        doc="Quantities of the Epileptor available to monitor.",
    )

    state_variables = ['u1', 'u2', 's', 'q1', 'q2', 'g']

    _nvar = 6
    cvar = numpy.array([0], dtype=numpy.int32)

    def dfun(self, x, c, local_coupling=0.0):
        x_ = x.reshape(x.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T

        if type(local_coupling) == float:
            loc11 = self.gamma11 * local_coupling * (0.5 * (numpy.sign(x[0, :, 0] - self.theta11) + 1.0))
            loc22 = self.gamma22 * local_coupling * (0.5 * (numpy.sign(x[3, :, 0] - self.theta22) + 1.0))
            loc12 = self.gamma12 * local_coupling * (0.5 * (numpy.sign(x[0, :, 0] - self.theta12) + 1.0))
        else:
            loc11 = self.gamma11 * local_coupling.dot(0.5 * (numpy.sign(x[0, :, 0] - self.theta11) + 1.0))
            loc22 = self.gamma22 * local_coupling.dot(0.5 * (numpy.sign(x[3, :, 0] - self.theta22) + 1.0))
            loc12 = self.gamma12 * local_coupling.dot(0.5 * (numpy.sign(x[0, :, 0] - self.theta12) + 1.0))

        deriv = _numba_dfun(x_, self.gamma_glob * c_,
                            self.x0, self.Iext, self.Iext2,
                            loc11, loc22, loc12,
                            self.tt, self.y0,
                            self.tau0, self.tau2, self.gamma)
        return deriv.T[..., numpy.newaxis]


@guvectorize([(float64[:],) * 14], '(n),(m)' + ',()'*11 + '->(n)', nopython=True, target='cpu')
def _numba_dfun(y, c_pop, x0, Iext, Iext2, loc11, loc22, loc12, tt, y0, tau0, tau2, gamma, ydot):
    "Gufunc for Epileptor model equations."

    # population 1
    if y[0] < 0.0:
        ydot[0] = y[0]**3 - 3 * y[0]**2
    else:
        ydot[0] = (y[3] - 0.6 * (y[2] - 4.0) ** 2) * y[0]

    ydot[0] = tt[0] * (y[1] - ydot[0] - y[2] + Iext[0] + loc11[0] + c_pop[0])
    ydot[1] = tt[0] * (y0[0] - 5*y[0]**2 - y[1])

    # energy
    if y[2] < 0.0:
        ydot[2] = - 0.1 * y[2] ** 7
    else:
        ydot[2] = 0.0
    ydot[2] = tt[0] * (1.0/tau0[0] * (4.0 * (y[0] - x0[0]) - y[2] + ydot[2]))

    # population 2
    ydot[3] = tt[0] * (-y[4] + y[3] - y[3] ** 3 + Iext2[0] + 2 * y[5] - 0.3 * (y[2] - 3.5) + loc22[0])
    if y[3] < -0.25:
        ydot[4] = 0.0
    else:
        ydot[4] = 6.0 * (y[3] + 0.25)
    ydot[4] = tt[0] * ((-y[4] + ydot[4]) / tau2[0])

    # filter
    ydot[5] = tt[0] * (-0.01 * y[5] + 0.003 * y[0] + 0.01 * loc12[0])