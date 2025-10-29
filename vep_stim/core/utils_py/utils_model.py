"""
Spatially extended 3D Epileptor model.

"""
import numpy
import numpy as np
from tvb.simulator.models.base import ModelNumbaDfun
from numba import guvectorize, float64, vectorize
from tvb.basic.neotraits.api import NArray, List, Range, Final, HasTraits, Attr
from tvb.datatypes.equations import SpatialApplicableEquation, FiniteSupportEquation
from tvb.simulator.lab import *
import scipy
from scipy.optimize import fsolve

class Spat3DEpi(ModelNumbaDfun):
    _ui_name = "Spat3DEpi"
    ui_configurable_parameters = []

    # y0 = NArray(
    #     label="y0",
    #     default=numpy.array([1]),
    #     doc="Additive coefficient for the second state variable")
    a = NArray(
        label=":math:`a`",
        default=numpy.array([1.0]),
        doc="Coefficient of the cubic term in the first state variable")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([3.0]),
        doc="Coefficient of the squared term in the first state variabel")

    c = NArray(
        label=":math:`c`",
        default=numpy.array([1.0]),
        doc="Additive coefficient for the second state variable, \
            called :math:`y_{0}` in Jirsa paper")

    d = NArray(
        label=":math:`d`",
        default=numpy.array([5.0]),
        doc="Coefficient of the squared term in the second state variable")

    r = NArray(
        label=":math:`r`",
        domain=Range(lo=0.0, hi=0.001, step=0.00005),
        default=numpy.array([0.00035]),
        doc="Temporal scaling in the third state variable, \
            called :math:`1/\\tau_{0}` in Jirsa paper")

    s = NArray(
        label=":math:`s`",
        default=numpy.array([4.0]),
        doc="Linear coefficient in the third state variable")

    x0 = NArray(
        label=":math:`x_0`",
        domain=Range(lo=-3.0, hi=-1.0, step=0.1),
        default=numpy.array([-1.6]),
        doc="Epileptogenicity parameter")

    Iext = NArray(
        label=":math:`I_{ext}`",
        domain=Range(lo=1.5, hi=5.0, step=0.1),
        default=numpy.array([3.1]),
        doc="External input current to the first population")

    # tau0 = NArray(
    #     label="tau0",
    #     default=numpy.array([2857.0]),
    #     doc="Temporal scaling in the third state variable")
    #
    # tau2 = NArray(
    #     label="tau2",
    #     default=numpy.array([10.0]),
    #     doc="Temporal scaling in the fifth state variable")
    # gamma = NArray(
    #     label="gamma",
    #     default=numpy.array([0.01]),
    #     doc="Temporal integration scaling"
    # )

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

    Kvf = NArray(
        label=":math:`K_{vf}`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=4.0, step=0.5),
        doc="Coupling scaling on a very fast time scale.")

    Kf = NArray(
        label=":math:`K_{f}`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=4.0, step=0.5),
        doc="Correspond to the coupling scaling on a fast time scale.")

    Ks = NArray(
        label=":math:`K_{s}`",
        default=numpy.array([0.0]),
        domain=Range(lo=-4.0, hi=4.0, step=0.1),
        doc="Permittivity coupling, that is from the fast time scale toward the slow time scale")

    tt = NArray(
        label="tt",
        default=numpy.array([1.0]),
        domain=Range(lo=0.001, hi=10.0, step=0.001),
        doc="Time scaling of the whole system")

    modification = NArray(
        dtype=bool,
        label=":math:`modification`",
        default=numpy.array([False]),
        doc="When modification is True, then use nonlinear influence on z. \
        The default value is False, i.e., linear influence.")

    Istim = NArray(
        label=":math:`I_{ext}`",
        domain=Range(lo=0, hi=50.0, step=0.1),
        default=numpy.array([0.]),
        doc="External input current from the stimuli applied to mysterious variables. ~BD")

    n_stim = NArray(
        label=":math:`n_stim`",
        default=numpy.array([0.]),
        doc="Counter for the number of stimulations applied to the model")

    state_variable_range = Final(
        default={
            "x1": numpy.array([-2., 1.]),
            "y1": numpy.array([-20., 2.]),
            "z": numpy.array([2.0, 5.0]),
            "m": numpy.array([-16.0, 6.0])
        },
        label="State variable ranges [lo, hi]",
        doc="Typical bounds on state variables in the Epileptor model.")


    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=('x1', 'y1', 'z', 'm'),
        default=('x1', 'z'),
        doc="Quantities of the Epileptor available to monitor.",
    )

    state_variables = ('x1', 'y1', 'z', 'm')
    _nvar = 4
    cvar = numpy.array([0], dtype=numpy.int32)
    # cvar = numpy.array([0,2,3], dtype=numpy.int32)  # should these not be constant Attr's?
    # TODO what is cvar ? document it

    def dfun(self, x, c, local_coupling=0.0, stimulus=0.0):

        if isinstance(stimulus, numpy.ndarray):  # and stimulus.any() > 0.0:
            # TODO fixme this is wrong here: stimulus[0][nr_region][0]

            ## 1 method of defining m(t)
            # self.slope = self.slope + stimulus[0][0][0]/2 # divide by two cause dfun is called twice in the HeunDeterministic integrator

            ## 2 method of defining m(t)
            # self.Istim[0] = stimulus[0][0][0]
            self.Istim = stimulus[0, :, 0]  # stimulus shape (nr_var, nr_regions, 1)

        elif isinstance(stimulus, float) and stimulus != 0.0:
            print("Error in stimulus argument in 3D epileptor model.")
            return

        x_ = x.reshape(x.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T

        if type(local_coupling) == float:
            loc11 = self.gamma11 * local_coupling * (0.5 * (numpy.sign(x[0, :, 0] - self.theta11) + 1.0))
            loc22 = self.gamma22 * local_coupling * (0.5 * (numpy.sign(x[3, :, 0] - self.theta22) + 1.0))
            # loc12 = self.gamma12 * local_coupling * (0.5 * (numpy.sign(x[0, :, 0] - self.theta12) + 1.0))
        else:
            loc11 = self.gamma11 * local_coupling.dot(0.5 * (numpy.sign(x[0, :, 0] - self.theta11) + 1.0))
            loc22 = self.gamma22 * local_coupling.dot(0.5 * (numpy.sign(x[3, :, 0] - self.theta22) + 1.0))
            # loc12 = self.gamma12 * local_coupling.dot(0.5 * (numpy.sign(x[0, :, 0] - self.theta12) + 1.0))

        deriv = _numba_dfun(x_, c_,
                         self.x0, self.Iext, self.a, self.b, self.tt, self.Kvf,
                         self.c, self.d, self.r, self.Ks, self.Kf, self.modification, self.Istim,
                         self.n_stim, loc11, loc22)
        return deriv.T[..., numpy.newaxis]


@vectorize([float64(float64, float64)])
def heaviside_impl(x1, x2):
    if x1 < 0:
        return 0.0
    elif x1 > 0:
        return 1.0
    else:
        return x2

@guvectorize([(float64[:],) * 19], '(n),(m)' + ',()'*16 + '->(n)', nopython=True)
def _numba_dfun(y, c_pop, x0, Iext, a, b, tt, Kvf, c, d, r, Ks, Kf, modification, Istim, n_stim, loc11, loc22, ydot):
    "Gufunc for Hindmarsh-Rose-Jirsa Epileptor model equations."

    c_pop1 = c_pop[0] #x1
    c_pop2 = c_pop[1] #z
    c_pop3 = c_pop[2] #m

    # population 1
    if y[0] < 0.0:
        ydot[0] = - a[0] * y[0] ** 2 + b[0] * y[0]
    else:
        #ydot[0] = slope[0] + 0.6 * (y[2] - 4.0) ** 2
        ydot[0] =  y[3] + 0.6 * (y[2] - 4.0) ** 2
    ydot[0] = tt[0] * (y[1] - y[2] + Iext[0] + 50 * Istim[0] + Kvf[0] * c_pop1 + ydot[0] * y[0] + loc11[0])
    ydot[1] = tt[0] * (c[0] - d[0] * y[0] ** 2 - y[1])

    # energy
    if y[2] < 0.0:
        ydot[2] = - 0.1 * y[2] ** 7
    else:
        ydot[2] = 0.0
    if modification[0]:
        h =  x0[0] + 3/(1 + numpy.exp(-(y[0]+0.5)/0.1))
    else:
        h = 4 * (y[0] - x0[0]) + ydot[2]
    ydot[2] = tt[0] * (r[0] * (h - y[2] + Ks[0] * c_pop1))

    ydot[3] = tt[0] * r[0] * (-y[3] + 25 * abs(Istim[0]) + Kf[0] * c_pop3 + loc22[0])

    # fixme nstim here could eventually be removed, but we would be modifying all x0s accross all epileptors
    # if n_stim[0] > 0:
    x0[0] = heaviside_impl(y[3]-1.8, 0) - 2.5
