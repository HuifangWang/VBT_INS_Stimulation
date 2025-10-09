from tvb.basic.neotraits.api import HasTraits, Attr, NArray, Float
from tvb.simulator.integrators import Integrator, IntegratorStochastic
import numpy


class HeunDeterministicAdapted(Integrator):
    """
    It is a simple example of a predictor-corrector method. It is also known as
    modified trapezoidal method, which uses the Euler method as its predictor.
    And it is also a implicit integration scheme.

    Here we adapt it so that the stimulation affects only the slope variable
    of the 3D Epileptor model we are using.
    """

    _ui_name = "Heun"

    def scheme(self, X, dfun, coupling, local_coupling, stimulus):
        r"""
        From [1]_:

        .. math::
            X_{n+1} &= X_n + dt (dX(t_n, X_n) +
                                 dX(t_{n+1}, \tilde{X}_{n+1})) / 2 \\
            \tilde{X}_{n+1} &= X_n + dt dX(t_n, X_n)

        cf. Equation 1.11, page 283.

        X  (4, 162, 1)
        m_dx_tn  (4, 162, 1)
        stimulus  (4, 162, 1)
        coupling  (2, 162, 1)

        """
        # import pdb; pdb.set_trace()
        # m_dx_tn = dfun(X, coupling, local_coupling)
        m_dx_tn = dfun(X, coupling, local_coupling, stimulus)
        # inter = X + self.dt * (m_dx_tn + stimulus)
        inter = X + self.dt * (m_dx_tn)
        if self.state_variable_boundaries is not None:
            self.bound_state(inter)
        if self.clamped_state_variable_values is not None:
            self.clamp_state(inter)

        # dX = (m_dx_tn + dfun(inter, coupling, local_coupling)) * self.dt / 2.0
        dX = (m_dx_tn + dfun(inter, coupling, local_coupling, stimulus)) * self.dt / 2.0

        # X_next = X + dX + self.dt * stimulus
        X_next = X + dX
        if self.state_variable_boundaries is not None:
            self.bound_state(X_next)
        if self.clamped_state_variable_values is not None:
            self.clamp_state(X_next)

        return X_next


class HeunStochasticAdapted(IntegratorStochastic):
    """
    It is a simple example of a predictor-corrector method. It is also known as
    modified trapezoidal method, which uses the Euler method as its predictor.

    """

    _ui_name = "Stochastic Heun Adapted"

    def scheme(self, X, dfun, coupling, local_coupling, stimulus):
        """
        From [2]_:

        .. math::
            X_i(t) = X_i(t-1) + dX(X_i(t)/2 + dX(X_i(t-1))) dt + g_i(X) Z_1

        in our case, :math:`noise = Z_1`

        See page 1180.

        """
        noise = self.noise.generate(X.shape)
        noise_gfun = self.noise.gfun(X)
        if (noise_gfun.shape != (1,) and noise.shape[0] != noise_gfun.shape[0]):
            msg = str("Got shape %s for noise but require %s."
                      " You need to reconfigure noise after you have changed your model." % (
                          noise_gfun.shape, (noise.shape[0], noise.shape[1])))
            print(msg)
            raise Exception(msg)

        # m_dx_tn = dfun(X, coupling, local_coupling)
        m_dx_tn = dfun(X, coupling, local_coupling, stimulus)

        noise *= noise_gfun

        # inter = X + self.dt * m_dx_tn + noise + self.dt * stimulus
        inter = X + self.dt * m_dx_tn + noise
        if self.state_variable_boundaries is not None:
            self.bound_state(inter)
        if self.clamped_state_variable_values is not None:
            self.clamp_state(inter)

        # dX = (m_dx_tn + dfun(inter, coupling, local_coupling)) * self.dt / 2.0
        dX = (m_dx_tn + dfun(inter, coupling, local_coupling, stimulus)) * self.dt / 2.0

        # X_next = X + dX + noise + self.dt * stimulus
        X_next = X + dX + noise
        if self.state_variable_boundaries is not None:
            self.bound_state(X_next)
        if self.clamped_state_variable_values is not None:
            self.clamp_state(X_next)

        return X_next