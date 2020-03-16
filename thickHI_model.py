
'''
Implement the optically-thick HI model from Braun 2009.
'''

import numpy as np
from lmfit import Model, Parameters, Minimizer, report_fit
import astropy.units as u
import matplotlib.pyplot as plt

sqrt2pi = np.sqrt(2 * np.pi)


def isoturbHI(x, log_Ts, sigma_nt, log_NH, vcent):
    '''
    Eq. 2 and 3 from Braun+09
    '''

    Ts = 10**log_Ts

    # sigma_th =0.093 * Ts**0.5
    sigma_th_sq = 0.093**2 * Ts
    sigmasq = (sigma_th_sq + sigma_nt**2)

    exp_term = np.exp(- 0.5 * ((x - vcent)**2 / sigmasq))
    tau_prefac = (5.49e-19 * 10**log_NH) / (Ts * sqrt2pi * sigmasq**0.5)

    tau = tau_prefac * exp_term

    return Ts * (1 - np.exp(-tau))


def fit_isoturbHI_model(vels, spec, vcent, delta_vcent=5 * u.km / u.s,
                        verbose=True, plot_fit=True,
                        use_emcee=False,
                        emcee_kwargs={}):

    vels = vels.copy().to(u.km / u.s)

    vel_min = (vcent - delta_vcent).to(vels.unit).value
    vel_max = (vcent + delta_vcent).to(vels.unit).value

    # Create the parameter list.
    pfit = Parameters()
    pfit.add(name='log_Ts', value=2., min=1.2, max=3.2)
    pfit.add(name='sigma_nt', value=15., min=2.0, max=31.6)
    pfit.add(name='log_NH', value=21., min=20., max=23.5)
    pfit.add(name='vcent', value=vcent.to(vels.unit).value,
             min=vel_min, max=vel_max)

    def residual(pars, x, data):
        model = isoturbHI(x, pars['log_Ts'], pars['sigma_nt'],
                          pars['log_NH'], pars['vcent'])
        return model - data

    mini = Minimizer(residual, pfit, fcn_args=(vels.value, spec.value))

    if use_emcee:
        out = mini.emcee(**emcee_kwargs)
    else:
        out = mini.leastsq()

    if verbose:
        report_fit(out.params)

    if plot_fit:

        plt.plot(vels.value, spec.value, drawstyle='steps-mid')

        pars = out.params
        model = isoturbHI(vels.value,
                          pars['log_Ts'].value, pars['sigma_nt'].value,
                          pars['log_NH'].value, pars['vcent'].value)
        plt.plot(vels.value, model)

    return out


def isoturbHI_simple(x, Ts, sigma, Tpeak, vcent):
    '''
    Eq. 2 and 3 from Braun+09.

    Since the NHI / (Ts * sigma) correlates the parameters,
    we will instead replace it with a single parameter Tpeak/Ts.
    We expect Tpeak and Ts to remain degenerate for optically-thin
    emission.

    Also, we won't separate thermal and non-thermal line widths.
    The issue is that the thermal contribution could be large, but
    it tau is low, we have no constraints on Ts. If it is constrained
    in the fits, we can try separating after.

    '''

    # sigma_th =0.093 * Ts**0.5
    # sigma_th_sq = 0.093**2 * Ts
    # sigmasq = (sigma_th_sq + sigma_nt**2)
    sigmasq = sigma**2

    exp_term = np.exp(- 0.5 * ((x - vcent)**2 / sigmasq))
    tau_prefac = Tpeak / Ts

    tau = tau_prefac * exp_term

    return Ts * (1 - np.exp(-tau))


def residual(pars, x, data, err):
    model = isoturbHI_simple(x, pars['Ts'], pars['sigma'],
                             pars['Tpeak'], pars['vcent'])
    return (model - data) / err


def fit_isoturbHI_model_simple(vels, spec, vcent, delta_vcent=5 * u.km / u.s,
                               err=None,
                               verbose=True, plot_fit=True,
                               return_model=False,
                               use_emcee=False,
                               emcee_kwargs={}):

    vels = vels.copy().to(u.km / u.s)

    vel_min = (vcent - delta_vcent).to(vels.unit).value
    vel_max = (vcent + delta_vcent).to(vels.unit).value

    # Create the parameter list.
    pfit = Parameters()
    pfit.add(name='Ts', value=100., min=10**1.2, max=8000)
    pfit.add(name='sigma', value=15., min=0.30, max=31.6)  # min v at Ts=16 K
    # pfit.add(name='log_NH', value=21., min=20., max=23.5)
    pfit.add(name='Tpeak', value=np.nanmax(spec.value), min=0)
    pfit.add(name='vcent', value=vcent.to(vels.unit).value,
             min=vel_min, max=vel_max)

    if err is not None:
        fcn_args = (vels.value, spec.value, err.value)
    else:
        fcn_args = (vels.value, spec.value, 1.)

    mini = Minimizer(residual, pfit, fcn_args=fcn_args,
                     maxfev=vels.size * 1000)

    out = mini.leastsq()

    if use_emcee:
        mini = Minimizer(residual, out.params,
                         fcn_args=fcn_args,
                         maxfev=vels.size * 1000)
        out = mini.emcee(**emcee_kwargs)

    if verbose:
        report_fit(out)

    pars = out.params
    model = isoturbHI_simple(vels.value,
                             pars['Ts'].value, pars['sigma'].value,
                             pars['Tpeak'].value, pars['vcent'].value)
    if plot_fit:

        plt.plot(vels.value, spec.value, drawstyle='steps-mid')

        plt.plot(vels.value, model)

    if return_model:
        return out, vels.value, model

    return out
