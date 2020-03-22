
'''
Implement the optically-thick HI model from Braun 2009.
'''

import numpy as np
from lmfit import Model, Parameters, Minimizer, report_fit
import astropy.units as u
import matplotlib.pyplot as plt
from spectral_cube import OneDSpectrum

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
    pfit.add(name='Tpeak', value=np.nanmax(spec.value), min=0,
             max=1.1 * np.nanmax(spec.value))
    pfit.add(name='vcent', value=vcent.to(vels.unit).value,
             min=vel_min, max=vel_max)

    finite_mask = np.isfinite(spec.filled_data[:])

    # Some cases are failing with a TypeError due to a lack
    # of data. This really shouldn't happen, but I'll throw in this
    # to catch those cases.
    if finite_mask.sum() <= 4:
        return None

    if err is not None:
        fcn_args = (vels[finite_mask].value, spec[finite_mask].value,
                    err.value)
    else:
        fcn_args = (vels[finite_mask].value, spec[finite_mask].value, 1.)

    try:
        mini = Minimizer(residual, pfit, fcn_args=fcn_args,
                         maxfev=vels.size * 1000,
                         nan_policy='omit')

        out = mini.leastsq()
    except TypeError:
        return None

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


def fit_func_simple(spec, noise_val,
                    vcent=None,
                    min_finite_chan=30,
                    downsamp_factor=1,
                    max_comp=10,
                    delta_vcent=5 * u.km / u.s):
    '''
    Wrapper function to work in parallelized map.
    Fits with fit_isoturbHI_model_simple.
    '''

    params_array = np.zeros((4,)) * np.NaN
    uncerts_array = np.zeros((4,)) * np.NaN

    if np.isfinite(spec.filled_data[:]).sum() < min_finite_chan:

        return params_array, uncerts_array, np.NaN

    # Downsample if needed
    if downsamp_factor > 1:
        new_width = np.diff(spec.spectral_axis)[0] * 2
        new_axis = np.arange(spec.spectral_axis.value[0],
                             spec.spectral_axis.value[-1],
                             new_width.value) * new_width.unit
        spec_conv = OneDSpectrum(spec.filled_data[:],
                                 wcs=spec.wcs)
        spec_conv = spec_conv.spectral_interpolate(new_axis)

        noise_val /= np.sqrt(downsamp_factor)

    elif downsamp_factor < 1:
        raise ValueError("Cannot upsample data.")
    else:
        spec_conv = spec

    # The function converts to km/s. Don't need to do it twice.
    vels = spec.spectral_axis

    # Still trying to catch this weird edge case with <4 finite
    # points to fit to.
    try:
        out = fit_isoturbHI_model_simple(vels, spec, vcent,
                                         delta_vcent=delta_vcent,
                                         err=noise_val,
                                         verbose=False,
                                         plot_fit=False,
                                         return_model=False,
                                         use_emcee=False,)
    except TypeError:
        out = None

    # Too few points to fit
    if out is None:
        return params_array, uncerts_array, np.NaN

    params_array = np.array([par.value for par in out.params.values()])
    uncerts_array = np.array([par.stderr if par.stderr is not None
                              else np.NaN
                              for par in out.params.values()])

    return params_array, uncerts_array, out.bic
