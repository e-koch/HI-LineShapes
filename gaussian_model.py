
'''
Individual spectrum fitter for comparison. There are a few different
approaches here:

- add gaussian components until a min. in AIC or BIC is found. Issues without
good starting guesses
- use AGD from gausspy to make initial guesses by looping over smoothing
lengths; remove components until all the integral over each component
is >N-sigma.

The latter method tends to be better at producing good models for the M31
spectra.
However, it likely overfits in some cases.
'''

import numpy as np
from lmfit import Model, Parameters, Minimizer, report_fit
# from lmfit.models import gaussian
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.convolution import convolve_fft, Gaussian1DKernel

from cube_analysis.spectral_fitting import sample_at_channels

from AGD_decomposer import (AGD_loop, vals_vec_from_lmfit,
                            errs_vec_from_lmfit)


FWHM2SIG = np.sqrt(8 * np.log(2))


def gaussian(x, amp, cent, sigma):
    return amp * np.exp(- (x - cent)**2 / (2 * sigma**2))


def multigaussian(x, pars):

    ncomp = len([par for par in pars if 'amp' in par])

    model = gaussian(x, pars["amp1"], pars['cent1'], pars['sigma1'])

    for nc in range(2, ncomp + 1):
        model += gaussian(x, pars[f"amp{nc}"],
                          pars[f'cent{nc}'], pars[f'sigma{nc}'])

    return model


def residual_single(pars, x, data, err):
    model = gaussian(x, pars['amp'], pars['cent'], pars['sigma'])
    return (model - data) / err


def residual_multigauss(pars, x, x_upsamp, data, err, discrete_fitter):

    if discrete_fitter:
        #  Evaluate model at upsampled values
        model = multigaussian(x_upsamp, pars)

        # This shouldn't be needed... but for some reason the
        # reverse velocity direction isn't working with the cython
        # function.
        if x[1] - x[0] > 0:
            disc_model = sample_at_channels(x, x_upsamp, model)
        else:
            disc_model = sample_at_channels(x[::-1], x_upsamp[::-1],
                                            model[::-1])[::-1]
    else:

        disc_model = multigaussian(x, pars)

    return (disc_model - data) / err


def fit_gaussian(spec,
                 vels=None,
                 vcent=None,
                 err=None,
                 amp_const=None,
                 cent_const=None,
                 sigma_const=None,
                 verbose=True,
                 plot_fit=True,
                 use_emcee=False,
                 emcee_kwargs={}):
    '''
    '''

    if vels is None:
        spec = spec.with_spectral_unit(u.km / u.s)
        vels = spec.spectral_axis

    # Set parameter limits:
    if amp_const is None:
        amp_min = 0.
        amp_max = 1.1 * np.nanmax(spec.value)
    else:
        amp_min = amp_const[0]
        amp_max = amp_const[1]

    if cent_const is None:
        cent_min = vels.value.min() - 0.1 * np.ptp(vels.value)
        cent_max = vels.value.max() + 0.1 * np.ptp(vels.value)
    else:
        cent_min = cent_const[0]
        cent_max = cent_const[1]

    if sigma_const is None:
        sig_min = np.abs(np.diff(vels.value)[0])
        sig_max = 0.3 * np.ptp(vels.value)
    else:
        sig_min = sigma_const[0]
        sig_max = sigma_const[1]

    if vcent is None:
        vcent = np.mean(vels.value)
    else:
        vcent = vcent.to(u.km / u.s).value

    pfit = Parameters()

    pfit.add(name='amp', value=20.,
             min=amp_min, max=amp_max)
    pfit.add(name='cent', value=vcent,
             min=cent_min, max=cent_max)
    pfit.add(name='sigma', value=10.,
             min=sig_min, max=sig_max)

    # valid_data = np.isfinite(spec.filled_data[:])
    # yfit = spec.filled_data[:].value[valid_data]
    # xfit = spec.spectral_axis.value[valid_data]

    yfit = spec.value
    xfit = vels.value

    mini = Minimizer(residual_single, pfit,
                     fcn_args=(xfit, yfit,
                               err if err is not None else 1.))

    out = mini.leastsq()

    if use_emcee:
        mini = Minimizer(residual_single, out.params,
                         fcn_args=(xfit, yfit,
                                   err if err is not None else 1.))
        out = mini.emcee(**emcee_kwargs)

    if plot_fit:

        plt.plot(vels.value, spec.value, drawstyle='steps-mid')

        model = gaussian(vels.value,
                         out.params["amp"],
                         out.params["cent"],
                         out.params["sigma"])
        plt.plot(vels.value, model)
        plt.plot(vels.value, spec.value - model, '--',
                 zorder=-10)
        plt.draw()

    return out


def fit_multigaussian(spec,
                      vcent=None,
                      err=None,
                      max_comp=10,
                      amp_const=None,
                      cent_const=None,
                      sigma_const=None,
                      sigma_init=10.,
                      verbose=True,
                      plot_fit=True,
                      min_delta_BIC=5.,
                      min_sigma_intensity=5,
                      return_model=False,
                      discrete_fitter=False,
                      discrete_oversamp=2):
    '''
    Increase the number of fitted Gaussians to find a minimum
    in AIC or BIC.
    '''

    spec = spec.with_spectral_unit(u.km / u.s)
    vels = spec.spectral_axis

    # Set parameter limits:
    if amp_const is None:
        amp_min = 0.
        amp_max = 1.1 * np.nanmax(spec.filled_data[:].value)
    else:
        amp_min = amp_const[0]
        amp_max = amp_const[1]

    if cent_const is None:
        cent_min = vels.value.min() - 0.1 * np.ptp(vels.value)
        cent_max = vels.value.max() + 0.1 * np.ptp(vels.value)
    else:
        cent_min = cent_const[0]
        cent_max = cent_const[1]

    if sigma_const is None:
        sig_min = np.abs(np.diff(vels.value)[0])
        sig_max = 0.3 * np.ptp(vels.value)
    else:
        sig_min = sigma_const[0]
        sig_max = sigma_const[1]

    if vcent is None:
        vcent = np.mean(vels.value)
    else:
        vcent = vcent.to(u.km / u.s).value

    # Currently assuming all spectra have some signal in them.
    aics = []
    bics = []
    fit_outputs = []

    pfit = Parameters()

    valid_data = np.isfinite(spec.filled_data[:])
    yfit = spec.filled_data[:].value[valid_data]
    xfit = spec.spectral_axis.value[valid_data]

    # Upsample for computing over discrete bins
    chan_width = np.abs(np.diff(vels.value)[0])

    order_sign = 1. if vels[-1] > vels[0] else -1.

    # You really need to rewrite this to be faster.
    assert discrete_oversamp > 1.
    xfit_upsamp = np.linspace(vels.value[0] - order_sign * 0.5 * chan_width,
                              vels.value[-1] + order_sign * 0.5 * chan_width,
                              vels.size * discrete_oversamp)

    for nc in range(1, max_comp + 1):

        if verbose:
            print(f"Now fitting with {nc} components.")

        # Place the centre at the largest positive residual within the bounds.
        if nc > 1:
            tpeak = 20
            vel_peakresid = spec.spectral_axis.value[np.argmax(fit_residual)]
            if vel_peakresid >= cent_min and vel_peakresid <= cent_max:
                v_guess = vel_peakresid
                tpeak = fit_residual[np.argmax(fit_residual)]
            else:
                v_guess = vcent

            if tpeak < amp_min:
                tpeak = 20.

            pfit.add(name=f'amp{nc}', value=tpeak,
                     min=amp_min, max=amp_max)
            pfit.add(name=f'cent{nc}', value=v_guess,
                     min=cent_min, max=cent_max)
        else:
            tpeak = 20.
            pfit.add(name=f'amp{nc}', value=tpeak,
                     min=amp_min, max=amp_max)
            pfit.add(name=f'cent{nc}', value=vcent,
                     min=cent_min, max=cent_max)

        # Setup a minimum relation between the amp. and line width.
        # pfit.add(name=f'integral{nc}',
        #          value=sigma_init * tpeak,
        #          min=err.value * sig_min * min_sigma_intensity,
        #          max=amp_max * sig_max)
        #          # expr=f'amp{nc} * sigma{nc}')

        pfit.add(name=f'sigma{nc}',
                 value=np.random.uniform(sigma_init - 2, sigma_init + 2),
                 # expr=f'integral{nc} / amp{nc}',
                 min=sig_min, max=sig_max,)

        mini = Minimizer(residual_multigauss, pfit,
                         fcn_args=(xfit, xfit_upsamp, yfit,
                                   err if err is not None else 1.,
                                   discrete_fitter),
                         maxfev=vels.size * 1000)

        out = mini.leastsq()
        # out = mini.minimize(method='differential_evolution')

        if not out.success:
            raise ValueError("Fit failed.")

        if verbose:
            report_fit(out)

        model = multigaussian(vels.value, out.params)

        if plot_fit:

            plt.plot(vels.value, spec.filled_data[:], drawstyle='steps-mid')

            plt.plot(vels.value, model)
            for n in range(1, nc + 1):
                plt.plot(vels.value, gaussian(vels.value,
                                              out.params[f"amp{n}"],
                                              out.params[f"cent{n}"],
                                              out.params[f"sigma{n}"]))
            plt.plot(vels.value, spec.filled_data[:].value - model, '--',
                     zorder=-10)
            plt.draw()
            input(f"{nc}?")
            plt.clf()

        if nc > 1:
            if verbose:
                print(f"BIC1: {out.bic}; BIC0: {bics[-1]}")

            if bics[-1] - out.bic < min_delta_BIC:
                if verbose:
                    print(f"Final model with {nc - 1} components.")
                break

        else:
            # n=1 cases tests against a noise model.

            err_norm = err.value if err is not None else 1.

            no_model_rss = np.nansum((yfit / err_norm)**2)
            no_model_bic = yfit.size * np.log(no_model_rss / yfit.size)

            no_fit_model = False

            if verbose:
                print(f"BIC1: {out.bic}; BIC0: {no_model_bic}")

            if no_model_bic - out.bic < min_delta_BIC:
                if verbose:
                    print("No components preferred. Consistent with noise.")
                no_fit_model = True

                bics.append(no_model_bic)

                pfit = Parameters()
                pfit.add('amp1', value=0.)
                pfit.add('cent1', value=0.)
                # pfit.add('integral1', value=0.)
                pfit.add('sigma1', value=0.)

                fit_outputs.append(pfit)

                break

        # Smooth the residual to ensure the peak chosen for
        # the next component
        # is not a single large noise value
        # fit_residual = np.abs(convolve_fft(yfit - model,
        #                             Gaussian1DKernel(3)))
        fit_residual = yfit - model

        aics.append(out.aic)
        bics.append(out.bic)
        fit_outputs.append(out)

        # Exit if max residual is small
        # if fit_residual.max() < 3 * err.value:
        #     if verbose:
        #         print("Max residual below 3-sigma.")
        #     break

        fit_residual = convolve_fft(fit_residual,
                                    Gaussian1DKernel(3))

        # Update parameters for next fit
        # With too few components, we often get bright, extremely wide
        # components
        # To avoid their influence, we will update the component amp and
        # cent, only.
        for ncc in range(1, nc + 1):
            pfit[f'amp{ncc}'].value = out.params[f'amp{ncc}'].value
            pfit[f'cent{ncc}'].value = out.params[f'cent{ncc}'].value

            # if out.params[f'sigma{ncc}'].value > sigma_init:
            #     pfit[f'integral{ncc}'].value = sigma_init * out.params[f'amp{ncc}'].value

            # else:
            #     pfit[f'integral{ncc}'].value = out.params[f'integral{ncc}'].value

            # pfit[f'sigma{ncc}'].value = min(out.params[f'sigma{ncc}'].value,
            #                                 sigma_init)

            # new_sigma = out.params[f'sigma{ncc}'].value

            # if new_sigma < sig_min:

            new_sigma = np.random.uniform(sigma_init - 2, sigma_init + 2)

            pfit[f'sigma{ncc}'].value = max(new_sigma, 2 * sig_min)

        # pfit = out.params.copy()

    if return_model:
        if no_fit_model:
            return fit_outputs[0], vels.value, np.zeros_like(vels.value)

        model = multigaussian(vels.value, fit_outputs[-1].params)
        return fit_outputs[-1], vels.value, model

    if no_fit_model:
        return fit_outputs[0]

    return fit_outputs[-1]


def fit_func(spec, noise_val,
             vcent=None,
             min_finite_chan=30,
             min_3sig_chan=10,
             downsamp_factor=1,
             max_comp=10,
             args=(),
             kwargs={'amp_const': None,
                     'cent_const': None,
                     'sigma_const': [1., 50.]}):
    '''
    Wrapper function to work in parallelized map.
    '''

    params_array = np.zeros((max_comp * 3,)) * np.NaN
    uncerts_array = np.zeros((max_comp * 3,)) * np.NaN

    if np.isfinite(spec.filled_data[:]).sum() < min_finite_chan:

        return params_array, uncerts_array

    # Find all valid pixels. Must have >5 channels above 3 sigma.
    if (spec.filled_data[:].value > 3 * noise_val).sum() < min_3sig_chan:

        return params_array, uncerts_array

    # Otherwise attempt the fit.
    # Remove influence of spectral response function by downsampling
    # by ~2x
    # But having multiple beams make this hard. Spoof since beams are
    # basically the same size.
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

    try:
        out = fit_multigaussian(spec_conv,
                                vcent=vcent,
                                err=noise_val,
                                max_comp=max_comp,
                                verbose=False,
                                plot_fit=False,
                                **kwargs)
    except ValueError:
        # This is coming from channel gaps in the mask.
        # And the data still has nans in it when trying to fit.
        # REALLY shouldn't be an issue for spectra with lots of
        # signal...
        return params_array, uncerts_array

    # Output the parameters to the array.
    ncomp = len(out.params) // 3

    for nc in range(ncomp):
        params_array[3 * nc] = out.params[f"amp{nc + 1}"].value
        params_array[3 * nc + 1] = out.params[f"cent{nc + 1}"].value
        params_array[3 * nc + 2] = out.params[f"sigma{nc + 1}"].value

        uncerts_array[3 * nc] = out.params[f"amp{nc + 1}"].stderr
        uncerts_array[3 * nc + 1] = out.params[f"cent{nc + 1}"].stderr
        uncerts_array[3 * nc + 2] = out.params[f"sigma{nc + 1}"].stderr

    return params_array, uncerts_array


def fit_func_gausspy(spec, noise_val,
                     alphas=[],
                     vcent=None,
                     min_finite_chan=30,
                     downsamp_factor=1,
                     max_comp=10,
                     agd_kwargs={}):
    '''
    Wrapper function to work in parallelized map.
    Uses AGD_loop to identify component guesses, then does a fit, iteratively
    removing insignificant components and refitting until all components are
    needed in the fit.
    '''

    params_array = np.zeros((max_comp * 3,)) * np.NaN
    uncerts_array = np.zeros((max_comp * 3,)) * np.NaN

    if np.isfinite(spec.filled_data[:]).sum() < min_finite_chan:

        return params_array, uncerts_array, np.NaN

    # Otherwise attempt the fit.
    # Remove influence of spectral response function by downsampling
    # by ~2x
    # But having multiple beams make this hard. Spoof since beams are
    # basically the same size.
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

    # try:
    multigauss_fit, vels, multigauss_fit_model = \
        AGD_loop(spec,
                 noise_val.value * np.ones_like(spec.value),
                 alphas,
                 **agd_kwargs)
    # except ValueError:
    #     # This is coming from channel gaps in the mask.
    #     # And the data still has nans in it when trying to fit.
    #     # REALLY shouldn't be an issue for spectra with lots of
    #     # signal...
    #     return params_array, uncerts_array, np.NaN

    # If no component is found in the fit, return empty arrays
    if multigauss_fit is None:
        return params_array, uncerts_array, np.NaN

    # Output the parameters to the array.
    ncomp = len(multigauss_fit.params) // 3

    params = vals_vec_from_lmfit(multigauss_fit.params)
    uncerts = errs_vec_from_lmfit(multigauss_fit.params)

    # print(multigauss_fit.params)
    # print(f"Fit parameters are: {params}")

    for nc in range(ncomp):
        # Amp
        params_array[3 * nc] = params[nc]
        # sigma (from FWHM)
        # print(f"FWHM is {params[ncomp + nc]}")
        params_array[3 * nc + 2] = params[ncomp + nc] / FWHM2SIG
        # Cent
        params_array[3 * nc + 1] = params[2 * ncomp + nc]

        uncerts_array[3 * nc] = uncerts[nc]
        uncerts_array[3 * nc + 2] = uncerts[ncomp + nc] / FWHM2SIG
        uncerts_array[3 * nc + 1] = uncerts[2 * ncomp + nc]

    # print(f"In the output array: {params_array}")

    return params_array, uncerts_array, multigauss_fit.bic
