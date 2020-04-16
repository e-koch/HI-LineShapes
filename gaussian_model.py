
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

import os
from tqdm import tqdm
import numpy as np
from lmfit import Parameters, Minimizer, report_fit
# from lmfit.models import gaussian
from astropy.io import fits
from spectral_cube import SpectralCube, Projection
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

    if ncomp == 0:
        return np.zeros_like(x)

    model = gaussian(x, pars["amp1"], pars['cent1'], pars['sigma1'])

    for nc in range(2, ncomp + 1):
        model += gaussian(x, pars[f"amp{nc}"],
                          pars[f'cent{nc}'], pars[f'sigma{nc}'])

    return model


def multigaussian_nolmfit(x, pars):

    ncomp = pars.size // 3

    model = gaussian(x, pars[0], pars[1], pars[2])

    if ncomp == 1:
        return model

    for nc in range(1, ncomp):
        model += gaussian(x, pars[3 * nc],
                          pars[3 * nc + 1],
                          pars[3 * nc + 2])

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


def refit_multigaussian(spec, init_params,
                        vels=None,
                        vcent=None,
                        err=None,
                        amp_const=None,
                        cent_const=None,
                        sigma_const=None,
                        component_sigma=5.,
                        nchan_component_sigma=3.,
                        discrete_fitter=False):
    '''
    Given full set of initial parameters, refit the spectrum.
    '''

    # if len(init_params) < 3:
    #     raise ValueError("Less than 3 initial parameters given.")

    if vels is None:
        spec = spec.with_spectral_unit(u.m / u.s)
        vels = spec.spectral_axis

    chan_width = np.abs(np.diff(vels)[:1]).value

    if err is None:
        # Can't remove components if we don't know what the error is
        def comp_sig(amp, sigma):
            return True

    else:

        def comp_sig(amp, sigma):
            return (amp * sigma) / \
                (err * chan_width * nchan_component_sigma)

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
        sig_max = 0.5 * np.ptp(vels.value) / 2.35
    else:
        sig_min = sigma_const[0]
        sig_max = sigma_const[1]

    # Create the fit parameter
    pars = Parameters()

    for i in range(len(init_params) // 3):
        pars.add(name=f'amp{i + 1}', value=init_params[3 * i],
                 min=amp_min, max=amp_max)
        pars.add(name=f'cent{i + 1}', value=init_params[3 * i + 1],
                 min=cent_min, max=cent_max)
        pars.add(name=f'sigma{i + 1}', value=init_params[3 * i + 2],
                 min=sig_min, max=sig_max)

    valid_data = np.isfinite(spec.filled_data[:])
    yfit = spec.filled_data[:].value[valid_data]
    xfit = vels.value[valid_data]

    if discrete_fitter:
        vels = xfit.copy()

        # Upsample for computing over discrete bins
        chan_width = np.abs(np.diff(vels.value)[0])

        order_sign = 1. if vels[-1] > vels[0] else -1.

        # You really need to rewrite this to be faster.
        discrete_oversamp = 4
        assert discrete_oversamp > 1.
        xfit_upsamp = np.linspace(vels.value[0] - order_sign * 0.5 * chan_width,
                                  vels.value[-1] + order_sign * 0.5 * chan_width,
                                  vels.size * discrete_oversamp)
    else:
        xfit_upsamp = None

    comp_deletes = []

    while True:

        mini = Minimizer(residual_multigauss, pars,
                         fcn_args=(xfit, xfit_upsamp, yfit,
                                   err if err is not None else 1.,
                                   discrete_fitter),
                         maxfev=vels.size * 1000)

        out = mini.leastsq()

        # Testing null model. Nothing to check
        if len(pars) == 0:
            break

        params_fit = [value.value for value in out.params.values()]
        params_fit = np.array(params_fit)

        # Make sure all components are significant
        component_signif = comp_sig(params_fit[::3],
                                    params_fit[2::3])

        if np.all(component_signif >= component_sigma):
            break

        comp_del = np.argmin(component_signif)
        comp_deletes.append(comp_del)

        remain_comps = np.arange(len(init_params) // 3)

        for dcomp in comp_deletes:
            remain_comps = np.delete(remain_comps, dcomp)

        pars = Parameters()

        for i, comp in enumerate(remain_comps):

            pars.add(name=f'amp{i + 1}', value=init_params[3 * i],
                     min=amp_min, max=amp_max)
            pars.add(name=f'cent{i + 1}', value=init_params[3 * i + 1],
                     min=cent_min, max=cent_max)
            pars.add(name=f'sigma{i + 1}', value=init_params[3 * i + 2],
                     min=sig_min, max=sig_max)

    return out


def neighbourhood_fit_comparison(cube_name, params_name, chunk_size=80000,
                                 diff_bic=10, err_map=None,
                                 use_ncomp_check=True,
                                 reverse_direction=False):
    '''
    Lazily account for spatial continuity by checking the fit
    of each pixel relative to its neighbours.
    If delta BIC < -10, will attempt to refit the pixel with that
    neighbour's model.
    If there is a difference in the number of components, will also
    attempt to refit with a different number of components initialized
    to the neighbour's fit.

    This is done in serial. Which isn't ideal but accounts for other pixels
    first being updated.
    '''

    with fits.open(params_name, memmap=False, mode='denywrite') as params_hdu:

        params_array = params_hdu[0].data
        uncerts_array = params_hdu[1].data
        bic_array = params_hdu[2].data

    ncomp_array = np.isfinite(params_array).sum(0) // 3

    cube = SpectralCube.read(cube_name)
    assert cube.shape[1:] == bic_array.shape

    if err_map is not None:

        if hasattr(err_map, 'value'):
            err_map = err_map.value.copy()

        assert err_map.shape == bic_array.shape

    # Number of pixels with valid fits.
    yposn, xposn = np.where(np.isfinite(bic_array) & (ncomp_array > 0))

    if reverse_direction:
        yposn = yposn[::-1]
        xposn = xposn[::-1]

    yshape, xshape = bic_array.shape

    basename = os.path.basename(cube_name)

    for i, (y, x) in tqdm(enumerate(zip(yposn, xposn)),
                          ascii=True,
                          desc=f"Rev. fit for: {basename[:15]}",
                          total=yposn.size):

        err = None if err_map is None else err_map[y, x]

        # Reload cube to release memory
        if i % chunk_size == 0 and i != 0:
            del cube
            cube = SpectralCube.read(cube_name)

        # Slice out 3x3 neighbourhood of y, x
        ymin = max(0, y - 1)
        ymax = min(yshape, y + 2)
        xmin = max(0, x - 1)
        xmax = min(xshape, x + 2)

        bic_neighb = bic_array[ymin:ymax, xmin:xmax].copy()
        ncomp_neighb = ncomp_array[ymin:ymax, xmin:xmax]
        bic_neighb[ncomp_neighb == 0] = np.NaN

        orig_posn = np.where(bic_neighb == bic_array[y, x])
        orig_index = (orig_posn[0][0], orig_posn[1][0])

        # If no valid neighbours, skip:
        if np.isfinite(bic_neighb).sum() == 1:
            continue

        # Condition 1: delta BIC
        if np.nanmax(bic_array[y, x] - bic_neighb) >= diff_bic:
            argmin = np.unravel_index(np.nanargmin(bic_neighb), (3, 3))

            yneighb = y + (argmin[0] - 1)
            xneighb = x + (argmin[1] - 1)

            # Refit
            spec = cube[:, y, x]

            init_params = params_array[:, yneighb, xneighb]
            init_params = init_params[np.isfinite(init_params)]

            # assert init_params.size > 0

            out_new = \
                refit_multigaussian(spec, init_params,
                                    vels=None,
                                    vcent=None,
                                    err=err,
                                    amp_const=None,
                                    cent_const=None,
                                    sigma_const=None,
                                    discrete_fitter=False)

            if bic_array[y, x] - out_new.bic >= diff_bic:
                # Update the parameter array
                params_array[:, y, x] = np.NaN
                params_array[:len(out_new.params), y, x] = \
                    [val.value for val in out_new.params.values()]

                uncerts_array[:, y, x] = np.NaN
                uncerts_array[:len(out_new.params), y, x] = \
                    [val.stderr if val.stderr is not None else np.NaN
                     for val in out_new.params.values()]

                bic_array[y, x] = out_new.bic

            continue

        # Condition 2: Change in # of components
        elif ((ncomp_array[y, x] - ncomp_neighb) != 0).any():

            if not use_ncomp_check:
                continue

            # We'll do this twice with the largest and smallest number of
            # components.
            # The lowest BIC fit will be kept.

            spec = cube[:, y, x]

            max_comp = ncomp_neighb.max()
            min_bic = bic_neighb[ncomp_neighb == max_comp].min()

            posn = np.where(bic_neighb == min_bic)
            argmax = (posn[0][0], posn[1][0])

            # Skip max if this is the original spectrum
            if argmax == orig_index:
                maxcomp_bic = bic_array[y, x]
            else:

                yneighb = y + (argmax[0] - 1)
                xneighb = x + (argmax[1] - 1)

                # Refit

                init_params_max = params_array[:, yneighb, xneighb]
                init_params_max = init_params_max[np.isfinite(init_params_max)]

                assert init_params_max.size > 0

                out_new_max = \
                    refit_multigaussian(spec, init_params_max,
                                        vels=None,
                                        vcent=None,
                                        err=err,
                                        amp_const=None,
                                        cent_const=None,
                                        sigma_const=None,
                                        discrete_fitter=False)

                maxcomp_bic = out_new_max.bic

            min_comp = ncomp_neighb[ncomp_neighb > 0].min()
            min_bic = bic_neighb[ncomp_neighb == min_comp].min()

            posn = np.where(bic_neighb == min_bic)
            argmin = (posn[0][0], posn[1][0])

            # Skip max if this is the original spectrum
            if argmin == orig_index:
                mincomp_bic = bic_array[y, x]
            else:

                yneighb = y + (argmin[0] - 1)
                xneighb = x + (argmin[1] - 1)

                # Refit

                init_params_min = params_array[:, yneighb, xneighb]
                init_params_min = init_params_min[np.isfinite(init_params_min)]

                assert init_params_min.size > 0

                out_new_min = \
                    refit_multigaussian(spec, init_params_min,
                                        vels=None,
                                        vcent=None,
                                        err=err,
                                        amp_const=None,
                                        cent_const=None,
                                        sigma_const=None,
                                        discrete_fitter=False)

                mincomp_bic = out_new_min.bic

            diff_maxcomp = (bic_array[y, x] - maxcomp_bic) >= diff_bic
            diff_mincomp = (bic_array[y, x] - mincomp_bic) >= diff_bic

            # Original fit is good.
            if not diff_mincomp and not diff_maxcomp:
                continue
            # Both are better than original. Take best.
            elif diff_mincomp and diff_maxcomp:
                if maxcomp_bic < mincomp_bic:
                    out_new = out_new_max
                else:
                    out_new = out_new_min
            # Update to max component fit.
            elif diff_maxcomp:
                out_new = out_new_max
            # Update to min component fit.
            else:
                out_new = out_new_min

            # Update the parameter array
            params_array[:, y, x] = np.NaN
            params_array[:len(out_new.params), y, x] = \
                [val.value for val in out_new.params.values()]

            uncerts_array[:, y, x] = np.NaN
            uncerts_array[:len(out_new.params), y, x] = \
                [val.stderr if val.stderr is not None else np.NaN
                 for val in out_new.params.values()]

            bic_array[y, x] = out_new.bic

        # Otherwise no refit is needed.
        else:
            continue

    del cube

    cube = SpectralCube.read(cube_name)

    # Grab the celestial header
    spat_header = cube[0].header
    del cube

    # Return a combined HDU that can be written out.
    params_hdu = fits.PrimaryHDU(params_array, spat_header.copy())
    params_hdu.header['BUNIT'] = ("", "Gaussian fit parameters")

    uncerts_hdu = fits.ImageHDU(uncerts_array, spat_header.copy())
    uncerts_hdu.header['BUNIT'] = ("", "Gaussian fit uncertainty")

    bics_hdu = fits.ImageHDU(bic_array, spat_header.copy())
    bics_hdu.header['BUNIT'] = ("", "Gaussian fit BIC")

    hdu_all = fits.HDUList([params_hdu, uncerts_hdu, bics_hdu])

    del params_array
    del uncerts_array
    del bic_array

    return hdu_all


def save_fitmodel(cube_name, params_name,
                  output_name,
                  chunk_size=80000,
                  save_sep_components=False):
    '''
    Output the fit model as a FITS cube. Optionally with
    extensions for every component.
    '''

    params_hdu = fits.open(params_name)

    params_array = params_hdu[0].data
    bic_array = params_hdu[2].data

    ncomp_array = np.isfinite(params_array).sum(0) // 3

    cube = SpectralCube.read(cube_name)
    assert cube.shape[1:] == bic_array.shape

    vels = cube.spectral_axis.to(u.m / u.s)
    vels_val = vels.value

    # Number of pixels with valid fits.
    yposn, xposn = np.where(np.isfinite(bic_array) & (ncomp_array > 0))

    yshape, xshape = bic_array.shape

    basename = os.path.basename(cube_name)

    # Create the output cube.
    from cube_analysis.io_utils import create_huge_fits
    create_huge_fits(output_name, cube.header, fill_nan=True)

    hdu = fits.open(output_name, mode='update')

    for i, (y, x) in tqdm(enumerate(zip(yposn, xposn)),
                          ascii=True,
                          desc=f"Model eval. for: {basename[:15]}",
                          total=yposn.size):

        # Reload cube to release memory
        if i % chunk_size == 0:
            hdu.flush()
            hdu.close()
            del hdu
            hdu = fits.open(output_name, mode='update')

        pars = params_array[:, y, x][np.isfinite(params_array[:, y, x])]
        hdu[0].data[:, y, x] = multigaussian_nolmfit(vels_val, pars)

    hdu.flush()
    hdu.close()


def remove_mw_components(params_name,
                         vcent_name,
                         delta_v=80 * u.km / u.s,
                         mwhi_mask=None,
                         return_mwcomps=True):
    '''
    All components get fit, including MW contamination.
    This is mostly an issue for M31 in C and D configs.
    This function only keeps components that are likely
    to be part of the galaxy.
    '''

    delta_v = delta_v.to(u.m / u.s)

    with fits.open(params_name) as params_hdu:

        params_array = params_hdu[0].data
        uncert_array = params_hdu[1].data
        bic_array = params_hdu[2].data

    ncomp_array = np.isfinite(params_array).sum(0) // 3

    max_comp = ncomp_array.max()

    if return_mwcomps:
        mwparams_array = np.ones((max_comp * 3,) +
                                 params_array.shape[1:]) * np.NaN
        mwuncert_array = np.ones((max_comp * 3,) +
                                 params_array.shape[1:]) * np.NaN

    vcent = Projection.from_hdu(fits.open(vcent_name))
    assert vcent.shape == bic_array.shape

    vcent = vcent.to(u.m / u.s)

    new_params_array = np.zeros_like(params_array) * np.NaN
    new_uncert_array = np.zeros_like(params_array) * np.NaN

    yposn, xposn = np.where(np.isfinite(bic_array) & (ncomp_array > 0))

    for i, (y, x) in enumerate(zip(yposn, xposn)):

        vcent_val = vcent.value[y, x]
        vmin = vcent_val - delta_v.value
        vmax = vcent_val + delta_v.value

        vfits = params_array[1::3, y, x][:ncomp_array[y, x]]

        valid_comps = np.logical_and(vfits >= vmin, vfits <= vmax)

        valids = np.where(valid_comps)[0]

        for j, comp in enumerate(valids):
            new_params_array[3 * j, y, x] = params_array[3 * comp, y, x]
            new_params_array[3 * j + 1, y, x] = params_array[3 * comp + 1, y, x]
            new_params_array[3 * j + 2, y, x] = params_array[3 * comp + 2, y, x]

            new_uncert_array[3 * j, y, x] = uncert_array[3 * comp, y, x]
            new_uncert_array[3 * j + 1, y, x] = uncert_array[3 * comp + 1, y, x]
            new_uncert_array[3 * j + 2, y, x] = uncert_array[3 * comp + 2, y, x]

        if return_mwcomps:
            mwvalids = np.where(~valid_comps)[0]
            for j, comp in enumerate(mwvalids):

                mwparams_array[3 * j, y, x] = params_array[3 * comp, y, x]
                mwparams_array[3 * j + 1, y, x] = params_array[3 * comp + 1, y, x]
                mwparams_array[3 * j + 2, y, x] = params_array[3 * comp + 2, y, x]

                mwuncert_array[3 * j, y, x] = uncert_array[3 * comp, y, x]
                mwuncert_array[3 * j + 1, y, x] = uncert_array[3 * comp + 1, y, x]
                mwuncert_array[3 * j + 2, y, x] = uncert_array[3 * comp + 2, y, x]

    # Return a combined HDU that can be written out.
    params_hdu = fits.PrimaryHDU(new_params_array, vcent.header.copy())
    params_hdu.header['BUNIT'] = ("", "Gaussian fit parameters")

    uncerts_hdu = fits.ImageHDU(new_uncert_array, vcent.header.copy())
    uncerts_hdu.header['BUNIT'] = ("", "Gaussian fit uncertainty")

    # Will need to update the BIC eventually...
    # bics_hdu = fits.ImageHDU(bic_array, vcent.header.copy())
    # bics_hdu.header['BUNIT'] = ("", "Gaussian fit BIC")

    hdu_all = fits.HDUList([params_hdu, uncerts_hdu])
    # hdu_all = fits.HDUList([params_hdu, uncerts_hdu, bics_hdu])

    if return_mwcomps:
        mwparams_hdu = fits.PrimaryHDU(mwparams_array, vcent.header.copy())
        mwparams_hdu.header['BUNIT'] = ("", "Gaussian fit parameters")

        mwuncerts_hdu = fits.ImageHDU(mwuncert_array, vcent.header.copy())
        mwuncerts_hdu.header['BUNIT'] = ("", "Gaussian fit uncertainty")

        hdu_mw = fits.HDUList([mwparams_hdu, mwuncerts_hdu])

        return hdu_all, hdu_mw

    return hdu_all


def remove_faint_components(params_name,
                            noise_val,
                            return_faintcomps=True,
                            min_lwidth_sub=25 * u.km / u.s,
                            max_amp_sub_sigma=3.,
                            logic_func=np.logical_or):
    '''
    Remove Gaussian components that are faint and/or wide from
    a cube. This is to produce a cube whose spectra contain the
    brightest component along the line-of-sight and then be fit
    by the thick HI single-component model to specifically test
    the shape of the brightest components.
    '''

    min_lwidth_sub = min_lwidth_sub.to(u.m / u.s)

    assert noise_val.unit == u.K

    max_peak = max_amp_sub_sigma * noise_val

    with fits.open(params_name) as params_hdu:

        params_array = params_hdu[0].data
        uncert_array = params_hdu[1].data
        bic_array = params_hdu[2].data

    ncomp_array = np.isfinite(params_array).sum(0) // 3

    max_comp = ncomp_array.max()

    if return_faintcomps:
        faint_params_array = np.ones((max_comp * 3,) +
                                     params_array.shape[1:]) * np.NaN
        faint_uncert_array = np.ones((max_comp * 3,) +
                                     params_array.shape[1:]) * np.NaN

    new_params_array = np.zeros_like(params_array) * np.NaN
    new_uncert_array = np.zeros_like(params_array) * np.NaN

    yposn, xposn = np.where(np.isfinite(bic_array) & (ncomp_array > 0))

    for i, (y, x) in enumerate(zip(yposn, xposn)):

        sig_fits = params_array[2::3, y, x][:ncomp_array[y, x]]
        amp_fits = params_array[::3, y, x][:ncomp_array[y, x]]

        valid_comps = logic_func(amp_fits >= max_peak.value,
                                 sig_fits <= min_lwidth_sub.value)

        valids = np.where(valid_comps)[0]

        for j, comp in enumerate(valids):
            new_params_array[3 * j, y, x] = params_array[3 * comp, y, x]
            new_params_array[3 * j + 1, y, x] = params_array[3 * comp + 1, y, x]
            new_params_array[3 * j + 2, y, x] = params_array[3 * comp + 2, y, x]

            new_uncert_array[3 * j, y, x] = uncert_array[3 * comp, y, x]
            new_uncert_array[3 * j + 1, y, x] = uncert_array[3 * comp + 1, y, x]
            new_uncert_array[3 * j + 2, y, x] = uncert_array[3 * comp + 2, y, x]

        if return_faintcomps:
            faint_valids = np.where(~valid_comps)[0]
            for j, comp in enumerate(faint_valids):

                faint_params_array[3 * j, y, x] = params_array[3 * comp, y, x]
                faint_params_array[3 * j + 1, y, x] = params_array[3 * comp + 1, y, x]
                faint_params_array[3 * j + 2, y, x] = params_array[3 * comp + 2, y, x]

                faint_uncert_array[3 * j, y, x] = uncert_array[3 * comp, y, x]
                faint_uncert_array[3 * j + 1, y, x] = uncert_array[3 * comp + 1, y, x]
                faint_uncert_array[3 * j + 2, y, x] = uncert_array[3 * comp + 2, y, x]

    # Return a combined HDU that can be written out.
    params_hdu = fits.PrimaryHDU(new_params_array, vcent.header.copy())
    params_hdu.header['BUNIT'] = ("", "Gaussian fit parameters")

    uncerts_hdu = fits.ImageHDU(new_uncert_array, vcent.header.copy())
    uncerts_hdu.header['BUNIT'] = ("", "Gaussian fit uncertainty")

    # Will need to update the BIC eventually...
    bics_hdu = fits.ImageHDU(bic_array, vcent.header.copy())
    bics_hdu.header['BUNIT'] = ("", "Gaussian fit BIC")

    hdu_all = fits.HDUList([params_hdu, uncerts_hdu, bics_hdu])

    if return_faintcomps:
        mwparams_hdu = fits.PrimaryHDU(mwparams_array, vcent.header.copy())
        mwparams_hdu.header['BUNIT'] = ("", "Gaussian fit parameters")

        mwuncerts_hdu = fits.ImageHDU(mwuncert_array, vcent.header.copy())
        mwuncerts_hdu.header['BUNIT'] = ("", "Gaussian fit uncertainty")

        hdu_mw = fits.HDUList([mwparams_hdu, mwuncerts_hdu])

        return hdu_all, hdu_mw

    return hdu_all


def remove_all_but_brightest(params_name,
                             model_cube_name,
                             noise_val,
                             return_othercomps=True,
                             min_lwidth_sub=25 * u.km / u.s,
                             max_amp_sub_sigma=3.):
    '''
    This function will remove all components except those whose
    intensity is a significant fraction of the brightest single peak.
    '''
    pass


def subtract_components(cube_name,
                        remove_params_name,
                        output_cube_name,
                        chunk_size=20000):
    '''
    Subtract Gaussian components given in remove_params_name
    '''

    with fits.open(remove_params_name) as params_hdu:

        params_array = params_hdu[0].data
        uncert_array = params_hdu[1].data
        # bic_array = params_hdu[2].data

    yposn, xposn = np.where(ncomp_array > 0)

    ncomp_array = np.isfinite(params_array).sum(0) // 3

    # Create huge fits

    # evaluate and subtract all components


def recalculate_bic():
    '''
    Add a function to recompute the BIC after masking all MW emission.
    '''
    pass
