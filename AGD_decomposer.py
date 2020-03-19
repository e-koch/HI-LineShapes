# @Author: Robert Lindner
# @Date:   Nov 10, 2014
# @Filename: AGD_decomposer.py
# @Last modified by:   riener
# @Last modified time: 2019-04-02T16:57:28+02:00

# Copied over from gausspyplus.
# Will need to add a license from gausspyplus

# Standard Libs
import time
from copy import copy

# Standard Third Party
import numpy as np
from scipy.interpolate import interp1d
from lmfit import minimize as lmfit_minimize
from lmfit import Parameters

import matplotlib.pyplot as plt
from numpy.linalg import lstsq
# from scipy.optimize import lsq_linear
from scipy.ndimage.filters import median_filter, convolve
import scipy.ndimage as nd

from gausspyplus.gausspy_py3.gp_plus import (try_to_improve_fitting,
                                             goodness_of_fit)
# from .gp_plus import try_to_improve_fitting, goodness_of_fit

# Python Regularized derivatives
# from . import tvdiff
from gausspyplus.gausspy_py3 import tvdiff
# from gausspyplus.gausspy_py3 import tvdiff_cyth as tvdiff


def vals_vec_from_lmfit(lmfit_params):
    """Return Python list of parameter values from LMFIT Parameters object."""
    vals = [value.value for value in lmfit_params.values()]
    return np.array(vals)


def errs_vec_from_lmfit(lmfit_params):
    """
    Return Python list of parameter uncertainties from LMFIT Parameters
    object.
    """
    errs = [value.stderr if value.stderr is not None else np.NaN
            for value in lmfit_params.values()]
    return np.array(errs)


def paramvec_to_lmfit(paramvec, max_amp, max_fwhm=None,
                      min_cent=None, max_cent=None,
                      free_amp=True, free_fwhm=True,
                      free_cent=True):
    """
    Transform a Python iterable of parameters into a LMFIT Parameters
    object.
    """
    ncomps = int(len(paramvec) / 3)
    params = Parameters()
    for i in range(len(paramvec)):
        if 0 <= i < ncomps:
            if max_amp is not None:
                params.add('p' + str(i + 1), value=paramvec[i],
                           min=0.0, max=max_amp, vary=free_amp)
            else:
                params.add('p' + str(i + 1), value=paramvec[i],
                           min=0.0, vary=free_amp)

        elif ncomps <= i < 2 * ncomps:
            if max_fwhm is not None:
                params.add('p' + str(i + 1), value=paramvec[i],
                           min=0.0, max=max_fwhm, vary=free_fwhm)
            else:
                params.add('p' + str(i + 1), value=paramvec[i],
                           min=0.0, vary=free_fwhm)
        else:
            params.add('p' + str(i + 1), value=paramvec[i],
                       min=min_cent, max=max_cent,
                       vary=free_cent)

    return params


def create_fitmask(size, offsets_i, di):
    """Return valid domain for intermediate fit in d2/dx2 space.

    fitmask = (0,1)
    fitmaskw = (True, False)
    """
    fitmask = np.zeros(size)
    for i in range(len(offsets_i)):
        fitmask[int(offsets_i[i] - di[i]):int(offsets_i[i] + di[i])] = 1.0
    fitmaskw = fitmask == 1.0
    return fitmask, fitmaskw


def say(message, verbose=False):
    """Diagnostic messages."""
    if verbose is True:
        print(message)


def split_params(params, ncomps):
    """Split params into amps, fwhms, offsets."""
    amps = params[0:ncomps]
    fwhms = params[ncomps:2 * ncomps]
    offsets = params[2 * ncomps:3 * ncomps]
    return amps, fwhms, offsets


def gaussian(peak, FWHM, mean):
    """Return a Gaussian function."""
    sigma = FWHM / 2.354820045  # (2 * sqrt( 2 * ln(2)))
    return lambda x: peak * np.exp(-(x - mean)**2 / 2. / sigma**2)


def func(x, *args):
    """Return multi-component Gaussian model F(x).

    Parameter vector kargs = [amp1, ..., ampN, width1, ..., widthN, mean1, ..., meanN],
    and therefore has len(args) = 3 x N_components.
    """
    ncomps = int(len(args) / 3)
    yout = x * 0.
    for i in range(ncomps):
        yout = yout + gaussian(args[i], args[i + ncomps],
                               args[i + 2 * ncomps])(x)
    return yout


def initialGuess(vel, data, errors=None, alpha=None, plot=False, mode='conv',
                 verbose=False, SNR_thresh=5.0, BLFrac=0.1, SNR2_thresh=5.0,
                 deblend=True):
    """Find initial parameter guesses (AGD algorithm).

    data,             Input data
    dv,             x-spacing absolute units
    alpha = No Default,     regularization parameter
    plot = False,     Show diagnostic plots?
    verbose = True    Diagnostic messages
    SNR_thresh = 5.0  Initial Spectrum S/N threshold
    BLFrac =          Edge fraction of data used for S/N threshold computation
    SNR2_thresh =   S/N threshold for Second derivative
    mode = Method for taking derivatives
    """
    say('\n\n  --> initialGuess() \n', verbose)
    say('Algorithm parameters: ', verbose)
    say('alpha = {0}'.format(alpha), verbose)
    say('SNR_thesh = {0}'.format(SNR_thresh), verbose)
    say('SNR2_thesh = {0}'.format(SNR2_thresh), verbose)
    # say('BLFrac = {0}'.format(BLFrac), verbose)

    if not alpha:
        print('Must choose value for alpha, no default.')
        return

    if np.any(np.isnan(data)):
        print('NaN-values in data, cannot continue.')
        return

    # Data inspection
    vel = np.array(vel)
    data = np.array(data)
    dv = np.abs(vel[1] - vel[0])
    # Converts from index -> x domain
    fvel = interp1d(np.arange(len(vel)), vel)
    # data_size = len(data)

    # Take regularized derivatives
    t0 = time.time()
    if mode == 'python':
        say('Taking python derivatives...', verbose)
        u = tvdiff.TVdiff(data, dx=dv, alph=alpha)
        u2 = tvdiff.TVdiff(u, dx=dv, alph=alpha)
        u3 = tvdiff.TVdiff(u2, dx=dv, alph=alpha)
        u4 = tvdiff.TVdiff(u3, dx=dv, alph=alpha)
    elif mode == 'conv':
        say('Convolution sigma [pixels]: {0}'.format(alpha), verbose)
        gauss_sigma = alpha
        gauss_sigma_int = np.max([np.fix(gauss_sigma), 5])
        gauss_dn = gauss_sigma_int * 6

        xx = np.arange(2*gauss_dn+2)-(gauss_dn) - 0.5
        gauss = np.exp(-xx**2/2./gauss_sigma**2)
        gauss = gauss / np.sum(gauss)
        gauss1 = np.diff(gauss) / dv
        gauss3 = np.diff(np.diff(gauss1)) / dv**2

        xx2 = np.arange(2*gauss_dn+1)-(gauss_dn)
        gauss2 = np.exp(-xx2**2/2./gauss_sigma**2)
        gauss2 = gauss2 / np.sum(gauss2)
        gauss2 = np.diff(gauss2) / dv
        gauss2 = np.diff(gauss2) / dv
        gauss4 = np.diff(np.diff(gauss2)) / dv**2

        u0 = convolve(data, gauss, mode='wrap')
        u = convolve(data, gauss1, mode='wrap')
        u2 = convolve(data, gauss2, mode='wrap')
        u3 = convolve(data, gauss3, mode='wrap')
        u4 = convolve(data, gauss4, mode='wrap')

    say('...took {0:4.2f} seconds per derivative.'.format(
        (time.time()-t0)/4.), verbose)

    # Decide on signal threshold
    if not errors:
        errors = np.std(data[data < abs(np.min(data))])  # added by M.Riener

    thresh = SNR_thresh * errors
    mask1 = np.array(data > thresh, dtype='int')[1:]  # Raw Data S/N
    mask3 = np.array(u4.copy()[1:] > 0., dtype='int')  # Positive 4th derivative

    if SNR2_thresh > 0.:
        wsort = np.argsort(np.abs(u2))
        # RMS based in +-1 sigma fluctuations
        RMSD2 = np.std(u2[wsort][0:int(0.5 * len(u2))]) / 0.377
        # RMSD2 = np.std(u2[data < 2 * errors]) * 0.377

        say('Second derivative noise: {0}'.format(RMSD2), verbose)
        thresh2 = -RMSD2 * SNR2_thresh
        say('Second derivative threshold: {0}'.format(thresh2), verbose)
    else:
        thresh2 = 0.
    # Negative second derivative
    mask4 = np.array(u2.copy()[1:] < thresh2, dtype='int')

    # Find optima of second derivative
    # --------------------------------
    zeros = np.abs(np.diff(np.sign(u3)))
    zeros = zeros * mask1 * mask3 * mask4

    # Some corner cases lead to many nearby (<few pix) offset points.
    # These are nearly always due to a bit of noise and should be reduced
    # down to 1.
    # offsets_data_i = np.array(np.where(zeros)).ravel()  # Index offsets

    # Label positions together if 2 pixels are called zeros.
    lab, num = nd.label(zeros)

    offsets = [np.where(lab == ii)[0].mean()
               for ii in range(1, num + 1)]

    # Remove cases where the zeros are smaller than the smoothing length.
    # Use FWHM.
    if mode == 'conv':
        diff_offsets = np.diff(offsets)

        offsets_to_remove = []
        for i, diff in enumerate(diff_offsets):

            if diff > alpha * 2.35:
                continue

            # Remove one with smaller u2
            if u2[int(offsets[i])] <= u2[int(offsets[i + 1])]:
                offsets_to_remove.append(offsets[i + 1])
            else:
                offsets_to_remove.append(offsets[i])

        offsets = list(set(offsets) - set(offsets_to_remove))

    else:
        # It would be nice to generalize this. Not happening now, though.
        pass

    offsets_data_i = np.floor(np.array(offsets)).astype(int)

    offsets = fvel(offsets_data_i + 0.5)  # Velocity offsets (Added 0.5 July 23)
    N_components = len(offsets)
    say('Components found for alpha={1}: {0}'.format(N_components, alpha),
        verbose=verbose)

    # Check if nothing was found, if so, return null
    # ----------------------------------------------
    if N_components == 0:
        odict = {'means': [], 'FWHMs': [], 'amps': [],
                 'u2': u2, 'errors': errors, 'thresh2': thresh2,
                 'thresh': thresh, 'N_components': N_components}

        return odict

    # Find points of inflection
    # inflection = np.abs(np.diff(np.sign(u2)))

    # Find Relative widths, then measure
    # peak-to-inflection distance for sharpest peak
    widths = np.sqrt(np.abs(u0 / u2)[offsets_data_i])

    if mode is 'python':

        median_window = 10**((np.log10(alpha) + 2.187) / 3.859)

    else:

        median_window = alpha

    # del_v = np.diff(vel[:2])[0]

    deconv_widths = np.sqrt(widths**2 - median_window**2)
    deconv_widths[widths < median_window] = median_window

    widths = deconv_widths

    FWHMs = widths * 2.355
    amps = np.array(u0[offsets_data_i])

    # Attempt deblending.
    # If Deblending results in all non-negative answers, keep.
    if deblend:
        FF_matrix = np.zeros([len(amps), len(amps)])
        for i in range(FF_matrix.shape[0]):
            for j in range(FF_matrix.shape[1]):
                FF_matrix[i, j] = np.exp(-(offsets[i]-offsets[j])**2/2./(FWHMs[j] / 2.355)**2)
        amps_new = lstsq(FF_matrix, amps, rcond=None)[0]
        if np.all(amps_new > 0):
            amps = amps_new
        else:
            say("Failed to deblend", verbose=verbose)

    odict = {'means': offsets, 'FWHMs': FWHMs, 'amps': amps,
             'u2': u2, 'errors': errors, 'thresh2': thresh2,
             'thresh': thresh, 'N_components': N_components}

    return odict


def AGD(vel, data, errors, idx=None, signal_ranges=None,
        noise_spike_ranges=None, improve_fitting_dict=None,
        alpha1=None, alpha2=None, plot=False, mode='conv', verbose=False,
        SNR_thresh=5.0, BLFrac=0.1, SNR2_thresh=5.0, deblend=True,
        perform_final_fit=True, phase='one',
        intermediate_fit=False,
        use_detect_wings=False):
    """ Autonomous Gaussian Decomposition."""
    dct = {}
    if improve_fitting_dict is not None:
        # TODO: check if max_amp causes problems
        # dct['improve_fitting'] = improve_fitting_dict['improve_fitting']
        # dct['min_fwhm'] = improve_fitting_dict['min_fwhm']
        # dct['max_fwhm'] = improve_fitting_dict['max_fwhm']
        # dct['snr_fit'] = improve_fitting_dict['snr_fit']
        # dct['significance'] = improve_fitting_dict['significance']
        # dct['min_offset'] = improve_fitting_dict['min_offset']
        # dct['max_amp_factor'] = improve_fitting_dict['max_amp_factor']
        # dct['max_amp'] = dct['max_amp_factor']*np.max(data)
        # dct['rchi2_limit'] = improve_fitting_dict['rchi2_limit']
        # dct['snr_negative'] = improve_fitting_dict['snr_negative']
        # dct['snr'] = improve_fitting_dict['snr']
        dct = improve_fitting_dict
        dct['max_amp'] = dct['max_amp_factor']*np.max(data)
    else:
        dct['improve_fitting'] = False
        dct['max_fwhm'] = 80.
        dct['max_amp'] = 1.1 * np.nanmax(data)

    if not isinstance(SNR_thresh, list):
        SNR_thresh = [SNR_thresh, SNR_thresh]
    if not isinstance(SNR2_thresh, list):
        SNR2_thresh = [SNR2_thresh, SNR2_thresh]

    say('\n  --> AGD() \n', verbose)

    if (not alpha2) and (phase == 'two'):
        print('alpha2 value required')
        return

    dv = np.abs(vel[1] - vel[0])
    v_to_i = interp1d(vel, np.arange(len(vel)))

    # -------------------------------------- #
    # Find phase-one guesses                 #
    # -------------------------------------- #
    agd1 = initialGuess(vel, data, errors=errors[0], alpha=alpha1, plot=plot,
                        mode=mode, verbose=verbose, SNR_thresh=SNR_thresh[0],
                        BLFrac=BLFrac, SNR2_thresh=SNR2_thresh[0],
                        deblend=deblend)

    amps_g1, widths_g1, offsets_g1, u2 = agd1['amps'], agd1['FWHMs'], agd1['means'], agd1['u2']
    params_g1 = np.append(np.append(amps_g1, widths_g1), offsets_g1)
    ncomps_g1 = int(len(params_g1) / 3)
    ncomps_g2 = 0  # Default
    ncomps_f1 = 0  # Default

    # ----------------------------#
    # Find phase-two guesses #
    # ----------------------------#
    if phase == 'two':
        say('Beginning phase-two AGD... ', verbose)
        ncomps_g2 = 0

        # ----------------------------------------------------------#
        # Produce the residual signal                               #
        #  -- Either the original data, or intermediate subtraction #
        # ----------------------------------------------------------#
        if ncomps_g1 == 0:
            say('Phase 2 with no narrow comps -> No intermediate subtration... ', verbose)
            residuals = data
            params_f1 = params_g1
            ncomps_f1 = int(len(params_f1) / 3)

        elif intermediate_fit:
            # "Else" Narrow components were found, and Phase == 2, so perform intermediate subtraction...

            # The "fitmask" is a collection of windows around the a list of phase-one components
            fitmask, fitmaskw = create_fitmask(len(vel), v_to_i(offsets_g1),
                                               widths_g1 / dv / 2.355 * 0.9)
            notfitmask = 1 - fitmask
            notfitmaskw = np.logical_not(fitmaskw)

            # Error function for intermediate optimization
            def objectiveD2_leastsq(paramslm):
                params = vals_vec_from_lmfit(paramslm)
                model0 = func(vel, *params)
                model2 = np.diff(np.diff(model0.ravel())) / dv / dv
                resids1 = fitmask[1:-1] * (model2 - u2[1:-1]) / errors[1:-1]
                resids2 = notfitmask * (model0 - data) / errors / 10.
                return np.append(resids1, resids2)

            # Perform the intermediate fit using LMFIT
            t0 = time.time()
            say('Running LMFIT on initial narrow components...', verbose)
            lmfit_params = paramvec_to_lmfit(params_g1, dct['max_amp'],
                                             max_fwhm=dct['max_fwhm'],
                                             free_fwhm=True,
                                             free_amp=True,
                                             free_cent=True)
            result = lmfit_minimize(objectiveD2_leastsq, lmfit_params,
                                    method='leastsq')
            params_f1 = np.asarray(vals_vec_from_lmfit(result.params))
            ncomps_f1 = int(len(params_f1) / 3)

            # Make "FWHMS" positive
            # params_f1[0:ncomps_f1][np.array(params_f1[0:ncomps_f1]) < 0.0] =\
            #     -1 * params_f1[0:ncomps_f1][np.array(params_f1[0:ncomps_f1]) < 0.0]

            del lmfit_params
            say('LMFIT fit took {0} seconds.'.format(time.time()-t0))

            if result.success:
                # Compute intermediate residuals
                # Median filter on 2x effective scale to remove poor subtractions of strong components
                intermediate_model = func(vel, *params_f1).ravel()  # Explicit final (narrow) model
                median_window = 2. * 10**((np.log10(alpha1) + 2.187) / 3.859)
                residuals = median_filter(data - intermediate_model, np.int(median_window))
                # residuals = data - intermediate_model
            else:
                residuals = data
                params_f1 = params_g1
                ncomps_f1 = int(len(params_f1) / 3)
            # Finished producing residual signal # ---------------------------
        else:
            # Just subtract off the estimated model from stage 1 without explicitly fitting.
            params_f1 = params_g1
            ncomps_f1 = int(len(params_f1) / 3)
            intermediate_model = func(vel, *params_f1).ravel()  # Explicit final (narrow) model
            median_window = 2. * 10**((np.log10(alpha1) + 2.187) / 3.859)
            residuals = median_filter(data - intermediate_model, np.int(median_window))
            # residuals = data - intermediate_model

        # Search for phase-two guesses
        agd2 = initialGuess(vel, residuals, errors=errors[0], alpha=alpha2,
                            mode=mode, verbose=verbose,
                            SNR_thresh=SNR_thresh[1], BLFrac=BLFrac,
                            SNR2_thresh=SNR2_thresh[1],  # June 9 2014, change
                            deblend=deblend, plot=plot)
        ncomps_g2 = agd2['N_components']
        if ncomps_g2 > 0:
            params_g2 = np.concatenate([agd2['amps'], agd2['FWHMs'],
                                        agd2['means']])

            # Look for double-counted "wings" where 1 Gaussian is appropriate
            # instead of 2
            if use_detect_wings:
                params_f1, params_g2 = detect_wings(vel, params_f1, params_g2)

            ncomps_g2 = len(params_g2) // 3

        else:
            params_g2 = []
        u22 = agd2['u2']

    else:
        params_f1 = params_g1
        ncomps_f1 = len(params_f1) // 3

        # END PHASE 2 <<<

    # Check for phase two components, make final guess list
    # ------------------------------------------------------
    if phase == 'two' and (ncomps_g2 > 0):
        # amps_gf = np.append(params_g1[0:ncomps_g1], params_g2[0:ncomps_g2])
        # widths_gf = np.append(params_g1[ncomps_g1:2*ncomps_g1], params_g2[ncomps_g2:2*ncomps_g2])
        # offsets_gf = np.append(params_g1[2*ncomps_g1:3*ncomps_g1], params_g2[2*ncomps_g2:3*ncomps_g2])
        amps_gf = np.append(params_f1[0:ncomps_f1], params_g2[0:ncomps_g2])
        widths_gf = np.append(params_f1[ncomps_f1:2*ncomps_f1], params_g2[ncomps_g2:2*ncomps_g2])
        offsets_gf = np.append(params_f1[2*ncomps_f1:3*ncomps_f1], params_g2[2*ncomps_g2:3*ncomps_g2])
        params_gf = np.concatenate([amps_gf, widths_gf, offsets_gf])
        ncomps_gf = int(len(params_gf) / 3)
    else:
        params_gf = params_f1
        ncomps_gf = int(len(params_gf) / 3)

    # Sort final guess list by amplitude
    # ----------------------------------
    say('N final parameter guesses: ' + str(ncomps_gf))
    amps_temp = params_gf[0:ncomps_gf]
    widths_temp = params_gf[ncomps_gf:2*ncomps_gf]
    offsets_temp = params_gf[2*ncomps_gf:3*ncomps_gf]
    w_sort_amp = np.argsort(amps_temp)[::-1]
    params_gf = np.concatenate([amps_temp[w_sort_amp], widths_temp[w_sort_amp],
                                offsets_temp[w_sort_amp]])

    if (ncomps_gf > 0):
        if perform_final_fit is True:
            say('\n\n  --> Final Fitting... \n', verbose)

            # Objective functions for final fit
            def objective_leastsq(paramslm):
                params = vals_vec_from_lmfit(paramslm)
                resids = (func(vel, *params).ravel() - data.ravel()) / errors
                return resids

            # Final fit using unconstrained parameters
            t0 = time.time()
            lmfit_params = paramvec_to_lmfit(params_gf, dct['max_amp'],
                                             max_fwhm=dct['max_fwhm'],)
            result2 = lmfit_minimize(objective_leastsq, lmfit_params, method='leastsq')
            params_fit = vals_vec_from_lmfit(result2.params)
            params_errs = errs_vec_from_lmfit(result2.params)

            del lmfit_params
            say('Final fit took {0} seconds.'.format(time.time()-t0), verbose)

            ncomps_fit = int(len(params_fit)/3)
            # Make "FWHMS" positive
            # params_fit[0:ncomps_fit][np.array(params_fit[0:ncomps_fit]) < 0.0] =\
            #     -1 * params_fit[0:ncomps_fit][np.array(params_fit[0:ncomps_fit]) < 0.0]

            best_fit_final = func(vel, *params_fit).ravel()
        else:
            best_fit_final = func(vel, *params_gf).ravel()

            params_fit = params_gf

            ncomps_fit = int(len(params_gf) / 3)

    # Try to improve the fit
    # ----------------------
    if dct['improve_fitting']:
        if ncomps_gf == 0:
            ncomps_fit = 0
            params_fit = []
        #  TODO: check if ncomps_fit should be ncomps_gf
        best_fit_list, N_neg_res_peak, N_blended, log_gplus =\
            try_to_improve_fitting(
                vel, data, errors, params_fit, ncomps_fit, dct,
                signal_ranges=signal_ranges, noise_spike_ranges=noise_spike_ranges)

        params_fit, params_errs, ncomps_fit, best_fit_final, residual,\
            rchi2, aicc, new_fit, params_min, params_max, pvalue, quality_control = best_fit_list

        ncomps_gf = ncomps_fit

    if plot:
        #                       P L O T T I N G
        datamax = np.max(data)
        try:
            print(("params_fit:", params_fit))
        except UnboundLocalError:
            pass

        if ncomps_gf == 0:
            ncomps_fit = 0
            best_fit_final = data * 0

        if dct['improve_fitting']:
            rchi2 = best_fit_list[5]
        else:
            #  TODO: define mask from signal_ranges
            rchi2 = goodness_of_fit(data, best_fit_final, errors, ncomps_fit)

        # Set up figure
        fig = plt.figure('AGD results', [16, 12])
        ax1 = fig.add_axes([0.1, 0.5, 0.4, 0.4])  # Initial guesses (alpha1)
        ax2 = fig.add_axes([0.5, 0.5, 0.4, 0.4])  # D2 fit to peaks(alpha2)
        ax3 = fig.add_axes([0.1, 0.1, 0.4, 0.4])  # Initial guesses (alpha2)
        ax4 = fig.add_axes([0.5, 0.1, 0.4, 0.4])  # Final fit

        # Decorations
        if dct['improve_fitting']:
            plt.figtext(0.52, 0.47, 'Final fit (GaussPy+)')
        else:
            plt.figtext(0.52, 0.47, 'Final fit (GaussPy)')
        if perform_final_fit:
            plt.figtext(0.52, 0.45, 'Reduced Chi2: {0:3.3f}'.format(rchi2))
            plt.figtext(0.52, 0.43, 'N components: {0}'.format(ncomps_fit))

        plt.figtext(0.12, 0.47, 'Phase-two initial guess')
        plt.figtext(0.12, 0.45, 'N components: {0}'.format(ncomps_g2))

        plt.figtext(0.12, 0.87, 'Phase-one initial guess')
        plt.figtext(0.12, 0.85, 'N components: {0}'.format(ncomps_g1))

        plt.figtext(0.52, 0.87, 'Intermediate fit')

        # Initial Guesses (Panel 1)
        # -------------------------
        ax1.xaxis.tick_top()
        u2_scale = 1. / np.max(np.abs(u2)) * datamax * 0.5
        ax1.axhline(color='black', linewidth=0.5)
        ax1.plot(vel, data, '-k')
        ax1.plot(vel, u2 * u2_scale, '-r')
        ax1.plot(vel, np.ones(len(vel)) * agd1['thresh'], '--k')
        ax1.plot(vel, np.ones(len(vel)) * agd1['thresh2'] * u2_scale, '--r')

        for i in range(ncomps_g1):
            one_component = gaussian(params_g1[i], params_g1[i+ncomps_g1], params_g1[i+2*ncomps_g1])(vel)
            ax1.plot(vel, one_component, '-g')

        # Plot intermediate fit components (Panel 2)
        # ------------------------------------------
        ax2.xaxis.tick_top()
        ax2.axhline(color='black', linewidth=0.5)
        ax2.plot(vel, data, '-k')
        ax2.yaxis.tick_right()
        for i in range(ncomps_f1):
            one_component = gaussian(params_f1[i], params_f1[i+ncomps_f1], params_f1[i+2*ncomps_f1])(vel)
            ax2.plot(vel, one_component, '-', color='blue')

        # Residual spectrum (Panel 3)
        # -----------------------------
        if phase == 'two':
            u22_scale = 1. / np.abs(u22).max() * np.max(residuals) * 0.5
            ax3.axhline(color='black', linewidth=0.5)
            ax3.plot(vel, residuals, '-k')
            ax3.plot(vel, np.ones(len(vel)) * agd2['thresh'], '--k')
            ax3.plot(vel, np.ones(len(vel)) * agd2['thresh2'] * u22_scale, '--r')
            ax3.plot(vel, u22 * u22_scale, '-r')
            for i in range(ncomps_g2):
                one_component = gaussian(params_g2[i], params_g2[i+ncomps_g2], params_g2[i+2*ncomps_g2])(vel)
                ax3.plot(vel, one_component, '-g')

        # Plot best-fit model (Panel 4)
        # -----------------------------
        # if perform_final_fit:
        ax4.yaxis.tick_right()
        ax4.axhline(color='black', linewidth=0.5)
        ax4.plot(vel, data, label='data', color='black')
        for i in range(ncomps_fit):
            one_component = gaussian(params_fit[i], params_fit[i + ncomps_fit],
                                     params_fit[i + 2 * ncomps_fit])(vel)
            ax4.plot(vel, one_component, '--', color='orange')
        ax4.plot(vel, best_fit_final, '-', color='orange', linewidth=2)

        plt.show()

    # Construct output dictionary (odict)
    # -----------------------------------
    odict = {}
    odict['initial_parameters'] = params_gf

    odict['N_components'] = ncomps_gf
    odict['index'] = idx
    if dct['improve_fitting']:
        odict['best_fit_rchi2'] = rchi2
        odict['best_fit_aicc'] = aicc
        odict['pvalue'] = pvalue

        odict['N_neg_res_peak'] = N_neg_res_peak
        odict['N_blended'] = N_blended
        odict['log_gplus'] = log_gplus
        odict['quality_control'] = quality_control

    if (perform_final_fit is True) and (ncomps_gf > 0):
        odict['best_fit_parameters'] = params_fit
        odict['best_fit_errors'] = params_errs

    return (1, odict)


def AGD_loop(spec, errors,
             alphas,
             idx=None, signal_ranges=None,
             improve_fitting_dict=None,
             plot=False, mode='conv', verbose=False,
             SNR_thresh1=5.0, SNR_thresh2=5.0,
             SNR2_thresh1=5.0, SNR2_thresh2=5.0,
             BLFrac=0.1, deblend=True,
             perform_final_fit=True,
             intermediate_fit=False,
             use_detect_wings=False,
             component_sigma=5.):
    """ Autonomous Gaussian Decomposition."""

    if len(alphas) == 0:
        raise ValueError("No smoothing lengths given.")

    # Grab data from the spectrum
    data = spec.filled_data[:].value
    vels = spec.spectral_axis.value
    vel = np.arange(vels.size)

    dct = {}
    if improve_fitting_dict is not None:
        dct = improve_fitting_dict
        dct['max_amp'] = dct['max_amp_factor'] * np.max(data)
    else:
        dct['improve_fitting'] = False
        dct['max_amp'] = 1.1 * np.nanmax(data)
        dct['max_fwhm'] = 0.5 * np.ptp(vel)
        dct['min_cent'] = vel.min() - 0.1 * np.ptp(vel)
        dct['max_cent'] = vel.max() + 0.1 * np.ptp(vel)

    dv = np.abs(vel[1] - vel[0])
    v_to_i = interp1d(vel, np.arange(len(vel)))

    # Loop through smoothing scales

    fit_results = {}

    for i, alpha in enumerate(alphas):

        if verbose:
            print(f"On alpha={alpha}")

        ncomps_gf = 0

        if i == 0:
            # -------------------------------------- #
            # Find phase-one guesses                 #
            # -------------------------------------- #
            agd1 = initialGuess(vel, data,
                                errors=errors[0],
                                alpha=alpha, plot=plot,
                                mode=mode, verbose=verbose,
                                SNR_thresh=SNR_thresh1,
                                BLFrac=BLFrac,
                                SNR2_thresh=SNR2_thresh1,
                                deblend=deblend)

            amps_g1, widths_g1, offsets_g1, u2 = \
                agd1['amps'], agd1['FWHMs'], agd1['means'], agd1['u2']
            params_g1 = np.append(np.append(amps_g1, widths_g1), offsets_g1)
            ncomps_g1 = int(len(params_g1) / 3)

            # Only one step. Output for final fit.
            if len(alphas) == 1:
                params_gf = params_g1
                params_fit = params_g1
                ncomps_gf = len(params_g1) // 3
        else:
            # Get components from last iteration
            try:
                params_g1 = params_fit
            except UnboundLocalError:
                params_g1 = params_gf

            ncomps_g1 = int(len(params_g1) / 3)
            widths_g1 = params_g1[ncomps_g1:2 * ncomps_g1]

        # Now we continue on from residuals

        ncomps_g2 = 0
        ncomps_f1 = 0  # Default

        # ----------------------------------------------------------#
        # Produce the residual signal                               #
        #  -- Either the original data, or intermediate subtraction #
        # ----------------------------------------------------------#
        if ncomps_g1 == 0:
            residuals = data
            params_f1 = params_g1
            ncomps_f1 = int(len(params_f1) / 3)

        elif intermediate_fit:
            # "Else" Narrow components were found, and Phase == 2,
            # so perform intermediate subtraction

            # The "fitmask" is a collection of windows around the a list of
            # phase-one components
            fitmask, fitmaskw = create_fitmask(len(vel), v_to_i(offsets_g1),
                                               widths_g1 / dv / 2.355 * 0.9)
            notfitmask = 1 - fitmask

            # Error function for intermediate optimization
            def objectiveD2_leastsq(paramslm):
                params = vals_vec_from_lmfit(paramslm)
                model0 = func(vel, *params)
                model2 = np.diff(np.diff(model0.ravel())) / dv / dv
                resids1 = fitmask[1:-1] * (model2 - u2[1:-1]) / errors[1:-1]
                resids2 = notfitmask * (model0 - data) / errors / 10.
                return np.append(resids1, resids2)

            # Perform the intermediate fit using LMFIT
            lmfit_params = paramvec_to_lmfit(params_g1, dct['max_amp'],
                                             max_fwhm=dct['max_fwhm'],
                                             free_fwhm=True,
                                             free_amp=True,
                                             free_cent=True)
            result = lmfit_minimize(objectiveD2_leastsq, lmfit_params,
                                    method='leastsq')
            params_f1 = np.asarray(vals_vec_from_lmfit(result.params))
            ncomps_f1 = int(len(params_f1) / 3)

            del lmfit_params

            if result.success:
                # Compute intermediate residuals
                # Median filter on 2x effective scale to remove poor
                # subtractions of strong components
                # Explicit final (narrow) model
                intermediate_model = func(vel, *params_f1).ravel()
                if mode == 'python':
                    median_window = 2. * 10**((np.log10(alpha) + 2.187) / 3.859)
                else:
                    median_window = alpha / 2.
                residuals = median_filter(data - intermediate_model,
                                          np.int(median_window))
            else:
                residuals = data
                params_f1 = params_g1
                ncomps_f1 = int(len(params_f1) / 3)

        else:
            # Just subtract off the estimated model from stage 1 without
            # explicitly fitting.
            params_f1 = params_g1
            ncomps_f1 = int(len(params_f1) / 3)
            intermediate_model = func(vel, *params_f1).ravel()
            if mode == 'python':
                median_window = 2. * 10**((np.log10(alpha) + 2.187) / 3.859)
            else:
                median_window = alpha / 2.
            residuals = median_filter(data - intermediate_model,
                                      np.int(median_window))

        # After first stage, propagate the intermediate fit forward.
        if i == 0:
            params_fit = params_f1

            continue

        # Search for phase-two guesses
        agd2 = initialGuess(vel, residuals,
                            # Adjust error after median window average.
                            errors=errors[0] / np.sqrt(alpha / 4.),
                            alpha=alpha,
                            mode=mode, verbose=verbose,
                            SNR_thresh=SNR_thresh2, BLFrac=BLFrac,
                            SNR2_thresh=SNR2_thresh2,  # June 9 2014, change
                            deblend=False, plot=plot)
        ncomps_g2 = agd2['N_components']
        if ncomps_g2 > 0:
            params_g2 = np.concatenate([agd2['amps'], agd2['FWHMs'],
                                        agd2['means']])

            # Look for double-counted "wings" where 1 Gaussian is appropriate
            # instead of 2
            if use_detect_wings:
                params_f1, params_g2 = detect_wings(vel, params_f1, params_g2)

            ncomps_g2 = len(params_g2) // 3

        else:
            params_g2 = []
        u22 = agd2['u2']

        # Check for phase two components, make final guess list
        # ------------------------------------------------------
        if ncomps_g2 > 0:

            amps_gf = np.append(params_f1[0:ncomps_f1],
                                params_g2[0:ncomps_g2])
            widths_gf = np.append(params_f1[ncomps_f1:2 * ncomps_f1],
                                  params_g2[ncomps_g2:2 * ncomps_g2])
            offsets_gf = np.append(params_f1[2 * ncomps_f1:3 * ncomps_f1],
                                   params_g2[2 * ncomps_g2:3 * ncomps_g2])
            params_gf = np.concatenate([amps_gf, widths_gf, offsets_gf])
            ncomps_gf = int(len(params_gf) / 3)

            # Attempt deblending.
            # If Deblending results in all non-negative answers, keep.
            if deblend:
                FF_matrix = np.zeros([len(amps_gf), len(amps_gf)])
                for i, j in np.ndindex(FF_matrix.shape):
                    FF_matrix[i, j] = np.exp(-(offsets_gf[j] - offsets_gf[i])**2/2./(widths_gf[j] / 2.355)**2)
                amps_new = lstsq(FF_matrix, amps_gf, rcond=None)[0]
                if np.all(amps_new > 0):
                    amps_gf = amps_new
                    params_gf[0:ncomps_gf] = amps_gf
                    say(f"Found new amplitudes {amps_gf}", verbose=verbose)
                else:
                    say("Failed to deblend", verbose=verbose)

        else:
            # Continue to the next smoothing scale if none are found.
            params_gf = params_f1
            ncomps_gf = int(len(params_gf) / 3)
            continue

        # Sort final guess list by integral
        # ----------------------------------
        amps_temp = params_gf[0:ncomps_gf]
        widths_temp = params_gf[ncomps_gf:2 * ncomps_gf]
        offsets_temp = params_gf[2 * ncomps_gf:3 * ncomps_gf]
        w_sort_amp = np.argsort(amps_temp * widths_temp)[::-1]
        params_gf = np.concatenate([amps_temp[w_sort_amp],
                                    widths_temp[w_sort_amp],
                                    offsets_temp[w_sort_amp]])

        if ncomps_gf > 0:
            if perform_final_fit:

                # Objective functions for final fit
                def objective_leastsq(paramslm):
                    params = vals_vec_from_lmfit(paramslm)
                    resids = (func(vel, *params).ravel() - data.ravel()) / errors
                    return resids

                # Final fit using unconstrained parameters
                lmfit_params = paramvec_to_lmfit(params_gf, dct['max_amp'],
                                                 max_fwhm=dct['max_fwhm'],)
                result2 = lmfit_minimize(objective_leastsq, lmfit_params,
                                         method='leastsq')
                params_fit = vals_vec_from_lmfit(result2.params)
                params_errs = errs_vec_from_lmfit(result2.params)

                fit_results[alpha] = lmfit_params

                ncomps_fit = len(params_fit) // 3

                best_fit_final = func(vel, *params_fit).ravel()
            else:
                best_fit_final = func(vel, *params_gf).ravel()

                params_fit = params_gf

                ncomps_fit = len(params_gf) // 3
        else:
            params_fit = []
            params_errs = []

        if plot:
            #                       P L O T T I N G
            datamax = np.max(data)
            try:
                print(("params_fit:", params_fit))
            except UnboundLocalError:
                pass

            if ncomps_gf == 0:
                ncomps_fit = 0
                best_fit_final = data * 0

            # if dct['improve_fitting']:
            #     rchi2 = best_fit_list[5]
            # else:
            #     #  TODO: define mask from signal_ranges
            rchi2 = goodness_of_fit(data, best_fit_final, errors, ncomps_fit)

            # Set up figure
            fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
            ax1, ax2, ax3, ax4 = axs.ravel()

            # Decorations
            if dct['improve_fitting']:
                plt.figtext(0.52, 0.47, 'Final fit (GaussPy+)')
            else:
                plt.figtext(0.52, 0.47, 'Final fit (GaussPy)')
            if perform_final_fit:
                plt.figtext(0.52, 0.45, 'Reduced Chi2: {0:3.3f}'.format(rchi2))
                plt.figtext(0.52, 0.43, 'N components: {0}'.format(ncomps_fit))

            plt.figtext(0.12, 0.47, 'Phase-two initial guess')
            plt.figtext(0.12, 0.45, 'N components: {0}'.format(ncomps_g2))

            plt.figtext(0.12, 0.87, 'Phase-one initial guess')
            plt.figtext(0.12, 0.85, 'N components: {0}'.format(ncomps_g1))

            plt.figtext(0.52, 0.87, 'Intermediate fit')

            # Initial Guesses (Panel 1)
            # -------------------------
            ax1.xaxis.tick_top()
            u2_scale = 1. / np.max(np.abs(u2)) * datamax * 0.5
            ax1.axhline(color='black', linewidth=0.5)
            ax1.plot(vel, data, '-k')
            ax1.plot(vel, u2 * u2_scale, '-r')
            ax1.plot(vel, np.ones(len(vel)) * agd1['thresh'], '--k')
            ax1.plot(vel, np.ones(len(vel)) * agd1['thresh2'] * u2_scale, '--r')

            for i in range(ncomps_g1):
                one_component = gaussian(params_g1[i], params_g1[i + ncomps_g1],
                                         params_g1[i + 2 * ncomps_g1])(vel)
                ax1.plot(vel, one_component, '-g')

            # Plot intermediate fit components (Panel 2)
            # ------------------------------------------
            ax2.xaxis.tick_top()
            ax2.axhline(color='black', linewidth=0.5)
            ax2.plot(vel, data, '-k')
            ax2.yaxis.tick_right()
            for i in range(ncomps_f1):
                one_component = gaussian(params_f1[i], params_f1[i + ncomps_f1],
                                         params_f1[i + 2 * ncomps_f1])(vel)
                ax2.plot(vel, one_component, '-', color='blue')

            # Residual spectrum (Panel 3)
            # -----------------------------
            u22_scale = 1. / np.abs(u22).max() * np.max(residuals) * 0.5
            ax3.axhline(color='black', linewidth=0.5)
            ax3.plot(vel, residuals, '-k')
            ax3.plot(vel, np.ones(len(vel)) * agd2['thresh'], '--k')
            ax3.plot(vel, np.ones(len(vel)) * agd2['thresh2'] * u22_scale,
                     '--r')
            ax3.plot(vel, u22 * u22_scale, '-r')
            for i in range(ncomps_g2):
                one_component = gaussian(params_g2[i],
                                         params_g2[i + ncomps_g2],
                                         params_g2[i + 2 * ncomps_g2])(vel)
                ax3.plot(vel, one_component, '-g')

            # Plot best-fit model (Panel 4)
            # -----------------------------
            if perform_final_fit:
                ax4.yaxis.tick_right()
                ax4.axhline(color='black', linewidth=0.5)
                ax4.plot(vel, data, label='data', color='black')
                for i in range(ncomps_fit):
                    one_component = gaussian(params_fit[i],
                                             params_fit[i + ncomps_fit],
                                             params_fit[i + 2 * ncomps_fit])(vel)
                    ax4.plot(vel, one_component, '--', color='orange')
                ax4.plot(vel, best_fit_final, '-', color='orange', linewidth=2)

            plt.draw()
            plt.show()

            input(f"On alpha={alpha}")

            plt.close()

    # One final fit at the end
    if ncomps_gf > 0:

        # Finally, we'll do the final fit, check for insignificant components,
        # and refit with fewer components, if needed.
        def comp_sig(amp, fwhm):
            return (amp * fwhm) / (errors[0] * alphas[0] * 2.35)

        has_prev = False
        while True:

            # Objective functions for final fit
            def objective_leastsq(paramslm):
                params = vals_vec_from_lmfit(paramslm)
                resids = (func(vel, *params).ravel() - data.ravel()) / errors
                return resids

            # Final fit using unconstrained parameters
            lmfit_params = paramvec_to_lmfit(params_gf, dct['max_amp'],
                                             max_fwhm=dct['max_fwhm'],
                                             min_cent=dct['min_cent'],
                                             max_cent=dct['max_cent'])
            result2 = lmfit_minimize(objective_leastsq, lmfit_params,
                                     method='leastsq',
                                     maxfev=int(1e4) * spec.size)

            params_fit = vals_vec_from_lmfit(result2.params)
            params_errs = errs_vec_from_lmfit(result2.params)

            ncomps_fit = len(params_fit) // 3

            component_signif = comp_sig(params_fit[:ncomps_fit],
                                        params_fit[ncomps_fit: 2 * ncomps_fit])

            say(f"Component sig: {component_signif}", verbose)

            if not (component_signif >= component_sigma).all():
                # All good. Continue with this fit.
                comp_del = np.argmin(component_signif)

                say(f"Removing component {comp_del}; below"
                    " significance threshold", verbose)

                params_gf = np.delete(params_gf, 2 * ncomps_fit + comp_del)
                params_gf = np.delete(params_gf, ncomps_fit + comp_del)
                params_gf = np.delete(params_gf, comp_del)
                continue

            if not has_prev:
                last_result = copy(result2)
                has_prev = True
                continue

            # Compare BIC with previous. Remove components
            # until BIC jumps with a difference >10.
            bic_diff = result2.bic - last_result.bic
            say(f"Previous BIC: {last_result.bic} "
                f"Current BIC: {result2.bic} "
                f"Difference: {bic_diff}.",
                verbose)
            if bic_diff > 10:
                # Result to previous fit.
                result2 = last_result
                say(f"Converged with {len(result2.params) // 3} components.",
                    verbose)
                break

            if len(result2.params) // 3 == 1:
                # Exit with one component
                break

            last_result = copy(result2)

            component_signif = comp_sig(params_fit[:ncomps_fit],
                                        params_fit[ncomps_fit: 2 * ncomps_fit])
            comp_del = np.argmin(component_signif)

            params_gf = np.delete(params_gf, 2 * ncomps_fit + comp_del)
            params_gf = np.delete(params_gf, ncomps_fit + comp_del)
            params_gf = np.delete(params_gf, comp_del)
            say(f"Reducing to {ncomps_fit} components.", verbose)

        # Final fit parameters
        params_fit = vals_vec_from_lmfit(result2.params)
        params_errs = errs_vec_from_lmfit(result2.params)

        ncomps_fit = len(params_fit) // 3

        fit_results = result2

        del lmfit_params

        best_fit_final = func(vel, *params_fit).ravel()

    else:
        fit_results = None
        best_fit_final = np.zeros_like(data)
        ncomps_fit = 0

    if plot:
        # P L O T T I N G
        datamax = np.max(data)
        try:
            print(("params_fit:", params_fit))
        except UnboundLocalError:
            pass

        if ncomps_gf == 0:
            ncomps_fit = 0
            best_fit_final = data * 0

        rchi2 = goodness_of_fit(data, best_fit_final, errors, ncomps_fit)

        # Set up figure
        ax = plt.subplot(111)

        # Decorations
        plt.figtext(0.75, 0.75, 'Final fit (GaussPy)')
        # if perform_final_fit:
        plt.figtext(0.75, 0.73, 'Reduced Chi2: {0:3.3f}'.format(rchi2))
        plt.figtext(0.75, 0.71, 'N components: {0}'.format(ncomps_fit))

        ax.yaxis.tick_right()
        ax.axhline(color='black', linewidth=0.5)
        ax.axhline(errors[0], color='black', linewidth=0.5)
        ax.axhline(-errors[0], color='black', linewidth=0.5)
        ax.plot(vel, data, label='data', color='black')
        for i in range(ncomps_fit):
            one_component = gaussian(params_fit[i],
                                     params_fit[i + ncomps_fit],
                                     params_fit[i + 2 * ncomps_fit])(vel)
            ax.plot(vel, one_component, '--', color='orange')
        ax.plot(vel, best_fit_final, '-', color='orange', linewidth=2)

        ax.plot(vel, data - best_fit_final, '-', color='gray', zorder=-10)

        plt.draw()
        plt.show()

        input("Final fit")

        plt.close()

    # Lastly, map to the spectral axis given.
    chan_width = vels[1] - vels[0]

    if ncomps_fit > 0:
        # Convert to physical units
        ncomps = len(fit_results.params) // 3
        for i, par in enumerate(fit_results.params):

            if i < ncomps:
                # Amplitude. Don't change.
                continue
            elif i < 2 * ncomps:
                # FWHM.
                fwhm_par = fit_results.params[par].value

                # May need to change the bounds.
                fit_results.params[par].min = 0.0
                fit_results.params[par].max = np.inf

                fit_results.params[par].value = fwhm_par * np.abs(chan_width)

                fwhm_std = fit_results.params[par].stderr
                if fwhm_std is not None:
                    fit_results.params[par].stderr = fwhm_std * np.abs(chan_width)

            elif i < 3 * ncomps:
                # Centroid.

                cent_par = fit_results.params[par].value

                # May need to change the bounds.
                fit_results.params[par].min = - np.inf
                fit_results.params[par].max = np.inf

                fit_results.params[par].value = vels[0] + chan_width * cent_par

                cent_std = fit_results.params[par].stderr
                if cent_std is not None:
                    fit_results.params[par].stderr = cent_std * np.abs(chan_width)

            else:
                raise ValueError("Unexpected parameter name")

    return fit_results, vels, best_fit_final


def detect_wings(vels, params_g1, params_g2):
    '''
    Accounts for cases where the central narrow peak is identified in
    stage 1 and the remaining residuals peak on either side, leading to
    2 components being added in stage 2 when one is likely needed.

    Uses params_g1 to identify wings as components that overlap with the
    stage 1 parameter. Removes the stage 2 component with a smaller intensity.
    '''

    ncomp_g1 = len(params_g1) // 3
    ncomp_g2 = len(params_g2) // 3

    # Can't have wings with <2 components.
    if ncomp_g1 == 0 or ncomp_g2 < 2:
        return params_g1, params_g2

    for g1_comp in range(ncomp_g1):

        par1 = [params_g1[g1_comp],
                params_g1[ncomp_g1 + g1_comp],
                params_g1[ncomp_g1 * 2 + g1_comp]]

        grouped = []

        # Loop through g2
        for g2_comp in range(ncomp_g2):

            par2 = [params_g2[g2_comp],
                    params_g2[ncomp_g2 + g2_comp],
                    params_g2[ncomp_g2 * 2 + g2_comp]]

            if gaussian_overlap(vels, par1, par2):
                grouped.append([g2_comp, par2[0] * par2[1]])

        if len(grouped) >= 2:
            # Keep the max component. Remove the others.
            max_comp = max(grouped, key=lambda x: x[1])
            grouped.remove(max_comp)

            # Now remove the remaining components from params_g2
            for n, group in enumerate(grouped):
                if n == 0:
                    indices = [group[0], group[0] + ncomp_g2,
                               group[0] + ncomp_g2 * 2]
                else:
                    indices.extend([group[0], group[0] + ncomp_g2,
                                    group[0] + ncomp_g2 * 2])

            params_g2 = np.delete(params_g2, indices)
            ncomp_g2 = len(params_g2) // 3

    return params_g1, params_g2


def gaussian_overlap(vels, params1, params2):
    '''
    Return whether 2 Gaussians overlap.
    '''

    comp1 = gaussian(params1[0], params1[1], params1[2])(vels)
    comp2 = gaussian(params2[0], params2[1], params2[2])(vels)

    if comp1.sum() >= comp2.sum():
        max_comp = comp1
        min_comp = comp2
    else:
        max_comp = comp2
        min_comp = comp1

    diff_comp = max_comp - min_comp

    # Nearly complete overlap
    if np.isclose(diff_comp.sum(0), 0., atol=1e-3):
        return True

    # If sum over positive values of max_comp - min_comp is
    # the sum of max_comp, then the 2 components don't or barely overlap.
    if np.isclose(diff_comp[diff_comp > 0.].sum(), max_comp.sum(), atol=1e-3):
        return False

    return True


def cluster_components(params, ncomp, data_shape):
    '''
    Cluster the components based on:

    min [comp1_i - comp2_j]**2

    Basically minimize the difference between neighbouring
    components.
    '''

    # Make a copy of the params that we'll populate.
    clustered_params = params.copy()

    comp1 = gaussian(params1[0], params1[1], params1[2])(vels)
    comp2 = gaussian(params2[0], params2[1], params2[2])(vels)

