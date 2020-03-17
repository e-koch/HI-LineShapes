
'''
Use the simplified optically-thick HI model and only
fit the peaks.

The assumption here is that any flattened tops from optical depth should
be associated with the brightest features in the spectra.

Only fitting the subcubes for now.
'''

import os
import numpy as np
from astropy import units as u
from astropy.convolution import Gaussian1DKernel
from astropy.utils.console import ProgressBar
from astropy.stats import sigma_clip, mad_std
from spectral_cube import SpectralCube
from scipy import ndimage as nd
from radio_beam import Beam
import matplotlib.pyplot as plt
from corner import corner


from bigcubefitter.AGD_decomposer import AGD_loop

from cube_analysis.spectral_stacking_models import find_peak_window


repo_path = os.path.expanduser("~/ownCloud/project_code/ThickHIFitting/")

constants_script = os.path.join(repo_path, "paths.py")
exec(compile(open(constants_script, "rb").read(), constants_script, 'exec'))

# model_script = os.path.join(repo_path, "thickHI_model.py")
# exec(compile(open(model_script, "rb").read(), model_script, 'exec'))

from thickHI_model import fit_isoturbHI_model_simple
from gaussian_model import fit_multigaussian

from glob import glob

osjoin = os.path.join


fifteenAtapercubes = glob(fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("braun09_subcubes") + "/*.fits")

fifteenAtapercubes.sort()

noise_val = 2.8 * u.K  # K

# Change to data directory
os.chdir(fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("braun09_subcubes", no_check=True))

for i, subcube_name in enumerate(fifteenAtapercubes[::-1]):
# for i, subcube_name in enumerate(fifteenAtapercubes):

    subcube_filename = os.path.split(subcube_name)[-1][:-5]

    subcube = SpectralCube.read(subcube_name)

    # Smooth to 25"
    # new_beam = Beam(25 * u.arcsec)
    new_beam = subcube.beams.common_beam()
    subcube = subcube.convolve_to(new_beam)

    # Downsample the spectral resolution to 2.3 km/s
    chan_width = np.diff(subcube.spectral_axis)[0]
    # target = 2.3 * u.km / u.s
    # target = chan_width * 2
    # spec_kern = Gaussian1DKernel(np.sqrt((target / 2.)**2 - chan_width**2).value)
    # subcube = subcube.spectral_smooth(spec_kern, verbose=True,
    #                                   num_cores=4)

    # New spectral axis
    # unit = subcube.spectral_axis.unit
    # newchan_width = target.to(unit).value
    # # if chan_width.value < 0:
    # #     newchan_width *= -1.
    # spec_axis = np.arange(subcube.spectral_axis[0].value,
    #                       subcube.spectral_axis[-1].value,
    #                       newchan_width) * unit
    # assert spec_axis.size > 0
    # subcube = subcube.spectral_interpolate(spec_axis)

    # err_arr = noise_val * np.ones(subcube.shape[1:])

    peaktemp = subcube.max(axis=0)
    vcent = subcube.moment1()

    peakchans = subcube.argmax(axis=0)
    peakvels = np.take_along_axis(subcube.spectral_axis[:, np.newaxis,
                                                        np.newaxis],
                                  peakchans[np.newaxis, :, :], 0)
    peakvels = peakvels.squeeze()
    peakvels = peakvels.to(u.km / u.s)

    # peak_name = fifteenA_HI_BCtaper_wEBHIS_HI_file_dict['PeakTemp']
    # peaktemp = Projection.from_hdu(fits.open(peak_name))

    # vcent_name = fourteenA_wEBHIS_HI_file_dict['Moment1']
    # vcent = Projection.from_hdu(fits.open(vcent_name)).to(u.km / u.s)

    # Restrict number of positions to fit.
    mask_peak = peaktemp >= 5 * u.K
    # Must have 5 channels above half the peak, following Braun+09
    mask_halfabovepeak = (subcube.filled_data[:] > 5 * u.K).sum(0) > 5

    mask_positions = np.where(np.logical_and(mask_peak,
                                             mask_halfabovepeak))

    # Try ordering by peak temperature.
    # mask_positions = np.unravel_index(np.argsort(peaktemp.value.ravel())[::-1],
    #                                   peaktemp.shape)

    # Parameters for the fit output
    thickHI_params = np.zeros((4,) + peaktemp.shape)
    thickHI_uncerts = np.zeros((4,) + peaktemp.shape)

    thickHI_model_cube = np.zeros(subcube.shape)

    max_gauss_comps = 10
    multigauss_params = np.zeros((3 * max_gauss_comps,) + peaktemp.shape)
    multigauss_uncerts = np.zeros((3 * max_gauss_comps,) + peaktemp.shape)

    multigauss_comps = np.zeros(peaktemp.shape)

    multigauss_model_cube = np.zeros(subcube.shape)

    fit_bics = np.zeros((2,) + peaktemp.shape)

    show_plots = False
    # show_plots = True

    pbar = ProgressBar(len(mask_positions[0]))

    for y, x in zip(mask_positions[0], mask_positions[1]):

        pbar.update()
        # print(f"{y}, {x}")

        spec = subcube[:, y, x].with_spectral_unit(u.km / u.s)

        # Fit that spectrum.

        thickHI_fit, vels, thickHI_fit_model = \
            fit_isoturbHI_model_simple(spec.spectral_axis,  # [spec_mask],
                                       spec,  # [spec_mask],
                                       peakvels[y, x],
                                       err=noise_val,
                                       delta_vcent=10 * u.km / u.s,
                                       verbose=show_plots,
                                       plot_fit=show_plots,
                                       use_emcee=False,
                                       return_model=True,
                                       emcee_kwargs={'nwalkers': 4 * 10,
                                                     'burn': 2000,
                                                     'steps': 10000,
                                                     'workers': 4})

        agd_kwargs = {"plot": show_plots,
                      "verbose": show_plots,
                      "SNR_thresh1": 5.,
                      "SNR_thresh2": 8.,
                      "SNR2_thresh1": 4.,
                      "SNR2_thresh2": 3.5,
                      "mode": "conv",
                      # "mode": "python",
                      "deblend": True,
                      "intermediate_fit": False,
                      "perform_final_fit": False,
                      "component_sigma": 5.}

        alphas = [5., 10., 15., 20., 30., 50.]

        multigauss_fit, vels, multigauss_fit_model = \
            AGD_loop(spec,
                     noise_val.value * np.ones_like(vels),
                     alphas,
                     **agd_kwargs)

        # multigauss_fit, vels, multigauss_fit_model = \
        #     fit_multigaussian(spec,
        #                       vcent=peakvels[y, x],
        #                       err=noise_val,
        #                       max_comp=max_gauss_comps,
        #                       amp_const=(0.5 * noise_val.value,
        #                                  1.1 * peaktemp[y, x].value),
        #                       cent_const=None,  # (valid_min, valid_max),
        #                       sigma_const=(0.8, 50.),
        #                       sigma_init=6.,
        #                       verbose=show_plots,
        #                       plot_fit=show_plots,
        #                       min_delta_BIC=10,
        #                       min_sigma_intensity=10,
        #                       return_model=True,
        #                       discrete_fitter=False)

        assert thickHI_fit.success is True
        assert multigauss_fit.success is True

        # plt.draw()

        # input("?")

        # plt.close()

        thickHI_model_cube[:, y, x] = thickHI_fit_model
        multigauss_model_cube[:, y, x] = multigauss_fit_model
        multigauss_comps[y, x] = len(multigauss_fit.params) // 3

        thickHI_params[:, y, x] = [thickHI_fit.params[par].value
                                   for par in thickHI_fit.params]
        if hasattr(thickHI_fit, 'covar'):
            thickHI_uncerts[:, y, x] = [thickHI_fit.params[par].stderr for par
                                        in thickHI_fit.params]
        else:
            thickHI_uncerts[:, y, x] = np.NaN

        for ii, par in enumerate(multigauss_fit.params):
            multigauss_params[ii, y, x] = multigauss_fit.params[par].value
            if hasattr(multigauss_fit, 'covar'):
                multigauss_uncerts[ii, y, x] = multigauss_fit.params[par].stderr
            else:
                multigauss_uncerts[ii, y, x] = np.NaN

        # for ncomp in range(len(multigauss_fit.params) // 3):
        #     par = [multigauss_fit.params[f'amp{ncomp + 1}'].value,
        #            multigauss_fit.params[f'cent{ncomp + 1}'].value,
        #            multigauss_fit.params[f'sigma{ncomp + 1}'].value]
        #     multigauss_params[3 * ncomp:3 * (ncomp + 1), y, x] = par

        #     stderr = [multigauss_fit.params[f'amp{ncomp + 1}'].stderr,
        #               multigauss_fit.params[f'cent{ncomp + 1}'].stderr,
        #               multigauss_fit.params[f'sigma{ncomp + 1}'].stderr]
        #     if hasattr(multigauss_fit, 'covar'):
        #         multigauss_uncerts[3 * ncomp:3 * (ncomp + 1), y, x] = stderr
        #     else:
        #         multigauss_uncerts[3 * ncomp:3 * (ncomp + 1), y, x] = np.NaN

        fit_bics[0, y, x] = thickHI_fit.bic
        fit_bics[1, y, x] = multigauss_fit.bic

        if show_plots:
            plt.draw()
            input(f"{y} {x}")
            # plt.close('all')
            plt.clf()
    print(argh)
