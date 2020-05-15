
'''
Fit multi-Gaussian models to synthetic thick HI spectra.
'''

from spectral_cube import OneDSpectrum
from astropy.io import fits
import astropy.wcs as wcs
import astropy.units as u
from astropy.table import Table
import os
import numpy as np
from lmfit import Parameters
from tqdm import tqdm

from cube_analysis.spectral_fitting import sample_at_channels

osjoin = os.path.join

repo_path = os.path.expanduser("~/ownCloud/project_code/ThickHIFitting/")

# Files should be small enough to keep here.
output_path = osjoin(repo_path, 'simrecovery')

constants_script = os.path.join(repo_path, "paths.py")
exec(compile(open(constants_script, "rb").read(), constants_script, 'exec'))

model_script = os.path.join(repo_path, "gaussian_model.py")
exec(compile(open(model_script, "rb").read(), model_script, 'exec'))

thickHI_model_script = os.path.join(repo_path, "thickHI_model.py")
exec(compile(open(thickHI_model_script, "rb").read(),
             thickHI_model_script, 'exec'))

modelcomp_script = os.path.join(repo_path, "model_selection_tools.py")
exec(compile(open(modelcomp_script, "rb").read(), modelcomp_script, 'exec'))


def generate_spectra(niter, Tp, Ts, sigma, noise_std, vcent=0 * u.km / u.s,
                     verbose=False):
    '''
    Yield `niter` spectra with the given parameters with noise added.
    '''

    pars = Parameters()
    pars.add(name='Ts', value=Ts)
    pars.add(name='sigma', value=sigma)
    pars.add(name='Tpeak', value=Tp)
    pars.add(name='vcent', value=vcent.value)

    for it in range(niter):

        model = isoturbHI_simple(vels_up.value, pars['Ts'],
                                 pars['sigma'],
                                 pars['Tpeak'], pars['vcent'])

        model_finchan = sample_at_channels(vels.value, vels_up.value, model)

        yield (model_finchan + np.random.normal(0., noise_std.value, vels.size)) * u.K


# Define fixed parameters
np.random.seed(4539803)

# Fit niter spectra at each point in the parameter space.
# niter = 100000
# niter = 10000
niter = 20000
nspec = 10

vel_min = -200 * u.km / u.s
vels_max = 200 * u.km / u.s

# Model evaluated at higher resolution
delta_v_up = 0.1 * u.km / u.s

vels_up = np.arange(vel_min.value, vels_max.value,
                    delta_v_up.value) * u.km / u.s

# Model averaged to final channel sizes set by observations
delta_v = 0.42 * u.km / u.s

vels = np.arange(vel_min.value, vels_max.value,
                 delta_v.value) * u.km / u.s

# Noise levels per 0.42 km/s channels
m31_noise = 2.8 * u.K
m33_noise = 2.4 * u.K / np.sqrt(2)


# Sample across the approx. observed distributions.
# TODO: update these to something the better represents the distribution shape

# tpeaks = np.random.uniform(10., 120., niter)  # K
tpeaks = np.random.uniform(30., 120., niter)  # K

tss = np.random.uniform(15., 8000., niter)  # K
# Force higher taus to compare
# taus = np.random.uniform(0.5, 3., niter)
# taus = np.random.uniform(0.01, 3., niter)
# Then calc the actual Ts
# tss = tpeaks / taus

sigmas = np.random.uniform(3., 30., niter)  # km/s
# sigmas = np.random.uniform(1., 30., niter)  # km/s

# Just keeping the centroid at one position as this doesn't test the
# phys. parameters

# Spectral WCS
my_wcs = wcs.WCS(header={'CDELT1': 0.42e3,
                         'CRVAL1': vels[0].value * 1000.,
                         'CUNIT1': 'm/s',
                         'CTYPE1': 'VRAD',
                         'RESTFRQ': 1.42040575177e9,
                         'CRPIX1': 1})


mgauss_BICs = np.empty(niter * nspec, dtype=float)
thick_BICs = np.empty(niter * nspec, dtype=float)
ncomps = np.empty(niter * nspec, dtype=float)

mgauss_BICs_taulim = np.empty(niter * nspec, dtype=float)
thick_BICs_taulim = np.empty(niter * nspec, dtype=float)
npts_taulim = np.empty(niter * nspec, dtype=float)
ncomps_taulim = np.empty(niter * nspec, dtype=float)
darkint_total = np.empty(niter * nspec, dtype=float)
optthinint_total = np.empty(niter * nspec, dtype=float)
darkint_taulim = np.empty(niter * nspec, dtype=float)
optthinint_taulim = np.empty(niter * nspec, dtype=float)


cur_iter = 0

check_params = []

for params in tqdm(zip(tpeaks, tss, sigmas),
                   ascii=True, desc="M31 tests",
                   total=niter):

    for spec in generate_spectra(nspec, params[0], params[1], params[2],
                                 m31_noise):

        # Fit the thickHI model for comparison
        out0 = fit_isoturbHI_model_simple(vels, spec,
                                          0.0 * u.km / u.s,
                                          delta_vcent=5 * u.km / u.s,
                                          err=m31_noise,
                                          verbose=False,
                                          # plot_fit=True,
                                          plot_fit=False,
                                          return_model=False,
                                          use_emcee=False,
                                          emcee_kwargs={})

        thick_BICs[cur_iter] = out0.bic

        # Make into a OneDSpectrum
        spec_obj = OneDSpectrum(spec, wcs=my_wcs)
        spec_obj = spec_obj.with_spectral_unit(u.m / u.s)

        agd_kwargs = {"plot": False,
                      "verbose": False,
                      "SNR_thresh1": 5.,
                      "SNR_thresh2": 8.,
                      "SNR2_thresh1": 4.,
                      "SNR2_thresh2": 3.5,
                      "mode": "conv",
                      "deblend": True,
                      "intermediate_fit": False,
                      "perform_final_fit": False,
                      "component_sigma": 5.}

        alphas = [5., 10., 15., 20., 30., 50.]

        out = fit_func_gausspy(spec_obj, m31_noise,
                               alphas=alphas,
                               vcent=None,
                               min_finite_chan=30,
                               downsamp_factor=1,
                               max_comp=10,
                               agd_kwargs=agd_kwargs)

        mgauss_BICs[cur_iter] = out[2]
        ncomps[cur_iter] = np.isfinite(out[0]).sum() // 3

        # Now compute the BICs only where tau > 0.5
        tau_min = 0.5
        min_pts = 10
        gausscomp_frac = 0.25

        # Dealing with small rounding errors, where the central channel is
        # evaluated lower than the fit due to the channel size
        if (out0.params['Tpeak'] / out0.params['Ts']) < (tau_min + 0.01):
            out_taulim = [np.NaN] * 8
        elif ncomps[cur_iter] == 0.:
            out_taulim = [np.NaN] * 8
        else:
            out_taulim = \
                compare_optthick_residual(spec_obj,
                                          np.array([val.value for val in out0.params.values()]),
                                          out[0][np.isfinite(out[0])],
                                          m31_noise,
                                          vels=vels,
                                          tau_min=tau_min,
                                          min_pts=min_pts,
                                          gausscomp_frac=gausscomp_frac)

        mgauss_BICs_taulim[cur_iter] = out_taulim[1]
        thick_BICs_taulim[cur_iter] = out_taulim[0]
        npts_taulim[cur_iter] = out_taulim[2]
        ncomps_taulim[cur_iter] = out_taulim[3]
        darkint_total[cur_iter] = out_taulim[4]
        optthinint_total[cur_iter] = out_taulim[5]
        darkint_taulim[cur_iter] = out_taulim[6]
        optthinint_taulim[cur_iter] = out_taulim[7]

        cur_iter += 1

        check_params.append(params)

# Convert list of results into a table and save as a CSV

tab = Table([np.tile(tpeaks, (nspec, 1)).T.ravel(),
             np.tile(tss, (nspec, 1)).T.ravel(),
             np.tile(sigmas, (nspec, 1)).T.ravel(),
             thick_BICs, mgauss_BICs, ncomps,
             thick_BICs_taulim, mgauss_BICs_taulim, ncomps_taulim,
             npts_taulim, darkint_total, optthinint_total,
             darkint_taulim, optthinint_taulim
             ],
            names=['Tpeak', 'Ts', 'sigma',
                   'thick_BIC', 'mgauss_BIC', 'mgauss_ncomps',
                   'thick_BIC_taulim', 'mgauss_BIC_taulim',
                   'mgauss_ncomps_taulim',
                   'npts_taulim',
                   'darkint_total', 'optthinint_total',
                   'darkint_taulim', 'optthinint_taulim'])

# output_name = 'm31_synthetic_thickHI_multigauss_comparison.csv'
output_name = 'm31_synthetic_thickHI_multigauss_comparison_with_lowtau.csv'
tab.write(osjoin(output_path, output_name), overwrite=True)


# # Now with M33 at a different noise level. Unlikely to change
# # much but this doesn't take that long to run.
# fit_results = []

# for params in tqdm(product(tpeaks, tss, sigmas),
#                    ascii=True, desc="M33 tests",
#                    total=npts**3):

#     pval_Ts = 0.
#     pval_Tp = 0.
#     pval_sigma = 0.

#     delta_BICs = np.empty(niter, dtype=float)

#     cur_iter = 0
#     for spec in generate_spectra(niter, params[0], params[1], params[2],
#                                  m31_noise):

#         # p.plot(vels.value, spec)
#         # p.draw()
#         # input("?")
#         # p.clf()

#         out = fit_isoturbHI_model_simple(vels, spec,
#                                          0.0 * u.km / u.s,
#                                          delta_vcent=5 * u.km / u.s,
#                                          err=m33_noise,
#                                          verbose=False, plot_fit=False,
#                                          return_model=False,
#                                          use_emcee=False,
#                                          emcee_kwargs={})

#         in_range = check_against_actual(params, out)

#         pval_Tp += in_range[0]
#         pval_Ts += in_range[1]
#         pval_sigma += in_range[2]

#         # We'll also fit a single Gaussian to test where the models can be
#         # distinguished.

#         out_gauss = fit_gaussian(spec,
#                                  vels=vels,
#                                  vcent=0 * u.km / u.s,
#                                  err=m33_noise,
#                                  amp_const=None,
#                                  cent_const=(-5, 5),
#                                  sigma_const=None,
#                                  verbose=False,
#                                  plot_fit=False,
#                                  use_emcee=False,
#                                  emcee_kwargs={})

#         # print(out_gauss.params)

#         delta_BICs[cur_iter] = out_gauss.bic - out.bic

#         cur_iter += 1

#     pval_Tp /= float(niter)
#     pval_Ts /= float(niter)
#     pval_sigma /= float(niter)

#     # Tp, Ts, sigma, pval_Tp, pval_Ts, pval_sigma, delta_BIC_mean,
#     # delta_BIC_std
#     fit_results.append([params[0], params[1], params[2],
#                         pval_Tp, pval_Ts, pval_sigma,
#                         np.nanmean(delta_BICs), np.nanstd(delta_BICs)])

# # Convert list of results into a table and save as a CSV

# tab = Table(np.array(fit_results),
#             names=['Tpeak', 'Ts', 'sigma',
#                    'p_Tpeak', 'p_Ts', 'p_sigma',
#                    'delta_BIC_mean', 'delta_BIC_std'])
# output_name = 'm33_synthetic_thickHI_recovery.csv'
# tab.write(osjoin(output_path, output_name))
