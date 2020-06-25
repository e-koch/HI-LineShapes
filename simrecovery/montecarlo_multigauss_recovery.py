
'''
Determine completeness limits for the M31 and M33
0.42 km/s multigauss fitting.

Will use range of observed parameters, but extended to
low amplitude and line widths for recovery tests.
Also, will draw from a dirichlet prior for number of
Gaussians.

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
from multiprocessing import Pool

from cube_analysis.spectral_fitting import sample_at_channels

osjoin = os.path.join

repo_path = os.path.expanduser("~/ownCloud/project_code/ThickHIFitting/")

# Save into bigdata as some file may be larger
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


def generate_spectra(niter, params, noise_std):
    '''
    Yield `niter` spectra with the given parameters with noise added.
    '''

    for it in range(niter):

        model = multigaussian_nolmfit(vels_up.value, params)

        model += np.random.normal(0., noise_std.value, vels_up.size)

        model_finchan = sample_at_channels(vels.value, vels_up.value, model)

        yield model_finchan * u.K


# Define fixed parameters
np.random.seed(4539803)

# Fit niter spectra at each point in the parameter space.
niter = 10000
nspec = 10

vel_min = -125 * u.km / u.s
vels_max = 125 * u.km / u.s

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


# Component numbers and weighting
ncomp_max = 8
ncomp_min = 1
weights = [0.4, 0.3, 0.18, 0.1, 0.01, 0.005, 0.0025, 0.0025]
assert len(weights) == ncomp_max

ncomp_max_fitter = 16

# Sample across the approx. observed distributions.
# TODO: update these to something the better represents the distribution shape

limits_dict = {'tpeak': [m31_noise.value, 120],
               'vcent': [-50, 50],
               'sigma': [delta_v.value, 60.]}

# Spectral WCS
my_wcs = wcs.WCS(header={'CDELT1': 0.42e3,
                         'CRVAL1': vels[0].value * 1000.,
                         'CUNIT1': 'm/s',
                         'CTYPE1': 'VRAD',
                         'RESTFRQ': 1.42040575177e9,
                         'CRPIX1': 1})


def parallel_fitter(spec):

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
                           max_comp=ncomp_max_fitter,
                           agd_kwargs=agd_kwargs)
    return out


ncomps_act = np.empty(niter, dtype=int)

mgauss_BICs = np.empty(niter * nspec, dtype=float)
ncomps_fit = np.empty(niter * nspec, dtype=int)

mgauss_act_params = np.empty((niter, ncomp_max * 3), dtype=float)
mgauss_fit_params = np.empty((niter * nspec,
                              ncomp_max_fitter * 3), dtype=float) * np.NaN
mgauss_fit_uncerts = np.empty((niter * nspec,
                               ncomp_max_fitter * 3), dtype=float) * np.NaN


cur_iter = 0

pool = Pool(10)

for it in tqdm(range(niter),
               ascii=True,
               desc="Synth MGauss tests",
               total=niter):

    # np.arange(8) is from 0 to 7. So add 1.
    ncomp = np.random.choice(ncomp_max, p=weights) + 1

    ncomps_act[it] = ncomp

    # print(f"Actual ncomp: {ncomp}")

    tpeaks = np.random.uniform(limits_dict['tpeak'][0],
                               limits_dict['tpeak'][1],
                               ncomp)

    vcents = np.random.uniform(limits_dict['vcent'][0],
                               limits_dict['vcent'][1],
                               ncomp)

    sigmas = np.random.uniform(limits_dict['sigma'][0],
                               limits_dict['sigma'][1],
                               ncomp)

    params = np.array(list(zip(tpeaks, vcents, sigmas))).ravel()

    mgauss_act_params[it, :len(params)] = params

    specs = [spec for spec in generate_spectra(nspec, params, m31_noise)]

    outs = pool.map(parallel_fitter, specs)

    for out in outs:

        mgauss_BICs[cur_iter] = out[2]

        ncomps_fit[cur_iter] = np.isfinite(out[0]).sum() // 3

        mgauss_fit_params[cur_iter] = out[0]
        mgauss_fit_uncerts[cur_iter] = out[1]

        cur_iter += 1

pool.close()
pool.join()

# Assign separate arrays for distinct vs. blended components

distinct_act = np.zeros((niter, ncomp_max), dtype=bool)

for i, pars in enumerate(tqdm(mgauss_act_params)):

    distinct, blend = distinct_vs_blended(pars, m31_noise * 3, vels.to(u.m / u.s),
                                          max_chan_diff=3,
                                          secderiv_fraction=0.75)

    distinct_act[i][distinct] = True


distinct_fit = np.zeros((niter * nspec, ncomp_max_fitter), dtype=bool)

for i, pars in enumerate(tqdm(mgauss_fit_params)):

    distinct, blend = distinct_vs_blended(pars, m31_noise * 3, vels,
                                          max_chan_diff=3,
                                          secderiv_fraction=0.75)

    distinct_fit[i][distinct] = True

# Convert list of results into a table and save as a CSV

tab = Table([np.tile(ncomps_act, (nspec, 1)).T.ravel(),
             ncomps_fit, mgauss_BICs],
            names=['ncomp_act', 'ncomp_fit', 'mgauss_BIC'])

output_name = 'multigauss_synthetic_fitting_tests.csv'
tab.write(osjoin(output_path, output_name), overwrite=True)

# Save params and fit params separately.
data_save_file = "multigauss_synthetic_fitting_tests.npz"

# The compressed file is ~1.6 MB.

np.savez_compressed(f"{output_path}/{data_save_file}",
                    mgauss_act_params=mgauss_act_params,
                    distinct_act=distinct_act,
                    mgauss_fit_params=mgauss_fit_params,
                    mgauss_fit_uncerts=mgauss_fit_uncerts,
                    distinct_fit=distinct_fit)
