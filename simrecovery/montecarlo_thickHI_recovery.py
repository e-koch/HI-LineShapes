
'''
What are the parameter space restrictions given the noise and resolution in
our VLA data?

Statistics to consider:
- delta BIC with a single Gaussian.
- two-tail p-values for fraction of fits where parameter falls within 1-sigma
fit uncertainty.

Will run a grid over the parameter space and fit some number of spectra at each
set.
'''

from spectral_cube import SpectralCube, Projection, OneDSpectrum
from astropy.io import fits
import astropy.units as u
from astropy.table import Table
import os
import numpy as np
from lmfit import Parameters
from itertools import product
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


np.random.seed(4539803)

# Fit niter spectra at each point in the parameter space.
niter = 100

vel_min = -100 * u.km / u.s
vels_max = 100 * u.km / u.s

# Model evaluated at higher resolution
delta_v_up = 0.1 * u.km / u.s

vels_up = np.arange(vel_min.value, vels_max.value,  #  + delta_v_up.value,
                    delta_v_up.value) * u.km / u.s

# Model averaged to final channel sizes set by observations
delta_v = 0.42 * u.km / u.s

vels = np.arange(vel_min.value, vels_max.value,  #  + delta_v.value,
                 delta_v.value) * u.km / u.s

# Noise levels per 0.42 km/s channels
m31_noise = 2.8 * u.K
m33_noise = 2.4 * u.K / np.sqrt(2)

# Will be fitting the simplified thickHI model
# Range in Tpeak.
npts = 15
tpeaks = np.linspace(10., 200., npts)  # K
tss = np.linspace(15., 1000., npts)  # K
sigmas = np.linspace(1., 30., npts)  # km/s
# Just keeping the centroid at one position as this doesn't test the
# phys. parameters


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


def check_against_actual(act_params, out, nsig=1):
    '''
    Return whether the fit parameters fall within n-sigma of the actual
    values.
    '''

    if out.params['Tpeak'].stderr is None:
        return 0., 0., 0.

    Tpeak = act_params[0]
    low_fit = out.params['Tpeak'].value - nsig * out.params['Tpeak'].stderr
    up_fit = out.params['Tpeak'].value + nsig * out.params['Tpeak'].stderr
    p_Tpeak = 1 if ((Tpeak >= low_fit) & (Tpeak <= up_fit)) else 0

    Ts = act_params[1]
    low_fit = out.params['Ts'].value - nsig * out.params['Ts'].stderr
    up_fit = out.params['Ts'].value + nsig * out.params['Ts'].stderr
    p_Ts = 1 if ((Ts >= low_fit) & (Ts <= up_fit)) else 0

    sigma = act_params[2]
    low_fit = out.params['sigma'].value - nsig * out.params['sigma'].stderr
    up_fit = out.params['sigma'].value + nsig * out.params['sigma'].stderr
    p_sigma = 1 if ((sigma >= low_fit) & (sigma <= up_fit)) else 0

    return p_Tpeak, p_Ts, p_sigma


# We'll append all info in each line and save as a table at the end.
fit_results = []

for params in tqdm(product(tpeaks, tss, sigmas),
                   ascii=True, desc="M31 tests",
                   total=npts**3):

    pval_Ts = 0.
    pval_Tp = 0.
    pval_sigma = 0.

    delta_BICs = np.empty(niter, dtype=float)

    cur_iter = 0
    for spec in generate_spectra(niter, params[0], params[1], params[2],
                                 m31_noise):

        # p.plot(vels.value, spec)
        # p.draw()
        # input("?")
        # p.clf()

        out = fit_isoturbHI_model_simple(vels, spec,
                                         0.0 * u.km / u.s,
                                         delta_vcent=5 * u.km / u.s,
                                         err=m31_noise,
                                         verbose=False, plot_fit=False,
                                         return_model=False,
                                         use_emcee=False,
                                         emcee_kwargs={})

        in_range = check_against_actual(params, out)

        pval_Tp += in_range[0]
        pval_Ts += in_range[1]
        pval_sigma += in_range[2]

        # We'll also fit a single Gaussian to test where the models can be
        # distinguished.

        out_gauss = fit_gaussian(spec,
                                 vels=vels,
                                 vcent=0 * u.km / u.s,
                                 err=m31_noise,
                                 amp_const=None,
                                 cent_const=(-5, 5),
                                 sigma_const=None,
                                 verbose=False,
                                 plot_fit=False,
                                 use_emcee=False,
                                 emcee_kwargs={})

        # print(out_gauss.params)

        delta_BICs[cur_iter] = out_gauss.bic - out.bic

        cur_iter += 1

    pval_Tp /= float(niter)
    pval_Ts /= float(niter)
    pval_sigma /= float(niter)

    # Tp, Ts, sigma, pval_Tp, pval_Ts, pval_sigma, delta_BIC_mean,
    # delta_BIC_std
    fit_results.append([params[0], params[1], params[2],
                        pval_Tp, pval_Ts, pval_sigma,
                        np.nanmean(delta_BICs), np.nanstd(delta_BICs)])

# Convert list of results into a table and save as a CSV

tab = Table(np.array(fit_results),
            names=['Tpeak', 'Ts', 'sigma',
                   'p_Tpeak', 'p_Ts', 'p_sigma',
                   'delta_BIC_mean', 'delta_BIC_std'])

output_name = 'm31_synthetic_thickHI_recovery.csv'
tab.write(osjoin(output_path, output_name))


# Now with M33 at a different noise level. Unlikely to change
# much but this doesn't take that long to run.
fit_results = []

for params in tqdm(product(tpeaks, tss, sigmas),
                   ascii=True, desc="M33 tests",
                   total=npts**3):

    pval_Ts = 0.
    pval_Tp = 0.
    pval_sigma = 0.

    delta_BICs = np.empty(niter, dtype=float)

    cur_iter = 0
    for spec in generate_spectra(niter, params[0], params[1], params[2],
                                 m31_noise):

        # p.plot(vels.value, spec)
        # p.draw()
        # input("?")
        # p.clf()

        out = fit_isoturbHI_model_simple(vels, spec,
                                         0.0 * u.km / u.s,
                                         delta_vcent=5 * u.km / u.s,
                                         err=m33_noise,
                                         verbose=False, plot_fit=False,
                                         return_model=False,
                                         use_emcee=False,
                                         emcee_kwargs={})

        in_range = check_against_actual(params, out)

        pval_Tp += in_range[0]
        pval_Ts += in_range[1]
        pval_sigma += in_range[2]

        # We'll also fit a single Gaussian to test where the models can be
        # distinguished.

        out_gauss = fit_gaussian(spec,
                                 vels=vels,
                                 vcent=0 * u.km / u.s,
                                 err=m33_noise,
                                 amp_const=None,
                                 cent_const=(-5, 5),
                                 sigma_const=None,
                                 verbose=False,
                                 plot_fit=False,
                                 use_emcee=False,
                                 emcee_kwargs={})

        # print(out_gauss.params)

        delta_BICs[cur_iter] = out_gauss.bic - out.bic

        cur_iter += 1

    pval_Tp /= float(niter)
    pval_Ts /= float(niter)
    pval_sigma /= float(niter)

    # Tp, Ts, sigma, pval_Tp, pval_Ts, pval_sigma, delta_BIC_mean,
    # delta_BIC_std
    fit_results.append([params[0], params[1], params[2],
                        pval_Tp, pval_Ts, pval_sigma,
                        np.nanmean(delta_BICs), np.nanstd(delta_BICs)])

# Convert list of results into a table and save as a CSV

tab = Table(np.array(fit_results),
            names=['Tpeak', 'Ts', 'sigma',
                   'p_Tpeak', 'p_Ts', 'p_sigma',
                   'delta_BIC_mean', 'delta_BIC_std'])
output_name = 'm33_synthetic_thickHI_recovery.csv'
tab.write(osjoin(output_path, output_name))
