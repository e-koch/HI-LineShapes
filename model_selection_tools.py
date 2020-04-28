
'''
Tools for closer comparisons between the multi-Gauss and opt-thick models:
* recalculate BIC
* change range where BIC is calculated
'''

import numpy as np
import astropy.units as u
import os
from astropy.io import fits
from spectral_cube import SpectralCube
from tqdm import tqdm

repo_path = os.path.expanduser("~/ownCloud/project_code/ThickHIFitting/")

gaussmodel_script = os.path.join(repo_path, "gaussian_model.py")
exec(compile(open(gaussmodel_script, "rb").read(), gaussmodel_script, 'exec'))
thickHImodel_script = os.path.join(repo_path, "thickHI_model.py")
exec(compile(open(thickHImodel_script, "rb").read(), thickHImodel_script, 'exec'))


def recalculate_bic(spec, spec_model, noise_val, Npar, mask=None):
    '''
    Add a function to recompute the BIC. Assumes a Gaussian likelihood
    function
    (https://lmfit.github.io/lmfit-py/fitting.html#akaike-and-bayesian-information-criteria)
    '''

    noisesq = noise_val.value**2.

    if mask is not None:

        assert mask.shape == spec.shape

        chisq = np.nansum((spec[mask].value - spec_model[mask])**2 / noisesq)

        Nfree = float(spec[mask].size)

    else:

        chisq = np.nansum((spec.value - spec_model)**2 / noisesq)

        Nfree = float(spec.size)

    return Nfree * np.log(chisq / (Nfree)) + np.log(Nfree) * Npar


def compare_optthick_residual(spec, params_thickHI, params_multigauss,
                              noise_val,
                              vels=None, tau_min=0.5,
                              min_pts=25,
                              gausscomp_frac=0.25,):
    '''
    Compare the thick and multi-Gauss model but only where tau is
    appreciable. From this, return a fit-like statistic.

    Parameters
    ----------
    tau_min : float, optional
        Minimum tau to consider in flattened-tops. Default is 0.5.
    min_pts : int, optional
        Minimum number of points required for the comparison. NaNs are
        returned when there are not enough points above the `min_tau`.
        Default is 25.
    gausscomp_frac : float, optional
        Include a Gaussian component within the `tau_mask` region when
        its area within the region exceed this fraction. Default is 0.25.

    '''

    assert gausscomp_frac > 0.
    assert gausscomp_frac <= 1.

    assert noise_val.unit == u.K

    if vels is None:
        vels = spec.with_spectral_unit(u.m / u.s).spectral_axis.value
    else:
        vels = vels.to(u.m / u.s).value

    mod_multigauss = multigaussian_nolmfit(vels, params_multigauss)
    mod_thickHI = isoturbHI_simple(vels / 1000., params_thickHI[0],
                                   params_thickHI[1],
                                   params_thickHI[2],
                                   params_thickHI[3])

    # Define where tau is appreciable and would flatten the spectrum.
    tau_profile = tau_func(vels / 1000., params_thickHI[0],
                           params_thickHI[1],
                           params_thickHI[2],
                           params_thickHI[3])

    # Region comparison over the high tau areas
    tau_mask = tau_profile >= tau_min

    if tau_mask.sum(0) == 0:
        raise ValueError("tau_min exceeds all tau values.")

    if tau_mask.sum(0) < min_pts:
        return [np.NaN] * 5

    # chisq_thickHI = np.nansum((spec.value[tau_mask] - mod_thickHI[tau_mask])**2 / noise_val.value**2)
    # chisq_multigauss = np.nansum((spec.value[tau_mask] - mod_multigauss[tau_mask])**2 / noise_val.value**2)

    # Can compare to expectation for pure noise.
    # chisq_noise = tau_mask.sum() * noise_val.value

    # return chisq_thickHI, chisq_multigauss

    # Use to calculate the BIC for direct comparison

    # Within that mask, find the Gaussian components that contribute
    # appreciably to best estimate the number of free parameters.
    Ncomp_region = 0
    for nc in range(params_multigauss.size // 3):

        # Evaluate that component
        mod_mg_comp = multigaussian_nolmfit(vels,
                                            params_multigauss[3 * nc: 3 * nc + 3])

        comp_frac = mod_mg_comp[tau_mask].sum() / mod_mg_comp.sum()

        if comp_frac >= gausscomp_frac:
            Ncomp_region += 1

    bic_thickHI = recalculate_bic(spec, mod_thickHI,
                                  noise_val, 4, mask=tau_mask)
    bic_multigauss = recalculate_bic(spec, mod_multigauss, noise_val,
                                     Ncomp_region * 3, mask=tau_mask)

    # Also compute the optically-thin limit to estimate amount
    # of integrated intensity lost in the thick model
    par = np.array([params_thickHI[2], params_thickHI[3],
                    params_thickHI[1]])
    mod_thickHI_thinlimit = multigaussian_nolmfit(vels, par)
    missing_intint = (mod_thickHI_thinlimit - mod_thickHI)[tau_mask].sum()

    return (bic_thickHI, bic_multigauss, tau_mask.sum(),
            Ncomp_region, missing_intint)


def compare_optthick_over_cube(cube_name, params_thickHI_name,
                               params_multigauss_name,
                               noise_map,
                               tau_min=0.5,
                               min_pts=25,
                               gausscomp_frac=0.25,
                               chunk_size=80000):
    '''
    Run `compare_optthick_residual` over a whole cube.
    '''

    with fits.open(params_multigauss_name) as params_hdu:

        params_hdr = params_hdu[0].header

        params_multigauss = params_hdu[0].data

        # Cut to max comps
        max_ncomp = np.max(np.isfinite(params_multigauss).sum(0))

        params_multigauss = params_multigauss[:3 * max_ncomp + 3]

    with fits.open(params_thickHI_name) as params_hdu:

        params_thickHI = params_hdu[0].data

    taupeak_map = params_thickHI[2] / params_thickHI[0]

    cube = SpectralCube.read(cube_name)
    assert cube.shape[1:] == params_thickHI.shape[1:]
    vels = cube.spectral_axis.to(u.m / u.s)

    yposn, xposn = np.indices(cube.shape[1:])

    # Output array for recalculated BICs
    # Last image is to return the number of points used in the
    # comparison
    out_bics = np.zeros((5,) + cube.shape[1:]) * np.NaN

    del cube._data
    del cube

    cube_hdu = fits.open(cube_name, mode='denywrite')

    for i, (y, x) in tqdm(enumerate(zip(yposn.ravel(),
                                        xposn.ravel())),
                          ascii=True,
                          desc=f"Model eval. thick HI only",
                          total=yposn.size):

        # Reload cube to release memory
        if i % chunk_size == 0:

            cube_hdu.close()
            del cube_hdu[0].data
            del cube_hdu
            cube_hdu = fits.open(cube_name, mode='denywrite')

        if np.isnan(taupeak_map[y, x]):
            continue

        # I got a weird rounding error. Bump up comparison by epsilon.
        if taupeak_map[y, x] < tau_min + 1e-3:
            continue

        spec = cube_hdu[0].data[:, y, x] * u.K

        params_mg = params_multigauss[:, y, x]

        if np.isnan(params_mg).all():
            continue

        params_mg = params_mg[np.isfinite(params_mg)]

        out = compare_optthick_residual(spec, params_thickHI[:, y, x],
                                        params_mg,
                                        noise_map[y, x],
                                        vels=vels,
                                        tau_min=tau_min,
                                        min_pts=min_pts,
                                        gausscomp_frac=gausscomp_frac)

        out_bics[:, y, x] = out

    del cube_hdu[0].data
    del cube_hdu

    # Output an HDU to keep the WCS info
    hdu_bic_thickHI = fits.PrimaryHDU(out_bics[0], params_hdr)
    hdu_bic_mg = fits.ImageHDU(out_bics[1], params_hdr)

    # Number of channels used in estimate
    hdu_npts = fits.ImageHDU(out_bics[2], params_hdr)

    # Number of Gaussian components within region
    hdu_gcomp = fits.ImageHDU(out_bics[3], params_hdr)

    # Apparent missing integrated intensity from the optically thick
    # model.
    hdu_miss = fits.ImageHDU(out_bics[4], params_hdr)

    # Record the peak taus for quick access
    hdu_peaktau = fits.ImageHDU(taupeak_map, params_hdr)

    return fits.HDUList([hdu_bic_thickHI, hdu_bic_mg, hdu_npts,
                         hdu_gcomp, hdu_miss, hdu_peaktau])
