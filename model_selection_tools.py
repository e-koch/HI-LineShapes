
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

    vel_diff = np.abs(vels[1] - vels[0])

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

    # if tau_mask.sum(0) == 0:
    #     raise ValueError("tau_min exceeds all tau values.")

    # Also compute the optically-thin limit to estimate amount
    # of integrated intensity lost in the thick model
    par = np.array([params_thickHI[2], params_thickHI[3],
                    params_thickHI[1]])
    mod_thickHI_thinlimit = multigaussian_nolmfit(vels / 1000., par)

    # Always return the total values

    missing_intint_total = (mod_thickHI_thinlimit - mod_thickHI).sum() * vel_diff
    optthin_intint_total = mod_thickHI_thinlimit.sum() * vel_diff

    if tau_mask.sum(0) < min_pts:
        return [np.NaN, np.NaN, np.NaN, np.NaN,
                missing_intint_total, optthin_intint_total,
                np.NaN, np.NaN]

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

    missing_intint_taulim = (mod_thickHI_thinlimit - mod_thickHI)[tau_mask].sum() * vel_diff

    optthin_intint_taulim = mod_thickHI_thinlimit[tau_mask].sum() * vel_diff


    return (bic_thickHI, bic_multigauss, tau_mask.sum(),
            Ncomp_region, missing_intint_total, optthin_intint_total,
            missing_intint_taulim, optthin_intint_taulim)


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
    out_bics = np.zeros((8,) + cube.shape[1:]) * np.NaN

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
        if taupeak_map[y, x] < tau_min + 1e-2:
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

    # And the total optically-thin integrated intensity for comparison
    hdu_optthin = fits.ImageHDU(out_bics[5], params_hdr)

    # Same but limited to only the >tau mask
    hdu_miss_taulim = fits.ImageHDU(out_bics[6], params_hdr)

    # And the total optically-thin integrated intensity for comparison
    hdu_optthin_taulim = fits.ImageHDU(out_bics[7], params_hdr)

    # Record the peak taus for quick access
    hdu_peaktau = fits.ImageHDU(taupeak_map, params_hdr)

    return fits.HDUList([hdu_bic_thickHI, hdu_bic_mg, hdu_npts,
                         hdu_gcomp,
                         hdu_miss, hdu_optthin,
                         hdu_miss_taulim, hdu_optthin_taulim,
                         hdu_peaktau])


def remove_offrot_components(params_name,
                             vcent_name,
                             cube_name,
                             noise_map,
                             delta_v=80 * u.km / u.s,
                             logic_func=np.logical_and,
                             mwhi_mask=None,
                             return_mwcomps=True,
                             chunk_size=80000):
    '''
    All components get fit, including MW contamination
    and other off-rotation features.
    The MW is mostly an issue for M31 in C and D configs.
    This function only keeps components that are likely
    to be part of the galaxy.

    Parameters
    ----------
    logic_func: function, optional
        Defaults to `np.logical_and`. Otherwise, a custom
        function can be given with three inputs for the data,
        greater condition, lesser condition. For example, to
        only flag MW components red-shifted of the source,
        `lambda: data, great, less: data >= great`.
    '''

    cube = SpectralCube.read(cube_name)
    vels = cube.spectral_axis.to(u.m / u.s).value
    del cube

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

    new_bic_array = np.zeros_like(bic_array) * np.NaN

    yposn, xposn = np.where(np.isfinite(bic_array) & (ncomp_array > 0))

    cube_hdu = fits.open(cube_name, mode='denywrite')

    for i, (y, x) in tqdm(enumerate(zip(yposn.ravel(),
                                        xposn.ravel())),
                          ascii=True,
                          desc=f"Split off-rotation components.",
                          total=yposn.size):

        # Reload cube to release memory
        if i % chunk_size == 0:

            cube_hdu.close()
            del cube_hdu[0].data
            del cube_hdu
            cube_hdu = fits.open(cube_name, mode='denywrite')

        vcent_val = vcent.value[y, x]
        vmin = vcent_val - delta_v.value
        vmax = vcent_val + delta_v.value

        vfits = params_array[1::3, y, x][:ncomp_array[y, x]]

        valid_comps = logic_func(vfits >= vmin, vfits <= vmax)

        # Check against interactive mask
        if mwhi_mask is not None:
            mask_spec = mwhi_mask[:, y, x]

            ranges = nd.find_objects(mask_spec)[0]

            inrange = np.zeros_like(vfits, dtype=bool)

            # Check whether each component is within a valid range
            for k, vf in enumerate(vfits):

                for rg in ranges:

                    min_vel = vels[rg.start]
                    max_vel = vels[rg.stop - 1]

                    if min_vel > max_vel:
                        max_vel, min_vel = min_vel, max_vel

                    isvalid = np.logical_and(vf >= min_vel, vf <= max_vel)

                    if isvalid:
                        inrange[k] = True
                        break

            valid_comps = np.logical_or(valid_comps, inrange)

        valids = np.where(valid_comps)[0]

        for j, comp in enumerate(valids):
            new_params_array[3 * j, y, x] = params_array[3 * comp, y, x]
            new_params_array[3 * j + 1, y, x] = params_array[3 * comp + 1, y, x]
            new_params_array[3 * j + 2, y, x] = params_array[3 * comp + 2, y, x]

            new_uncert_array[3 * j, y, x] = uncert_array[3 * comp, y, x]
            new_uncert_array[3 * j + 1, y, x] = uncert_array[3 * comp + 1, y, x]
            new_uncert_array[3 * j + 2, y, x] = uncert_array[3 * comp + 2, y, x]

        # Recalculate the BIC for just the non-MW components.
        new_params = new_params_array[:, y, x]
        new_params = new_params[np.isfinite(new_params)]
        if new_params.size == 0:
            spec_model = np.zeros_like(vels)
        else:
            spec_model = multigaussian_nolmfit(vels, new_params)

        new_bic_array[y, x] = recalculate_bic(cube_hdu[0].data[:, y, x] * u.K,
                                              spec_model,
                                              noise_map[y, x],
                                              len(valids) * 3,
                                              mask=None)

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

    bics_hdu = fits.ImageHDU(new_bic_array, vcent.header.copy())
    bics_hdu.header['BUNIT'] = ("", "Gaussian fit BIC")

    # hdu_all = fits.HDUList([params_hdu, uncerts_hdu])
    hdu_all = fits.HDUList([params_hdu, uncerts_hdu, bics_hdu])

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

        params_header = params_hdu[0].header

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

        recalculate_bic(spec, spec_model, noise_val, Npar, mask=None)

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
    params_hdu = fits.PrimaryHDU(new_params_array, params_header.copy())
    params_hdu.header['BUNIT'] = ("", "Gaussian fit parameters")

    uncerts_hdu = fits.ImageHDU(new_uncert_array, params_header.copy())
    uncerts_hdu.header['BUNIT'] = ("", "Gaussian fit uncertainty")

    # Will need to update the BIC eventually...
    bics_hdu = fits.ImageHDU(bic_array, params_header.copy())
    bics_hdu.header['BUNIT'] = ("", "Gaussian fit BIC")

    hdu_all = fits.HDUList([params_hdu, uncerts_hdu, bics_hdu])

    if return_faintcomps:
        mwparams_hdu = fits.PrimaryHDU(faint_params_array,
                                       params_header.copy())
        mwparams_hdu.header['BUNIT'] = ("", "Gaussian fit parameters")

        mwuncerts_hdu = fits.ImageHDU(faint_uncert_array,
                                      params_header.copy())
        mwuncerts_hdu.header['BUNIT'] = ("", "Gaussian fit uncertainty")

        hdu_mw = fits.HDUList([mwparams_hdu, mwuncerts_hdu])

        return hdu_all, hdu_mw

    return hdu_all


def remove_all_but_brightest(params_name,
                             model_cube_name,
                             noise_val,
                             return_othercomps=True,
                             min_temp=25 * u.K,
                             max_lwidth_sub=25 * u.km / u.s,):
    '''
    This function will remove all components except those whose
    intensity is a significant fraction of the brightest single peak.
    '''
    pass


def find_distinct_features(params_name,
                           cube_name,
                           noise_val,
                           return_blendcomps=True,
                           nsigma=5.,
                           max_chan_diff=3,
                           secderiv_fraction=0.75):
    '''
    This function will remove all components except those whose
    intensity is a significant fraction of the brightest single peak.
    '''

    assert noise_val.unit == u.K

    with fits.open(params_name) as params_hdu:

        params_array = params_hdu[0].data
        uncert_array = params_hdu[1].data
        if len(params_hdu) > 2:
            bic_array = params_hdu[2].data
        else:
            bic_array = None

        params_header = params_hdu[0].header

    ncomp_array = np.isfinite(params_array).sum(0) // 3

    max_comp = ncomp_array.max()

    if return_blendcomps:
        blend_params_array = np.ones((max_comp * 3,) +
                                     params_array.shape[1:]) * np.NaN
        blend_uncert_array = np.ones((max_comp * 3,) +
                                     params_array.shape[1:]) * np.NaN

    keep_params_array = np.zeros_like(params_array) * np.NaN
    keep_uncert_array = np.zeros_like(params_array) * np.NaN

    if bic_array is None:
        yposn, xposn = np.where(ncomp_array > 0)
    else:
        yposn, xposn = np.where(np.isfinite(bic_array) & (ncomp_array > 0))

    cube = SpectralCube.read(cube_name)
    vels = cube.spectral_axis.to(u.m / u.s).value

    for i, (y, x) in enumerate(zip(yposn, xposn)):

        ncomp = ncomp_array[y, x]

        params = params_array[:3 * ncomp, y, x]
        uncerts = uncert_array[:3 * ncomp, y, x]

        noise_threshold = nsigma * noise_val

        # Do the split between distinct and blended.
        keeps, blends = distinct_vs_blended(params, noise_threshold, vels,
                                            max_chan_diff=max_chan_diff,
                                            secderiv_fraction=secderiv_fraction)

        for k, comp in enumerate(keeps):
            keep_params_array[3 * k:3 * k + 3, y, x] = params[3 * comp:3 * comp + 3]
            keep_uncert_array[3 * k:3 * k + 3, y, x] = uncerts[3 * comp:3 * comp + 3]

        if return_blendcomps:

            for k, comp in enumerate(blends):
                blend_params_array[3 * k:3 * k + 3, y, x] = params[3 * comp:3 * comp + 3]
                blend_uncert_array[3 * k:3 * k + 3, y, x] = uncerts[3 * comp:3 * comp + 3]

    del cube

    # Return a combined HDU that can be written out.
    params_hdu = fits.PrimaryHDU(keep_params_array, params_header.copy())
    params_hdu.header['BUNIT'] = ("", "Gaussian fit parameters")

    uncerts_hdu = fits.ImageHDU(keep_uncert_array, params_header.copy())
    uncerts_hdu.header['BUNIT'] = ("", "Gaussian fit uncertainty")

    # Will need to update the BIC eventually...
    # bics_hdu = fits.ImageHDU(bic_array, params_header.copy())
    # bics_hdu.header['BUNIT'] = ("", "Gaussian fit BIC")

    # hdu_all = fits.HDUList([params_hdu, uncerts_hdu, bics_hdu])
    hdu_all = fits.HDUList([params_hdu, uncerts_hdu])

    if return_blendcomps:
        blendparams_hdu = fits.PrimaryHDU(blend_params_array,
                                          params_header.copy())
        blendparams_hdu.header['BUNIT'] = ("", "Gaussian fit parameters")

        blenduncerts_hdu = fits.ImageHDU(blend_uncert_array,
                                         params_header.copy())
        blenduncerts_hdu.header['BUNIT'] = ("", "Gaussian fit uncertainty")

        hdu_mw = fits.HDUList([blendparams_hdu, blenduncerts_hdu])

        return hdu_all, hdu_mw

    return hdu_all


def distinct_vs_blended(params, noise_threshold, vels,
                        max_chan_diff=3, secderiv_fraction=0.75):
    '''
    Define distinct vs. blended Gaussian components from a set of components.
    "Distinct" is defined by matching the ratio of 2nd derivative minima between the
    full model and each component. A distinct peak will have a 2nd moment that is similar
    to the individual component in the complete model.
    '''

    params = params.copy
    params[params == 0.0] = np.NaN

    ncomp = np.isfinite(params).sum(0) // 3

    mod_spec = multigaussian_nolmfit(vels, params)

    if mod_spec.max() <= noise_threshold.value:
        continue

    bright_regions = np.where(mod_spec > noise_threshold.value)[0]

    diffs = np.diff(bright_regions)

    bright_regions_split = np.array_split(bright_regions,
                                          np.where(diffs != 1)[0] + 1)

    if len(bright_regions_split) == 0:
        raise ValueError("No valid regions found. But not skipped.")

    # Pick the brightest region:
    for i, region in enumerate(bright_regions_split):

        if len(region) == 0:
            continue

        max_val_reg = np.nanmax(mod_spec[region])

        if i == 0:
            max_val = max_val_reg
            regnum = 0
            continue

        if max_val_reg > max_val:
            max_val = max_val_reg
            regnum = i

    bright_region = bright_regions_split[regnum]

    # Identify peaks in total model
    deriv1 = np.gradient(mod_spec)
    deriv2 = np.gradient(deriv1)
    deriv3 = np.gradient(deriv2)

    zeros = np.abs(np.diff(np.sign(deriv3))) > 0

    mask_peaks = np.logical_and(deriv2[bright_region] < 0.,
                                zeros[bright_region])
    peaks = np.where(mask_peaks)[0] + bright_region[0] + 1

    cent_chans = np.array([np.argmin(np.abs(vels - cent))
                           for cent in params[1::3]])

    # Determine a peaks independence by the ratio of the 2nd derivative
    # of that component vs. the whole model.
    keeps = []

    for peak in peaks:

        diff_chan = np.abs(cent_chans - peak)

        min_diff = diff_chan.min()

        if min_diff > max_chan_diff:
            continue

        match = diff_chan.argmin()

        mod_comp = multigaussian_nolmfit(vels, params[3 * match:3 * match + 3])

        diff2_comp = np.gradient(np.gradient(mod_comp))

        # Now check the ratio between the 2nd deriv minima
        diff2_frac = deriv2[peak] / diff2_comp[cent_chans[match]]

        if diff2_frac >= secderiv_fraction:
            keeps.append(match)

    # Could have duplicates. Fail here is this happens so I can check
    if len(np.unique(keeps)) != len(keeps):
        ValueError("Duplicates shouldn't really happen!"
                   " You should check this.")

    blends = list(set(range(ncomp)) - set(keeps))

    return keeps, blends


def find_bright_narrow(params_name,
                       model_cube_name,
                       noise_val,
                       return_othercomps=True,
                       min_temp=25 * u.K,
                       max_lwidth_sub=25 * u.km / u.s,
                       max_amp_sub_sigma=3.):
    '''
    This function will remove all components except those whose
    intensity is a significant fraction of the brightest single peak.
    '''
    pass
