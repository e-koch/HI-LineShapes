'''
Figure of the BIC for the full cube models.
'''

from astropy.io import fits
import astropy.units as u
import os
import numpy as np
from spectral_cube import Projection
from tqdm import tqdm

from uncertainties import ufloat, unumpy


def isoturbHI_simple_with_ufunc(x, Ts, sigma, Tpeak, vcent):
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

    tau = tau_func_with_uncert(x, Ts, sigma, Tpeak, vcent)

    return Ts * (1 - unumpy.exp(-tau))


def tau_func_with_uncert(x, Ts, sigma, Tpeak, vcent):

    # sigma_th =0.093 * Ts**0.5
    # sigma_th_sq = 0.093**2 * Ts
    # sigmasq = (sigma_th_sq + sigma_nt**2)
    sigmasq = sigma**2

    exp_term = unumpy.exp(- 0.5 * ((x - vcent)**2 / sigmasq))
    tau_prefac = Tpeak / Ts

    tau = tau_prefac * exp_term
    return tau


def gaussian_with_uncert(x, amp, cent, sigma):
    return amp * unumpy.exp(- (x - cent)**2 / (2 * sigma**2))



def multigaussian_nolmfit_with_uncert(x, pars):

    ncomp = pars.size // 3

    model = gaussian_with_uncert(x, pars[0], pars[1], pars[2])

    if ncomp == 1:
        return model

    for nc in range(1, ncomp):
        model += gaussian_with_uncert(x, pars[3 * nc],
                          pars[3 * nc + 1],
                          pars[3 * nc + 2])

    return model


def nominal_and_stddev(arr):
    return unumpy.nominal_values(arr), unumpy.std_devs(arr)

def darkNHI_fraction(vels, params, uncerts, spec):
    '''
    Compute the fraction of apparent dark NHI
    
    params: Ts, sigma, Tpeak, vcent
    '''
    
    if not np.isfinite(uncerts).all():
        uncerts = np.array([0, 0, 0, 0])
    
    # isoturbHI_simple wants km/s. Check the units in one place.
    # force converting to km/s everywhere
    
    params = unumpy.uarray(params, uncerts)

    mod_thickHI = isoturbHI_simple_with_ufunc(vels.to(u.km / u.s).value,
                                              params[0],
                                              params[1],
                                              params[2],
                                              params[3])

    
    # Compute the model residuals. These are used in the Braun+09 style "correction"
    # maps for total NHI (their Eq. 5):
    residsum_thickHI, residsum_thickHI_stddev = nominal_and_stddev((spec.value - mod_thickHI).sum())

    # Make equivalent "optically-thin" models to calculate inferred missing intensity
    # fractions, etc.
    par = np.array([params[2], params[3], params[1]])
    
    mod_thickHI_thinlimit = multigaussian_nolmfit_with_uncert(vels.to(u.km / u.s).value, par)
    missing_intint, missing_intint_stddev = nominal_and_stddev((mod_thickHI_thinlimit - mod_thickHI).sum())
    
    # Catch fit failures with Tpeak=0.
    if unumpy.nominal_values(params[2]) > 0.:

        # Missing vs. total opt.thin actual
        missing_intint_frac, missing_intint_frac_stddev = nominal_and_stddev(missing_intint / mod_thickHI_thinlimit.sum())
        
        # Detected vs. dark
        missing_intint_corr, missing_intint_corr_stddev = nominal_and_stddev(missing_intint / mod_thickHI.sum())
    else:
        # These aren't defined:
        missing_intint_frac, missing_intint_frac_stddev = np.NaN, np.NaN
        missing_intint_corr, missing_intint_corr_stddev = np.NaN, np.NaN

    diff_vel = np.abs(vels[1] - vels[0]).to(u.km / u.s).value
    
    return [missing_intint * diff_vel * u.K * u.km / u.s,
            missing_intint_stddev * diff_vel * u.K * u.km / u.s,
            missing_intint_frac, missing_intint_frac_stddev,
            missing_intint_corr, missing_intint_corr_stddev,
            residsum_thickHI * diff_vel * u.K * u.km / u.s,
            residsum_thickHI_stddev * diff_vel * u.K * u.km / u.s]

if __name__ == "__main__":

    osjoin = os.path.join

    repo_path = os.path.expanduser("~/ownCloud/project_code/ThickHIFitting/")

    paths_script = os.path.join(repo_path, "paths.py")
    exec(compile(open(paths_script, "rb").read(), paths_script, 'exec'))

    plotstyle_script = os.path.join(repo_path, "plotting_styles.py")
    exec(compile(open(plotstyle_script, "rb").read(), plotstyle_script, 'exec'))

    model_script = os.path.join(repo_path, "gaussian_model.py")
    exec(compile(open(model_script, "rb").read(), model_script, 'exec'))

    thickHI_model_script = os.path.join(repo_path, "thickHI_model.py")
    exec(compile(open(thickHI_model_script, "rb").read(),
                thickHI_model_script, 'exec'))

    import warnings

    compute_Idark = True

    run_m31 = False
    run_m33 = True


    # M31
    if run_m31:

        m31_cubename_K = f"{fifteenA_HI_BCtaper_wEBHIS_HI_file_dict['Cube'].rstrip('.fits')}_K.fits"

        m31_cube = SpectralCube.read(m31_cubename_K, use_dask=False)
        print(f'Opening cube {m31_cubename_K}')

        m31_vels = m31_cube.spectral_axis.to(u.m / u.s)
            
        # del m31_cube

        m31_mom0 = Projection.from_hdu(fits.open(fifteenA_HI_BCtaper_wEBHIS_HI_file_dict['Moment0'])).to(u.K * u.km / u.s)
            
        m31_multigauss_name = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_multigaussian_gausspy_fits_neighbcheck2_nomw.fits")
        # m31_multigauss_name = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_multigaussian_gausspy_fits_neighbcheck2.fits")
        m31_multigauss_hdu = fits.open(m31_multigauss_name)

        m31_ngauss = np.isfinite(m31_multigauss_hdu[0].data).sum(0) // 3

        m31_thickHI_name = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_simplethick_HI_fits_5kms_centlimit.fits")
        m31_thickHI_hdu = fits.open(m31_thickHI_name)

        m31_thickHI80_name = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_simplethick_HI_fits_80kms_centlimit.fits")
        m31_thickHI80_hdu = fits.open(m31_thickHI_name)

        # Keep only where the fit parameters are valid
        m31_multigauss_hdu[2].data[m31_ngauss == 0] = np.NaN

        m31_multigauss_bic_proj = Projection.from_hdu(m31_multigauss_hdu[2])
            
        # Split the different fit statistics
        m31_thickHI_bic_proj = Projection.from_hdu(fits.PrimaryHDU(m31_thickHI_hdu[2].data[0], m31_thickHI_hdu[2].header))
        m31_thickHI80_bic_proj = Projection.from_hdu(fits.PrimaryHDU(m31_thickHI80_hdu[2].data[0], m31_thickHI_hdu[2].header))
            
        m31_thickHI_rchi_proj = Projection.from_hdu(fits.PrimaryHDU(m31_thickHI_hdu[2].data[2], m31_thickHI_hdu[2].header))
        m31_thickHI80_rchi_proj = Projection.from_hdu(fits.PrimaryHDU(m31_thickHI80_hdu[2].data[2], m31_thickHI_hdu[2].header))
            
        # Lastly, the recomputed statistics limited to where tau > 0.5
        m31_modcompare_name = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_recomp_bic_tau_gt_0p5_5kms_centlimit.fits")
        m31_modcompare_hdu = fits.open(m31_modcompare_name)
            
        m31_modcompare80_name = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_recomp_bic_tau_gt_0p5_80kms_centlimit.fits")
        m31_modcompare80_hdu = fits.open(m31_modcompare_name)
        
    # M33

    if run_m33:

        cube_name = fourteenB_HI_data_wGBT_path("M33_14B-088_HI.clean.image.GBT_feathered.pbcov_gt_0.5_masked.fits")

        downsamp_cube_name = f"{cube_name.rstrip('.fits')}_0p42kms_K.fits"
                
        m33_cube = SpectralCube.read(downsamp_cube_name, use_dask=False)
        print(f'Opening cube {downsamp_cube_name}')

        m33_vels = m33_cube.spectral_axis.to(u.m / u.s)
            
        # del m33_cube

        m33_mom0_name = "M33_14B-088_HI.clean.image.GBT_feathered.pbcov_gt_0.5_masked.moment0_Kkms.fits"
        m33_mom0 = Projection.from_hdu(fits.open(fourteenB_HI_data_wGBT_path(m33_mom0_name))).to(u.K * u.km / u.s)

        m33_multigauss_name = fourteenB_HI_data_wGBT_path("individ_multigaussian_gausspy_fits_neighbcheck2_nomw.fits")
        # m33_multigauss_name = fourteenB_HI_data_wGBT_path("individ_multigaussian_gausspy_fits_neighbcheck2.fits")
        m33_multigauss_hdu = fits.open(m33_multigauss_name)

        m33_ngauss = np.isfinite(m33_multigauss_hdu[0].data).sum(0) // 3

        m33_thickHI_name = fourteenB_HI_data_wGBT_path("individ_simplethick_HI_fits_5kms_centlimit.fits")
        m33_thickHI_hdu = fits.open(m33_thickHI_name)

        m33_thickHI80_name = fourteenB_HI_data_wGBT_path("individ_simplethick_HI_fits_80kms_centlimit.fits")
        m33_thickHI80_hdu = fits.open(m33_thickHI_name)

        # Keep only where the fit parameters are valid
        m33_multigauss_hdu[2].data[m33_ngauss == 0] = np.NaN

        m33_multigauss_bic_proj = Projection.from_hdu(m33_multigauss_hdu[2])

        # Split the different fit statistics
        m33_thickHI_bic_proj = Projection.from_hdu(fits.PrimaryHDU(m33_thickHI_hdu[2].data[0], m33_thickHI_hdu[2].header))
        m33_thickHI80_bic_proj = Projection.from_hdu(fits.PrimaryHDU(m33_thickHI80_hdu[2].data[0], m33_thickHI_hdu[2].header))

        m33_thickHI_rchi_proj = Projection.from_hdu(fits.PrimaryHDU(m33_thickHI_hdu[2].data[2], m33_thickHI_hdu[2].header))
        m33_thickHI80_rchi_proj = Projection.from_hdu(fits.PrimaryHDU(m33_thickHI80_hdu[2].data[2], m33_thickHI_hdu[2].header))

        # Lastly, the recomputed statistics limited to where tau > 0.5
        m33_modcompare_name = fourteenB_HI_data_wGBT_path("individ_recomp_bic_tau_gt_0p5_5kms_centlimit.fits")
        m33_modcompare_hdu = fits.open(m33_modcompare_name)
            
        m33_modcompare80_name = fourteenB_HI_data_wGBT_path("individ_recomp_bic_tau_gt_0p5_80kms_centlimit.fits")
        m33_modcompare80_hdu = fits.open(m33_modcompare_name)


    # Compute all the dark NHI fractions
    # Further, compute upper and low limits on everything.

    if compute_Idark:

        if run_m31:

            print("Running M31")

            m31_darknhi = np.zeros_like(m31_thickHI_bic_proj.value) * np.NaN * u.K * u.km / u.s
            m31_darknhi_stddev = np.zeros_like(m31_thickHI_bic_proj.value) * np.NaN * u.K * u.km / u.s

            m31_darknhi80 = np.zeros_like(m31_thickHI_bic_proj.value) * np.NaN * u.K * u.km / u.s
            m31_darknhi80_stddev = np.zeros_like(m31_thickHI_bic_proj.value) * np.NaN * u.K * u.km / u.s

            m31_darknhi_frac = np.zeros_like(m31_thickHI_bic_proj.value) * np.NaN
            m31_darknhi_frac_stddev = np.zeros_like(m31_thickHI_bic_proj.value) * np.NaN

            m31_darknhi80_frac = np.zeros_like(m31_thickHI_bic_proj.value) * np.NaN
            m31_darknhi80_frac_stddev = np.zeros_like(m31_thickHI_bic_proj.value) * np.NaN

            m31_darknhi_corr = np.zeros_like(m31_thickHI_bic_proj.value) * np.NaN
            m31_darknhi_corr_stddev = np.zeros_like(m31_thickHI_bic_proj.value) * np.NaN

            m31_darknhi80_corr = np.zeros_like(m31_thickHI_bic_proj.value) * np.NaN
            m31_darknhi80_corr_stddev = np.zeros_like(m31_thickHI_bic_proj.value) * np.NaN

            m31_residsum = np.zeros_like(m31_thickHI_bic_proj.value) * np.NaN * u.K * u.km / u.s
            m31_residsum_stddev = np.zeros_like(m31_thickHI_bic_proj.value) * np.NaN * u.K * u.km / u.s

            m31_residsum80 = np.zeros_like(m31_thickHI_bic_proj.value) * np.NaN * u.K * u.km / u.s
            m31_residsum80_stddev = np.zeros_like(m31_thickHI_bic_proj.value) * np.NaN * u.K * u.km / u.s

            # yy, xx = np.indices(m31_darknhi.shape)
            # This will only include where the uncertainties can be estimated from the fit
            # yy, xx = np.where(np.isfinite(m31_thickHI_hdu[1].data[0]))

            # This will include everywhere where the fit parameters are at least fit
            yy, xx = np.where(np.isfinite(m31_thickHI_hdu[0].data[0]))

            for (y, x) in tqdm(zip(yy.ravel(), xx.ravel()), total=yy.size):

                with warnings.catch_warnings():
                
                    warnings.simplefilter("ignore")
                
                    out = \
                        darkNHI_fraction(m31_vels,
                                        m31_thickHI_hdu[0].data[:, y, x],
                                        m31_thickHI_hdu[1].data[:, y, x],
                                        m31_cube[:, y, x])
                
                m31_darknhi[y, x], m31_darknhi_stddev[y, x], \
                m31_darknhi_frac[y, x], m31_darknhi_frac_stddev[y, x], \
                m31_darknhi_corr[y, x], m31_darknhi_corr_stddev[y, x], \
                m31_residsum[y, x], m31_residsum_stddev[y, x] = out
            
        #     m31_darknhi80[y, x], m31_darknhi80_stddev[y, x], m31_darknhi80_up[y, x], \
        #     m31_darknhi80_frac[y, x], m31_darknhi80_frac_stddev[y, x], m31_darknhi80_frac_up[y, x], \
        #     m31_darknhi80_corr[y, x], m31_darknhi80_corr_stddev[y, x], m31_darknhi80_corr_up[y, x], \
        #     m31_residsum80[y, x], m31_residsum80_stddev[y, x], m31_residsum80_up[y, x] = \
        #         darkNHI_fraction(m31_vels,
        #                          m31_thickHI80_hdu[0].data[:, y, x],
        #                          m31_thickHI80_hdu[1].data[:, y, x],
        #                          m31_cube[:, y, x])
            
            m31_darkhi_hdu = fits.HDUList([fits.PrimaryHDU(m31_darknhi.value, m31_mom0.header),
                                        fits.ImageHDU(m31_darknhi_stddev.value, m31_mom0.header),
                                        fits.ImageHDU(m31_residsum.value, m31_mom0.header),
                                        fits.ImageHDU(m31_residsum_stddev.value, m31_mom0.header),
                                        fits.ImageHDU(m31_darknhi_frac, m31_mom0.header),
                                        fits.ImageHDU(m31_darknhi_frac_stddev, m31_mom0.header),
                                        fits.ImageHDU(m31_darknhi_corr, m31_mom0.header),
                                        fits.ImageHDU(m31_darknhi_corr_stddev, m31_mom0.header),
                                        ])

            m31_darkhi_hdu.writeto(fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_simplethick_HI_fits_5kms_centlimit_darkHI.fits", no_check=True),
                                overwrite=True)

        if run_m33:

            print("Running M33")

            m33_darknhi = np.zeros_like(m33_thickHI_bic_proj.value) * np.NaN * u.K * u.km / u.s
            m33_darknhi_stddev = np.zeros_like(m33_thickHI_bic_proj.value) * np.NaN * u.K * u.km / u.s

            m33_darknhi80 = np.zeros_like(m33_thickHI_bic_proj.value) * np.NaN * u.K * u.km / u.s
            m33_darknhi80_stddev = np.zeros_like(m33_thickHI_bic_proj.value) * np.NaN * u.K * u.km / u.s

            m33_darknhi_frac = np.zeros_like(m33_thickHI_bic_proj.value) * np.NaN
            m33_darknhi_frac_stddev = np.zeros_like(m33_thickHI_bic_proj.value) * np.NaN

            m33_darknhi80_frac = np.zeros_like(m33_thickHI_bic_proj.value) * np.NaN
            m33_darknhi80_frac_stddev = np.zeros_like(m33_thickHI_bic_proj.value) * np.NaN

            m33_darknhi_corr = np.zeros_like(m33_thickHI_bic_proj.value) * np.NaN
            m33_darknhi_corr_stddev = np.zeros_like(m33_thickHI_bic_proj.value) * np.NaN

            m33_darknhi80_corr = np.zeros_like(m33_thickHI_bic_proj.value) * np.NaN
            m33_darknhi80_corr_stddev = np.zeros_like(m33_thickHI_bic_proj.value) * np.NaN


            m33_residsum = np.zeros_like(m33_thickHI_bic_proj.value) * np.NaN * u.K * u.km / u.s
            m33_residsum_stddev = np.zeros_like(m33_thickHI_bic_proj.value) * np.NaN * u.K * u.km / u.s

            m33_residsum80 = np.zeros_like(m33_thickHI_bic_proj.value) * np.NaN * u.K * u.km / u.s
            m33_residsum80_stddev = np.zeros_like(m33_thickHI_bic_proj.value) * np.NaN * u.K * u.km / u.s

            # yy, xx = np.indices(m33_darknhi.shape)
            # yy, xx = np.where(np.isfinite(m33_thickHI_hdu[1].data[0]))

            yy, xx = np.where(np.isfinite(m33_thickHI_hdu[0].data[0]))

            for (y, x) in tqdm(zip(yy.ravel(), xx.ravel()), total=yy.size):
                
                with warnings.catch_warnings():
                
                    warnings.simplefilter("ignore")
                    
                    m33_darknhi[y, x], m33_darknhi_stddev[y, x], \
                    m33_darknhi_frac[y, x], m33_darknhi_frac_stddev[y, x], \
                    m33_darknhi_corr[y, x], m33_darknhi_corr_stddev[y, x], \
                    m33_residsum[y, x], m33_residsum_stddev[y, x] = \
                        darkNHI_fraction(m33_vels,
                                        m33_thickHI_hdu[0].data[:, y, x],
                                        m33_thickHI_hdu[1].data[:, y, x],
                                        m33_cube[:, y, x])
                
            #     m33_darknhi80[y, x], m33_darknhi80_stddev[y, x], m33_darknhi80_up[y, x], \
            #     m33_darknhi80_frac[y, x], m33_darknhi80_frac_stddev[y, x], m33_darknhi80_frac_up[y, x], \
            #     m33_darknhi80_corr[y, x], m33_darknhi80_corr_stddev[y, x], m33_darknhi80_corr_up[y, x], \
            #     m33_residsum80[y, x], m33_residsum80_stddev[y, x], m33_residsum80_up[y, x] = \
            #         darkNHI_fraction(m33_vels,
            #                          m33_thickHI80_hdu[0].data[:, y, x],
            #                          m33_thickHI80_hdu[1].data[:, y, x],
            #                          m33_cube[:, y, x])
        

            m33_darkhi_hdu = fits.HDUList([fits.PrimaryHDU(m33_darknhi.value, m33_mom0.header),
                                        fits.ImageHDU(m33_darknhi_stddev.value, m33_mom0.header),
                                        fits.ImageHDU(m33_residsum.value, m33_mom0.header),
                                        fits.ImageHDU(m33_residsum_stddev.value, m33_mom0.header),
                                        fits.ImageHDU(m33_darknhi_frac, m33_mom0.header),
                                        fits.ImageHDU(m33_darknhi_frac_stddev, m33_mom0.header),
                                        fits.ImageHDU(m33_darknhi_corr, m33_mom0.header),
                                        fits.ImageHDU(m33_darknhi_corr_stddev, m33_mom0.header),
                                        ])

            m33_darkhi_hdu.writeto(fourteenB_HI_data_wGBT_path("individ_simplethick_HI_fits_5kms_centlimit_darkHI_recomp.fits", no_check=True),
                                overwrite=True)

    # del m31_darkhi_hdu, m33_darkhi_hdu