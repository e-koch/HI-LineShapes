
'''
Fit a multigaussian model from gausspy
'''


from spectral_cube import SpectralCube, Projection, OneDSpectrum
from astropy.io import fits
import astropy.units as u
import os
import numpy as np

from cube_analysis.spectral_fitting import cube_fitter

osjoin = os.path.join

repo_path = os.path.expanduser("~/ownCloud/project_code/ThickHIFitting/")

constants_script = os.path.join(repo_path, "paths.py")
exec(compile(open(constants_script, "rb").read(), constants_script, 'exec'))

model_script = os.path.join(repo_path, "gaussian_model.py")
exec(compile(open(model_script, "rb").read(), model_script, 'exec'))

modsel_script = os.path.join(repo_path, "model_selection_tools.py")
exec(compile(open(modsel_script, "rb").read(), modsel_script, 'exec'))


run_D = False
run_BCDtaper = True
run_BCD = False

run_fit = False
run_neighbcheck = False
run_writemodel = False
run_component_selection = True

if run_D:
    # 14A cube.

    cube_name = fourteenA_wEBHIS_HI_file_dict['Cube']

    # Use the interactive mask to remove MW contamination
    # Use a spatial mask from the signal masking, restricted by the interactive mask
    maskint_name = fourteenA_HI_data_wEBHIS_path("M31_14A_HI_contsub_width_04kms.image.pbcor.EBHIS_feathered_interactive_mask.fits")
    maskint_hdu = fits.open(maskint_name)[0]

    mask_name = fourteenA_wEBHIS_HI_file_dict['Source_Mask']
    mask_hdu = fits.open(mask_name)[0]

    spat_mask = np.logical_and(maskint_hdu.data.sum(0) > 10, mask_hdu.data.sum(0) > 10)

    del maskint_hdu, mask_hdu

    # Load in PB plane to account for varying uncertainty
    pb = fits.open(fourteenA_HI_file_dict['PB'], mode='denywrite')
    pb_plane = pb[0].data[0].copy()
    del pb

    # Need peak temp and centroid maps.

    # peak_name = fourteenA_wEBHIS_HI_file_dict['PeakTemp']
    # peaktemp = Projection.from_hdu(fits.open(peak_name))

    vcent_name = fourteenA_wEBHIS_HI_file_dict['Moment1']
    vcent = Projection.from_hdu(fits.open(vcent_name)).to(u.km / u.s)

    noise_val = 0.72 * u.K

    # Set max number of gaussians to something ridiculous.
    # Just so we don't have a failure putting into the output array
    max_comp = 30

    err_map = noise_val / pb_plane

    params_name = fourteenA_HI_data_wEBHIS_path("individ_multigaussian_gausspy_fits.fits", no_check=True)

    if run_fit:

        agd_kwargs = {"plot": False,
                      "verbose": False,
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

        params_array, uncerts_array, bic_array = \
            cube_fitter(cube_name, fit_func_gausspy,
                        mask_name=None,
                        npars=max_comp * 3,
                        args=(),
                        kwargs={'downsamp_factor': 1,
                                'min_finite_chan': 30,
                                'alphas': alphas,
                                'agd_kwargs': agd_kwargs,
                                'max_comp': max_comp},
                        spatial_mask=spat_mask,
                        err_map=err_map,
                        vcent_map=None,
                        num_cores=3,
                        chunks=80000)

        # Save the parameters

        params_hdu = fits.PrimaryHDU(params_array, vcent.header.copy())
        params_hdu.header['BUNIT'] = ("", "Gaussian fit parameters")

        uncerts_hdu = fits.ImageHDU(uncerts_array, vcent.header.copy())
        uncerts_hdu.header['BUNIT'] = ("", "Gaussian fit uncertainty")

        bics_hdu = fits.ImageHDU(bic_array, vcent.header.copy())
        bics_hdu.header['BUNIT'] = ("", "Gaussian fit BIC")

        hdu_all = fits.HDUList([params_hdu, uncerts_hdu, bics_hdu])

        hdu_all.writeto(params_name, overwrite=True)

    # Now we'll loop through the fits to check against the 3x3 neighbourhood.
    # This should even out nearby spectral fits and the number of components.
    params_name_rev = fourteenA_HI_data_wEBHIS_path("individ_multigaussian_gausspy_fits_neighbcheck.fits", no_check=True)

    params_name_rev2 = fourteenA_HI_data_wEBHIS_path("individ_multigaussian_gausspy_fits_neighbcheck2.fits", no_check=True)

    if run_neighbcheck:
        hdu_all_revised = neighbourhood_fit_comparison(cube_name,
                                                       params_name,
                                                       chunk_size=80000,
                                                       diff_bic=10,
                                                       err_map=err_map,
                                                       use_ncomp_check=True,
                                                       reverse_direction=False)

        hdu_all_revised.writeto(params_name_rev, overwrite=True)

        hdu_all_revised2 = neighbourhood_fit_comparison(cube_name,
                                                        params_name_rev,
                                                        chunk_size=80000,
                                                        diff_bic=10,
                                                        err_map=err_map,
                                                        use_ncomp_check=True,
                                                        reverse_direction=True)

        hdu_all_revised2.writeto(params_name_rev2, overwrite=True)

    # Make a model cube from the last fit.
    model_outname = fourteenA_HI_data_wEBHIS_path("individ_multigaussian_gausspy_fits_neighbcheck2_model.fits", no_check=True)

    if run_writemodel:
        overwrite = True
        if overwrite and os.path.exists(model_outname):
            os.system(f"rm {model_outname}")

        save_fitmodel(cube_name, params_name_rev2, model_outname,
                      chunk_size=80000,
                      save_sep_components=False)

    # Make selection cuts for:
    # 1. MW foreground
    # 2. Wide and faint components.
    # 3. Everthing but the brightest peak
    if run_component_selection:

        # 1. MW foreground

        # Use the interactive MW HI mask to find overlapping components
        maskint_hdu = fits.open(maskint_name)
        maskint_data = maskint_hdu[0].data > 0

        maskint_hdu.close()
        del maskint_hdu

        params_name_nomw = fourteenA_HI_data_wEBHIS_path("individ_multigaussian_gausspy_fits_neighbcheck2_nomw.fits", no_check=True)
        params_name_mw = fourteenA_HI_data_wEBHIS_path("individ_multigaussian_gausspy_fits_neighbcheck2_mw.fits", no_check=True)

        hdu_nomw, hdu_mw = remove_offrot_components(params_name_rev2,
        # hdu_nomw, hdu_mw = remove_offrot_components(params_name,
                                                vcent_name,
                                                cube_name,
                                                delta_v=80 * u.km / u.s,
                                                logic_func=np.logical_and,
                                                mwhi_mask=maskint_data,
                                                return_mwcomps=True)

        hdu_nomw.writeto(params_name_nomw, overwrite=True)
        hdu_mw.writeto(params_name_mw, overwrite=True)

        cube_name_nomw = f"{cube_name.rstrip('.fits')}_MWsub.fits"

        overwrite = True
        if overwrite and os.path.exists(cube_name_nomw):
            os.system(f"rm {cube_name_nomw}")

        # Make a cube version without the "MW" components
        subtract_components(cube_name,
                            params_name_mw,
                            cube_name_nomw,
                            chunk_size=20000)

        # 2. Wide and faint components.
        params_name_bright = fourteenA_HI_data_wEBHIS_path("individ_multigaussian_gausspy_fits_neighbcheck2_bright.fits", no_check=True)
        params_name_faint = fourteenA_HI_data_wEBHIS_path("individ_multigaussian_gausspy_fits_neighbcheck2_faint.fits", no_check=True)

        hdu_bright, hdu_faint = \
            remove_faint_components(params_name_rev2,
                                    noise_val,
                                    min_lwidth_sub=25 * u.km / u.s,
                                    max_amp_sub_sigma=3.,
                                    logic_func=np.logical_or)

        hdu_bright.writeto(params_name_bright, overwrite=True)
        hdu_faint.writeto(params_name_faint, overwrite=True)

        cube_name_nofaint = f"{cube_name.rstrip('.fits')}_faintsub.fits"

        overwrite = True
        if overwrite and os.path.exists(cube_name_nofaint):
            os.system(f"rm {cube_name_nofaint}")

        # Make a cube version without the "MW" components
        subtract_components(cube_name,
                            params_name_faint,
                            cube_name_nofaint,
                            chunk_size=20000)

        # Separate distinct and blended components
        # Use the non-MW components

if run_BCDtaper:
    # 14A+15A cube tapered to C-config.

    # Want the K version. Just do it manually
    # cube_name = fifteenA_HI_BCtaper_wEBHIS_HI_file_dict['Cube']
    cube_name = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("M31_15A_B_C_14A_HI_contsub_width_0_4kms.image.pbcor.EBHIS_feathered_K.fits")


    # Use the interactive mask to remove MW contamination
    # Use a spatial mask from the signal masking, restricted by the interactive mask
    maskint_name = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("M31_15A_taper_interactive_mask.fits")
    maskint_hdu = fits.open(maskint_name)[0]

    mask_name = fifteenA_HI_BCtaper_wEBHIS_HI_file_dict['Source_Mask']
    mask_hdu = fits.open(mask_name)[0]

    spat_mask = np.logical_and(maskint_hdu.data.sum(0) > 10, mask_hdu.data.sum(0) > 10)

    del maskint_hdu, mask_hdu

    # Load in PB plane to account for varying uncertainty
    pb = fits.open(fifteenA_HI_BCtaper_file_dict['PB'], mode='denywrite')
    pb_plane = pb[0].data[0].copy()
    del pb

    # Remove spectra with strong absorption
    mom0_name = fifteenA_HI_BCtaper_wEBHIS_HI_file_dict['Moment0']
    mom0 = Projection.from_hdu(fits.open(mom0_name))

    # Remove the spectra with strong absorption.
    spat_mask = np.logical_and(spat_mask, mom0.value > 0.)

    # Need peak temp and centroid maps.

    # peak_name = fifteenA_HI_BCtaper_wEBHIS_HI_file_dict['PeakTemp']
    # peaktemp = Projection.from_hdu(fits.open(peak_name))

    vcent_name = fifteenA_HI_BCtaper_wEBHIS_HI_file_dict['Moment1']
    vcent = Projection.from_hdu(fits.open(vcent_name)).to(u.km / u.s)

    noise_val = 2.8 * u.K

    # Set max number of gaussians to something ridiculous.
    # Just so we don't have a failure putting into the output array
    max_comp = 30

    err_map = noise_val / pb_plane

    params_name = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_multigaussian_gausspy_fits.fits", no_check=True)

    if run_fit:
        agd_kwargs = {"plot": False,
                      "verbose": False,
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

        params_array, uncerts_array, bic_array = \
            cube_fitter(cube_name, fit_func_gausspy,
                        mask_name=None,
                        npars=max_comp * 3,
                        args=(),
                        kwargs={'downsamp_factor': 1,
                                'min_finite_chan': 30,
                                'alphas': alphas,
                                'agd_kwargs': agd_kwargs,
                                'max_comp': max_comp},
                        spatial_mask=spat_mask,
                        err_map=err_map,
                        vcent_map=None,
                        num_cores=6,
                        chunks=30000)

        # Save the parameters
        # Cut to max number of components
        max_comp = np.isfinite(params_array).sum(0).max() // 3

        params_array = params_array[:3 * max_comp]
        uncerts_array = uncerts_array[:3 * max_comp]

        params_hdu = fits.PrimaryHDU(params_array, vcent.header.copy())
        params_hdu.header['BUNIT'] = ("", "Gaussian fit parameters")

        uncerts_hdu = fits.ImageHDU(uncerts_array, vcent.header.copy())
        uncerts_hdu.header['BUNIT'] = ("", "Gaussian fit uncertainty")

        bics_hdu = fits.ImageHDU(bic_array, vcent.header.copy())
        bics_hdu.header['BUNIT'] = ("", "Gaussian fit BIC")

        hdu_all = fits.HDUList([params_hdu, uncerts_hdu, bics_hdu])

        hdu_all.writeto(params_name, overwrite=True)

        # Delete the parameter arrays to recoup memory
        del params_array, uncerts_array, bic_array

        hdu_all.close()
        del hdu_all

    # Now we'll loop through the fits to check against the 3x3 neighbourhood.
    # This should even out nearby spectral fits and the number of components.
    params_name_rev = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_multigaussian_gausspy_fits_neighbcheck.fits", no_check=True)
    # params_name_rev = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_multigaussian_gausspy_fits_neighbcheck_nocompcheck.fits", no_check=True)

    params_name_rev2 = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_multigaussian_gausspy_fits_neighbcheck2.fits", no_check=True)

    if run_neighbcheck:
        hdu_all_revised = neighbourhood_fit_comparison(cube_name,
                                                       params_name,
                                                       chunk_size=80000,
                                                       diff_bic=10,
                                                       err_map=err_map,
                                                       use_ncomp_check=True,
                                                       reverse_direction=False)

        hdu_all_revised.writeto(params_name_rev, overwrite=True)

        hdu_all_revised.close()
        del hdu_all_revised

        hdu_all_revised2 = neighbourhood_fit_comparison(cube_name,
                                                        params_name_rev,
                                                        chunk_size=80000,
                                                        diff_bic=10,
                                                        err_map=err_map,
                                                        use_ncomp_check=True,
                                                        reverse_direction=True)

        hdu_all_revised2.writeto(params_name_rev2, overwrite=True)

        hdu_all_revised2.close()
        del hdu_all_revised2

    # Make a model cube from the last fit.
    model_outname = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_multigaussian_gausspy_fits_neighbcheck2_model.fits", no_check=True)

    if run_writemodel:
        overwrite = True
        if overwrite and os.path.exists(model_outname):
            os.system(f"rm {model_outname}")

        save_fitmodel(cube_name, params_name_rev2, model_outname,
                      chunk_size=80000,
                      save_sep_components=False)

    # Make selection cuts for:
    # 1. MW foreground
    # 2. Wide and faint components.
    # 3. Everthing but the brightest peak
    if run_component_selection:

        # 1. MW foreground

        # Use the interactive MW HI mask to find overlapping components
        maskint_name = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("M31_15A_taper_interactive_mask.fits")
        maskint_hdu = fits.open(maskint_name)
        maskint_data = maskint_hdu[0].data > 0

        maskint_hdu.close()
        del maskint_hdu

        params_name_nomw = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_multigaussian_gausspy_fits_neighbcheck2_nomw.fits", no_check=True)
        params_name_mw = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_multigaussian_gausspy_fits_neighbcheck2_mw.fits", no_check=True)

        hdu_nomw, hdu_mw = remove_offrot_components(params_name_rev2,
                                                    vcent_name,
                                                    cube_name,
                                                    err_map,
                                                    delta_v=80 * u.km / u.s,
                                                    logic_func=np.logical_and,
                                                    mwhi_mask=maskint_data,
                                                    return_mwcomps=True)

        hdu_nomw.writeto(params_name_nomw, overwrite=True)
        hdu_mw.writeto(params_name_mw, overwrite=True)

        cube_name_nomw = f"{cube_name.rstrip('.fits')}_MWsub.fits"

        overwrite = True
        if overwrite and os.path.exists(cube_name_nomw):
            os.system(f"rm {cube_name_nomw}")

        # Make a cube version without the "MW" components
        subtract_components(cube_name,
                            params_name_mw,
                            cube_name_nomw,
                            chunk_size=20000)

        # 2. Wide and faint components.
        params_name_bright = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_multigaussian_gausspy_fits_neighbcheck2_bright.fits", no_check=True)
        params_name_faint = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_multigaussian_gausspy_fits_neighbcheck2_faint.fits", no_check=True)

        hdu_bright, hdu_faint = \
            remove_faint_components(params_name_rev2,
                                    noise_val,
                                    min_lwidth_sub=25 * u.km / u.s,
                                    max_amp_sub_sigma=3.,
                                    logic_func=np.logical_or)

        hdu_bright.writeto(params_name_bright, overwrite=True)
        hdu_faint.writeto(params_name_faint, overwrite=True)

        cube_name_nofaint = f"{cube_name.rstrip('.fits')}_faintsub.fits"

        overwrite = True
        if overwrite and os.path.exists(cube_name_nofaint):
            os.system(f"rm {cube_name_nofaint}")

        # Make a cube version without the "MW" components
        subtract_components(cube_name,
                            params_name_faint,
                            cube_name_nofaint,
                            chunk_size=20000)

        # 3. Separate distinct and blended components
        # Use the non-MW components
        params_name_distinct = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_multigaussian_gausspy_fits_neighbcheck2_noMW_distinct.fits", no_check=True)
        params_name_blend = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_multigaussian_gausspy_fits_neighbcheck2_noMW_blend.fits", no_check=True)

        hdu_distinct, hdu_blend = \
            find_distinct_features(params_name_nomw,
                                   cube_name,
                                   noise_val,
                                   return_blendcomps=True,
                                   nsigma=5.,
                                   max_chan_diff=3,
                                   secderiv_fraction=0.75)

        hdu_distinct.writeto(params_name_distinct, overwrite=True)
        hdu_blend.writeto(params_name_blend, overwrite=True)

        cube_name_noblend = f"{cube_name.rstrip('.fits')}_blendsub.fits"

        # overwrite = True
        # if overwrite and os.path.exists(cube_name_noblend):
        #     os.system(f"rm {cube_name_noblend}")

        # # Make a cube version without the "MW" components
        # subtract_components(cube_name_nomw,
        #                     params_name_blend,
        #                     cube_name_noblend,
        #                     chunk_size=20000)

if run_BCD:

    # 14A+15A cube tapered to C-config.

    # Want the K version. Just do it manually
    # cube_name = fifteenA_HI_BCtaper_wEBHIS_HI_file_dict['Cube']
    cube_name = fifteenA_HI_BC_1_2kms_data_wEBHIS_path("M31_15A_B_C_14A_HI_contsub_width_1_2kms.image.pbcor.EBHIS_feathered_K.fits")


    # Use the interactive mask to remove MW contamination
    # Use a spatial mask from the signal masking, restricted by the interactive mask
    maskint_name = fifteenA_HI_BC_1_2kms_data_wEBHIS_path("M31_15A_interactive_mask.fits")
    maskint_hdu = fits.open(maskint_name)[0]

    mask_name = fifteenA_HI_wEBHIS_HI_file_dict['Source_Mask']
    mask_hdu = fits.open(mask_name)[0]

    spat_mask = np.logical_and(maskint_hdu.data.sum(0) > 6, mask_hdu.data.sum(0) > 6)

    del maskint_hdu, mask_hdu

    # Load in PB plane to account for varying uncertainty
    pb = fits.open(fifteenA_HI_file_dict['PB'], mode='denywrite')
    pb_plane = pb[0].data[0].copy()
    del pb

    # Remove spectra with strong absorption
    mom0_name = fifteenA_HI_wEBHIS_HI_file_dict['Moment0']
    mom0 = Projection.from_hdu(fits.open(mom0_name))

    # Remove the spectra with strong absorption.
    spat_mask = np.logical_and(spat_mask, mom0.value > 0.)

    # Need peak temp and centroid maps.

    # peak_name = fifteenA_HI_wEBHIS_HI_file_dict['PeakTemp']
    # peaktemp = Projection.from_hdu(fits.open(peak_name))

    vcent_name = fifteenA_HI_wEBHIS_HI_file_dict['Moment1']
    vcent = Projection.from_hdu(fits.open(vcent_name)).to(u.km / u.s)

    noise_val = 4.8 * u.K

    # Set max number of gaussians to something ridiculous.
    # Just so we don't have a failure putting into the output array
    max_comp = 10

    err_map = noise_val / pb_plane

    params_name = fifteenA_HI_BC_1_2kms_data_wEBHIS_path("individ_multigaussian_gausspy_fits.fits", no_check=True)

    if run_fit:
        agd_kwargs = {"plot": False,
                      "verbose": False,
                      "SNR_thresh1": 5.,
                      "SNR_thresh2": 5.,
                      "SNR2_thresh1": 4.,
                      "SNR2_thresh2": 3.5,
                      "mode": "conv",
                      # "mode": "python",
                      "deblend": True,
                      "intermediate_fit": False,
                      "perform_final_fit": False,
                      "component_sigma": 5.}

        alphas = [2., 5., 10., 15., 20., 30., 50.]

        params_array, uncerts_array, bic_array = \
            cube_fitter(cube_name, fit_func_gausspy,
                        mask_name=None,
                        npars=max_comp * 3,
                        args=(),
                        kwargs={'downsamp_factor': 1,
                                'min_finite_chan': 30,
                                'alphas': alphas,
                                'agd_kwargs': agd_kwargs,
                                'max_comp': max_comp},
                        spatial_mask=spat_mask,
                        err_map=err_map,
                        vcent_map=None,
                        num_cores=8,
                        chunks=10000)

        # Save the parameters
        # Cut to max number of components
        max_comp = np.isfinite(params_array).sum(0).max() // 3

        params_array = params_array[:3 * max_comp]
        uncerts_array = uncerts_array[:3 * max_comp]

        params_hdu = fits.PrimaryHDU(params_array, vcent.header.copy())
        params_hdu.header['BUNIT'] = ("", "Gaussian fit parameters")

        uncerts_hdu = fits.ImageHDU(uncerts_array, vcent.header.copy())
        uncerts_hdu.header['BUNIT'] = ("", "Gaussian fit uncertainty")

        bics_hdu = fits.ImageHDU(bic_array, vcent.header.copy())
        bics_hdu.header['BUNIT'] = ("", "Gaussian fit BIC")

        hdu_all = fits.HDUList([params_hdu, uncerts_hdu, bics_hdu])

        hdu_all.writeto(params_name, overwrite=True)

        # Delete the parameter arrays to recoup memory
        del params_array, uncerts_array, bic_array

        hdu_all.close()
        del hdu_all

    # Now we'll loop through the fits to check against the 3x3 neighbourhood.
    # This should even out nearby spectral fits and the number of components.
    params_name_rev = fifteenA_HI_BC_1_2kms_data_wEBHIS_path("individ_multigaussian_gausspy_fits_neighbcheck.fits", no_check=True)
    # params_name_rev = fifteenA_HI_BC_1_2kms_data_wEBHIS_path("individ_multigaussian_gausspy_fits_neighbcheck_nocompcheck.fits", no_check=True)

    params_name_rev2 = fifteenA_HI_BC_1_2kms_data_wEBHIS_path("individ_multigaussian_gausspy_fits_neighbcheck2.fits", no_check=True)

    if run_neighbcheck:
        hdu_all_revised = neighbourhood_fit_comparison(cube_name,
                                                       params_name,
                                                       chunk_size=10000,
                                                       diff_bic=10,
                                                       err_map=err_map,
                                                       use_ncomp_check=True,
                                                       reverse_direction=False)

        hdu_all_revised.writeto(params_name_rev, overwrite=True)

        hdu_all_revised.close()
        del hdu_all_revised

        hdu_all_revised2 = neighbourhood_fit_comparison(cube_name,
                                                        params_name_rev,
                                                        chunk_size=10000,
                                                        diff_bic=10,
                                                        err_map=err_map,
                                                        use_ncomp_check=True,
                                                        reverse_direction=True)

        hdu_all_revised2.writeto(params_name_rev2, overwrite=True)

        hdu_all_revised2.close()
        del hdu_all_revised2

    # Make a model cube from the last fit.
    model_outname = fifteenA_HI_BC_1_2kms_data_wEBHIS_path("individ_multigaussian_gausspy_fits_neighbcheck2_model.fits", no_check=True)

    if run_writemodel:
        overwrite = True
        if overwrite and os.path.exists(model_outname):
            os.system(f"rm {model_outname}")

        save_fitmodel(cube_name, params_name_rev2, model_outname,
                      chunk_size=10000,
                      save_sep_components=False)
