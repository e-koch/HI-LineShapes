
'''
Fit a multigaussian model from gausspy to the entire M33 cubes.
'''

from spectral_cube import SpectralCube, Projection, OneDSpectrum
from astropy.io import fits
import astropy.units as u
import os
import numpy as np
import scipy.ndimage as nd

from cube_analysis.spectral_fitting import cube_fitter

from cube_analysis.cube_utils import spectral_interpolate, convert_K

osjoin = os.path.join

repo_path = os.path.expanduser("~/ownCloud/project_code/ThickHIFitting/")

constants_script = os.path.join(repo_path, "paths.py")
exec(compile(open(constants_script, "rb").read(), constants_script, 'exec'))

model_script = os.path.join(repo_path, "gaussian_model.py")
exec(compile(open(model_script, "rb").read(), model_script, 'exec'))

modsel_script = os.path.join(repo_path, "model_selection_tools.py")
exec(compile(open(modsel_script, "rb").read(), modsel_script, 'exec'))


run_C = True
run_BC = False

run_fit = False
run_neighbcheck = False
run_writemodel = False
run_component_selection = True

if run_C:
    # 14B cube

    # cube_name = fourteenB_wGBT_HI_file_dict['Cube']
    cube_name = fourteenB_HI_data_wGBT_path("M33_14B-088_HI.clean.image.GBT_feathered.pbcov_gt_0.5_masked.fits")

    # Create a downsampled version of the cube with 0.42 km/s channels.
    # This will match the M31 spectral resolution.
    downsamp_cube_name = f"{cube_name.rstrip('.fits')}_0p42kms_K.fits"

    if not os.path.exists(downsamp_cube_name):

        if not os.path.exists(f"{cube_name.rstrip('.fits')}_K.fits"):
            convert_K(cube_name,
                      fourteenB_HI_data_wGBT_path(""),
                      verbose=True)

        spectral_interpolate(f"{cube_name.rstrip('.fits')}_K.fits",
                             downsamp_cube_name,
                             2, verbose=True)

    mask_name = fourteenB_wGBT_HI_file_dict['Source_Mask']
    mask_hdu = fits.open(mask_name)[0]

    # We'll keep all positions with 10 pixels in the mask for the
    # downsampled cube (so 2x the original cube channels)
    spat_mask = mask_hdu.data.sum(0) > 20

    del mask_hdu

    # Load in PB plane to account for varying uncertainty
    pb = fits.open(fourteenB_HI_data_path("M33_14B-088_pbcov.fits"), mode='denywrite')
    # pb_plane = pb[0].data[0].copy()
    pb_plane = pb[0].data.copy()
    pb_plane = pb_plane[nd.find_objects(pb_plane > 0.5)[-1]]
    del pb

    # Need peak temp and centroid maps.

    # peak_name = fourteenB_wGBT_HI_file_dict['PeakTemp']
    # peaktemp = Projection.from_hdu(fits.open(peak_name))

    vcent_name = fourteenB_wGBT_HI_file_dict['Moment1']
    vcent = Projection.from_hdu(fits.open(vcent_name)).to(u.km / u.s)

    # Noise lowered for the downsampled cube.
    noise_val = 2.8 * u.K / np.sqrt(2)

    # Set max number of gaussians to something ridiculous.
    # Just so we don't have a failure putting into the output array
    max_comp = 30

    err_map = noise_val / pb_plane

    params_name = fourteenB_HI_data_wGBT_path("individ_multigaussian_gausspy_fits.fits", no_check=True)

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
            cube_fitter(downsamp_cube_name, fit_func_gausspy,
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

        hdu_all.close()
        del hdu_all

    # Stage 2: Do a neighbourhood check to ensure smoother fit solutions.
    # Now we'll loop through the fits to check against the 3x3 neighbourhood.
    # This should even out nearby spectral fits and the number of components.
    params_name_rev = fourteenB_HI_data_wGBT_path("individ_multigaussian_gausspy_fits_neighbcheck.fits", no_check=True)
    params_name_rev2 = fourteenB_HI_data_wGBT_path("individ_multigaussian_gausspy_fits_neighbcheck2.fits", no_check=True)

    if run_neighbcheck:
        hdu_all_revised = neighbourhood_fit_comparison(downsamp_cube_name,
                                                       params_name,
                                                       chunk_size=80000,
                                                       diff_bic=10,
                                                       err_map=err_map,
                                                       use_ncomp_check=True,
                                                       reverse_direction=False)

        hdu_all_revised.writeto(params_name_rev, overwrite=True)

        hdu_all_revised2 = neighbourhood_fit_comparison(downsamp_cube_name,
                                                        params_name_rev,
                                                        chunk_size=80000,
                                                        diff_bic=10,
                                                        err_map=err_map,
                                                        use_ncomp_check=True,
                                                        reverse_direction=True)

        hdu_all_revised2.writeto(params_name_rev2, overwrite=True)

    # Make a model cube from the last fit.
    model_outname = fourteenB_HI_data_wGBT_path("individ_multigaussian_gausspy_fits_neighbcheck2_model.fits", no_check=True)

    if run_writemodel:
        overwrite = True
        if overwrite and os.path.exists(model_outname):
            os.system(f"rm {model_outname}")

        save_fitmodel(downsamp_cube_name, params_name_rev2, model_outname,
                      chunk_size=80000,
                      save_sep_components=False)

    # Make selection cuts for:
    # 1. MW foreground
    # 2. Wide and faint components.
    # 3. Everthing but the brightest peak
    if run_component_selection:

        # 1. MW foreground

        params_name_nomw = fourteenB_HI_data_wGBT_path("individ_multigaussian_gausspy_fits_neighbcheck2_nomw.fits", no_check=True)
        params_name_mw = fourteenB_HI_data_wGBT_path("individ_multigaussian_gausspy_fits_neighbcheck2_mw.fits", no_check=True)

        hdu_nomw, hdu_mw = remove_offrot_components(params_name_rev2,
                                                    vcent_name,
                                                    downsamp_cube_name,
                                                    err_map,
                                                    delta_v=50 * u.km / u.s,
                                                    logic_func=np.logical_and,
                                                    mwhi_mask=None,
                                                    return_mwcomps=True)

        hdu_nomw.writeto(params_name_nomw, overwrite=True)
        hdu_mw.writeto(params_name_mw, overwrite=True)

        downsamp_cube_name_nomw = f"{downsamp_cube_name.rstrip('.fits')}_MWsub.fits"

        overwrite = True
        if overwrite and os.path.exists(downsamp_cube_name_nomw):
            os.system(f"rm {downsamp_cube_name_nomw}")

        # Make a cube version without the "MW" components
        subtract_components(downsamp_cube_name,
                            params_name_mw,
                            downsamp_cube_name_nomw,
                            chunk_size=20000)

        # 2. Wide and faint components.
        params_name_bright = fourteenB_HI_data_wGBT_path("individ_multigaussian_gausspy_fits_neighbcheck2_bright.fits", no_check=True)
        params_name_faint = fourteenB_HI_data_wGBT_path("individ_multigaussian_gausspy_fits_neighbcheck2_faint.fits", no_check=True)

        hdu_bright, hdu_faint = \
            remove_faint_components(params_name_rev2,
                                    noise_val,
                                    min_lwidth_sub=25 * u.km / u.s,
                                    max_amp_sub_sigma=3.,
                                    logic_func=np.logical_or)

        hdu_bright.writeto(params_name_bright, overwrite=True)
        hdu_faint.writeto(params_name_faint, overwrite=True)

        downsamp_cube_name_nofaint = f"{downsamp_cube_name.rstrip('.fits')}_faintsub.fits"

        overwrite = True
        if overwrite and os.path.exists(downsamp_cube_name_nofaint):
            os.system(f"rm {downsamp_cube_name_nofaint}")

        # Make a cube version without the "MW" components
        subtract_components(downsamp_cube_name,
                            params_name_faint,
                            downsamp_cube_name_nofaint,
                            chunk_size=20000)

        # 3. Separate distinct and blended components
        # Use the non-MW components
        params_name_distinct = fourteenB_HI_data_wGBT_path("individ_multigaussian_gausspy_fits_neighbcheck2_noMW_distinct.fits", no_check=True)
        params_name_blend = fourteenB_HI_data_wGBT_path("individ_multigaussian_gausspy_fits_neighbcheck2_noMW_blend.fits", no_check=True)

        hdu_distinct, hdu_blend = \
            find_distinct_features(params_name_nomw,
                                   downsamp_cube_name,
                                   noise_val,
                                   return_blendcomps=True,
                                   nsigma=5.,
                                   max_chan_diff=3,
                                   secderiv_fraction=0.75)

        hdu_distinct.writeto(params_name_distinct, overwrite=True)
        hdu_blend.writeto(params_name_blend, overwrite=True)

        # cube_name_noblend = f"{cube_name.rstrip('.fits')}_blendsub.fits"


if run_BC:
    # 14B+17B cube

    # cube_name = fourteenB_wGBT_HI_file_dict['Cube']
    cube_name = seventeenB_HI_data_1kms_wGBT_path("M33_14B_17B_HI_contsub_width_1kms.image.pbcor.GBT_feathered_K.fits")


    mask_name = seventeenB_1kms_wGBT_HI_file_dict['Source_Mask']
    mask_hdu = fits.open(mask_name)[0]

    spat_mask = mask_hdu.data.sum(0) > 6

    del mask_hdu

    # Load in PB plane to account for varying uncertainty
    pb = fits.open(seventeenB_HI_data_1kms_path("M33_14B_17B_HI_contsub_width_1kms.pb.fits"), mode='denywrite')
    pb_plane = pb[0].data[0].copy()
    # pb_plane = pb_plane[nd.find_objects(pb_plane > 0.5)[-1]]
    del pb

    # Need peak temp and centroid maps.

    vcent_name = seventeenB_1kms_wGBT_HI_file_dict['Moment1']
    vcent = Projection.from_hdu(fits.open(vcent_name)).to(u.km / u.s)

    # Noise lowered for the downsampled cube.
    noise_val = 6.6 * u.K

    # Set max number of gaussians to something ridiculous.
    # Just so we don't have a failure putting into the output array
    # First time fitting there was max of 5. Noise is higher here
    # so don't expect to be able to detect a lot of faint emission
    max_comp = 8

    err_map = noise_val / pb_plane

    params_name = seventeenB_HI_data_1kms_wGBT_path("individ_multigaussian_gausspy_fits.fits", no_check=True)

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
                        num_cores=5,
                        chunks=20000)

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

    # Stage 2: Do a neighbourhood check to ensure smoother fit solutions.
    # Now we'll loop through the fits to check against the 3x3 neighbourhood.
    # This should even out nearby spectral fits and the number of components.
    params_name_rev = seventeenB_HI_data_1kms_wGBT_path("individ_multigaussian_gausspy_fits_neighbcheck.fits", no_check=True)
    params_name_rev2 = seventeenB_HI_data_1kms_wGBT_path("individ_multigaussian_gausspy_fits_neighbcheck2.fits", no_check=True)

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
    model_outname = seventeenB_HI_data_1kms_wGBT_path("individ_multigaussian_gausspy_fits_neighbcheck2_model.fits", no_check=True)

    if run_writemodel:
        overwrite = True
        if overwrite and os.path.exists(model_outname):
            os.system(f"rm {model_outname}")

        save_fitmodel(cube_name, params_name_rev2, model_outname,
                      chunk_size=80000,
                      save_sep_components=False)
