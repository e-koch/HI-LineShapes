
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

run_D = False
run_BCDtaper = True
run_BCD = False


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

    hdu_all.writeto(fourteenA_HI_data_wEBHIS_path("individ_multigaussian_gausspy_fits.fits", no_check=True),
                    overwrite=True)

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

    params_hdu = fits.PrimaryHDU(params_array, vcent.header.copy())
    params_hdu.header['BUNIT'] = ("", "Gaussian fit parameters")

    uncerts_hdu = fits.ImageHDU(uncerts_array, vcent.header.copy())
    uncerts_hdu.header['BUNIT'] = ("", "Gaussian fit uncertainty")

    bics_hdu = fits.ImageHDU(bic_array, vcent.header.copy())
    bics_hdu.header['BUNIT'] = ("", "Gaussian fit BIC")

    hdu_all = fits.HDUList([params_hdu, uncerts_hdu, bics_hdu])

    params_name = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_multigaussian_gausspy_fits.fits", no_check=True)

    hdu_all.writeto(params_name, overwrite=True)

    # Now we'll loop through the fits to check against the 3x3 neighbourhood.
    # This should even out nearby spectral fits and the number of components.
    hdu_all_revised = neighbourhood_fit_comparison(cube_name,
                                                   params_name,
                                                   chunk_size=80000,
                                                   diff_bic=10,
                                                   err_map=err_map)

    params_name_rev = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_multigaussian_gausspy_fits_neighbcheck.fits", no_check=True)

    hdu_all_revised.writeto(params_name_rev, overwrite=True)
