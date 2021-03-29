
'''
Use functions from model_selection_tools.py to compare the
models where the thickHI model has large tau (roughly where Tspin
is constrained in the fit).
'''

from astropy.io import fits
import numpy as np
import scipy.ndimage as nd
import astropy.units as u
import os

osjoin = os.path.join

repo_path = os.path.expanduser("~/ownCloud/project_code/ThickHIFitting/")

constants_script = os.path.join(repo_path, "paths.py")
exec(compile(open(constants_script, "rb").read(), constants_script, 'exec'))

model_script = os.path.join(repo_path, "model_selection_tools.py")
exec(compile(open(model_script, "rb").read(), model_script, 'exec'))

run_D = True
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

    noise_val = 0.72 * u.K
    err_map = noise_val / pb_plane

    params_name_rev2 = fourteenA_HI_data_wEBHIS_path("individ_multigaussian_gausspy_fits_neighbcheck2.fits",
                                                   no_check=True)

    # Different opt thick model constraints
    params_name_thHI_vconst = fourteenA_HI_data_wEBHIS_path("individ_simplethick_HI_fits_5kms_centlimit.fits")

    recomp_bic_hdu_vconst = \
        compare_optthick_over_cube(cube_name,
                                   params_name_thHI_vconst,
                                   params_name_rev2,
                                   err_map,
                                   tau_min=0.5,
                                   min_pts=10,
                                   gausscomp_frac=0.25,
                                   chunk_size=80000)

    recomp_bic_vconst_name = fourteenA_HI_data_wEBHIS_path("individ_recomp_bic_tau_gt_0p5_5kms_centlimit.fits",
                                                         no_check=True)

    recomp_bic_hdu_vconst.writeto(recomp_bic_vconst_name, overwrite=True)

    params_name_thHI = fourteenA_HI_data_wEBHIS_path("individ_simplethick_HI_fits_80kms_centlimit.fits")

    recomp_bic_hdu = \
        compare_optthick_over_cube(cube_name,
                                   params_name_thHI,
                                   params_name_rev2,
                                   err_map,
                                   tau_min=0.5,
                                   min_pts=10,
                                   gausscomp_frac=0.25,
                                   chunk_size=80000)

    recomp_bic_vconst_name = fourteenA_HI_data_wEBHIS_path("individ_recomp_bic_tau_gt_0p5_80kms_centlimit.fits",
                                                           no_check=True)

    recomp_bic_hdu.writeto(recomp_bic_vconst_name, overwrite=True)

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

    noise_val = 2.8 * u.K

    err_map = noise_val / pb_plane


    params_name_rev2 = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_multigaussian_gausspy_fits_neighbcheck2.fits",
                                                   no_check=True)

    # Different opt thick model constraints
    params_name_thHI_vconst = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_simplethick_HI_fits_5kms_centlimit.fits")

    recomp_bic_hdu_vconst = \
        compare_optthick_over_cube(cube_name,
                                   params_name_thHI_vconst,
                                   params_name_rev2,
                                   err_map,
                                   tau_min=0.5,
                                   min_pts=10,
                                   gausscomp_frac=0.25,
                                   chunk_size=80000)

    recomp_bic_vconst_name = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_recomp_bic_tau_gt_0p5_5kms_centlimit.fits",
                                                         no_check=True)

    recomp_bic_hdu_vconst.writeto(recomp_bic_vconst_name, overwrite=True)

    params_name_thHI = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_simplethick_HI_fits_80kms_centlimit.fits")

    recomp_bic_hdu = \
        compare_optthick_over_cube(cube_name,
                                   params_name_thHI,
                                   params_name_rev2,
                                   err_map,
                                   tau_min=0.5,
                                   min_pts=10,
                                   gausscomp_frac=0.25,
                                   chunk_size=80000)

    recomp_bic_vconst_name = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_recomp_bic_tau_gt_0p5_80kms_centlimit.fits",
                                                           no_check=True)

    recomp_bic_hdu.writeto(recomp_bic_vconst_name, overwrite=True)

if run_BCD:
    pass
