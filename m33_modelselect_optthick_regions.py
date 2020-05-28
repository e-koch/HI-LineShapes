
'''
Use functions from model_selection_tools.py to compare the
models where the thickHI model has large tau (roughly where Tspin
is constrained in the fit).
'''

from astropy.io import fits
import numpy as np
import scipy.ndimage as nd
import os
import astropy.units as u

osjoin = os.path.join

repo_path = os.path.expanduser("~/ownCloud/project_code/ThickHIFitting/")

constants_script = os.path.join(repo_path, "paths.py")
exec(compile(open(constants_script, "rb").read(), constants_script, 'exec'))

model_script = os.path.join(repo_path, "model_selection_tools.py")
exec(compile(open(model_script, "rb").read(), model_script, 'exec'))


run_C = True
run_BC = False


if run_C:

    cube_name = fourteenB_HI_data_wGBT_path("M33_14B-088_HI.clean.image.GBT_feathered.pbcov_gt_0.5_masked.fits")

    # Create a downsampled version of the cube with 0.42 km/s channels.
    # This will match the M31 spectral resolution.
    downsamp_cube_name = f"{cube_name.rstrip('.fits')}_0p42kms_K.fits"

    # Load in PB plane to account for varying uncertainty
    pb = fits.open(fourteenB_HI_data_path("M33_14B-088_pbcov.fits"), mode='denywrite')
    # pb_plane = pb[0].data[0].copy()
    pb_plane = pb[0].data.copy()
    pb_plane = pb_plane[nd.find_objects(pb_plane > 0.5)[-1]]
    del pb

    noise_val = 2.8 * u.K / np.sqrt(2)
    err_map = noise_val / pb_plane

    params_name_rev2 = fourteenB_HI_data_wGBT_path("individ_multigaussian_gausspy_fits_neighbcheck2.fits",
                                                   no_check=True)

    # Different opt thick model constraints
    params_name_thHI_vconst = fourteenB_HI_data_wGBT_path("individ_simplethick_HI_fits_5kms_centlimit.fits")

    recomp_bic_hdu_vconst = \
        compare_optthick_over_cube(downsamp_cube_name,
                                   params_name_thHI_vconst,
                                   params_name_rev2,
                                   err_map,
                                   tau_min=0.5,
                                   min_pts=10,
                                   gausscomp_frac=0.25,
                                   chunk_size=80000)

    recomp_bic_vconst_name = fourteenB_HI_data_wGBT_path("individ_recomp_bic_tau_gt_0p5_5kms_centlimit.fits",
                                                         no_check=True)

    recomp_bic_hdu_vconst.writeto(recomp_bic_vconst_name, overwrite=True)

    params_name_thHI = fourteenB_HI_data_wGBT_path("individ_simplethick_HI_fits_80kms_centlimit.fits")

    recomp_bic_hdu = \
        compare_optthick_over_cube(downsamp_cube_name,
                                   params_name_thHI,
                                   params_name_rev2,
                                   err_map,
                                   tau_min=0.5,
                                   min_pts=10,
                                   gausscomp_frac=0.25,
                                   chunk_size=80000)

    recomp_bic_vconst_name = fourteenB_HI_data_wGBT_path("individ_recomp_bic_tau_gt_0p5_80kms_centlimit.fits",
                                                         no_check=True)

    recomp_bic_hdu.writeto(recomp_bic_vconst_name, overwrite=True)


if run_BC:
    pass
