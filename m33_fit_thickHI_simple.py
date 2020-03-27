
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

model_script = os.path.join(repo_path, "thickHI_model.py")
exec(compile(open(model_script, "rb").read(), model_script, 'exec'))

run_C = True
run_BC = False

# Enable to limit the thick HI vel. centroid to \pm 5 km/s
# of the centroid velocity.
with_vcent_constraint = False

if with_vcent_constraint:
    delta_vcent = 5 * u.km / u.s
else:
    delta_vcent = 80 * u.km / u.s

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
    pb_plane[pb_plane < 0.5] = np.NaN
    del pb

    # Need peak temp and centroid maps.

    peak_name = fourteenB_wGBT_HI_file_dict['PeakTemp']
    peaktemp = Projection.from_hdu(fits.open(peak_name))

    vcent_name = fourteenB_wGBT_HI_file_dict['Moment1']
    vcent = Projection.from_hdu(fits.open(vcent_name)).to(u.km / u.s)

    # Noise lowered for the downsampled cube.
    noise_val = 2.8 * u.K / np.sqrt(2)

    err_map = noise_val / pb_plane

    params_array, uncerts_array, bic_array = \
        cube_fitter(downsamp_cube_name, fit_func_simple,
                    mask_name=None,
                    npars=4,
                    args=(),
                    kwargs={'downsamp_factor': 1,
                            'min_finite_chan': 30,
                            "delta_vcent": delta_vcent},
                    spatial_mask=spat_mask,
                    err_map=err_map,
                    vcent_map=vcent,
                    num_cores=6,
                    chunks=80000)

    # Save the parameters

    params_hdu = fits.PrimaryHDU(params_array, vcent.header.copy())
    params_hdu.header['BUNIT'] = ("", "Simple thick HI fit parameters")

    uncerts_hdu = fits.ImageHDU(uncerts_array, vcent.header.copy())
    uncerts_hdu.header['BUNIT'] = ("", "Simple thick HI fit uncertainty")

    bics_hdu = fits.ImageHDU(bic_array, vcent.header.copy())
    bics_hdu.header['BUNIT'] = ("", "Simple thick HI fit BIC")

    hdu_all = fits.HDUList([params_hdu, uncerts_hdu, bics_hdu])

    if with_vcent_constraint:
        out_name = "individ_simplethick_HI_fits_5kms_centlimit.fits"
    else:
        out_name = "individ_simplethick_HI_fits_80kms_centlimit.fits"

    hdu_all.writeto(fourteenB_HI_data_wGBT_path(out_name, no_check=True),
                    overwrite=True)

if run_BC:
    pass
