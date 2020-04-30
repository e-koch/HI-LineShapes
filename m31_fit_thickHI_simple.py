
'''
Fit the simplified optically-thick HI models to the whole M31 cubes.
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

model_script = os.path.join(repo_path, "thickHI_model.py")
exec(compile(open(model_script, "rb").read(), model_script, 'exec'))

run_D = True
run_BCDtaper = True
run_BCD = False

if run_D:
    # 14A cube.

    # Enable to limit the thick HI vel. centroid to \pm 5 km/s
    # of the centroid velocity.

    for with_vcent_constraint in [True, False]:

        if with_vcent_constraint:
            delta_vcent = 5 * u.km / u.s
        else:
            delta_vcent = 80 * u.km / u.s

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

        mom0_name = fourteenA_wEBHIS_HI_file_dict['Moment0']
        mom0 = Projection.from_hdu(fits.open(mom0_name))

        # Remove the spectra with strong absorption.
        spat_mask = np.logical_and(spat_mask, mom0.value > 0.)

        peak_name = fourteenA_wEBHIS_HI_file_dict['PeakTemp']
        peaktemp = Projection.from_hdu(fits.open(peak_name))


        # Need centroid map.
        vcent_name = fourteenA_wEBHIS_HI_file_dict['Moment1']
        vcent = Projection.from_hdu(fits.open(vcent_name)).to(u.km / u.s)

        noise_val = 0.72 * u.K

        err_map = noise_val / pb_plane

        params_array, uncerts_array, bic_array = \
            cube_fitter(cube_name, fit_func_simple,
                        mask_name=None,
                        npars=4,
                        args=(),
                        kwargs={'downsamp_factor': 1,
                                'min_finite_chan': 30,
                                "delta_vcent": delta_vcent},
                        spatial_mask=spat_mask,
                        err_map=err_map,
                        vcent_map=vcent,
                        num_cores=4,
                        chunks=20000)

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

        hdu_all.writeto(fourteenA_HI_data_wEBHIS_path(out_name, no_check=True),
                        overwrite=True)

if run_BCDtaper:
    # 14A+15A cube tapered to C-config.

    for with_vcent_constraint in [True, False]:

        if with_vcent_constraint:
            delta_vcent = 5 * u.km / u.s
        else:
            delta_vcent = 80 * u.km / u.s

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

        mom0_name = fifteenA_HI_BCtaper_wEBHIS_HI_file_dict['Moment0']
        mom0 = Projection.from_hdu(fits.open(mom0_name))

        # Remove the spectra with strong absorption.
        spat_mask = np.logical_and(spat_mask, mom0.value > 0.)

        # Need centroid map.
        vcent_name = fifteenA_HI_BCtaper_wEBHIS_HI_file_dict['Moment1']
        vcent = Projection.from_hdu(fits.open(vcent_name)).to(u.km / u.s)

        noise_val = 2.8 * u.K

        err_map = noise_val / pb_plane

        params_array, uncerts_array, bic_array = \
            cube_fitter(cube_name, fit_func_simple,
                        mask_name=None,
                        npars=4,
                        args=(),
                        kwargs={'downsamp_factor': 1,
                                'min_finite_chan': 30,
                                "delta_vcent": delta_vcent},
                        spatial_mask=spat_mask,
                        err_map=err_map,
                        vcent_map=vcent,
                        num_cores=4,
                        chunks=20000)

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

        hdu_all.writeto(fifteenA_HI_BCtaper_04kms_data_wEBHIS_path(out_name, no_check=True),
                        overwrite=True)
