'''
Figure of the BIC for the full cube models.
'''

from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.convolution import Gaussian1DKernel

from time import time
import os
import numpy as np
import matplotlib.pyplot as plt
from spectral_cube import Projection, SpectralCube
from radio_beam import Beam

from scipy import ndimage as nd
from astropy.stats import histogram as astro_hist
from tqdm import tqdm
from corner import hist2d

from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()

osjoin = os.path.join

repo_path = os.path.expanduser("~/ownCloud/project_code/ThickHIFitting/")

figures_path_png = osjoin(repo_path, "figures/png")
figures_path_pdf = osjoin(repo_path, "figures/pdf")


def save_figure(fig, plot_name):
    fig.savefig(f"{figures_path_pdf}/{plot_name}.pdf")
    fig.savefig(f"{figures_path_png}/{plot_name}.png")


paths_script = os.path.join(repo_path, "paths.py")
exec(compile(open(paths_script, "rb").read(), paths_script, 'exec'))

plotstyle_script = os.path.join(repo_path, "plotting_styles.py")
exec(compile(open(plotstyle_script, "rb").read(), plotstyle_script, 'exec'))

thickHI_model_script = os.path.join(repo_path, "thickHI_model.py")
exec(compile(open(thickHI_model_script, "rb").read(),
             thickHI_model_script, 'exec'))


# M31

m31_cubename_K = f"{fifteenA_HI_BCtaper_wEBHIS_HI_file_dict['Cube'].rstrip('.fits')}_K.fits"

m31_cube = SpectralCube.read(m31_cubename_K, use_dask=True)
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


region_slice = (slice(850, 1050), slice(900, 1100))

subcube = m31_cube[(slice(None),) + region_slice]


# Smooth to 30"
# new_beam = Beam(30 * u.arcsec)
# conv_cube = subcube.convolve_to(new_beam, save_to_tmp_dir=True)

# Skipping the spatial smooth to emphasize only the affect of degraded spec res.
# BUT we still need a common beam convolve.
conv_cube = subcube.convolve_to(subcube.beams.common_beam(), save_to_tmp_dir=True)

# Downsample the spectral resolution to 2.27 km/s
target = 2.27 * u.km / u.s
chan_width = np.diff(conv_cube.spectral_axis)[0].to(u.km / u.s)
spec_kern = Gaussian1DKernel(np.sqrt((target / 2.)**2 - chan_width**2).value)
conv_specsmooth_cube = conv_cube.spectral_smooth(spec_kern, save_to_tmp_dir=True)

# downsamp_factor = int(np.floor(np.abs(target / chan_width).value))
downsamp_factor = 6
matched_cube = conv_specsmooth_cube.downsample_axis(downsamp_factor, 0, save_to_tmp_dir=True)

# Make moment maps needed for the fitting:

peaktemp = matched_cube.max(axis=0)
vcent = matched_cube.moment1()

peakchans = matched_cube.argmax(axis=0)
peakvels = np.take_along_axis(matched_cube.spectral_axis[:, np.newaxis,
                                                    np.newaxis],
                                peakchans[np.newaxis, :, :], 0)
peakvels = peakvels.squeeze()
peakvels = peakvels.to(u.km / u.s)

# Save the subcube:

subcube_filename = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("match_braun09/M31_HI_subcube_K_30arcsec_2p4kms.fits", no_check=True)

if not os.path.exists(fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("match_braun09", no_check=True)):
    os.mkdir(fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("match_braun09", no_check=True))

matched_cube.write(subcube_filename, overwrite=True)


from cube_analysis.spectral_fitting import cube_fitter

delta_vcent = 5 * u.km / u.s

noise_val = 0.65 * u.K

pb = fits.open(fifteenA_HI_BCtaper_file_dict['PB'], mode='denywrite')
pb_plane = pb[0].data[0].copy()[region_slice]
del pb

err_map = noise_val / pb_plane

t0 = time()
params_array, uncerts_array, bic_array = \
    cube_fitter(subcube_filename, fit_func_simple,
                mask_name=None,
                npars=4,
                nfit_stats=3,
                args=(),
                kwargs={'downsamp_factor': 1,
                        'min_finite_chan': 30,
                        "delta_vcent": delta_vcent},
                spatial_mask=None,
                err_map=err_map,
                vcent_map=vcent.quantity,
                num_cores=1,
                chunks=100000)

t1 = time()

print(f"{(t1-t0) / 60.} minutes to run the opaque fit")

# Save the parameters

params_hdu = fits.PrimaryHDU(params_array, vcent.header.copy())
params_hdu.header['BUNIT'] = ("", "Simple thick HI fit parameters")

uncerts_hdu = fits.ImageHDU(uncerts_array, vcent.header.copy())
uncerts_hdu.header['BUNIT'] = ("", "Simple thick HI fit uncertainty")

bics_hdu = fits.ImageHDU(bic_array, vcent.header.copy())
bics_hdu.header['BUNIT'] = ("", "Simple thick HI fit BIC")

hdu_all = fits.HDUList([params_hdu, uncerts_hdu, bics_hdu])

hdu_all.writeto(fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("braun09_subcubes/matched_region_opaque_fit_comparisons.fits", no_check=True), overwrite=True)

# Now multi-Gaussians!
model_script = os.path.join(repo_path, "gaussian_model.py")
exec(compile(open(model_script, "rb").read(), model_script, 'exec'))

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

# Use the interactive mask to remove MW contamination
# Use a spatial mask from the signal masking, restricted by the interactive mask
maskint_name = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("M31_15A_taper_interactive_mask.fits")
maskint_hdu = fits.open(maskint_name)[0]

mask_name = fifteenA_HI_BCtaper_wEBHIS_HI_file_dict['Source_Mask']
mask_hdu = fits.open(mask_name)[0]

spat_mask = np.logical_and(maskint_hdu.data.sum(0) > 10, mask_hdu.data.sum(0) > 10)[region_slice]

del maskint_hdu, mask_hdu

mom0_name = fifteenA_HI_BCtaper_wEBHIS_HI_file_dict['Moment0']
mom0 = Projection.from_hdu(fits.open(mom0_name))[region_slice]

spat_mask = np.logical_and(spat_mask, mom0.value > 0.)

# Wayy in excess
max_comp = 30

t2 = time()

params_array_mulgauss, uncerts_array_mulgauss, bic_array_mulgauss = \
    cube_fitter(subcube_filename, fit_func_gausspy,
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
                vcent_map=vcent.quantity,
                num_cores=6,
                chunks=100000)

t3 = time()

print(f"{(t3-t2) / 60.} minutes to run the multi-Gauss fit")

# Save the parameters
# Cut to max number of components
max_comp = np.isfinite(params_array_mulgauss).sum(0).max() // 3

params_array_mulgauss = params_array_mulgauss[:3 * max_comp]
uncerts_array_mulgauss = uncerts_array_mulgauss[:3 * max_comp]

params_hdu = fits.PrimaryHDU(params_array_mulgauss, vcent.header.copy())
params_hdu.header['BUNIT'] = ("", "Gaussian fit parameters")

uncerts_hdu = fits.ImageHDU(uncerts_array_mulgauss, vcent.header.copy())
uncerts_hdu.header['BUNIT'] = ("", "Gaussian fit uncertainty")

bics_hdu = fits.ImageHDU(bic_array_mulgauss, vcent.header.copy())
bics_hdu.header['BUNIT'] = ("", "Gaussian fit BIC")

hdu_all = fits.HDUList([params_hdu, uncerts_hdu, bics_hdu])

hdu_all.writeto(fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("braun09_subcubes/matched_region_multigauss_fit_comparisons.fits", no_check=True), overwrite=True)


# Compare the fit statistics to those from the full resolution cube.

# m31_multigauss_bic_proj[region_slice]
