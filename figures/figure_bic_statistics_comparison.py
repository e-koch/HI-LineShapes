
'''
Figure of the BIC for the full cube models.
'''

from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
import os
import numpy as np
import matplotlib.pyplot as plt
from spectral_cube import Projection
from scipy import ndimage as nd
from astropy.stats import histogram as astro_hist

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

# M31

m31_multigauss_name = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_multigaussian_gausspy_fits.fits")
m31_multigauss_hdu = fits.open(m31_multigauss_name)

m31_ngauss = np.isfinite(m31_multigauss_hdu[0].data).sum(0) // 3

m31_thickHI_name = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_simplethick_HI_fits_5kms_centlimit.fits")
m31_thickHI_hdu = fits.open(m31_thickHI_name)

m31_multigauss_bic_proj = Projection.from_hdu(m31_multigauss_hdu[2])
m31_thickHI_bic_proj = Projection.from_hdu(m31_thickHI_hdu[2])

# Slice out to zoom into the valid data region.
spat_slice_zoom = tuple([slice(vals.min() - 10, vals.max() + 10) for vals in
                         np.where(np.isfinite(m31_multigauss_hdu[2].data))])


twocolumn_figure()

diff_bic = m31_multigauss_bic_proj - m31_thickHI_bic_proj
diff_bic_zoom = diff_bic[spat_slice_zoom]

fig = plt.figure()

ax = fig.add_subplot(projection=diff_bic_zoom.wcs)

# Scale to the 95% in image
vmin, vmax = np.nanpercentile(diff_bic, [2.5, 97.5])

# im = ax.imshow(diff_bic_zoom.value, vmin=vmin, vmax=vmax, cmap=plt.cm.RdGy_r)
im = ax.imshow(diff_bic_zoom.value, vmin=vmin, vmax=vmax,
               cmap=plt.cm.Oranges_r)

# ax.contour(diff_bic_zoom, levels=[20], colors='g')

cbar = plt.colorbar(im)
cbar.set_label(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")

# TODO: Add a 'signif. threshold' at -10.

save_figure(fig, 'm31_delta_bic_map')
plt.close()

# Show a collapsed histogram of delta BIC

fig = plt.figure()
ax = fig.add_subplot()

cts, bin_edges = astro_hist(diff_bic.value[np.isfinite(diff_bic)],
                            bins='knuth')
bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.

ax.semilogy(bin_centres, cts, drawstyle='steps-mid', label='All')

ax.set_xlabel(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")
ax.set_ylabel("Number of Spectra")

ax.axvline(-10., color='k', linestyle='--')
ax.axvline(10., color='k', linestyle='--')

ax.text(-80, 2e5, "Gaussian model\npreferred",
        verticalalignment='center',
        horizontalalignment='right')
ax.text(80, 2e5, "Optically-thick model\npreferred",
        verticalalignment='center',
        horizontalalignment='left')

# Now split by number of gaussians
# Max is 8. We'll do each number up to 5, then >5

for nc in range(1, 5):

    comp_mask = np.logical_and(np.isfinite(diff_bic),
                               m31_ngauss == nc)

    cts = astro_hist(diff_bic.value[comp_mask],
                     bins=bin_edges)[0]

    ax.semilogy(bin_centres, cts, drawstyle='steps-mid',
                label=f'{nc} Gaussians' if nc > 1 else "1 Gaussian")
# "the rest"
comp_mask = np.logical_and(np.isfinite(diff_bic),
                           m31_ngauss >= 5)

cts = astro_hist(diff_bic.value[comp_mask],
                 bins=bin_edges)[0]

ax.semilogy(bin_centres, cts, drawstyle='steps-mid',
            label=r'$\geq$5 Gaussians')

leg = ax.legend(loc='upper left')

ax.grid()

save_figure(fig, 'm31_delta_bic_hist_all')
plt.close()

# Histogram but limited to where the spin temperature is constrained.
constTs_mask = m31_thickHI_hdu[0].data[0] / m31_thickHI_hdu[1].data[0] > 1.
constTs_mask = np.logical_and(constTs_mask,
                              np.isfinite(m31_thickHI_hdu[1].data[0]))
constTs_mask = np.logical_and(constTs_mask,
                              np.isfinite(diff_bic))

fig = plt.figure()
ax = fig.add_subplot()

cts, bin_edges = astro_hist(diff_bic.value[constTs_mask],
                            bins='knuth')
bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.

ax.semilogy(bin_centres, cts, drawstyle='steps-mid', label='All')

ax.set_xlabel(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")
ax.set_ylabel("Number of Spectra")

ax.axvline(-10., color='k', linestyle='--')
ax.axvline(10., color='k', linestyle='--')

ax.text(-80, 2e4, "Gaussian model\npreferred",
        verticalalignment='center',
        horizontalalignment='right')
ax.text(80, 2e4, "Optically-thick\nmodel preferred",
        verticalalignment='center',
        horizontalalignment='left')

# Now split by number of gaussians
# Max is 8. We'll do each number up to 5, then >5

for nc in range(1, 5):

    comp_mask = np.logical_and(constTs_mask,
                               m31_ngauss == nc)

    cts = astro_hist(diff_bic.value[comp_mask],
                     bins=bin_edges)[0]

    ax.semilogy(bin_centres, cts, drawstyle='steps-mid',
                label=f'{nc} Gaussians' if nc > 1 else "1 Gaussian")

# "the rest"
comp_mask = np.logical_and(np.isfinite(diff_bic),
                           m31_ngauss >= 5)

cts = astro_hist(diff_bic.value[comp_mask],
                 bins=bin_edges)[0]

ax.semilogy(bin_centres, cts, drawstyle='steps-mid',
            label=r'$\geq$5 Gaussians')

leg = ax.legend(loc='upper left')

ax.grid()

save_figure(fig, 'm31_delta_bic_hist_constrained_Ts')
plt.close()


# Histogram of LOS where the thick HI peak opt. depth is >0.5
thick_mask = m31_thickHI_hdu[0].data[2] / m31_thickHI_hdu[0].data[0] > 0.5
thick_mask = np.logical_and(thick_mask,
                            np.isfinite(m31_thickHI_hdu[1].data[0]))
thick_mask = np.logical_and(thick_mask,
                            np.isfinite(diff_bic))

fig = plt.figure()
ax = fig.add_subplot()

cts, bin_edges = astro_hist(diff_bic.value[thick_mask],
                            bins='knuth')
bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.

ax.semilogy(bin_centres, cts, drawstyle='steps-mid', label='All')

ax.set_xlabel(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")
ax.set_ylabel("Number of Spectra")

ax.axvline(-10., color='k', linestyle='--')
ax.axvline(10., color='k', linestyle='--')

ax.text(-80, 2e4, "Gaussian model\npreferred",
        verticalalignment='center',
        horizontalalignment='right')
ax.text(80, 2e4, "Optically-thick\nmodel preferred",
        verticalalignment='center',
        horizontalalignment='left')

# Now split by number of gaussians
# Max is 8. We'll do each number up to 5, then >5

for nc in range(1, 5):

    comp_mask = np.logical_and(thick_mask,
                               m31_ngauss == nc)

    cts = astro_hist(diff_bic.value[comp_mask],
                     bins=bin_edges)[0]

    ax.semilogy(bin_centres, cts, drawstyle='steps-mid',
                label=f'{nc} Gaussians' if nc > 1 else "1 Gaussian")

# "the rest"
comp_mask = np.logical_and(np.isfinite(diff_bic),
                           m31_ngauss >= 5)

cts = astro_hist(diff_bic.value[comp_mask],
                 bins=bin_edges)[0]

ax.semilogy(bin_centres, cts, drawstyle='steps-mid',
            label=r'$\geq$5 Gaussians')

leg = ax.legend(loc='upper left')

ax.grid()

save_figure(fig, 'm31_delta_bic_hist_tau_gt_05')
plt.close()


# Histogram binned with peak tau
peaktau_map = m31_thickHI_hdu[0].data[2] / m31_thickHI_hdu[0].data[0]
thick_mask = peaktau_map > 0.5
thick_mask = np.logical_and(thick_mask,
                            np.isfinite(m31_thickHI_hdu[1].data[0]))
thick_mask = np.logical_and(thick_mask,
                            np.isfinite(diff_bic))

fig = plt.figure()
ax = fig.add_subplot()

cts, bin_edges = astro_hist(diff_bic.value[thick_mask],
                            bins='knuth')
bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.

ax.semilogy(bin_centres, cts, drawstyle='steps-mid', label='All')

ax.set_xlabel(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")
ax.set_ylabel("Number of Spectra")

ax.axvline(-10., color='k', linestyle='--')
ax.axvline(10., color='k', linestyle='--')

ax.text(-80, 2e4, "Gaussian model\npreferred",
        verticalalignment='center',
        horizontalalignment='right')
ax.text(80, 2e4, "Optically-thick\nmodel preferred",
        verticalalignment='center',
        horizontalalignment='left')

# Now split by number of gaussians
# Max is 8. We'll do each number up to 5, then >5

for ltau, utau in zip([0.5, 1, 2, 3, 4], [1, 2, 3, 4, 5]):

    tau_mask = np.logical_and(peaktau_map >= ltau,
                              peaktau_map < utau)

    comp_mask = np.logical_and(thick_mask,
                               tau_mask)

    cts = astro_hist(diff_bic.value[comp_mask],
                     bins=bin_edges)[0]

    ax.semilogy(bin_centres, cts, drawstyle='steps-mid',
                label=r'{0} $<\,\tau_0\,<$ {1}'.format(int(ltau), int(utau)))

leg = ax.legend(loc='upper left')

ax.grid()

save_figure(fig, 'm31_delta_bic_hist_tau_steps')
plt.close()
