
'''
Compare the Gaussian properties in M31 and M33.
'''

from astropy.io import fits
import astropy.units as u
import astropy.constants as c
import os
import numpy as np
import matplotlib.pyplot as plt
from spectral_cube import Projection
from scipy import ndimage as nd
from astropy.stats import histogram as astro_hist

from galaxies import Galaxy

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


m31_multigauss_name = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_multigaussian_gausspy_fits.fits")
m31_multigauss_hdu = fits.open(m31_multigauss_name)

m31_ngauss = np.isfinite(m31_multigauss_hdu[0].data).sum(0) // 3
m31_maxgauss = np.nanmax(m31_ngauss)

m31_amps = m31_multigauss_hdu[0].data[::3][:m31_maxgauss] * u.K
m31_cents = m31_multigauss_hdu[0].data[1::3][:m31_maxgauss] * u.m / u.s
m31_lwidths = m31_multigauss_hdu[0].data[2::3][:m31_maxgauss] * u.m / u.s

m31_cents = m31_cents.to(u.km / u.s)
m31_lwidths = m31_lwidths.to(u.km / u.s)

m33_multigauss_name = fourteenB_HI_data_wGBT_path("individ_multigaussian_gausspy_fits.fits")
m33_multigauss_hdu = fits.open(m33_multigauss_name)

m33_ngauss = np.isfinite(m33_multigauss_hdu[0].data).sum(0) // 3
m33_maxgauss = np.nanmax(m33_ngauss)

m33_amps = m33_multigauss_hdu[0].data[::3][:m33_maxgauss] * u.K
m33_cents = m33_multigauss_hdu[0].data[1::3][:m33_maxgauss] * u.m / u.s
m33_lwidths = m33_multigauss_hdu[0].data[2::3][:m33_maxgauss] * u.m / u.s

m33_cents = m33_cents.to(u.km / u.s)
m33_lwidths = m33_lwidths.to(u.km / u.s)


twocolumn_figure()

cts_lw_m31, bin_edges_lw_m31 = astro_hist(m31_lwidths[np.isfinite(m31_lwidths) &
                                                      (m31_lwidths.value < 55)],
                                          bins='knuth', density=True)
bin_centres_lw_m31 = (bin_edges_lw_m31[1:] + bin_edges_lw_m31[:-1]) / 2.

cts_lw_m33, bin_edges_lw_m33 = astro_hist(m33_lwidths[np.isfinite(m33_lwidths) &
                                                      (m33_lwidths.value < 45)].value,
                                          bins='knuth', density=True)
bin_centres_lw_m33 = (bin_edges_lw_m33[1:] + bin_edges_lw_m33[:-1]) / 2.


fig = plt.figure()
ax = fig.add_subplot()

plt.semilogy(bin_centres_lw_m31, cts_lw_m31,
             drawstyle='steps-mid',
             linestyle='-',
             label="M31")

plt.semilogy(bin_centres_lw_m33, cts_lw_m33,
             drawstyle='steps-mid',
             linestyle='--',
             label="M33")

plt.legend(loc='upper right', frameon=True)

plt.grid()

plt.xlabel(r"Line width (km s$^{-1})$")
plt.ylabel("")


def thermwidth(temp):
    return np.sqrt(c.k_B * temp / (1.4 * c.m_p)).to(u.km / u.s)

plt.axvline(thermwidth(1000 * u.K).value, linestyle=':', color='k', zorder=-1)
plt.axvline(thermwidth(6000 * u.K).value, linestyle=':', color='k', zorder=-1)


# Split M33 line widths by radius.
gal = Galaxy('M33')

radii = gal.radius(header=m33_multigauss_hdu[2].header).to(u.kpc)

bins = np.linspace(0, 10, 11) * u.kpc

fig = plt.figure()

ax = fig.add_subplot(111)

max_val = 50 * u.km / u.s

for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):

    bin_locs = np.where(np.logical_and(radii >= low, radii < high))

    lwidth_vals = []

    for j in range(m33_lwidths.shape[0]):
        lwidth_val = m33_lwidths[j][bin_locs].value
        lwidth_vals.append(lwidth_val[np.isfinite(lwidth_val) &
                                      (lwidth_val < max_val.value)])

    lwidth_vals = np.hstack(lwidth_vals)

    if i == 0:
        cts_lw_rad, bin_edges_lw_rad = astro_hist(lwidth_vals,
                                                  bins='knuth',
                                                  density=True)
    else:
        cts_lw_rad, bin_edges_lw_rad = astro_hist(lwidth_vals,
                                                  bins=bin_edges_lw_rad0,
                                                  density=True)

    if i == 0:
        bin_edges_lw_rad0 = bin_edges_lw_rad
        bin_centres_lw_rad = (bin_edges_lw_rad[1:] + bin_edges_lw_rad[:-1]) / 2.

    ax.semilogy(bin_centres_lw_rad, cts_lw_rad,
                drawstyle='steps-mid',
                linestyle='-',
                label=f"{int(low.value)}<R<{int(high.value)}")

ax.legend(frameon=True, loc='upper right')

ax.axvline(0.42, linestyle=':', color='k')
ax.axvline(5 * 0.42, linestyle=':', color='k')
ax.grid(True)

save_figure(fig, "m33_linewidth_rgal_hist")
plt.close()

fig = plt.figure()

ax = fig.add_subplot(111)

for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):

    bin_locs = np.where(np.logical_and(radii >= low, radii < high))

    amp_vals = []

    for j in range(m33_amps.shape[0]):
        amp_val = m33_amps[j][bin_locs].value
        amp_vals.append(amp_val[np.isfinite(amp_val)])

    amp_vals = np.hstack(amp_vals)

    if i == 0:
        cts_vc_rad, bin_edges_vc_rad = astro_hist(amp_vals,
                                                  bins='knuth',
                                                  density=True)
    else:
        cts_vc_rad, bin_edges_vc_rad = astro_hist(amp_vals,
                                                  bins=bin_edges_vc_rad0,
                                                  density=True)

    if i == 0:
        bin_edges_vc_rad0 = bin_edges_vc_rad
        bin_centres_vc_rad = (bin_edges_vc_rad[1:] + bin_edges_vc_rad[:-1]) / 2.

    ax.semilogy(bin_centres_vc_rad, cts_vc_rad,
                drawstyle='steps-mid',
                linestyle='-',
                label=f"{int(low.value)}<R<{int(high.value)}")

ax.legend(frameon=True, loc='upper right', ncol=2)

ax.set_xlabel(f"Amplitude (K)")

ax.axvline(noise_val.value, linestyle=':', color='k')
ax.axvline(5 * noise_val.value, linestyle=':', color='k')
ax.grid(True)

save_figure(fig, "m33_amp_rgal_hist")
plt.close()


fig = plt.figure()

ax = fig.add_subplot(111)

for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):

    bin_locs = np.where(np.logical_and(radii >= low, radii < high))

    intint_vals = []

    for j in range(m33_amps.shape[0]):
        amp_val = m33_amps[j][bin_locs].value
        lwidth_val = m33_lwidths[j][bin_locs].value
        intint_val = np.sqrt(2 * np.pi) * amp_val * lwidth_val
        # intint_val *= 0.0198  # Msol /pc^2
        # intint_val *= np.cos(gal.inclination).value  # M33 inclination
        intint_vals.append(intint_val[np.isfinite(amp_val)])

    intint_vals = np.hstack(intint_vals)

    if i == 0:
        cts_vc_rad, bin_edges_vc_rad = astro_hist(intint_vals,
                                                  bins='knuth',
                                                  density=True)
    else:
        cts_vc_rad, bin_edges_vc_rad = astro_hist(intint_vals,
                                                  bins=bin_edges_vc_rad0,
                                                  density=True)

    if i == 0:
        bin_edges_vc_rad0 = bin_edges_vc_rad
        bin_centres_vc_rad = (bin_edges_vc_rad[1:] + bin_edges_vc_rad[:-1]) / 2.

    ax.semilogy(bin_centres_vc_rad, cts_vc_rad,
                drawstyle='steps-mid',
                linestyle='-',
                label=f"{int(low.value)}<R<{int(high.value)}")

ax.legend(frameon=True, loc='upper right', ncol=2)

ax.set_xlabel(r"Integrated Intensity (K km s$^{-1}$)")

min_sig = 5 * 0.42 / 2.35
ax.axvline(5 * noise_val.value * min_sig * np.sqrt(2 * np.pi),
           linestyle=':', color='k')
ax.grid(True)

save_figure(fig, "m33_intint_rgal_hist")
plt.close()
