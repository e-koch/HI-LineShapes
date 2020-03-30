
'''
Figure of the BIC for the full cube models.
No 5 km/s vcent restriction to see if that removes the preference
for any thick HI models (within reason).
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

run_m31 = True
run_m33 = True


# M31
if run_m31:

    m31_multigauss_name = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_multigaussian_gausspy_fits.fits")
    m31_multigauss_hdu = fits.open(m31_multigauss_name)

    m31_ngauss = np.isfinite(m31_multigauss_hdu[0].data).sum(0) // 3

    m31_thickHI_name = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("individ_simplethick_HI_fits_80kms_centlimit.fits")
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

    save_figure(fig, 'm31_delta_bic_map_novcentlim')
    plt.close()

    # Show a collapsed histogram of delta BIC
    # We'll loop over three types of plots:
    # 1. Histogram
    # 2. Histogram and CDF
    # 3. 2. plus a map with contours from histogram splits

    # Offset by none so I don't need an n-1 index
    file_suffixes = [None, "", "_cdf", "_cdf_map"]

    for fig_type in [1, 2, 3]:

        fig = plt.figure()
        if fig_type == 1:
            ax = fig.add_subplot()
        elif fig_type == 2:
            ax = fig.add_subplot(211)
            ax2 = fig.add_subplot(212, sharex=ax)
        else:
            ax = fig.add_subplot(221)
            ax2 = fig.add_subplot(223, sharex=ax)
            ax3 = fig.add_subplot(122, projection=diff_bic_zoom.wcs)

    cts_all, bin_edges = astro_hist(diff_bic.value[np.isfinite(diff_bic)],
                                    bins='knuth')
    bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.

    ax.semilogy(bin_centres, cts_all, drawstyle='steps-mid', label='All',
                linewidth=2, color='k')

    if fig_type == 1:
        ax.set_xlabel(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")

    ax.set_ylabel("Number of Spectra")

    if fig_type > 1:
        ax2.plot(bin_centres, np.cumsum(cts_all) / cts_all.sum(),
                 drawstyle='steps-mid', label='All',
                 linewidth=2, color='k')
        ax2.set_xlabel(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")
        ax2.set_ylabel("CDF")

        ax2.axvline(0., color='k', linestyle='--')

    ax.axvline(0., color='k', linestyle='--')

    ax.text(-80, 1e5, "Gaussian model\npreferred",
            verticalalignment='center',
            horizontalalignment='right')
    ax.text(80, 1e5, "Optically-thick\nmodel preferred",
            verticalalignment='center',
            horizontalalignment='left')

    # Make delta BIC image
    if fig_type == 3:
        im = ax3.imshow(diff_bic_zoom.value, vmin=vmin, vmax=vmax,
                        cmap=plt.cm.Oranges_r)

        cbar = plt.colorbar(im)
        cbar.set_label(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")

    # Now split by number of gaussians
    # Max is 8. We'll do each number up to 5, then >5

    for nc, lstyle in zip(range(1, 5), linestyles_list):

        col = sb.color_palette()[nc - 1]

        comp_mask = np.logical_and(np.isfinite(diff_bic),
                                   m31_ngauss == nc)

        cts = astro_hist(diff_bic.value[comp_mask],
                         bins=bin_edges)[0]

        ax.semilogy(bin_centres, cts, drawstyle='steps-mid',
                    label=f'{nc} Gaussians' if nc > 1 else "1 Gaussian",
                    linestyle=lstyle)

        if fig_type > 1:
            ax2.plot(bin_centres, np.cumsum(cts) / cts_all.sum(),
                     drawstyle='steps-mid',
                     label=f'{nc} Gaussians' if nc > 1 else "1 Gaussian",
                     linestyle=lstyle)

        if fig_type == 3:
            comp_mask = np.logical_and(np.isfinite(diff_bic_zoom),
                                       m31_ngauss[spat_slice_zoom] == nc)

            ax3.contour(comp_mask,
                        linestyles=[lstyle],
                        levels=[0.5],
                        colors=[col])

    # "the rest"
    comp_mask = np.logical_and(np.isfinite(diff_bic),
                               m31_ngauss >= 5)

    cts = astro_hist(diff_bic.value[comp_mask],
                     bins=bin_edges)[0]

    ax.semilogy(bin_centres, cts, drawstyle='steps-mid',
                label=r'$\geq$5 Gaussians',
                linestyle=linestyles_list[5])

    if fig_type > 1:
        ax2.plot(bin_centres, np.cumsum(cts) / cts_all.sum(),
                 label=r'$\geq$5 Gaussians',
                 drawstyle='steps-mid',
                 linestyle=linestyles_list[5])

    if fig_type == 3:
        comp_mask = np.logical_and(np.isfinite(diff_bic_zoom),
                                   m31_ngauss[spat_slice_zoom] >= 5)

        col = sb.color_palette()[4]

        ax3.contour(comp_mask,
                    linestyles=[linestyles_list[5]],
                    levels=[0.5],
                    colors=[col])

    if fig_type > 1:
        leg = ax2.legend(loc='upper left')
        ax2.grid()
    else:
        leg = ax.legend(loc='upper left')

    ax.grid()

    save_figure(fig, f'm31_delta_bic_hist_all_novcentlim{file_suffixes[fig_type]}')
    plt.close()

    # Histogram but limited to where the spin temperature is constrained.
    constTs_mask = m31_thickHI_hdu[0].data[0] / m31_thickHI_hdu[1].data[0] > 1.
    constTs_mask = np.logical_and(constTs_mask,
                                  np.isfinite(m31_thickHI_hdu[1].data[0]))
    constTs_mask = np.logical_and(constTs_mask,
                                  np.isfinite(diff_bic))

    fig = plt.figure()
    if fig_type == 1:
        ax = fig.add_subplot()
    elif fig_type == 2:
        ax = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax)
    else:
        ax = fig.add_subplot(221)
        ax2 = fig.add_subplot(223, sharex=ax)
        ax3 = fig.add_subplot(122, projection=diff_bic_zoom.wcs)

    ax.set_ylabel("Number of Spectra")

    cts_all, bin_edges = astro_hist(diff_bic.value[constTs_mask],
                                    bins='knuth')
    bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.

    ax.semilogy(bin_centres, cts_all, drawstyle='steps-mid', label='All',
                linewidth=2, color='k')

    if fig_type == 1:
        ax.set_xlabel(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")

    ax.set_ylabel("Number of Spectra")

    if fig_type > 1:
        ax2.plot(bin_centres, np.cumsum(cts_all) / cts_all.sum(),
                 drawstyle='steps-mid', label='All',
                 linewidth=2, color='k')
        ax2.set_xlabel(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")
        ax2.set_ylabel("CDF")

        ax2.axvline(0., color='k', linestyle='--')

    ax.axvline(0., color='k', linestyle='--')

    ax.text(-80, 2e4, "Gaussian model\npreferred",
            verticalalignment='center',
            horizontalalignment='right')
    ax.text(80, 2e4, "Optically-thick\nmodel preferred",
            verticalalignment='center',
            horizontalalignment='left')

    if fig_type == 3:
        im = ax3.imshow(diff_bic_zoom.value, vmin=vmin, vmax=vmax,
                        cmap=plt.cm.Oranges_r)

        cbar = plt.colorbar(im)
        cbar.set_label(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")

    # Now split by number of gaussians
    # Max is 8. We'll do each number up to 5, then >5

    for nc, lstyle in zip(range(1, 5), linestyles_list):

        col = sb.color_palette()[nc - 1]

        comp_mask = np.logical_and(constTs_mask,
                                   m31_ngauss == nc)

        cts = astro_hist(diff_bic.value[comp_mask],
                         bins=bin_edges)[0]

        ax.semilogy(bin_centres, cts, drawstyle='steps-mid',
                    label=f'{nc} Gaussians' if nc > 1 else "1 Gaussian",
                    linestyle=lstyle)

        if fig_type > 1:
            ax2.plot(bin_centres, np.cumsum(cts) / cts_all.sum(),
                     drawstyle='steps-mid',
                     label=f'{nc} Gaussians' if nc > 1 else "1 Gaussian",
                     linestyle=lstyle)

        if fig_type == 3:
            comp_mask = np.logical_and(constTs_mask[spat_slice_zoom],
                                       m31_ngauss[spat_slice_zoom] == nc)

            ax3.contour(comp_mask,
                        linestyles=[lstyle],
                        levels=[0.5],
                        colors=[col])

    # "the rest"
    comp_mask = np.logical_and(constTs_mask,
                               m31_ngauss >= 5)

    cts = astro_hist(diff_bic.value[comp_mask],
                     bins=bin_edges)[0]

    ax.semilogy(bin_centres, cts, drawstyle='steps-mid',
                label=r'$\geq$5 Gaussians',
                linestyle=linestyles_list[5])

    if fig_type > 1:
        ax2.plot(bin_centres, np.cumsum(cts) / cts_all.sum(),
                 label=r'$\geq$5 Gaussians',
                 drawstyle='steps-mid',
                 linestyle=linestyles_list[5])

    if fig_type == 3:
        comp_mask = np.logical_and(constTs_mask[spat_slice_zoom],
                                   m31_ngauss[spat_slice_zoom] >= 5)

        col = sb.color_palette()[4]

        ax3.contour(comp_mask,
                    linestyles=[linestyles_list[5]],
                    levels=[0.5],
                    colors=[col])

    if fig_type > 1:
        leg = ax2.legend(loc='upper left')
        ax2.grid()
    else:
        leg = ax.legend(loc='upper left')

        ax.grid()

        save_figure(fig, f'm31_delta_bic_hist_constrained_Ts_novcentlim{file_suffixes[fig_type]}')
        plt.close()


        # Histogram of LOS where the thick HI peak opt. depth is >0.5
        thick_mask = m31_thickHI_hdu[0].data[2] / m31_thickHI_hdu[0].data[0] > 0.5
        thick_mask = np.logical_and(thick_mask,
                                    np.isfinite(m31_thickHI_hdu[1].data[0]))
        thick_mask = np.logical_and(thick_mask,
                                    np.isfinite(diff_bic))

        fig = plt.figure()
        if fig_type == 1:
            ax = fig.add_subplot()
        elif fig_type == 2:
            ax = fig.add_subplot(211)
            ax2 = fig.add_subplot(212, sharex=ax)
        else:
            ax = fig.add_subplot(221)
            ax2 = fig.add_subplot(223, sharex=ax)
            ax3 = fig.add_subplot(122, projection=diff_bic_zoom.wcs)

        cts_all, bin_edges = astro_hist(diff_bic.value[thick_mask],
                                        bins='knuth')
        bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.

        ax.semilogy(bin_centres, cts_all, drawstyle='steps-mid', label='All',
                    linewidth=2, color='k')

        if fig_type == 1:
            ax.set_xlabel(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")

        ax.set_ylabel("Number of Spectra")

        if fig_type > 1:
            ax2.plot(bin_centres, np.cumsum(cts_all) / cts_all.sum(),
                     drawstyle='steps-mid', label='All',
                     linewidth=2, color='k')
            ax2.set_xlabel(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")
            ax2.set_ylabel("CDF")

            ax2.axvline(0., color='k', linestyle='--')

        ax.axvline(0., color='k', linestyle='--')

        ax.text(-80, 2e4, "Gaussian model\npreferred",
                verticalalignment='center',
                horizontalalignment='right')
        ax.text(80, 2e4, "Optically-thick\nmodel preferred",
                verticalalignment='center',
                horizontalalignment='left')

        if fig_type == 3:
            im = ax3.imshow(diff_bic_zoom.value, vmin=vmin, vmax=vmax,
                            cmap=plt.cm.Oranges_r)

            cbar = plt.colorbar(im)
            cbar.set_label(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")

        # Now split by number of gaussians
        # Max is 8. We'll do each number up to 5, then >5

        for nc, lstyle in zip(range(1, 5), linestyles_list):

            col = sb.color_palette()[nc - 1]

            comp_mask = np.logical_and(thick_mask,
                                       m31_ngauss == nc)

            cts = astro_hist(diff_bic.value[comp_mask],
                             bins=bin_edges)[0]

            ax.semilogy(bin_centres, cts, drawstyle='steps-mid',
                        label=f'{nc} Gaussians' if nc > 1 else "1 Gaussian",
                        linestyle=lstyle)

            if fig_type > 1:
                ax2.plot(bin_centres, np.cumsum(cts) / cts_all.sum(),
                         drawstyle='steps-mid',
                         label=f'{nc} Gaussians' if nc > 1 else "1 Gaussian",
                         linestyle=lstyle)

            if fig_type == 3:
                comp_mask = np.logical_and(thick_mask[spat_slice_zoom],
                                           m31_ngauss[spat_slice_zoom] == nc)

                ax3.contour(comp_mask,
                            linestyles=[lstyle],
                            levels=[0.5],
                            colors=[col])

        # "the rest"
        comp_mask = np.logical_and(thick_mask,
                                   m31_ngauss >= 5)

        cts = astro_hist(diff_bic.value[comp_mask],
                         bins=bin_edges)[0]

        ax.semilogy(bin_centres, cts, drawstyle='steps-mid',
                    label=r'$\geq$5 Gaussians',
                    linestyle=linestyles_list[5])

        if fig_type > 1:
            ax2.plot(bin_centres, np.cumsum(cts) / cts_all.sum(),
                     label=r'$\geq$5 Gaussians',
                     drawstyle='steps-mid',
                     linestyle=linestyles_list[5])

        if fig_type == 3:
            comp_mask = np.logical_and(thick_mask[spat_slice_zoom],
                                       m31_ngauss[spat_slice_zoom] >= 5)

            col = sb.color_palette()[4]

            ax3.contour(comp_mask,
                        linestyles=[linestyles_list[5]],
                        levels=[0.5],
                        colors=[col])

        if fig_type > 1:
            leg = ax2.legend(loc='upper left')
            ax2.grid()
        else:
            leg = ax.legend(loc='upper left')

        ax.grid()
        save_figure(fig, f'm31_delta_bic_hist_tau_gt_0p5_novcentlim{file_suffixes[fig_type]}')
        plt.close()


        # Histogram binned with peak tau
        peaktau_map = m31_thickHI_hdu[0].data[2] / m31_thickHI_hdu[0].data[0]
        thick_mask = peaktau_map > 0.5
        thick_mask = np.logical_and(thick_mask,
                                    np.isfinite(m31_thickHI_hdu[1].data[0]))
        thick_mask = np.logical_and(thick_mask,
                                    np.isfinite(diff_bic))

        fig = plt.figure()
        if fig_type == 1:
            ax = fig.add_subplot()
        elif fig_type == 2:
            ax = fig.add_subplot(211)
            ax2 = fig.add_subplot(212, sharex=ax)
        else:
            ax = fig.add_subplot(221)
            ax2 = fig.add_subplot(223, sharex=ax)
            ax3 = fig.add_subplot(122, projection=diff_bic_zoom.wcs)

        cts_all, bin_edges = astro_hist(diff_bic.value[thick_mask],
                                        bins='knuth')
        bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.

        ax.semilogy(bin_centres, cts_all, drawstyle='steps-mid', label='All',
                    linewidth=2, color='k')

        if fig_type == 1:
            ax.set_xlabel(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")

        ax.set_ylabel("Number of Spectra")

        if fig_type > 1:
            ax2.plot(bin_centres, np.cumsum(cts_all) / cts_all.sum(),
                     drawstyle='steps-mid', label='All',
                     linewidth=2, color='k')
            ax2.set_xlabel(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")
            ax2.set_ylabel("CDF")

            ax2.axvline(0., color='k', linestyle='--')

        ax.axvline(0., color='k', linestyle='--')

        # ax.axvline(-10., color='k', linestyle='--')
        # ax.axvline(10., color='k', linestyle='--')

        ax.text(-80, 2e4, "Gaussian model\npreferred",
                verticalalignment='center',
                horizontalalignment='right')
        ax.text(80, 2e4, "Optically-thick\nmodel preferred",
                verticalalignment='center',
                horizontalalignment='left')

        if fig_type == 3:
            im = ax3.imshow(diff_bic_zoom.value, vmin=vmin, vmax=vmax,
                            cmap=plt.cm.Oranges_r)

            cbar = plt.colorbar(im)
            cbar.set_label(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")

        # Now split by number of gaussians
        # Max is 8. We'll do each number up to 5, then >5

        for ltau, utau, lstyle in zip([0.5, 1, 2, 3, 4], [1, 2, 3, 4, 5],
                                      linestyles_list):

            col = sb.color_palette()[utau - 1]

            tau_mask = np.logical_and(peaktau_map >= ltau,
                                      peaktau_map < utau)

            comp_mask = np.logical_and(thick_mask,
                                       tau_mask)

            cts = astro_hist(diff_bic.value[comp_mask],
                             bins=bin_edges)[0]

            ax.semilogy(bin_centres, cts, drawstyle='steps-mid',
                        label=r'{0} $<\,\tau_0\,<$ {1}'.format(int(ltau), int(utau)),
                        linestyle=lstyle)

            if fig_type > 1:
                ax2.plot(bin_centres, np.cumsum(cts) / cts_all.sum(),
                         drawstyle='steps-mid',
                         label=r'{0} $<\,\tau_0\,<$ {1}'.format(int(ltau), int(utau)),
                         linestyle=lstyle)

            if fig_type == 3:
                comp_mask = np.logical_and(thick_mask[spat_slice_zoom],
                                           tau_mask[spat_slice_zoom])

                ax3.contour(comp_mask,
                            linestyles=[lstyle],
                            levels=[0.5],
                            colors=[col])

        if fig_type > 1:
            leg = ax2.legend(loc='upper left')
            ax2.grid()
        else:
            leg = ax.legend(loc='upper left')

        ax.grid()

        save_figure(fig, f'm31_delta_bic_hist_tau_steps_novcentlim{file_suffixes[fig_type]}')
        plt.close()


    # Make a box plot of the peak tau vs. number of Gaussians

    # Remove 0
    ngausses = np.unique(m31_ngauss[thick_mask])[1:]

    data = [peaktau_map[thick_mask & (m31_ngauss == nc)] for nc in ngausses]

    fig = plt.figure()

    ax = fig.add_subplot()

    _ = ax.boxplot(data)

    ax.set_ylabel(r"$\tau_0$")
    ax.set_xlabel("Number of Gaussians")

    save_figure(fig, 'm31_tau_ngauss_boxplot_novcentlim')
    plt.close()


# M33
if run_m33:

    m33_multigauss_name = fourteenB_HI_data_wGBT_path("individ_multigaussian_gausspy_fits.fits")
    m33_multigauss_hdu = fits.open(m33_multigauss_name)

    m33_ngauss = np.isfinite(m33_multigauss_hdu[0].data).sum(0) // 3

    m33_thickHI_name = fourteenB_HI_data_wGBT_path("individ_simplethick_HI_fits_80kms_centlimit.fits")
    m33_thickHI_hdu = fits.open(m33_thickHI_name)

    m33_multigauss_bic_proj = Projection.from_hdu(m33_multigauss_hdu[2])
    m33_thickHI_bic_proj = Projection.from_hdu(m33_thickHI_hdu[2])

    # Slice out to zoom into the valid data region.
    # spat_slice_zoom = tuple([slice(vals.min() - 10, vals.max() + 10) for vals in
    #                          np.where(np.isfinite(m33_multigauss_hdu[2].data))])
    spat_slice_zoom = (slice(None), slice(None))


    twocolumn_figure()

    diff_bic = m33_multigauss_bic_proj - m33_thickHI_bic_proj
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

    save_figure(fig, 'm33_delta_bic_map_novcentlim')
    plt.close()

    # Show a collapsed histogram of delta BIC
    # We'll loop over three types of plots:
    # 1. Histogram
    # 2. Histogram and CDF
    # 3. 2. plus a map with contours from histogram splits

    # Offset by none so I don't need an n-1 index
    file_suffixes = [None, "", "_cdf", "_cdf_map"]

    for fig_type in [1, 2, 3]:

        fig = plt.figure()
        if fig_type == 1:
            ax = fig.add_subplot()
        elif fig_type == 2:
            ax = fig.add_subplot(211)
            ax2 = fig.add_subplot(212, sharex=ax)
        else:
            ax = fig.add_subplot(221)
            ax2 = fig.add_subplot(223, sharex=ax)
            ax3 = fig.add_subplot(122, projection=diff_bic_zoom.wcs)

        cts_all, bin_edges = astro_hist(diff_bic.value[np.isfinite(diff_bic)],
                                        bins='knuth')
        bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.

        ax.semilogy(bin_centres, cts_all, drawstyle='steps-mid', label='All',
                    linewidth=2, color='k')

        if fig_type == 1:
            ax.set_xlabel(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")

        ax.set_ylabel("Number of Spectra")

        if fig_type > 1:
            ax2.plot(bin_centres, np.cumsum(cts_all) / cts_all.sum(),
                     drawstyle='steps-mid', label='All',
                     linewidth=2, color='k')
            ax2.set_xlabel(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")
            ax2.set_ylabel("CDF")

            ax2.axvline(0., color='k', linestyle='--')

        ax.axvline(0., color='k', linestyle='--')

        ax.text(-80, 1e5, "Gaussian model\npreferred",
                verticalalignment='center',
                horizontalalignment='right')
        ax.text(80, 1e5, "Optically-thick\nmodel preferred",
                verticalalignment='center',
                horizontalalignment='left')

        # Make delta BIC image
        if fig_type == 3:
            im = ax3.imshow(diff_bic_zoom.value, vmin=vmin, vmax=vmax,
                            cmap=plt.cm.Oranges_r)

            cbar = plt.colorbar(im)
            cbar.set_label(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")

        # Now split by number of gaussians
        # Max is 8. We'll do each number up to 5, then >5

        for nc, lstyle in zip(range(1, 5), linestyles_list):

            col = sb.color_palette()[nc - 1]

            comp_mask = np.logical_and(np.isfinite(diff_bic),
                                       m33_ngauss == nc)

            cts = astro_hist(diff_bic.value[comp_mask],
                             bins=bin_edges)[0]

            ax.semilogy(bin_centres, cts, drawstyle='steps-mid',
                        label=f'{nc} Gaussians' if nc > 1 else "1 Gaussian",
                        linestyle=lstyle)

            if fig_type > 1:
                ax2.plot(bin_centres, np.cumsum(cts) / cts_all.sum(),
                         drawstyle='steps-mid',
                         label=f'{nc} Gaussians' if nc > 1 else "1 Gaussian",
                         linestyle=lstyle)

            if fig_type == 3:
                comp_mask = np.logical_and(np.isfinite(diff_bic_zoom),
                                           m33_ngauss[spat_slice_zoom] == nc)

                ax3.contour(comp_mask,
                            linestyles=[lstyle],
                            levels=[0.5],
                            colors=[col])

        # "the rest"
        comp_mask = np.logical_and(np.isfinite(diff_bic),
                                   m33_ngauss >= 5)

        cts = astro_hist(diff_bic.value[comp_mask],
                         bins=bin_edges)[0]

        ax.semilogy(bin_centres, cts, drawstyle='steps-mid',
                    label=r'$\geq$5 Gaussians',
                    linestyle=linestyles_list[5])

        if fig_type > 1:
            ax2.plot(bin_centres, np.cumsum(cts) / cts_all.sum(),
                     label=r'$\geq$5 Gaussians',
                     drawstyle='steps-mid',
                     linestyle=linestyles_list[5])

        if fig_type == 3:
            comp_mask = np.logical_and(np.isfinite(diff_bic_zoom),
                                       m33_ngauss[spat_slice_zoom] >= 5)

            col = sb.color_palette()[4]

            ax3.contour(comp_mask,
                        linestyles=[linestyles_list[5]],
                        levels=[0.5],
                        colors=[col])

        if fig_type > 1:
            leg = ax2.legend(loc='upper left')
            ax2.grid()
        else:
            leg = ax.legend(loc='upper left')

        ax.grid()

        save_figure(fig, f'm33_delta_bic_hist_all_novcentlim{file_suffixes[fig_type]}')
        plt.close()

        # Histogram but limited to where the spin temperature is constrained.
        constTs_mask = m33_thickHI_hdu[0].data[0] / m33_thickHI_hdu[1].data[0] > 1.
        constTs_mask = np.logical_and(constTs_mask,
                                      np.isfinite(m33_thickHI_hdu[1].data[0]))
        constTs_mask = np.logical_and(constTs_mask,
                                      np.isfinite(diff_bic))

        fig = plt.figure()
        if fig_type == 1:
            ax = fig.add_subplot()
        elif fig_type == 2:
            ax = fig.add_subplot(211)
            ax2 = fig.add_subplot(212, sharex=ax)
        else:
            ax = fig.add_subplot(221)
            ax2 = fig.add_subplot(223, sharex=ax)
            ax3 = fig.add_subplot(122, projection=diff_bic_zoom.wcs)

        ax.set_ylabel("Number of Spectra")

        cts_all, bin_edges = astro_hist(diff_bic.value[constTs_mask],
                                        bins='knuth')
        bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.

        ax.semilogy(bin_centres, cts_all, drawstyle='steps-mid', label='All',
                    linewidth=2, color='k')

        if fig_type == 1:
            ax.set_xlabel(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")

        ax.set_ylabel("Number of Spectra")

        if fig_type > 1:
            ax2.plot(bin_centres, np.cumsum(cts_all) / cts_all.sum(),
                     drawstyle='steps-mid', label='All',
                     linewidth=2, color='k')
            ax2.set_xlabel(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")
            ax2.set_ylabel("CDF")

            ax2.axvline(0., color='k', linestyle='--')

        ax.axvline(0., color='k', linestyle='--')

        ax.text(-80, 2e4, "Gaussian model\npreferred",
                verticalalignment='center',
                horizontalalignment='right')
        ax.text(80, 2e4, "Optically-thick\nmodel preferred",
                verticalalignment='center',
                horizontalalignment='left')

        if fig_type == 3:
            im = ax3.imshow(diff_bic_zoom.value, vmin=vmin, vmax=vmax,
                            cmap=plt.cm.Oranges_r)

            cbar = plt.colorbar(im)
            cbar.set_label(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")

        # Now split by number of gaussians
        # Max is 8. We'll do each number up to 5, then >5

        for nc, lstyle in zip(range(1, 5), linestyles_list):

            col = sb.color_palette()[nc - 1]

            comp_mask = np.logical_and(constTs_mask,
                                       m33_ngauss == nc)

            cts = astro_hist(diff_bic.value[comp_mask],
                             bins=bin_edges)[0]

            ax.semilogy(bin_centres, cts, drawstyle='steps-mid',
                        label=f'{nc} Gaussians' if nc > 1 else "1 Gaussian",
                        linestyle=lstyle)

            if fig_type > 1:
                ax2.plot(bin_centres, np.cumsum(cts) / cts_all.sum(),
                         drawstyle='steps-mid',
                         label=f'{nc} Gaussians' if nc > 1 else "1 Gaussian",
                         linestyle=lstyle)

            if fig_type == 3:
                comp_mask = np.logical_and(constTs_mask[spat_slice_zoom],
                                           m33_ngauss[spat_slice_zoom] == nc)

                ax3.contour(comp_mask,
                            linestyles=[lstyle],
                            levels=[0.5],
                            colors=[col])

        # "the rest"
        comp_mask = np.logical_and(constTs_mask,
                                   m33_ngauss >= 5)

        cts = astro_hist(diff_bic.value[comp_mask],
                         bins=bin_edges)[0]

        ax.semilogy(bin_centres, cts, drawstyle='steps-mid',
                    label=r'$\geq$5 Gaussians',
                    linestyle=linestyles_list[5])

        if fig_type > 1:
            ax2.plot(bin_centres, np.cumsum(cts) / cts_all.sum(),
                     label=r'$\geq$5 Gaussians',
                     drawstyle='steps-mid',
                     linestyle=linestyles_list[5])

        if fig_type == 3:
            comp_mask = np.logical_and(constTs_mask[spat_slice_zoom],
                                       m33_ngauss[spat_slice_zoom] >= 5)

            col = sb.color_palette()[4]

            ax3.contour(comp_mask,
                        linestyles=[linestyles_list[5]],
                        levels=[0.5],
                        colors=[col])

        if fig_type > 1:
            leg = ax2.legend(loc='upper left')
            ax2.grid()
        else:
            leg = ax.legend(loc='upper left')

        ax.grid()

        save_figure(fig, f'm33_delta_bic_hist_constrained_Ts_novcentlim{file_suffixes[fig_type]}')
        plt.close()


        # Histogram of LOS where the thick HI peak opt. depth is >0.5
        thick_mask = m33_thickHI_hdu[0].data[2] / m33_thickHI_hdu[0].data[0] > 0.5
        thick_mask = np.logical_and(thick_mask,
                                    np.isfinite(m33_thickHI_hdu[1].data[0]))
        thick_mask = np.logical_and(thick_mask,
                                    np.isfinite(diff_bic))

        fig = plt.figure()
        if fig_type == 1:
            ax = fig.add_subplot()
        elif fig_type == 2:
            ax = fig.add_subplot(211)
            ax2 = fig.add_subplot(212, sharex=ax)
        else:
            ax = fig.add_subplot(221)
            ax2 = fig.add_subplot(223, sharex=ax)
            ax3 = fig.add_subplot(122, projection=diff_bic_zoom.wcs)

        cts_all, bin_edges = astro_hist(diff_bic.value[thick_mask],
                                        bins='knuth')
        bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.

        ax.semilogy(bin_centres, cts_all, drawstyle='steps-mid', label='All',
                    linewidth=2, color='k')

        if fig_type == 1:
            ax.set_xlabel(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")

        ax.set_ylabel("Number of Spectra")

        if fig_type > 1:
            ax2.plot(bin_centres, np.cumsum(cts_all) / cts_all.sum(),
                     drawstyle='steps-mid', label='All',
                     linewidth=2, color='k')
            ax2.set_xlabel(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")
            ax2.set_ylabel("CDF")

            ax2.axvline(0., color='k', linestyle='--')

        ax.axvline(0., color='k', linestyle='--')

        ax.text(-80, 2e4, "Gaussian model\npreferred",
                verticalalignment='center',
                horizontalalignment='right')
        ax.text(80, 2e4, "Optically-thick\nmodel preferred",
                verticalalignment='center',
                horizontalalignment='left')

        if fig_type == 3:
            im = ax3.imshow(diff_bic_zoom.value, vmin=vmin, vmax=vmax,
                            cmap=plt.cm.Oranges_r)

            cbar = plt.colorbar(im)
            cbar.set_label(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")

        # Now split by number of gaussians
        # Max is 8. We'll do each number up to 5, then >5

        for nc, lstyle in zip(range(1, 5), linestyles_list):

            col = sb.color_palette()[nc - 1]

            comp_mask = np.logical_and(thick_mask,
                                       m33_ngauss == nc)

            cts = astro_hist(diff_bic.value[comp_mask],
                             bins=bin_edges)[0]

            ax.semilogy(bin_centres, cts, drawstyle='steps-mid',
                        label=f'{nc} Gaussians' if nc > 1 else "1 Gaussian",
                        linestyle=lstyle)

            if fig_type > 1:
                ax2.plot(bin_centres, np.cumsum(cts) / cts_all.sum(),
                         drawstyle='steps-mid',
                         label=f'{nc} Gaussians' if nc > 1 else "1 Gaussian",
                         linestyle=lstyle)

            if fig_type == 3:
                comp_mask = np.logical_and(thick_mask[spat_slice_zoom],
                                           m33_ngauss[spat_slice_zoom] == nc)

                ax3.contour(comp_mask,
                            linestyles=[lstyle],
                            levels=[0.5],
                            colors=[col])

        # "the rest"
        comp_mask = np.logical_and(thick_mask,
                                   m33_ngauss >= 5)

        cts = astro_hist(diff_bic.value[comp_mask],
                         bins=bin_edges)[0]

        ax.semilogy(bin_centres, cts, drawstyle='steps-mid',
                    label=r'$\geq$5 Gaussians',
                    linestyle=linestyles_list[5])

        if fig_type > 1:
            ax2.plot(bin_centres, np.cumsum(cts) / cts_all.sum(),
                     label=r'$\geq$5 Gaussians',
                     drawstyle='steps-mid',
                     linestyle=linestyles_list[5])

        if fig_type == 3:
            comp_mask = np.logical_and(thick_mask[spat_slice_zoom],
                                       m33_ngauss[spat_slice_zoom] >= 5)

            col = sb.color_palette()[4]

            ax3.contour(comp_mask,
                        linestyles=[linestyles_list[5]],
                        levels=[0.5],
                        colors=[col])

        if fig_type > 1:
            leg = ax2.legend(loc='upper left')
            ax2.grid()
        else:
            leg = ax.legend(loc='upper left')

        ax.grid()
        save_figure(fig, f'm33_delta_bic_hist_tau_gt_0p5_novcentlim{file_suffixes[fig_type]}')
        plt.close()


        # Histogram binned with peak tau
        peaktau_map = m33_thickHI_hdu[0].data[2] / m33_thickHI_hdu[0].data[0]
        thick_mask = peaktau_map > 0.5
        thick_mask = np.logical_and(thick_mask,
                                    np.isfinite(m33_thickHI_hdu[1].data[0]))
        thick_mask = np.logical_and(thick_mask,
                                    np.isfinite(diff_bic))

        fig = plt.figure()
        if fig_type == 1:
            ax = fig.add_subplot()
        elif fig_type == 2:
            ax = fig.add_subplot(211)
            ax2 = fig.add_subplot(212, sharex=ax)
        else:
            ax = fig.add_subplot(221)
            ax2 = fig.add_subplot(223, sharex=ax)
            ax3 = fig.add_subplot(122, projection=diff_bic_zoom.wcs)

        cts_all, bin_edges = astro_hist(diff_bic.value[thick_mask],
                                        bins='knuth')
        bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.

        ax.semilogy(bin_centres, cts_all, drawstyle='steps-mid', label='All',
                    linewidth=2, color='k')

        if fig_type == 1:
            ax.set_xlabel(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")

        ax.set_ylabel("Number of Spectra")

        if fig_type > 1:
            ax2.plot(bin_centres, np.cumsum(cts_all) / cts_all.sum(),
                     drawstyle='steps-mid', label='All',
                     linewidth=2, color='k')
            ax2.set_xlabel(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")
            ax2.set_ylabel("CDF")

            ax2.axvline(0., color='k', linestyle='--')

        ax.axvline(0., color='k', linestyle='--')

        # ax.axvline(-10., color='k', linestyle='--')
        # ax.axvline(10., color='k', linestyle='--')

        ax.text(-80, 2e4, "Gaussian model\npreferred",
                verticalalignment='center',
                horizontalalignment='right')
        ax.text(80, 2e4, "Optically-thick\nmodel preferred",
                verticalalignment='center',
                horizontalalignment='left')

        if fig_type == 3:
            im = ax3.imshow(diff_bic_zoom.value, vmin=vmin, vmax=vmax,
                            cmap=plt.cm.Oranges_r)

            cbar = plt.colorbar(im)
            cbar.set_label(r"BIC$_{\rm Gauss}$ - BIC$_{\rm Thick}$")

        # Now split by number of gaussians
        # Max is 8. We'll do each number up to 5, then >5

        for ltau, utau, lstyle in zip([0.5, 1, 2, 3, 4], [1, 2, 3, 4, 5],
                                      linestyles_list):

            col = sb.color_palette()[utau - 1]

            tau_mask = np.logical_and(peaktau_map >= ltau,
                                      peaktau_map < utau)

            comp_mask = np.logical_and(thick_mask,
                                       tau_mask)

            cts = astro_hist(diff_bic.value[comp_mask],
                             bins=bin_edges)[0]

            ax.semilogy(bin_centres, cts, drawstyle='steps-mid',
                        label=r'{0} $<\,\tau_0\,<$ {1}'.format(int(ltau), int(utau)),
                        linestyle=lstyle)

            if fig_type > 1:
                ax2.plot(bin_centres, np.cumsum(cts) / cts_all.sum(),
                         drawstyle='steps-mid',
                         label=r'{0} $<\,\tau_0\,<$ {1}'.format(int(ltau), int(utau)),
                         linestyle=lstyle)

            if fig_type == 3:
                comp_mask = np.logical_and(thick_mask[spat_slice_zoom],
                                           tau_mask[spat_slice_zoom])

                ax3.contour(comp_mask,
                            linestyles=[lstyle],
                            levels=[0.5],
                            colors=[col])

        if fig_type > 1:
            leg = ax2.legend(loc='upper left')
            ax2.grid()
        else:
            leg = ax.legend(loc='upper left')

        ax.grid()

        save_figure(fig, f'm33_delta_bic_hist_tau_steps_novcentlim{file_suffixes[fig_type]}')
        plt.close()


    # Make a box plot of the peak tau vs. number of Gaussians

    # Remove 0
    ngausses = np.unique(m33_ngauss[thick_mask])[1:]

    data = [peaktau_map[thick_mask & (m33_ngauss == nc)] for nc in ngausses]

    fig = plt.figure()

    ax = fig.add_subplot()

    _ = ax.boxplot(data)

    ax.set_ylabel(r"$\tau_0$")
    ax.set_xlabel("Number of Gaussians")

    save_figure(fig, 'm33_tau_ngauss_boxplot_novcentlim')
    plt.close()

