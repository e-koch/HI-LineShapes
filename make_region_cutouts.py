
from spectral_cube import SpectralCube, Projection
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import os
import astropy.units as u

from paths import (fourteenA_HI_data_wEBHIS_path,
                   fifteenA_HI_BCtaper_04kms_data_wEBHIS_path,
                   fifteenA_HI_BC_1_2kms_data_wEBHIS_path)


# Grab all triplets across a region highlighted in Figures 9,10 of
# Braun+2009


def find_closest_pixel(coord, spatial_coords):

    min_posn = coord.separation(spatial_coords).argmin()

    return np.unravel_index(min_posn, spatial_coords.shape)


def make_cube_cutout(cube, coords, pix_pad_size=20):
    '''
    Given a set of sky coordinates, find the subcube that includes all
    in a square region.
    '''

    spatcoord_quantity = cube.spatial_coordinate_map
    spatial_coords = SkyCoord(spatcoord_quantity[1], spatcoord_quantity[0])

    pixel_positions = []
    for coord in coords:
        pixel_positions.append(find_closest_pixel(coord, spatial_coords))

    pixel_positions = np.array(pixel_positions)

    cutout_shape = np.ptp(pixel_positions, axis=0)

    # Set the shape to the maximum to make a square. Then append on the pad size,
    # if needed.
    cutout_shape = cutout_shape.max() + pix_pad_size

    assert cutout_shape > 0

    centre_position = tuple(np.round(pixel_positions.mean(0), 0).astype(int))

    slicer = (slice(None),
              slice(centre_position[0] - cutout_shape // 2,
                    centre_position[0] + cutout_shape // 2 + 1),
              slice(centre_position[1] - cutout_shape // 2,
                    centre_position[1] + cutout_shape // 2 + 1))

    return cube[slicer], pixel_positions, centre_position

tab = Table.read("braun09_coords.csv")

tab = tab[tab['fig'] < 11]

coords = SkyCoord(tab['ra'], tab['dec'])

cube_14A = SpectralCube.read(fourteenA_HI_data_wEBHIS_path("M31_14A_HI_contsub_width_04kms.image.pbcor.EBHIS_feathered_K.fits"))

# Subcube out directory
out_path = fourteenA_HI_data_wEBHIS_path("braun09_subcubes", no_check=True)
if not os.path.exists(out_path):
    os.mkdir(out_path)

for i in range(6):

    coord_set = coords[3 * i:3 * i + 3]

    subcube_i = make_cube_cutout(cube_14A, coord_set, pix_pad_size=20)[0]

    # Write out the subcubes into their own directory
    subcube_name = f"M31_14A_HI_contsub_width_04kms.image.pbcor.EBHIS_feathered_K_braun09_subcube_{i}.fits"

    subcube_i.write(os.path.join(out_path, subcube_name), overwrite=True)

# Cutout some subcubes for the higher-res cube. But only where we have data.

tab = Table.read("braun09_coords_in15A_mosaic.csv")

tab = tab[tab['fig'] < 11]

coords = SkyCoord(tab['ra'], tab['dec'])

cube_15A_taper = SpectralCube.read(fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("M31_15A_B_C_14A_HI_contsub_width_0_4kms.image.pbcor.EBHIS_feathered.fits"))


out_path = fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("braun09_subcubes",
                                                      no_check=True)
if not os.path.exists(out_path):
    os.mkdir(out_path)

for i in range(2):

    coord_set = coords[3 * i:3 * i + 3]

    subcube_i = make_cube_cutout(cube_15A_taper, coord_set, pix_pad_size=20)[0]

    # I haven't made a K version of the full cube yet
    subcube_i = subcube_i.to(u.K)

    # Write out the subcubes into their own directory
    subcube_name = f"M31_15A_B_C_14A_HI_contsub_width_0_4kms.image.pbcor.EBHIS_feathered_K_braun09_subcube_{i}.fits"

    subcube_i.write(os.path.join(out_path, subcube_name), overwrite=True)


cube_15A = SpectralCube.read(fifteenA_HI_BC_1_2kms_data_wEBHIS_path("M31_15A_B_C_14A_HI_contsub_width_1_2kms.image.pbcor.EBHIS_feathered_K.fits"))


out_path = fifteenA_HI_BC_1_2kms_data_wEBHIS_path("braun09_subcubes",
                                                  no_check=True)
if not os.path.exists(out_path):
    os.mkdir(out_path)

for i in range(2):

    coord_set = coords[3 * i:3 * i + 3]

    subcube_i = make_cube_cutout(cube_15A, coord_set, pix_pad_size=20)[0]

    # The last few channels have some weird artifacts. Just remove those since
    # there's no signal there anymore.
    subcube_i = subcube_i[:240]

    # Write out the subcubes into their own directory
    subcube_name = f"M31_15A_B_C_14A_HI_contsub_width_1_2kms.image.pbcor.EBHIS_feathered_K_braun09_subcube_{i}.fits"

    subcube_i.write(os.path.join(out_path, subcube_name), overwrite=True)
