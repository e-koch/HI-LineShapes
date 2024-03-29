import os
import socket
from functools import partial
import glob

'''
Common set of paths giving the location of data products.
'''


def name_return_check(filename, path, no_check=False):
    full_path = os.path.join(path, filename)

    if not os.path.exists(full_path) and not no_check:
        raise OSError("{} does not exist.".format(full_path))

    return full_path


if socket.gethostname() == 'ewk':
    root = os.path.expanduser('~/ownCloud/project_code/ThickHIFitting/')
    m31_data_path = "/home/eric/cirrus/bigdata/ekoch/M31/"
    m33_data_path = "/home/eric/cirrus/bigdata/ekoch/M31/"
elif "segfault" == socket.gethostname():
    root = os.path.expanduser("~/ownCloud/project_code/ThickHIFitting/")
    m31_data_path = "/mnt/bigdata/ekoch/M31"
    m33_data_path = "/mnt/bigdata/ekoch/M33"
# elif "cedar.computecanada" in socket.gethostname():
#     root = "/home/ekoch/code/LocalGroup-VLA/"
#     m31_data_path = "/home/ekoch/project/ekoch/"
elif 'ewk-laptop' in socket.gethostname():
    root = os.path.expanduser("~/ownCloud/project_code/ThickHIFitting/")
    m31_data_path = os.path.expanduser("~/storage/M31")
    m33_data_path = os.path.expanduser("~/storage/M33")
elif 'SI38' in socket.gethostname():
    root = os.path.expanduser("~/ownCloud/project_code/ThickHIFitting/")
    m31_data_path = os.path.expanduser("~/storage/M31")
    m33_data_path = os.path.expanduser("~/storage/M33")


# M31 data paths
fourteenA_HI_data_path = \
    partial(name_return_check,
            path=os.path.join(m31_data_path, "VLA/14A-235/HI/full_imaging_noSD/"))
fourteenA_HI_data_wEBHIS_path = \
    partial(name_return_check,
            path=os.path.join(m31_data_path, "VLA/14A-235/HI/full_imaging_wEBHIS/"))

fifteenA_HI_BC_1_2kms_data_path = \
    partial(name_return_check,
            path=os.path.join(m31_data_path, "VLA/15A-175/HI/full_imaging_1_2kms_noSD/"))
fifteenA_HI_BC_1_2kms_data_wEBHIS_path = \
    partial(name_return_check,
            path=os.path.join(m31_data_path, "VLA/15A-175/HI/full_imaging_1_2kms_wEBHIS/"))

fifteenA_HI_BCtaper_04kms_data_path = \
    partial(name_return_check,
            path=os.path.join(m31_data_path, "VLA/15A-175/HI/full_imaging_BCD_taper_0_42kms_noSD/"))
fifteenA_HI_BCtaper_04kms_data_wEBHIS_path = \
    partial(name_return_check,
            path=os.path.join(m31_data_path, "VLA/15A-175/HI/full_imaging_BCD_taper_0_42kms_wEBHIS/"))


ebhis_m31_HI_data_path = \
    partial(name_return_check,
            path=os.path.join(m31_data_path, "EBHIS/"))

# M33 data paths
fourteenB_HI_data_path = \
    partial(name_return_check,
            path=os.path.join(m33_data_path, "VLA/14B-088/HI/full_imaging_noSD/"))
fourteenB_HI_data_wGBT_path = \
    partial(name_return_check,
            path=os.path.join(m33_data_path, "VLA/14B-088/HI/full_imaging_wGBT/"))


seventeenB_HI_data_1kms_path = \
    partial(name_return_check,
            path=os.path.join(m33_data_path,
                              "VLA/17B-162/HI/full_imaging_1kms_noSD/"))
seventeenB_HI_data_1kms_wGBT_path = \
    partial(name_return_check,
            path=os.path.join(m33_data_path,
                              "VLA/17B-162/HI/full_imaging_1kms_wGBT/"))


# Proposal Figures
varfig_path = os.path.expanduser("~/ownCloud/Various Plots/Proposals")
proposal_figures_path = lambda x: os.path.join(varfig_path, x)

# All figures
fig_path = os.path.expanduser("~/ownCloud/Various Plots/M31/")
allfigs_path = lambda x: os.path.join(fig_path, x)
alltables_path = lambda x: os.path.join(fig_path, "tables", x)


def find_dataproduct_names(path):
    '''
    Given a path, return a dictionary of the data products with the name
    convention used in this repository.
    '''

    search_dict = {"Moment0": "mom0.",
                   "Moment0_err": "mom0_err",
                   "Moment1": "mom1.",
                   "LWidth": "lwidth",
                   "Skewness": "skewness",
                   "Kurtosis": "kurtosis",
                   "PeakTemp": "peaktemps",
                   "PeakVels": "peakvels.",
                   "Source_Mask": "_source_mask.fits",
                   # "Source_Mask": "masked_source_mask.fits",
                   "CentSub_Cube": "masked.centroid_corrected",
                   "CentSub_Mask": "masked_source_mask.centroid_corrected",
                   "RotSub_Cube": "masked.rotation_corrected",
                   "RotSub_Mask": "masked_source_mask.rotation_corrected",
                   "PeakSub_Cube": "masked.peakvels_corrected",
                   "PeakSub_Mask": "masked_source_mask.peakvels_corrected",
                   "Flux_Spec": "total_flux_spec.fits",
                   "Cube": ".image.",
                   "PB": ".pb."}
                   # "Cube": ".masked.fits",}

    found_dict = {}

    for filename in glob.glob(os.path.join(path, "*.fits")):

        for key in search_dict:
            if search_dict[key] in filename:
                found_dict[key] = filename
                search_dict.pop(key)
                break

    return found_dict


# Return dictionaries with names for the existing directories
fourteenA_HI_file_dict = \
    find_dataproduct_names(fourteenA_HI_data_path("", no_check=True))
fourteenA_wEBHIS_HI_file_dict = \
    find_dataproduct_names(fourteenA_HI_data_wEBHIS_path("", no_check=True))

fifteenA_HI_file_dict = \
  find_dataproduct_names(fifteenA_HI_BC_1_2kms_data_path("", no_check=True))
fifteenA_HI_wEBHIS_HI_file_dict = \
  find_dataproduct_names(fifteenA_HI_BC_1_2kms_data_wEBHIS_path("", no_check=True))

fifteenA_HI_BCtaper_file_dict = \
  find_dataproduct_names(fifteenA_HI_BCtaper_04kms_data_path("", no_check=True))
fifteenA_HI_BCtaper_wEBHIS_HI_file_dict = \
  find_dataproduct_names(fifteenA_HI_BCtaper_04kms_data_wEBHIS_path("", no_check=True))

fourteenB_HI_file_dict = \
    find_dataproduct_names(fourteenB_HI_data_path("", no_check=True))
fourteenB_wGBT_HI_file_dict = \
    find_dataproduct_names(fourteenB_HI_data_wGBT_path("", no_check=True))

seventeenB_1kms_HI_file_dict = \
    find_dataproduct_names(seventeenB_HI_data_1kms_path("", no_check=True))
seventeenB_1kms_wGBT_HI_file_dict = \
    find_dataproduct_names(seventeenB_HI_data_1kms_wGBT_path("",
                                                             no_check=True))

if __name__ == "__main__":

    # Append the repo directory to the path so paths is importable
    os.sys.path.append(root)
