import Visual
import os
import pandas
import numpy as np
import re
import pydicom ,dicom
import glob
import os
import copy
import timeit
import pickle
import pydicom
import datetime
import numpy as np
import pandas as pd
import SimpleITK as sitk
from statistics import mode
import glob
import dicom
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from skimage import morphology
import zipfile , os
#combine_slices.py
#modified from: https://github.com/innolitics/dicom-numpy
import logging
import numpy as np

logger = logging.getLogger(__name__)

class DicomImportException(Exception):
    pass

def isclose(a, b, rel_tol=1e-9, abs_tol=0.0):
    '''
    This function is implemented in Python 3.
    To support Python 2, we include our own implementation.
    '''
    return abs(a-b) <= max(rel_tol*max(abs(a), abs(b)), abs_tol)

def combine_slices_func(slice_datasets, rescale=None):
    '''
    Given a list of pydicom datasets for an image series, stitch them together into a
    three-dimensional numpy array.  Also calculate a 4x4 affine transformation
    matrix that converts the ijk-pixel-indices into the xyz-coordinates in the
    DICOM patient's coordinate system.
    Returns a two-tuple containing the 3D-ndarray and the affine matrix.
    If `rescale` is set to `None` (the default), then the image array dtype
    will be preserved, unless any of the DICOM images contain either the
    `Rescale Slope
    <https://dicom.innolitics.com/ciods/ct-image/ct-image/00281053>`_ or the
    `Rescale Intercept <https://dicom.innolitics.com/ciods/ct-image/ct-image/00281052>`_
    attributes.  If either of these attributes are present, they will be
    applied to each slice individually.
    If `rescale` is `True` the voxels will be cast to `float32`, if set to
    `False`, the original dtype will be preserved even if DICOM rescaling information is present.
    The returned array has the column-major byte-order.
    This function requires that the datasets:
    - Be in same series (have the same
      `Series Instance UID <https://dicom.innolitics.com/ciods/ct-image/general-series/0020000e>`_,
      `Modality <https://dicom.innolitics.com/ciods/ct-image/general-series/00080060>`_,
      and `SOP Class UID <https://dicom.innolitics.com/ciods/ct-image/sop-common/00080016>`_).
    - The binary storage of each slice must be the same (have the same
      `Bits Allocated <https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280100>`_,
      `Bits Stored <https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280101>`_,
      `High Bit <https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280102>`_, and
      `Pixel Representation <https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280103>`_).
    - The image slice must approximately form a grid. This means there can not
      be any missing internal slices (missing slices on the ends of the dataset
      are not detected).
    - It also means that  each slice must have the same
      `Rows <https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280010>`_,
      `Columns <https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280011>`_,
      `Pixel Spacing <https://dicom.innolitics.com/ciods/ct-image/image-plane/00280030>`_, and
      `Image Orientation (Patient) <https://dicom.innolitics.com/ciods/ct-image/image-plane/00200037>`_
      attribute values.
    - The direction cosines derived from the
      `Image Orientation (Patient) <https://dicom.innolitics.com/ciods/ct-image/image-plane/00200037>`_
      attribute must, within 1e-4, have a magnitude of 1.  The cosines must
      also be approximately perpendicular (their dot-product must be within
      1e-4 of 0).  Warnings are displayed if any of these approximations are
      below 1e-8, however, since we have seen real datasets with values up to
      1e-4, we let them pass.
    - The `Image Position (Patient) <https://dicom.innolitics.com/ciods/ct-image/image-plane/00200032>`_
      values must approximately form a line.
    If any of these conditions are not met, a `dicom_numpy.DicomImportException` is raised.
    '''
    if len(slice_datasets) == 0:
        raise DicomImportException("Must provide at least one DICOM dataset")

    _validate_slices_form_uniform_grid(slice_datasets)

    voxels, slice_positions = _merge_slice_pixel_arrays(slice_datasets, rescale)
    transform = _ijk_to_patient_xyz_transform_matrix(slice_datasets)
    
    return voxels, slice_positions, transform


def _merge_slice_pixel_arrays(slice_datasets, rescale=None):
    first_dataset = slice_datasets[0]
    num_rows = first_dataset.Rows
    num_columns = first_dataset.Columns
    num_slices = len(slice_datasets)
    
    pos_and_data = _sort_by_slice_z_positions(slice_datasets)
    sorted_slice_datasets = [data for (pos, data) in pos_and_data]
    slice_positions = [pos for (pos, data) in pos_and_data]

    if rescale is None:
        rescale = any(_requires_rescaling(d) for d in sorted_slice_datasets)

    if rescale:
        voxels = np.empty((num_columns, num_rows, num_slices), dtype=np.float32, order='F')
        for k, dataset in enumerate(sorted_slice_datasets):
            slope = float(getattr(dataset, 'RescaleSlope', 1))
            intercept = float(getattr(dataset, 'RescaleIntercept', 0))
            voxels[:, :, k] = dataset.pixel_array.T.astype(np.float32) * slope + intercept
    else:
        dtype = first_dataset.pixel_array.dtype
        voxels = np.empty((num_columns, num_rows, num_slices), dtype=dtype, order='F')
        for k, dataset in enumerate(sorted_slice_datasets):
            voxels[:, :, k] = dataset.pixel_array.T

    return voxels, slice_positions


def _requires_rescaling(dataset):
    return hasattr(dataset, 'RescaleSlope') or hasattr(dataset, 'RescaleIntercept')


def _ijk_to_patient_xyz_transform_matrix(slice_datasets):
    pos_and_data = _sort_by_slice_z_positions(slice_datasets)
    first_dataset = [data for (pos, data) in pos_and_data][0]
    
    image_orientation = first_dataset.ImageOrientationPatient
    row_cosine, column_cosine, slice_cosine = _extract_cosines(image_orientation)

    row_spacing, column_spacing = first_dataset.PixelSpacing
    slice_spacing = _slice_spacing(slice_datasets)

    transform = np.identity(4, dtype=np.float32)

    transform[:3, 0] = row_cosine * column_spacing
    transform[:3, 1] = column_cosine * row_spacing
    transform[:3, 2] = slice_cosine * slice_spacing

    transform[:3, 3] = first_dataset.ImagePositionPatient

    return transform


def _validate_slices_form_uniform_grid(slice_datasets):
    '''
    Perform various data checks to ensure that the list of slices form a
    evenly-spaced grid of data.
    Some of these checks are probably not required if the data follows the
    DICOM specification, however it seems pertinent to check anyway.
    '''
    invariant_properties = [
        'Modality',
        'SOPClassUID',
        'SeriesInstanceUID',
        'Rows',
        'Columns',
        'PixelSpacing',
        'PixelRepresentation',
        'BitsAllocated',
        'BitsStored',
        'HighBit',
    ]

    for property_name in invariant_properties:
        _slice_attribute_equal(slice_datasets, property_name)

    _validate_image_orientation(slice_datasets[0].ImageOrientationPatient)
    _slice_ndarray_attribute_almost_equal(slice_datasets, 'ImageOrientationPatient', 1e-5)

    slice_positions = _slice_positions(slice_datasets)
    _check_for_missing_slices(slice_positions)


def _validate_image_orientation(image_orientation):
    '''
    Ensure that the image orientation is supported
    - The direction cosines have magnitudes of 1 (just in case)
    - The direction cosines are perpendicular
    '''
    row_cosine, column_cosine, slice_cosine = _extract_cosines(image_orientation)

    if not _almost_zero(np.dot(row_cosine, column_cosine), 1e-4):
        raise DicomImportException("Non-orthogonal direction cosines: {}, {}".format(row_cosine, column_cosine))
    elif not _almost_zero(np.dot(row_cosine, column_cosine), 1e-8):
        logger.warning("Direction cosines aren't quite orthogonal: {}, {}".format(row_cosine, column_cosine))

    if not _almost_one(np.linalg.norm(row_cosine), 1e-4):
        raise DicomImportException("The row direction cosine's magnitude is not 1: {}".format(row_cosine))
    elif not _almost_one(np.linalg.norm(row_cosine), 1e-8):
        logger.warning("The row direction cosine's magnitude is not quite 1: {}".format(row_cosine))

    if not _almost_one(np.linalg.norm(column_cosine), 1e-4):
        raise DicomImportException("The column direction cosine's magnitude is not 1: {}".format(column_cosine))
    elif not _almost_one(np.linalg.norm(column_cosine), 1e-8):
        logger.warning("The column direction cosine's magnitude is not quite 1: {}".format(column_cosine))


def _almost_zero(value, abs_tol):
    return isclose(value, 0.0, abs_tol=abs_tol)


def _almost_one(value, abs_tol):
    return isclose(value, 1.0, abs_tol=abs_tol)


def _extract_cosines(image_orientation):
    row_cosine = np.array(image_orientation[:3])
    column_cosine = np.array(image_orientation[3:])
    slice_cosine = np.cross(row_cosine, column_cosine)
    return row_cosine, column_cosine, slice_cosine


def _slice_attribute_equal(slice_datasets, property_name):
    initial_value = getattr(slice_datasets[0], property_name, None)
    for dataset in slice_datasets[1:]:
        #print(dataset)
        value = getattr(dataset, property_name, None)
        if value != initial_value:
            msg = 'All slices must have the same value for "{}": {} != {}'
            raise DicomImportException(msg.format(property_name, value, initial_value))


def _slice_ndarray_attribute_almost_equal(slice_datasets, property_name, abs_tol):
    print(slice_datasets[0].ImageOrientationPatient)
    initial_value = getattr(slice_datasets[0], property_name, None)
    print(initial_value)
    for dataset in slice_datasets[1:]:
        print(len(slice_datasets))
        value = getattr(dataset, property_name, None)
        print(value)
        if not np.allclose(value, initial_value, atol=abs_tol):
            msg = 'All slices must have the same value for "{}" within "{}": {} != {}'
            raise DicomImportException(msg.format(property_name, abs_tol, value, initial_value))


def _slice_positions(slice_datasets):
    print("*************************************")
    print(slice_datasets[0])
    image_orientation = slice_datasets[0].ImageOrientationPatient
    row_cosine, column_cosine, slice_cosine = _extract_cosines(image_orientation)
    return [np.dot(slice_cosine, d.ImagePositionPatient) for d in slice_datasets]


def _check_for_missing_slices(slice_positions):
    slice_positions_diffs = np.diff(sorted(slice_positions))
    if not np.allclose(slice_positions_diffs, slice_positions_diffs[0], atol=0, rtol=1e-1):
        print('\tWarning: combine_slices detects that there may be missing slices')
        
        
def _slice_spacing(slice_datasets):
    if len(slice_datasets) > 1:
        slice_positions = _slice_positions(slice_datasets)
        slice_positions_diffs = np.diff(sorted(slice_positions))
        return np.mean(slice_positions_diffs)
    else:
        return 0.0


def _sort_by_slice_z_positions(slice_datasets):
    """Example:
    >>> slice_spacing = [1.1,7,2.2,100]
    >>> slice_datasets=['a','b','c','d']
    >>> sorted(zip(slice_spacing, slice_datasets), reverse=True)
    [(100, 'd'), (7, 'b'), (2.2, 'c'), (1.1, 'a')]
    >>> [d for (s, d) in sorted(zip(slice_spacing, slice_datasets), reverse=True)]
    ['d', 'b', 'c', 'a']
    """
    slice_positions = _slice_positions(slice_datasets)
    return sorted(zip(slice_positions, slice_datasets), reverse=True)

#dicom_numpy code is from https://github.com/innolitics/dicom-numpy/blob/master/dicom_numpy/combine_slices.py
#downloaded on September 19, 2019
#see dicom_numpy/LICENSE.txt for the dicom_numpy license 
class CleanCTScans(object):
    def __init__(self, mode,folder_file, photo_path,patient,save_npz_folder,augmentation =False,list_augmentation= []):
        #Folder_path : where dicom's are saved 
        self.mode = mode
        self.Rprob = []
        assert self.mode == 'testing' or self.mode == 'run'
        self.logdir = os.path.join(photo_path,datetime.datetime.today().strftime('%Y-%m-%d')+patient + '_volume_preprocessing')
        if not os.path.isdir(self.logdir):
            os.mkdir(self.logdir)
        self.set_up_logdf()
        self.patient_name = patient
        self.augmentation = augmentation
        self.directory = "" + self.patient_name
        self.list_augmentation = list_augmentation
        self.save_npz_folder = save_npz_folder
        self.x = self.run(folder_file)
     
    def set_up_logdf(self):
        #there is no preexisting log file; initialize from scratch
        column_names = ['status','status_reason','full_filename_pkl','full_filename_npz',
                        'orig_square','orig_numslices','orig_slope',
                        'orig_inter','orig_yxspacing',
                        'orig_zpositions_all','orig_zdiff_all','orig_zdiff_set','orig_zdiff','zdiffs_all_equal',
                        'orig_orientation','orig_gantry_tilt','final_square',
                        'final_numslices','final_spacing','transform']
        self.logdf = pd.DataFrame([["","","","","","","","","","","","","","","","","","","",""]],columns = column_names)
        for colname in ['status','status_reason','full_filename_pkl','full_filename_npz',
                        'orig_square','orig_numslices','orig_slope',
                        'orig_inter','orig_yxspacing',
                        'orig_zpositions_all','orig_zdiff_all','orig_zdiff_set','orig_zdiff','zdiffs_all_equal',
                        'orig_orientation','orig_gantry_tilt','final_square',
                        'final_numslices','final_spacing','transform']:
            self.logdf[colname]=''

        #Ensure correct dtypes
        for colname in ['transform','orig_zpositions_all','orig_zdiff_all','orig_zdiff_set']:
            self.logdf[colname]=self.logdf[colname].astype('object')
    def run(self,fpath):
        #--------------------------------------------------------------------------------------------------------#
        #Here we need  in raw variable: 
                                        #Load volume. Format: python list. Each element of the list is a
                                        #pydicom.dataset.FileDataset that contains metadata as well as pixel
                                        #data. Each pydicom.dataset.FileDataset corresponds to one slice.
        #--------------------------------------------------------------------------------------------------------#
        print("......Gathering dicom's for you.......")

        raw = [(pydicom.dcmread(file)) for file in fpath]

        #slices = [(pydicom.dcmread( file), file) for file in fpath]
        ctvol = self.process_ctvol(raw)
        if "contrast" in str.lower(raw[0].ProtocolName):
             np.savez_compressed(self.save_npz_folder + "/" + self.patient_name,ct=ctvol)
        else:
          #if ctvol != None:
             np.savez_compressed(self.save_npz_folder  + "/" + self.patient_name,ct=ctvol)
        return raw
    def process_ctvol(self, raw):
        """Read in the pickled CT volume and return the CT volume as a numpy
        array. Save to the log file important characteristics. Includes sanity checks."""
      
        
        #Extract information from all the slices (function
        #includes for loop over slices) and save to self.logdf
        if self.mode=='testing': print('running extract_info()')
        self.logdf = CleanCTScans.extract_info(self,raw, self.logdf)
        
        #Create the volume by stacking the slices in the right order:
        if self.mode=='testing': print('running create_volume()')
        ctvol, self.logdf = CleanCTScans.create_volume(raw, self.logdf)
        RescaleSlope = self.logdf ['orig_slope'].values
        RescaleIntercept = self.logdf ['orig_inter'].values
        

        print("resampling")
        print(ctvol.shape)
        #Resample
        if self.mode=='testing': print('running resample_volume()')
        orig_yxspacing = self.logdf['orig_yxspacing']
        orig_zspacing = self.logdf['orig_zdiff']
        #z first, then square (xy) for the sitk resampling function:
        #verify if we need to augment then augment then verify if its the augmented list#
        if self.augmentation:
          if self.patient_name in self.list_augmentation:
            ctvol = ctvol
          else:
            return None
        original_spacing = [float(x) for x in [orig_zspacing, orig_yxspacing, orig_yxspacing]] 
        ctvol, self.logdf = CleanCTScans.resample_volume(ctvol, original_spacing, self.logdf, self.mode)

        #Represent the volume more efficiently by casting to float16 and
        #clipping the Hounsfield Units:
       
        ctvol = CleanCTScans.represent_volume_efficiently_and_transpose(ctvol)
        for i in range(len(ctvol)):
          ctvol[i] = windowing_size_crop_noise(ctvol[i],RescaleSlope,RescaleIntercept)
     
        ctvol = minmax_normalization(ctvol)
        if self.augmentation:
          temp = np.load(self.directory)
          size = temp['ct'].shape[0]
          size1 = ctvol.shape[0]
          if size < size1 :
            diffrence =size1 - size
            ctvol = ctvol[diffrence:]
          print("augementation {}".format(ctvol.shape))
        else:
          print("removing blanks")
          ctvol = remove_blanks(ctvol)
  
        if self.mode == 'testing': self.visualize(ctvol,'_final')
 
        
        #Return the final volume. ctvol is a 3D numpy array with float16 elements
        return ctvol
    @staticmethod
    def extract_info(self,raw, logdf):
        """Process the CT data contained in <raw> and save properties of the
        data in <logdf> 
        Variables:
        <raw> is a list of pydicom.dataset.FileDataset objects, one for
            each slice."""
        #Initialize empty lists for collecting info on each slice
        #These values should all be identical for each slice but we want to
        #double check that, because TRUST NOTHING
        gathered_slopes = []
        gathered_inters = []
        gathered_spacing = []
        gathered_orientation = []
        #I believe the gantry tilt should always be zero for chest CTs
        #but it makes sense to save it and record it just in case.
        gathered_gantry_tilt = []
        RP = False
        for index in range(len(raw)):
            oneslice = raw[index]
         
            #Gather the scaling slope and intercept: https://blog.kitware.com/dicom-rescale-intercept-rescale-slope-and-itk/
            scl_slope = oneslice.data_element('RescaleSlope').value #example: "1"
            gathered_slopes.append(scl_slope)
            scl_inter = oneslice.data_element('RescaleIntercept').value #example: "-1024"
            gathered_inters.append(scl_inter)
            #print(type(oneslice.data_element('RescaleIntercept').value))
            if oneslice.data_element('RescaleIntercept').value != 0 :
              #assert oneslice.data_element('RescaleType').value == 'HU', 'Error: RescaleType not equal to HU'
                RP = False
            else:
                RP = True
            #Gather the spacing: example: ['0.585938', '0.585938']; what we save is 0.585938
            #for DICOM, first vlaue is row spacing (vertical spacing) and
            #the second value is column spacing (horizontal spacing)
            yxspacing = oneslice.data_element('PixelSpacing').value
            assert float(yxspacing[0])==float(yxspacing[1]), 'Error: non-square pixels: yxspacing[0] not equal to yxspacing[1]' #i.e. verify that they are square pixels
            gathered_spacing.append(yxspacing[0])
            
            #Check the orientation
            #example: ['1.000000', '0.000000', '0.000000', '0.000000', '1.000000', '0.000000']
            orient = [float(x) for x in oneslice.data_element('ImageOrientationPatient').value]
            assert orient==[1.0,0.0,0.0,0.0,1.0,0.0], 'Error: nonstandard ImageOrientationPatient'
            
            #Save the gantry tilt. example: "0.000000"
            gathered_gantry_tilt.append(oneslice.data_element('GantryDetectorTilt').value)
        
        #Make sure the values for all the slices are the same
        #Slopes and intercepts:
        #assert len(set(gathered_slopes)) == 1, 'Error: more than one slope'
        #assert len(set(gathered_inters)) == 1, 'Error: more than one intercept'
        logdf['orig_slope'] = list(set(gathered_slopes))[0]
        logdf['orig_inter'] = list(set(gathered_inters))[0]
        
        #yxspacing
        assert len(set(gathered_spacing)) == 1, 'Error: more than one yxspacing'
        logdf['orig_yxspacing'] = list(set(gathered_spacing))[0]
        
        #orientations
        logdf['orig_orientation'] = '1,0,0,0,1,0'       
        
        #gantry tilt
        assert len(set(gathered_gantry_tilt))==1, 'Error: more than one gantry tilt'
        gtilt = list(set(gathered_gantry_tilt))[0]
        if float(gtilt)!=0: print('gantry tilt nonzero:',gtilt)
        logdf['orig_gantry_tilt'] = gtilt
        if RP:
          self.Rprob.append(self.patient_name)
        return logdf
    @staticmethod
    def create_volume(raw, logdf):
        """Concatenate the slices in the correct order and return a 3D numpy
        array. Also rescale using the slope and intercept."""
        #According to this website https://itk.org/pipermail/insight-users/2003-September/004762.html
        #the only reliable way to order slices is to use ImageOrientationPatient
        #and ImagePositionPatient. You can't even trust Image Number because
        #for some scanners it doesn't work.
        #Just for reference, you access ImagePositionPatient like this:
        #positionpatient = oneslice.data_element('ImagePositionPatient').value
        #and the values are e.g. ['-172.100', '-177.800', '-38.940'] (strings
        #of floats).
        #We can't use this for ordering the volume:
        #slice_number = int(oneslice.data_element('InstanceNumber').value)
        #because on some CT scanners the InstanceNumber is unreliable.
        #We will use a concatenation implementation from combine_slices.py:
        #ctvol has shape [num_columns, num_rows, num_slices] i.e. square, square, slices
        
        #Fields in the logdf that we fill out:
        #'orig_zpositions_all': a list of the raw z position values in order
        #'orig_zdiff_all': a list of the z diffs in order
        #'orig_zdiff_set': a set of the unique z diff values. Should have 1 member
        #'orig_zdiff': the final zdiff value. If 'orig_zdiff_set' has one
        #    member then that is the final zdiff value. If 'orig_zdiff_set'
        #    has more than one member then the final zdiff value is the mode.
        #'zdiffs_all_equal': True if the orig_zdiff_set has only one member.
        #    False otherwise.
        ctvol, slice_positions, transform = combine_slices_func(raw,rescale=True)
        assert slice_positions == sorted(slice_positions,reverse=True), 'Error: combine_slices did not sort slice_positions correctly'
        logdf['orig_zpositions_all'] = pd.Series([round(x,4) for x in slice_positions])
        
        #figure out the z spacing by taking the difference in the z positions
        #of adjacent slices. Z spacing should be consistent throughout the
        #entire volume:
        zdiffs = [abs(round(x,4)) for x in np.ediff1d(slice_positions)] #round so you don't get floating point arithmetic problems. abs because distances can't be negative.
        logdf['orig_zdiff_all'] = pd.Series(zdiffs)
        logdf['orig_zdiff_set'] = pd.Series(list(set(zdiffs)))
        if len(list(set(zdiffs))) == 1:
            logdf['orig_zdiff'] = pd.Series(list(set(zdiffs))[0])
            logdf['zdiffs_all_equal'] = True
        else:
            #choose the zdiff value as the mode, not as the min.
            #if you choose the min you will get warped resamplings sometimes.
            #you care about what is most frequently the zdiff;
            #it's usually around 0.625
            logdf['orig_zdiff'] = pd.Series(mode(list(zdiffs)))
            logdf['zdiffs_all_equal'] = False
        
        #save other characteristics:
        assert ctvol.shape[0] == ctvol.shape[1], 'Error: non-square axial slices'
        logdf['orig_square'] = ctvol.shape[0]
        logdf['orig_numslices'] = ctvol.shape[2]
        logdf['transform'] = [transform]
        return ctvol, logdf
    @staticmethod
    def resample_volume(ctvol, original_spacing, logdf, mode):
        """Resample the numpy array <ctvol> to [0.8,0.8,0.8] spacing and return.
        There are a lot of internal checks in this function to make sure
        all the dimensions are right, because:
        - converting a numpy array to a sitk image permutes the axes
        - we need to be sure that the original_spacing z axis is in the same
          place as the sitk image z axis
        - when we convert back to a numpy array at the end, we need to be sure
          that the z axis is once again in the place it used to be in the
          original input numpy array.
        
        If <mode>=='testing' then print more output.
        Modified from https://medium.com/tensorflow/an-introduction-to-biomedical-image-analysis-with-tensorflow-and-dltk-2c25304e7c13
        """
        if mode=='testing': print('ctvol before resampling',ctvol.shape) #e.g. [512, 512, 518]
        assert ctvol.shape[0]==ctvol.shape[1], 'Error in resample_volume: non-square axial slices in input ctvol'
        ctvol_itk = sitk.GetImageFromArray(ctvol)
        ctvol_itk.SetSpacing(original_spacing)
        original_size = ctvol_itk.GetSize()
        if mode=='testing': print('ctvol original shape after sitk conversion:',original_size) #e.g. [518, 512, 512]
        if mode=='testing': print('ctvol original spacing:',original_spacing) #e.g. [0.6, 0.732421875, 0.732421875]
        
        #Double check that the square positions (x and y) are in slots 1 and 2
        #for both the original size and the original spacing:
        #(which means that the z axis, or slices, is in position 0)
        assert original_size[1]==original_size[2], 'Error in resample_volume: non-square axial slices in the original_size'
        assert original_spacing[1]==original_spacing[2], 'Error in resample_volume: non-square pixels in the original_spacing'
        
        #Calculate out shape:
        out_spacing=[1,1,1]
        #Relationship: (origshape x origspacing) = (outshape x outspacing)
        #in other words, we want to be sure we are still representing
        #the same real-world lengths in each direction.
        out_shape = [int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
            int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
            int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
        if mode=='testing': print('desired out shape:',out_shape) #e.g. [388, 469, 469]
        
        #Perform resampling:
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(out_shape)
        resample.SetOutputDirection(ctvol_itk.GetDirection())
        resample.SetOutputOrigin(ctvol_itk.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(ctvol_itk.GetPixelIDValue())
        resample.SetInterpolator(sitk.sitkBSpline)
        resampled_ctvol = resample.Execute(ctvol_itk)
        if mode=='testing': print('actual out shape in sitk:',resampled_ctvol.GetSize()) #e.g. [388, 469, 469]
        assert [x for x in resampled_ctvol.GetSize()]==out_shape, 'Error in resample_volume: incorrect sitk resampling shape obtained' #make sure we got the shape we wanted
        assert out_shape[1]==out_shape[2], 'Error in resample_volume: non-square sitk axial slices after resampling ' #make sure square is in slots 1 and 2
        
        #Get numpy array. Note that in the transformation from a Simple ITK
        #image to a numpy array, the axes are permuted (1,2,0) so that
        #we have the z axis (slices) at the end again (in position 2)
        #In other words the z axis gets moved from position 0 (in sitk)
        #to position 2 (in numpy)
        final_result = sitk.GetArrayFromImage(resampled_ctvol)
        if mode=='testing': print('actual out shape in numpy:',final_result.shape) #e.g. [469, 469, 388]
        assert [x for x in final_result.shape] == [out_shape[1], out_shape[2], out_shape[0]], 'Error in resample_volume: incorrect numpy resampling shape obtained'
        #we're back to having the z axis (slices) in position 2 just like
        #they were in the original ctvol: [square, square, slices]
        assert final_result.shape[0]==final_result.shape[1], 'Error in resample_volume: non-square numpy axial slices after resampling'
        
        #Update the logdf
        logdf['final_square'] = final_result.shape[0]
        logdf['final_numslices'] = final_result.shape[2]
        logdf['final_spacing']=1
        np.swapaxes(final_result,0,2)

        return final_result, logdf
    
    def visualize(self, ctvol, pathtosave):
        """Visualize a CT volume"""
        outprefix = os.path.join(self.logdir,pathtosave)
        
        Visual.plot_hu_histogram(ctvol, outprefix)
        print('finished plotting histogram')
        
        #Uncomment the next lines to plot the 3D skeleton (this step is slow)
        #visualize_volumes.plot_3d_skeleton(ctvol,'HU',outprefix)
        #print('finished plotting 3d skeleton')
        
        gifpath = os.path.join(self.logdir,'gifs')
        if not os.path.exists(gifpath):
            os.mkdir(gifpath)
        print(gifpath)
        Visual.make_gifs(ctvol,gifpath,chosen_views=['axial'])
        print('finished making gifs')
        
        np.save(outprefix+'.npy',ctvol)
        print('finished saving npy file')
    @staticmethod
    def represent_volume_efficiently_and_transpose(ctvol):
        """Clip the Hounsfield units and cast from float32 to int16 to
        dramatically reduce the amount of space it will take to store a
        compressed version of the <ctvol>. Also transpose.
        
        We don't care about the Hounsfield units less than -1000 (the HU of air)
        and we don't care about values higher than +1000 (bone).
        Quote, "The CT Hounsfield scale places water density at a value of
        zero with air and bone at opposite extreme values of -1000HU
        and +1000HU."
        From https://www.sciencedirect.com/topics/medicine-and-dentistry/hounsfield-scale
        By clipping the values we dramatically reduce the size of the
        compressed ctvol.
        
        It is a waste of precious space to save the CT volumes as float32
        (the default type.) Even float16 results in a final dataset size too big
        to fit in the 3.5 TB hard drive and HUs are supposed to be ints anyway
        (we only get floats due to resampling step.)
        Thus represent pixel values using int16."""
        ctvol = np.transpose(ctvol, axes=[2,1,0]) #so we get slices, square, square
        #Round and cast to integer:
        ctvol_int = np.rint(ctvol) #round each element to the nearest integer
        ctvol_int = ctvol_int.astype(np.int16) #cast to int16 data type
        assert np.isfinite(ctvol_int).all(), 'Error: infinite values created when casting to integer' #check that no np.infs created in rounding/casting
        if not np.amax(np.absolute(ctvol - ctvol_int)) < 1:
          print('Error: difference from original is too great when casting to integer')
          return ctvol
           #check that no element is off by more than 1 HU from the original
        assert isinstance(ctvol_int[0,0,0],np.int16), 'Error: casting to int16 failed'
        return ctvol_int

def Normalization_zeromean(image):
    """Whitening. Normalises image to zero mean and unit variance."""

    mean = np.mean(image)
    std = np.std(image)

    if std > 0:
        ret = (image - mean) / std
    else:
        ret = image * 0.
    return ret

def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    
    return window_image
def remove_noise(image,RescaleSlope,RescaleIntercept, display=False):

 
    #print("hu",image.shape)
    brain_image = window_image(image, 40, 400)
    #print("BRAIN",brain_image.shape)
    # morphology.dilation creates a segmentation of the image
    # If one pixel is between the origin and the edge of a square of size
    # 5x5, the pixel belongs to the same class
    
    # We can instead use a circule using: morphology.disk(2)
    # In this case the pixel belongs to the same class if it's between the origin
    # and the radius
    
    segmentation = morphology.dilation(brain_image, np.ones((5, 5)))
    labels, label_nb = ndimage.label(segmentation)
    
    label_count = np.bincount(labels.ravel().astype(np.int))
    # The size of label_count is the number of classes/segmentations found
    
    # We don't use the first class since it's the background
    label_count[0] = 0
    
    # We create a mask with the class with more pixels
    # In this case should be the brain
    mask = labels == label_count.argmax()
    
    # Improve the brain mask
    mask = morphology.dilation(mask, np.ones((5, 5)))
    mask = ndimage.morphology.binary_fill_holes(mask)
    mask = morphology.dilation(mask, np.ones((3, 3)))
    
    # Since the the pixels in the mask are zero's and one's
    # We can multiple the original image to only keep the brain region
    masked_image = mask * brain_image

    if display:
        plt.figure(figsize=(50, 40))
        plt.subplot(143)
        plt.imshow(masked_image)
        plt.title('Final Image')
        plt.axis('off')
    
    return masked_image


def windowing_size_crop_noise(image,RescaleSlope,RescaleIntercept,display=False):
    final_image = remove_noise(image,RescaleSlope,RescaleIntercept, display=False)
    #print(final_image.shape)
    return final_image
def remove_blanks(x):  
  blanks = []
  check = 0
  for index,slice in enumerate(x):
    plt.imshow(slice, cmap = 'gray')
    plt.show()
    if (np.all(slice[0:250] <  -0.4)):
      blanks.append(index)
      print("its  blank")
    else:
      print("its not blank")
      if check > 0:
        break
      check += 1
  if len(blanks) > 0:
    return x[blanks[-1] + 2:]
  else:
    return x

def minmax_normalization(image):
        print("go")
        max = np.max(image)
        min = np.min(image)

        final_image = (image - min)/(max - min)

        return final_image


