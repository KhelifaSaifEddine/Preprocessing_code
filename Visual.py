from PIL import Image
import SimpleITK as sitk

import os
import imageio
import timeit
import datetime
import operator
import numpy as np
import pandas as pd
from numpy.linalg import inv, det, norm
from math import sqrt, pi
from functools import partial
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pydicom
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.ioff()

import ipywidgets as ipyw

class ImageSliceViewer3D:
    #Copied from https://github.com/mohakpatel/ImageSliceViewer3D
    """ 
    ImageSliceViewer3D is for viewing volumetric image slices in jupyter or
    ipython notebooks. 
    
    User can interactively change the slice plane selection for the image and 
    the slice plane being viewed. 
    Argumentss:
    Volume = 3D input image (3D numpy array)
    figsize = default(8,8), to set the size of the figure
    cmap = default('plasma'), string for the matplotlib colormap. You can find 
    more matplotlib colormaps on the following link:
    https://matplotlib.org/users/colormaps.html
    
    """
    
    def __init__(self, volume, figsize=(8,8)):
        self.volume = volume
        self.figsize = figsize
        self.cmap = plt.cm.gray #alternative: cmap='plasma'
        self.v = [np.min(volume), np.max(volume)]
        
        # Call to select slice plane
        ipyw.interact(self.view_selection, view=ipyw.RadioButtons(
                        options=['x-y','y-z', 'z-x'], value='x-y', 
            description='Slice plane selection:', disabled=False,
            style={'description_width': 'initial'}))
    
    def view_selection(self, view):
        # Transpose the volume to orient according to the slice plane selection
        orient = {"y-z":[1,2,0], "z-x":[2,0,1], "x-y": [0,1,2]}
        self.vol = np.transpose(self.volume, orient[view])
        maxZ = self.vol.shape[2] - 1
        
        # Call to view a slice within the selected slice plane
        ipyw.interact(self.plot_slice, 
            z=ipyw.IntSlider(min=0, max=maxZ, step=1, continuous_update=False, 
            description='Image Slice:'))
        
    def plot_slice(self, z):
        # Plot slice for the given plane and slice
        self.fig = plt.figure(figsize=self.figsize)
        plt.imshow(self.vol[:,:,z], cmap=plt.get_cmap(self.cmap), 
            vmin=self.v[0], vmax=self.v[1])

def plot_hu_histogram(ctvol, outprefix):
    #Histogram of HUs https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    plt.hist(ctvol.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.savefig(outprefix+'_HU_Hist.png')
    plt.close()

def plot_middle_slice(ctvol, outprefix):
    # Show some slice in the middle
    plt.imshow(ctvol[100,:,:], cmap=plt.cm.gray)
    plt.savefig(outprefix+'_Middle_Slice.png')
    plt.close()

def plot_3d_skeleton(ctvol, units, outprefix):
    """Make a 3D plot of the skeleton.
    <units> either 'HU' or 'processed' (normalized) determines the thresholds"""
    #from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    #Get high threshold to show mostly bones
    if units == 'HU':
        threshold = 400
    elif units == 'processed':
        threshold = 0.99
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = ctvol.transpose(2,1,0)
    p = np.flip(p, axis = 0) #need this line or else the patient is upside-down
    
    #https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.marching_cubes_lewiner
    verts, faces, _ignore1, _ignore2 = measure.marching_cubes_lewiner(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    
    plt.savefig(outprefix+'_3D_Bones.png')
    plt.close()    

def make_gifs(ctvol, outprefix, chosen_views):
    """Save GIFs of the <ctvol> in the axial, sagittal, and coronal planes.
    This assumes the final orientation produced by the preprocess_volumes.py
    script: [slices, square, square].
    
    <chosen_views> is a list of strings that can contain any or all of
        ['axial','coronal','sagittal']. It specifies which view(s) will be
        made into gifs."""
    #https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
    
    #First fix the grayscale colors.
    #imageio assumes only 256 colors (uint8): https://stackoverflow.com/questions/41084883/imageio-how-to-increase-quality-of-output-gifs
    #If you do not truncate to a 256 range, imageio will do so on a per-slice
    #basis, which creates weird brightening and darkening artefacts in the gif.
    #Thus, before making the gif, you should truncate to the 0-256 range
    #and cast to a uint8 (the dtype imageio requires):
    #how to truncate to 0-256 range: https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
    print(ctvol.shape)
    Visualise_how_much(10, ctvol[50:]) 


    ctvol = np.clip(ctvol, a_min=-0, a_max=1)
    ctvol = (  ((ctvol+0)*(255))/(1+0)  ).astype('uint8')
    
    #Now create the gifs in each plane
    if 'axial' in chosen_views:
        images = []
        for slicenum in range(ctvol.shape[2]):
            images.append(ctvol[slicenum,:,:])
        imageio.mimsave(outprefix+'_axial.gif',images)
        print('\t\tdone with axial gif')
    
    if 'coronal' in chosen_views:
        images = []
        for slicenum in range(ctvol.shape[1]):
            images.append(ctvol[:,slicenum,:])
        imageio.mimsave(outprefix+'_coronal.gif',images)
        print('\t\tdone with coronal gif')
    
    if 'sagittal' in chosen_views:
        images = []
        for slicenum in range(ctvol.shape[2]):
            images.append(ctvol[:,:,slicenum])
        imageio.mimsave(outprefix+'_sagittal.gif',images)
        print('\t\tdone with sagittal gif')
    

def Visualise_how_much(len, slices):
    fig = plt.figure()
    for num,each_slice in enumerate(slices[:len]):
        y = fig.add_subplot(3,4,num+1)
        #new_img = cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE))
        y.imshow(slices[num] , cmap='gray')
    plt.show()
def Visualise_how_much_inversed(len, slices,max_size):
    fig = plt.figure()
    for num,each_slice in enumerate(slices[:len]):
        y = fig.add_subplot(3,4,num+1)
        #new_img = cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE))
        print(max_size - num)
        y.imshow(slices[max_size - num] , cmap='gray')
    plt.show()

def show_alone(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    image = np.stack([s.pixel_array for s in slices])
    Visualise_how_much_inversed(12,image ,image.shape[0]-1) 