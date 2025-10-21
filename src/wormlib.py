#!/usr/bin/env python
# coding: utf-8

# # WormLib: open source image analysis library for *C. elegans* 

# Standard library imports
import csv
import os
import time
import warnings
from datetime import datetime
import pathlib
from pathlib import Path
# Scientific computing
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, label, gaussian_filter, rotate, center_of_mass, zoom
from scipy.spatial import cKDTree

# Image processing
import cv2
import tifffile
import tifffile as tiff
from PIL import Image
from skimage import measure, morphology, filters
from skimage.draw import polygon2mask
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import square, dilation, erosion, disk, binary_erosion
from skimage.transform import resize
from skimage.util import img_as_float

# Machine learning
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Visualization
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
#from matplotlib.path import Path
from matplotlib.path import Path as MPLPath  # Explicit alias for matplotlib
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import seaborn as sns

# Specialized libraries
import bigfish
import bigfish.stack as stack
import bigfish.plot as plot
import bigfish.multistack as multistack
import bigfish.detection as detection
import cellpose
from cellpose import models, utils
import nd2

# PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle, SimpleDocTemplate

# Jupyter/IPython
from IPython.display import Image, display

# Multiprocessing
import multiprocessing as mp
import torch


# load models located in main directory 
# src_dir = next(parent / 'src' for parent in Path().absolute().parents if (parent / 'src').is_dir())
# main_dir = Path(src_dir.parents[0])
# models_dir = main_dir / 'models'


def create_cellpose_model(prefer_mps: bool = False):
    """
    Return a Cellpose model on a safe device.
    Default: CPU to avoid MPS sparse op crashes on macOS.
    Set prefer_mps=True to try MPS; auto-falls back to CPU.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if prefer_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            dev = torch.device("mps")
            model = models.Cellpose(gpu=True, device=dev)
            # tiny dry run to fail fast if MPS is missing ops
            import numpy as np
            _ = model.eval(np.zeros((2,2), np.float32), channels=[0,0], diameter=30)
            print("Cellpose using MPS")
            return model
        except Exception as e:
            print(f"⚠️ MPS failed ({e}); falling back to CPU.")

    print("Cellpose using CPU")
    return models.Cellpose(gpu=False, device=torch.device("cpu"))



def load_images(image_path, output_directory, channel_names, slice_to_plot=0):
    """
    Automatically detect image type (DV, ND2, or TIFF) and load images accordingly.
    """
    
    # Convert to Path object
    image_path = Path(image_path)
    
    # Determine if it's a file or directory
    if image_path.is_file():
        folder_path = image_path.parent
        list_filenames = [image_path.name]
    elif image_path.is_dir():
        folder_path = image_path
        list_filenames = sorted([
            f for f in os.listdir(folder_path)
            if not f.startswith('.') and not f.startswith('_')
        ])
    else:
        print(f"Path does not exist: {image_path}")
        return None
    
    if len(list_filenames) == 0:
        print("No image files found.")
        return None
    
    # Extract channel names (use .get() to avoid KeyError)
    Cy5 = channel_names.get('Cy5', None)
    mCherry = channel_names.get('mCherry', None)
    FITC = channel_names.get('FITC', None)
    DAPI = channel_names.get('DAPI', None)
    brightfield = channel_names.get('brightfield', None)
    
    # Detect image type
    dv_files = [f for f in list_filenames if f.endswith('.dv')]
    nd2_files = [f for f in list_filenames if f.endswith('.nd2')]
    tiff_files = [f for f in list_filenames if f.endswith(('.tif', '.tiff'))]
    
    if dv_files:
        image_type = 'dv'
        print("Detected DeltaVision (.dv) images")
    elif nd2_files:
        image_type = 'nd2'
        print("Detected Nikon (.nd2) images")
    elif tiff_files:
        image_type = 'tiff'
        print("Detected TIFF (.tif/.tiff) images")
    else:
        print("No supported image format found (.dv, .nd2, .tif, .tiff)")
        return None
    
    # Initialize return variables
    bf = None
    image_Cy5 = None
    image_mCherry = None
    image_FITC = None
    image_nuclei = None
    Cy5_array = None
    mCherry_array = None
    FITC_array = None
    nuclei_array = None
    grid_width = 80
    grid_height = 80
    
    # === DELTAVISION LOADING ===
    if image_type == 'dv':
        path_files = [os.path.join(folder_path, f) for f in dv_files]
        
        list_images = []
        for image_file in path_files:
            image_stack = stack.read_dv(image_file)
            list_images.append(image_stack)
        
        # Extract Image ID
        dv_filename = dv_files[0]
        if '_R3D_REF.dv' in dv_filename:
            image_name = dv_filename.replace("_R3D_REF.dv", "")
        elif '_R3D.dv' in dv_filename:
            image_name = dv_filename.replace("_R3D.dv", "")
        else:
            image_name = dv_filename.replace(".dv", "")
        
        print(f'Image ID: {image_name}\n')
        
        # Process the image stack
        image_stack = list_images[0].astype(np.uint16)
        print(f'Image shape: {image_stack.shape}')
        
        # === KEY FIX: Check dimensions FIRST, then assign based on what's requested ===
        if image_stack.ndim == 2:
            # 2D image - assign to brightfield ONLY if requested
            if brightfield is not None:
                bf = image_stack
                print("Loaded 2D brightfield image")
            
        elif image_stack.ndim == 4:
            # 4D stack [C, Z, Y, X] - assign color channels ONLY if requested
            image_colors = image_stack
            
            if Cy5 is not None and image_colors.shape[0] > 0:
                Cy5_array = image_colors[0, :, :, :]
                image_Cy5 = np.max(Cy5_array, axis=0)
            
            if mCherry is not None and image_colors.shape[0] > 1:
                mCherry_array = image_colors[1, :, :, :]
                image_mCherry = np.max(mCherry_array, axis=0)
            
            if FITC is not None and image_colors.shape[0] > 2:
                FITC_array = image_colors[2, :, :, :]
                image_FITC = np.max(FITC_array, axis=0)
            
            if DAPI is not None and image_colors.shape[0] > 3:
                nuclei_array = image_colors[3, :, :, :]
                image_nuclei = np.max(nuclei_array, axis=0)
            
            print("Loaded 4D color channel stack")
        
        grid_width, grid_height = 80, 80
    
    # === ND2 LOADING ===
    elif image_type == 'nd2':
        path_files = [os.path.join(folder_path, f) for f in nd2_files]
        
        list_images = []
        for image_file in path_files:
            image_stack = nd2.imread(image_file)
            list_images.append(image_stack)
        
        image_name = nd2_files[0].replace('.nd2', '')
        print(f"Image ID: {image_name}\n")
        
        image_colors = list_images[0]
        print(f"Image colors shape: {image_colors.shape}\n")
        
        # Assign channels [T, C, Y, X]
        if Cy5 is not None and image_colors.shape[1] > 0:
            Cy5_array = image_colors[:, 0, :, :]
            image_Cy5 = np.max(Cy5_array, axis=0)
        
        if mCherry is not None and image_colors.shape[1] > 1:
            mCherry_array = image_colors[:, 1, :, :]
            image_mCherry = np.max(mCherry_array, axis=0)
        
        if FITC is not None and image_colors.shape[1] > 2:
            FITC_array = image_colors[:, 2, :, :]
            image_FITC = np.max(FITC_array, axis=0)
        
        if DAPI is not None and image_colors.shape[1] > 3:
            nuclei_array = image_colors[:, 3, :, :]
            image_nuclei = np.max(nuclei_array, axis=0)
        
        if brightfield is not None and image_colors.shape[1] > 4:
            bf_stack = image_colors[:, 4, :, :]
            bf = np.max(bf_stack, axis=0)
        
        grid_width, grid_height = 80, 60
    
    # === TIFF LOADING ===
    else:
        path_files = [os.path.join(folder_path, f) for f in tiff_files]
        list_images = [tiff.imread(img) for img in path_files]
        
        image_stack = list_images[0]
        image_name = tiff_files[0].replace('.tif', '').replace('.tiff', '')
        print(f"Image ID: {image_name}")
        print(f"Image shape: {image_stack.shape}")
        
        plt.figure(figsize=(4, 4))
        if image_stack.ndim == 3:
            total_z = image_stack.shape[0]
            if slice_to_plot >= total_z:
                slice_to_plot = 0
            # create figure with size 4X4
            plt.imshow(image_stack[slice_to_plot], cmap='gray')
            plt.title(f"Slice {slice_to_plot} of {total_z}")
        else:
            plt.imshow(image_stack, cmap='gray')
            plt.title("2D Image")
        plt.axis('off')
        plt.show()
        
        if brightfield is not None:
            bf = image_stack
        
        grid_width, grid_height = 80, 80
    
    # === PLOT CHANNELS (skip for TIFF) ===
    if image_type != 'tiff':
        images = [image_Cy5, image_mCherry, image_FITC, image_nuclei, bf]
        titles = [Cy5, mCherry, FITC, DAPI, brightfield]
        filtered_images = [(img, title) for img, title in zip(images, titles) 
                          if img is not None and title is not None]
        if len(filtered_images) > 0:
            fig, ax = plt.subplots(1, len(filtered_images), figsize=(4 * len(filtered_images), 4))
            if len(filtered_images) == 1:
                ax = [ax]
            for i, (img, title) in enumerate(filtered_images):
                ax[i].imshow(img, cmap="gray")
                ax[i].set_title(title, size=20)
                ax[i].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(output_directory, f'channels_{image_name}.png'))
            plt.show()
            plt.close()
    
    return {
        'image_type': image_type,
        'image_name': image_name,
        'bf': bf,
        'image_Cy5': image_Cy5,
        'image_mCherry': image_mCherry,
        'image_FITC': image_FITC,
        'image_nuclei': image_nuclei,
        'Cy5_array': Cy5_array,
        'mCherry_array': mCherry_array,
        'FITC_array': FITC_array,
        'nuclei_array': nuclei_array,
        'grid_width': grid_width,
        'grid_height': grid_height
    }



# ## 3. Segmentation

# #### 3.1 Single cell segmentation (optimized for <4-cell embryos)

# In[7]:


#### 1.1 Single cell segmentation (up to 4-cell embryos)

# Additional functions used for segmentation
def is_nucleus_in_cytosol(mask_n, mask_c):
    mask_n[mask_n>1]=1
    mask_c[mask_c>1]=1
    size_mask_n = np.count_nonzero(mask_n)
    size_mask_c = np.count_nonzero(mask_c)
    min_size =np.min( (size_mask_n,size_mask_c) )
    mask_combined =  mask_n + mask_c
    sum_mask = np.count_nonzero(mask_combined[mask_combined==2])
    if (sum_mask> min_size*0.8) and (min_size>200): # the element is inside if the two masks overlap over the 80% of the smaller mask.
        return 1
    else:
        return 0
    
def remove_lonely_masks(masks_0, masks_1,is_nuc=None):
    n_mask_0 = np.max(masks_0)
    n_mask_1 = np.max(masks_1)
    if (n_mask_0>0) and (n_mask_1>0):
        for ind_0 in range(1,n_mask_0+1):
            tested_mask_0 = erosion(np.where(masks_0 == ind_0, 1, 0))
            array_paired= np.zeros(n_mask_1)
            for ind_1 in range(1,n_mask_1+1):
                tested_mask_1 = erosion(np.where(masks_1 == ind_1, 1, 0))
                array_paired[ind_1-1] = is_nucleus_in_cytosol(tested_mask_1, tested_mask_0)
                if (is_nuc =='nuc') and (np.count_nonzero(tested_mask_0) > np.count_nonzero(tested_mask_1) ):
                    # condition that rejects images with nucleus bigger than the cytosol
                    array_paired[ind_1-1] = 0
                elif (is_nuc is None ) and (np.count_nonzero(tested_mask_1) > np.count_nonzero(tested_mask_0) ):
                    array_paired[ind_1-1] = 0
            if any (array_paired) == False: # If the cytosol is not associated with any mask.
                masks_0 = np.where(masks_0 == ind_0, 0, masks_0)
            masks_with_pairs = masks_0
    else:
        masks_with_pairs = np.zeros_like(masks_0)
    return masks_with_pairs

def matching_masks(masks_cyto, masks_nuclei):
    n_mask_cyto = np.max(masks_cyto)
    n_mask_nuc = np.max(masks_nuclei)
    new_masks_nuclei = np.zeros_like(masks_cyto)
    reordered_mask_nuclei = np.zeros_like(masks_cyto)
    if (n_mask_cyto>0) and (n_mask_nuc>0):
        for mc in range(1,n_mask_cyto+1):
            tested_mask_cyto = np.where(masks_cyto == mc, 1, 0)
            for mn in range(1,n_mask_nuc+1):
                mask_paired = False
                tested_mask_nuc = np.where(masks_nuclei == mn, 1, 0)
                mask_paired = is_nucleus_in_cytosol(tested_mask_nuc, tested_mask_cyto)
                if mask_paired == True:
                    if np.count_nonzero(new_masks_nuclei) ==0:
                        new_masks_nuclei = np.where(masks_nuclei == mn, -mc, masks_nuclei)
                    else:
                        new_masks_nuclei = np.where(new_masks_nuclei == mn, -mc, new_masks_nuclei)
            reordered_mask_nuclei = np.absolute(new_masks_nuclei)
    return reordered_mask_nuclei

def remove_extreme_values(image,min_percentile=0.1, max_percentile=99.5):
    max_val = np.percentile(image, max_percentile)
    min_val = np.percentile(image, min_percentile)
    image [image < min_val] = min_val
    image [image > max_val] = max_val
    return image

def metric_max_cells_and_area(masks):
    n_masks = np.max(masks)
    if n_masks > 1: # detecting if more than 1 mask are detected per cell
        size_mask = []
        for nm in range (1, n_masks+1): # iterating for each mask in a given cell. The mask has values from 0 for background, to int n, where n is the number of detected masks.
            approximated_radius = np.sqrt(np.sum(masks == nm)/np.pi)  # a=  pi r2
            size_mask.append(approximated_radius) #np.sum(masks == nm)) # creating a list with the size of each mask
        size_masks_array = np.array(size_mask)
        metric = np.mean(size_masks_array).astype(int) * n_masks
    elif n_masks == 1: # do nothing if only a single mask is detected per image.
        approximated_radius = np.sqrt(np.sum(masks == 1)/np.pi) 
        metric = approximated_radius.astype(int)
    else:  # return zero if no mask are detected
        metric = 0  
    return metric   

def nuclear_segmentation(image_nuclei):
    MIN_CELL_SIZE = 1000
    list_masks_nuclei = []
    list_thresholds = np.arange(0.7,0.95, 0.05) # for nd2 images
    array_number_detected_masks = np.zeros(len(list_thresholds))
    for i,tested_ts in enumerate(list_thresholds):
        image_nuclei_binary = image_nuclei.copy()
        max_value_image = np.max(image_nuclei_binary)
        image_nuclei_binary[image_nuclei_binary < max_value_image*tested_ts] = 0
        image_nuclei_binary[image_nuclei_binary > max_value_image*tested_ts] = 1
        labels = measure.label(image_nuclei_binary)
        filtered_labels = morphology.remove_small_objects(labels, min_size=MIN_CELL_SIZE)
        unique_filtered_labels = np.unique(filtered_labels)
        tested_masks_nuclei = np.zeros_like(filtered_labels)
        for idx, old_label in enumerate(unique_filtered_labels):
            tested_masks_nuclei[filtered_labels == old_label] = idx
        list_masks_nuclei.append(tested_masks_nuclei)
        array_number_detected_masks[i]= metric_max_cells_and_area( tested_masks_nuclei) 
    selected_index = np.argmax(array_number_detected_masks)
    masks_nuclei = list_masks_nuclei [selected_index]
    return masks_nuclei

def cytosol_segmentation(image_cytosol,second_image_cytosol,cytosol_diameter):
    flow_ts=1
    MIN_CELL_SIZE = 1000 #100000 -> segmented the entire embryo
    model = models.Cellpose(model_type='cyto2') # model_type='cyto', 'cyto2' or model_type='nuclei'
    if not (second_image_cytosol is None):
        merged_image_cytosol = np.concatenate((image_cytosol[:, :, np.newaxis], second_image_cytosol[:, :, np.newaxis]), axis=2)
        masks_cytosol_unfiltered = model.eval(merged_image_cytosol, diameter=cytosol_diameter, flow_threshold=flow_ts, channels=[0,1])[0]
    else:
        masks_cytosol_unfiltered = model.eval(image_cytosol, diameter=cytosol_diameter, flow_threshold=flow_ts, channels=[0,0])[0]
    filtered_cyto = morphology.remove_small_objects(masks_cytosol_unfiltered, min_size=MIN_CELL_SIZE)
    unique_filtered_cyto = np.unique(filtered_cyto)
    masks_cytosol = np.zeros_like(filtered_cyto)
    for idx, old_label in enumerate(unique_filtered_cyto):
        masks_cytosol[filtered_cyto == old_label] = idx
    return masks_cytosol


def segmentation_optimization(image_cytosol,image_nuclei,cytosol_diameter,second_image_cytosol=None):
    # Cytosol segmentation
    masks_cytosol =cytosol_segmentation(image_cytosol,second_image_cytosol,cytosol_diameter)
    # Nuclear segmentation
    masks_nuclei = nuclear_segmentation(image_nuclei)
    # reordering nuclei masks
    masks_nuclei = matching_masks(masks_cytosol,masks_nuclei)
    # remove masks without nuclei
    masks_nuclei= remove_lonely_masks(masks_0=masks_nuclei , masks_1=masks_cytosol,is_nuc='nuc')
    masks_cytosol= remove_lonely_masks(masks_0=masks_cytosol , masks_1=masks_nuclei)
    # calculate size of masks
    number_masks_cyto = np.max(masks_cytosol)
    list_masks_cyto_sizes =[]
    for i in range (1, number_masks_cyto+1):
        list_masks_cyto_sizes.append(len(masks_cytosol[masks_cytosol==i]) )
    number_masks_nuc = np.max(masks_nuclei)
    list_masks_nuc_sizes =[]
    for i in range (1, number_masks_nuc+1):
        list_masks_nuc_sizes.append(len(masks_nuclei[masks_nuclei==i]) )
    return masks_nuclei, masks_cytosol,list_masks_nuc_sizes, list_masks_cyto_sizes


# Codes used to segment the nucleus and the cytosol
def segmentation(image_cytosol,image_nuclei, second_image_cytosol=None,output_directory= 'temp_output'):
    # removing outliers in image
    image_cytosol = remove_extreme_values(image=image_cytosol,min_percentile=0.1, max_percentile=99.5)
    if not (second_image_cytosol is None):
        second_image_cytosol = remove_extreme_values(image=second_image_cytosol,min_percentile=0.1, max_percentile=99.5)
    image_nuclei = remove_extreme_values(image=image_nuclei,min_percentile=0.1, max_percentile=99.5)
    # Optimization segmentation
    list_masks_nuclei = []
    list_masks_cytosol=[]
    list_masks_nuc_sizes =[]
    list_masks_cyto_sizes=[]
    list_flow_thresholds = np.arange(40, 200, 10)
    array_number_detected_masks = np.zeros(len(list_flow_thresholds))
    for i,tested_ts in enumerate(list_flow_thresholds):
        tested_masks_nuclei, tested_masks_cytosol, tested_list_masks_nuc_sizes, tested_list_masks_cyto_sizes = segmentation_optimization(image_cytosol,image_nuclei,cytosol_diameter=tested_ts,second_image_cytosol=second_image_cytosol)
        list_masks_nuclei.append(tested_masks_nuclei)
        list_masks_cytosol.append(tested_masks_cytosol)
        list_masks_nuc_sizes.append(tested_list_masks_nuc_sizes)
        list_masks_cyto_sizes.append(tested_list_masks_cyto_sizes)
        array_number_detected_masks[i]= metric_max_cells_and_area( tested_masks_cytosol) + metric_max_cells_and_area( tested_masks_nuclei)
    selected_index = np.argmax(array_number_detected_masks)
    masks_nuclei = list_masks_nuclei [selected_index]
    masks_cytosol = list_masks_cytosol [selected_index]
    masks_nuc_sizes = list_masks_nuc_sizes[selected_index]
    masks_cyto_sizes = list_masks_cyto_sizes[selected_index]
    
    # Save cyto masks as arrays
    masks_cytosol = np.array(masks_cytosol)
    # if output_directory does not exist, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    tiff.imwrite(os.path.join(output_directory, "masks_cytosol.tif"), masks_cytosol.astype(masks_cytosol.dtype))
    
    
        # Plotting
    color_map = 'Greys_r'
    fig, ax = plt.subplots(1,4, figsize=(14, 4))
    # Plotting the heatmap of a section in the image
    ax[0].imshow(image_nuclei,cmap=color_map)
    ax[1].imshow(masks_nuclei,cmap=color_map)
    ax[2].imshow(image_cytosol,cmap=color_map)
    ax[3].imshow(masks_cytosol,cmap=color_map)
    ax[0].set(title='DAPI'); ax[0].axis('off');ax[0].grid(False)
    ax[1].set(title='mask nuclei'); ax[1].axis('off');ax[1].grid(False)
    ax[2].set(title='brightfield'); ax[2].axis('off');ax[2].grid(False)
    ax[3].set(title='mask cytosol'); ax[3].axis('off');ax[3].grid(False)


    return masks_cytosol, masks_nuclei, masks_cyto_sizes, masks_nuc_sizes




# #Sort images based on nuclei
def get_cell_stage_and_size_filtered(masks_nuclei, voxel_size, min_fraction_median=0.2, verbose=True):
    # Label connected components
    labeled_mask, num_nuclei = label(masks_nuclei, return_num=True)
    
    # XY pixel size in nm
    pixel_size_xy = voxel_size[1]
    
    # First pass: get all areas
    all_areas = np.array([region.area for region in regionprops(labeled_mask)])
    if len(all_areas) == 0:
        print("No nuclei detected")
        return "no-nuclei", [], np.zeros_like(masks_nuclei)  # <-- modified to return 3 values
    
    median_area = np.median(all_areas)
    min_area_pixels = median_area * min_fraction_median  # dynamically filter small outliers
    
    # Map number of nuclei to stage (after filtering)
    filtered_mask = np.zeros_like(labeled_mask)
    nuclei_sizes = []
    label_idx = 1
    for region in regionprops(labeled_mask):
        if region.area < min_area_pixels:
            continue  # skip very small outliers
        
        # keep in filtered mask
        filtered_mask[labeled_mask == region.label] = label_idx
        
        # Approximate diameter in pixels (assume circular)
        diameter_pixels = 2 * np.sqrt(region.area / np.pi)
        area_um2 = region.area * (pixel_size_xy / 1000)**2
        diameter_um = diameter_pixels * (pixel_size_xy / 1000)
        
        nuclei_sizes.append({
            "label": label_idx,
            "area_pixels": region.area,
            "diameter_pixels": diameter_pixels,
            "area_um2": area_um2,
            "diameter_um": diameter_um
        })
        label_idx += 1
    
    # Determine cell stage based on filtered nuclei
    num_filtered = len(nuclei_sizes)
    stage_map = {1: "1-cell", 2: "2-cell", 4: "4-cell"}
    stage = stage_map.get(num_filtered, f"{num_filtered}-cell")
    
    print(f"Detected {num_filtered} nuclei after filtering → Stage: {stage}")

    if verbose:
        run_2cell_classifier = stage == "2-cell"
        run_4cell_classifier = stage == "4-cell"
        run_embryo_segmentation = stage in ["no-nuclei"] or num_filtered not in [2, 4]
        
        print(f"run_2cell_classifier = {run_2cell_classifier}")
        print(f"run_4cell_classifier = {run_4cell_classifier}")
        print(f"run_embryo_segmentation = {run_embryo_segmentation}")
    
    return stage, nuclei_sizes, filtered_mask




# Suppress just the InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


# run_2cell_classifier = False
# run_4cell_classifier = False

# if run_2cell_classifier:
#     # --- Extract features of unseen brightfield image for classification ---
#     props_unseen = regionprops_table(
#         masks_cytosol,
#         intensity_image=bf,
#         properties=[
#             'label', 'area', 'perimeter', 'eccentricity', 'solidity', 'extent',
#             'major_axis_length', 'minor_axis_length', 'mean_intensity',
#             'bbox', 'centroid', 'orientation'
#         ]
#     )

#     # Convert to DataFrame
#     features_df = pd.DataFrame(props_unseen)

#     # Rename centroids for convenience
#     features_df['centroid_y'] = features_df['centroid-0']
#     features_df['centroid_x'] = features_df['centroid-1']

#     # Load trained Random Forest model
#     rf = joblib.load("models/2-cell_classification_RFmodel.joblib")

#     # --- Select the features used during training ---
#     X_new = features_df[
#         [
#             'area', 'perimeter', 'eccentricity', 'solidity', 'extent',
#             'major_axis_length', 'minor_axis_length', 'mean_intensity',
#             'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3',
#             'centroid-0', 'centroid-1',
#             'orientation'
#         ]
#     ]

#     # --- Predict cell type and probabilities ---
#     predictions = rf.predict(X_new)
#     proba = rf.predict_proba(X_new)
#     classes = rf.classes_

#     # --- Get confidence scores ---
#     predicted_class_indices = [list(classes).index(pred) for pred in predictions]
#     prediction_confidence = [proba[i][idx] for i, idx in enumerate(predicted_class_indices)]

#     # Add predictions and confidence to dataframe
#     features_df["initial_prediction"] = predictions
#     features_df["prediction_confidence"] = prediction_confidence
    
#     # --- Fit ellipse to entire embryo ---
#     binary_image = (masks_cytosol > 0).astype(np.uint8)
#     contours = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

#     if contours and len(contours[0]) >= 5:
#         cnt = max(contours, key=cv2.contourArea)
#         ellipse = cv2.fitEllipse(cnt)
#         (xc, yc), (d1, d2), angle = ellipse  # center, axes, rotation
#         theta = np.deg2rad(angle)
#         minor_axis_vec = np.array([-np.sin(theta), np.cos(theta)])  # along minor axis

#         # Compute relative minor-axis positions for each cell
#         rel_positions = []
#         for _, row in features_df.iterrows():
#             centroid = np.array([row['centroid_x'], row['centroid_y']])
#             vec_from_center = centroid - np.array([xc, yc])
#             rel_pos_minor = np.dot(vec_from_center, minor_axis_vec) / d2  # normalized
#             rel_positions.append(rel_pos_minor)
#         features_df['rel_pos_minor'] = rel_positions

#     # ---- Fail-safe logic: Ensure exactly one AB and one P1 ----
#     ab_idx = list(classes).index("AB")
#     p1_idx = list(classes).index("P1")

#     # Rank cells by their AB and P1 confidence
#     features_df["AB_conf"] = proba[:, ab_idx]
#     features_df["P1_conf"] = proba[:, p1_idx]

#     # Assign highest AB_conf as AB, highest P1_conf as P1
#     ab_row = features_df.loc[features_df["AB_conf"].idxmax()]
#     p1_row = features_df.loc[features_df["P1_conf"].idxmax()]

#     # Assign labels
#     features_df["highest_confidence_label"] = "Unassigned"
#     features_df.loc[ab_row.name, "highest_confidence_label"] = "AB"
#     features_df.loc[p1_row.name, "highest_confidence_label"] = "P1"
    
#     # --- Extract AB and P1 masks ---
#     ab_mask = (masks_cytosol == ab_row['label'])
#     p1_mask = (masks_cytosol == p1_row['label'])

#     # Dilate slightly to ensure boundary contact detection
#     ab_dilated = binary_dilation(ab_mask, iterations=1)
#     p1_dilated = binary_dilation(p1_mask, iterations=1)

#     touching = np.any(ab_dilated & p1_mask) or np.any(p1_dilated & ab_mask)

#     if not touching:
#         print(f"Fail-safe triggered for {image_name}: AB and P1 are not touching.")
#         features_df.loc[ab_row.name, "highest_confidence_label"] = "Unassigned"
#         features_df.loc[p1_row.name, "highest_confidence_label"] = "Unassigned"
#         features_df["nearby_cells"] = False
#     else:
#         features_df["nearby_cells"] = True

#     # --- Plot prediction labels ---
#     mask_image = np.max(masks_cytosol, axis=0) if masks_cytosol.ndim == 3 else masks_cytosol
#     plt.figure(figsize=(6, 6))
#     plt.imshow(mask_image, cmap='nipy_spectral')
#     plt.axis('off')

#     for _, row in features_df.iterrows():
#         label = row['label']
#         pred_label = row['highest_confidence_label']
#         y, x = center_of_mass(mask_image == label)
#         plt.text(x, y, pred_label, color='white', fontsize=16,
#                  ha='center', va='center', weight='bold')

#     plt.title("Predicted Labels on Cytosol Masks")
#     predicted_label_filename = os.path.join(output_directory, f'predicted_label_{image_name}.png')
#     plt.savefig(predicted_label_filename, dpi=300, bbox_inches='tight')
#     plt.show()

#     # --- Save outputs ---
#     features_df_output = os.path.join(output_directory, f'features_df_{image_name}.csv')
#     features_df.to_csv(features_df_output, index=False)

#     print(features_df.tail())
    
#     # --- Plot cell centroid position ---
#     plt.figure(figsize=(6,6))
#     for label, group in features_df.groupby('highest_confidence_label'):
#         plt.scatter(group['centroid_x'], group['centroid_y'], 
#                     s=group['area']/100, label=label)

#     plt.gca().invert_yaxis()  # Match image coordinates
#     plt.xlabel("X position")
#     plt.ylabel("Y position")
#     plt.title("Cell positions and sizes")

#     # Move legend outside the plot
#     plt.legend(title="Cell Type", loc='center left', bbox_to_anchor=(1, 0.7))

#     # Save figure
#     centroid_position_plot_filename = os.path.join(output_directory, f'centroid_position_plot_{image_name}.png')
#     plt.savefig(centroid_position_plot_filename, dpi=300, bbox_inches='tight')
#     plt.show()
#     # --- Plot prediction confidence ---
#     conf_cols = ['AB_conf', 'P1_conf']
#     conf_df = features_df.melt(id_vars='label', value_vars=conf_cols,
#                                var_name='class', value_name='confidence')
#     conf_df['class'] = conf_df['class'].str.replace('_conf','')

#     plt.figure(figsize=(8,4))
#     sns.barplot(data=conf_df, x='label', y='confidence', hue='class')
#     plt.ylabel("Classifier Confidence")
#     plt.xlabel("Cell Mask Label")
#     plt.title("Confidence Scores per Cell and Class")
#     plt.legend(title="Class")
#     cell_confidence_plot_filename = os.path.join(output_directory, f'cell_confidence_plot_{image_name}.png')
#     plt.savefig(cell_confidence_plot_filename, dpi=300, bbox_inches='tight')
#     plt.show()
# else:
#     print("Skipping 2-cell classifier...")






# ### 4-cell classifier

# # Suppress just the InconsistentVersionWarning
# warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# if run_4cell_classifier:
#     # --- Extract base features ---
#     props_unseen = regionprops_table(
#         masks_cytosol,
#         intensity_image=bf,
#         properties=[
#             'label', 'area', 'perimeter', 'eccentricity', 'solidity', 'extent',
#             'major_axis_length', 'minor_axis_length', 'mean_intensity',
#             'bbox', 'centroid', 'orientation'
#         ]
#     )
#     features_df = pd.DataFrame(props_unseen)
#     # --- define centroid_x / centroid_y ---
#     features_df['centroid_y'] = features_df['centroid-0']
#     features_df['centroid_x'] = features_df['centroid-1']

#     # --- Apply filters once to whole BF image ---
#     bf_float = img_as_float(bf)
#     smooth = filters.gaussian(bf_float, sigma=1)
#     sobel_edges = filters.sobel(smooth)
#     median_filtered = filters.rank.median(
#         (bf / bf.max() * 255).astype(np.uint8),
#         disk(3)
#     )

#     # --- Add per-cell filtered stats ---
#     extra_features = []
#     for lbl in features_df['label']:
#         mask = (masks_cytosol == lbl)
#         extra_features.append({
#             "smooth_mean": np.mean(smooth[mask]),
#             "smooth_std": np.std(smooth[mask]),
#             "smooth_median": np.median(smooth[mask]),
#             "sobel_mean": np.mean(sobel_edges[mask]),
#             "sobel_std": np.std(sobel_edges[mask]),
#             "sobel_median": np.median(sobel_edges[mask]),
#             "medianf_mean": np.mean(median_filtered[mask]),
#             "medianf_std": np.std(median_filtered[mask]),
#             "medianf_median": np.median(median_filtered[mask])
#         })
#     extra_df = pd.DataFrame(extra_features)
#     features_df = pd.concat([features_df, extra_df], axis=1)

#     # --- Load model ---
#     rf = joblib.load("models/4-cell_classification_RFmodel.joblib")

#     # --- Match training feature order ---
#     X_new = features_df[
#         [
#             'area', 'perimeter', 'eccentricity', 'solidity', 'extent',
#             'major_axis_length', 'minor_axis_length', 'mean_intensity',
#             'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3',
#             'centroid-0', 'centroid-1',
#             'orientation',
#             'smooth_mean', 'smooth_std', 'smooth_median',
#             'sobel_mean', 'sobel_std', 'sobel_median',
#             'medianf_mean', 'medianf_std', 'medianf_median'
#         ]
#     ]

#     # --- Predict ---
#     proba = rf.predict_proba(X_new)
#     classes = rf.classes_
#     initial_preds = rf.predict(X_new)

#     # --- Confidence scores ---
#     predicted_class_indices = [list(classes).index(pred) for pred in initial_preds]
#     prediction_confidence = [proba[i][idx] for i, idx in enumerate(predicted_class_indices)]
#     features_df["initial_prediction"] = initial_preds
#     features_df["prediction_confidence"] = prediction_confidence

#     # --- Confidence for each class ---
#     for cname in ["ABa", "ABp", "EMS", "P2"]:
#         features_df[f"{cname}_conf"] = proba[:, list(classes).index(cname)]

#     # --- Fail-safe: ensure one label per class ---
#     features_df["highest_confidence_label"] = "Unassigned"
#     for cname in ["ABa", "ABp", "EMS", "P2"]:
#         features_df.loc[features_df[f"{cname}_conf"].idxmax(), "highest_confidence_label"] = cname

#     # --- Positional fail-safe using ellipse ---
#     # Fit ellipse to entire embryo
#     binary_image = (masks_cytosol > 0).astype(np.uint8)
#     contours = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

#     if contours and len(contours[0]) >= 5:
#         cnt = max(contours, key=cv2.contourArea)
#         ellipse = cv2.fitEllipse(cnt)
#         (xc, yc), (d1, d2), angle = ellipse  # center, axes, rotation
#         theta = np.deg2rad(angle)
#         minor_axis_vec = np.array([-np.sin(theta), np.cos(theta)])  # along minor axis

#         # Compute relative minor-axis positions for each cell
#         rel_positions = []
#         for idx, row in features_df.iterrows():
#             centroid = np.array([row['centroid_x'], row['centroid_y']])
#             vec_from_center = centroid - np.array([xc, yc])
#             rel_pos_minor = np.dot(vec_from_center, minor_axis_vec) / d2  # normalized
#             rel_positions.append(rel_pos_minor)
#         features_df['rel_pos_minor'] = rel_positions

#         # Assign ABa / P2 at ends
#         left_cell = features_df.loc[features_df['rel_pos_minor'].idxmin()]
#         right_cell = features_df.loc[features_df['rel_pos_minor'].idxmax()]
#         if left_cell['area'] > right_cell['area']:
#             features_df.loc[left_cell.name, 'highest_confidence_label'] = 'ABa'
#             features_df.loc[right_cell.name, 'highest_confidence_label'] = 'P2'
#         else:
#             features_df.loc[left_cell.name, 'highest_confidence_label'] = 'P2'
#             features_df.loc[right_cell.name, 'highest_confidence_label'] = 'ABa'

#         # Middle cells: EMS = smaller, ABp = larger
#         middle_cells = features_df.drop([left_cell.name, right_cell.name])
#         ems_cell = middle_cells.loc[middle_cells['area'].idxmin()]
#         abp_cell = middle_cells.loc[middle_cells['area'].idxmax()]
#         features_df.loc[ems_cell.name, 'highest_confidence_label'] = 'EMS'
#         features_df.loc[abp_cell.name, 'highest_confidence_label'] = 'ABp'

#     # --- Plot results ---
#     mask_image = np.max(masks_cytosol, axis=0) if masks_cytosol.ndim == 3 else masks_cytosol
#     plt.figure(figsize=(6, 6))
#     plt.imshow(mask_image, cmap='nipy_spectral')
#     plt.axis('off')

#     for idx, row in features_df.iterrows():
#         y, x = center_of_mass(mask_image == row['label'])
#         plt.text(x, y, row['highest_confidence_label'], color='white',
#                  fontsize=16, ha='center', va='center', weight='bold')

#     plt.title("Predicted Labels on Cytosol Masks")

#     # --- Save outputs ---
#     features_df_output = os.path.join(output_directory, f'features_df_{image_name}.csv')
#     features_df.to_csv(features_df_output, index=False)
#     predicted_label_filename = os.path.join(output_directory, f'predicted_label_{image_name}.png')
#     plt.savefig(predicted_label_filename, dpi=300, bbox_inches='tight')
#     plt.show()

#     print(features_df.tail())

#    ## plot cell centroid position
#     plt.figure(figsize=(6,6))
#     for label, group in features_df.groupby('highest_confidence_label'):
#         plt.scatter(group['centroid_x'], group['centroid_y'], 
#                     s=group['area']/50, label=label)

#     plt.gca().invert_yaxis()  # Match image coordinates
#     plt.xlabel("X position")
#     plt.ylabel("Y position")
#     plt.legend()
#     plt.title("Cell positions and sizes")
#     plt.show()
    
#     # Save the figure
#     centroid_position_plot_filename = os.path.join(output_directory, f'centroid_position_plot_{image_name}.png')
#     plt.savefig(centroid_position_plot_filename, dpi=300, bbox_inches='tight')
#     plt.show()



#     ### plot prediction confidence per cell
#     # Reshape dataframe for plotting
#     conf_cols = ['ABa_conf', 'ABp_conf', 'EMS_conf', 'P2_conf']
#     conf_df = features_df.melt(id_vars='label', value_vars=conf_cols,
#                                var_name='class', value_name='confidence')

#     # Strip "_conf" from class names
#     conf_df['class'] = conf_df['class'].str.replace('_conf','')

#     plt.figure(figsize=(8,4))
#     sns.barplot(data=conf_df, x='label', y='confidence', hue='class')
#     plt.ylabel("Classifier Confidence")
#     plt.xlabel("Cell Mask Label")
#     plt.title("Confidence Scores per Cell and Class")
#     plt.legend(title="Class")
#     plt.show()
    
#     # Save the figure
#     cell_confidence_plot_filename = os.path.join(output_directory, f'cell_confidence_plot_{image_name}.png')
#     plt.savefig(cell_confidence_plot_filename, dpi=300, bbox_inches='tight')
#     plt.show()

# else:
#     print("Skipping 4-cell classifier...")




def classify_2cell(masks_cytosol, bf, image_name, output_directory, 
                   model_path="models/2-cell_classification_RFmodel.joblib", 
                   verbose=True):
    """
    Classify 2-cell embryo into AB and P1 cells using Random Forest classifier.
    
    Parameters:
    -----------
    masks_cytosol : ndarray
        Cytosol segmentation masks
    bf : ndarray
        Brightfield image
    image_name : str
        Image identifier for saving outputs
    output_directory : str or Path
        Directory to save output files
    model_path : str
        Path to trained Random Forest model (default: "models/2-cell_classification_RFmodel.joblib")
    verbose : bool
        Whether to print statements and show plots (default: True)
    
    Returns:
    --------
    features_df : pd.DataFrame
        DataFrame with cell features, predictions, and confidence scores
    """
    
    # --- Extract features of unseen brightfield image for classification ---
    props_unseen = regionprops_table(
        masks_cytosol,
        intensity_image=bf,
        properties=[
            'label', 'area', 'perimeter', 'eccentricity', 'solidity', 'extent',
            'major_axis_length', 'minor_axis_length', 'mean_intensity',
            'bbox', 'centroid', 'orientation'
        ]
    )

    # Convert to DataFrame
    features_df = pd.DataFrame(props_unseen)

    # Rename centroids for convenience
    features_df['centroid_y'] = features_df['centroid-0']
    features_df['centroid_x'] = features_df['centroid-1']

    # Load trained Random Forest model
    rf = joblib.load(model_path)

    # --- Select the features used during training ---
    X_new = features_df[
        [
            'area', 'perimeter', 'eccentricity', 'solidity', 'extent',
            'major_axis_length', 'minor_axis_length', 'mean_intensity',
            'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3',
            'centroid-0', 'centroid-1',
            'orientation'
        ]
    ]

    # --- Predict cell type and probabilities ---
    predictions = rf.predict(X_new)
    proba = rf.predict_proba(X_new)
    classes = rf.classes_

    # --- Get confidence scores ---
    predicted_class_indices = [list(classes).index(pred) for pred in predictions]
    prediction_confidence = [proba[i][idx] for i, idx in enumerate(predicted_class_indices)]

    # Add predictions and confidence to dataframe
    features_df["initial_prediction"] = predictions
    features_df["prediction_confidence"] = prediction_confidence
    
    # --- Fit ellipse to entire embryo ---
    binary_image = (masks_cytosol > 0).astype(np.uint8)
    contours = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    if contours and len(contours[0]) >= 5:
        cnt = max(contours, key=cv2.contourArea)
        ellipse = cv2.fitEllipse(cnt)
        (xc, yc), (d1, d2), angle = ellipse  # center, axes, rotation
        theta = np.deg2rad(angle)
        minor_axis_vec = np.array([-np.sin(theta), np.cos(theta)])  # along minor axis

        # Compute relative minor-axis positions for each cell
        rel_positions = []
        for _, row in features_df.iterrows():
            centroid = np.array([row['centroid_x'], row['centroid_y']])
            vec_from_center = centroid - np.array([xc, yc])
            rel_pos_minor = np.dot(vec_from_center, minor_axis_vec) / d2  # normalized
            rel_positions.append(rel_pos_minor)
        features_df['rel_pos_minor'] = rel_positions

    # ---- Fail-safe logic: Ensure exactly one AB and one P1 ----
    ab_idx = list(classes).index("AB")
    p1_idx = list(classes).index("P1")

    # Rank cells by their AB and P1 confidence
    features_df["AB_conf"] = proba[:, ab_idx]
    features_df["P1_conf"] = proba[:, p1_idx]

    # Assign highest AB_conf as AB, highest P1_conf as P1
    ab_row = features_df.loc[features_df["AB_conf"].idxmax()]
    p1_row = features_df.loc[features_df["P1_conf"].idxmax()]

    # Assign labels
    features_df["highest_confidence_label"] = "Unassigned"
    features_df.loc[ab_row.name, "highest_confidence_label"] = "AB"
    features_df.loc[p1_row.name, "highest_confidence_label"] = "P1"
    
    # --- Extract AB and P1 masks ---
    ab_mask = (masks_cytosol == ab_row['label'])
    p1_mask = (masks_cytosol == p1_row['label'])

    # Dilate slightly to ensure boundary contact detection
    ab_dilated = binary_dilation(ab_mask, iterations=1)
    p1_dilated = binary_dilation(p1_mask, iterations=1)

    touching = np.any(ab_dilated & p1_mask) or np.any(p1_dilated & ab_mask)

    if not touching:
        if verbose:
            print(f"Fail-safe triggered for {image_name}: AB and P1 are not touching.")
        features_df.loc[ab_row.name, "highest_confidence_label"] = "Unassigned"
        features_df.loc[p1_row.name, "highest_confidence_label"] = "Unassigned"
        features_df["nearby_cells"] = False
    else:
        features_df["nearby_cells"] = True

    # --- Plot prediction labels ---
    if verbose:
        mask_image = np.max(masks_cytosol, axis=0) if masks_cytosol.ndim == 3 else masks_cytosol
        plt.figure(figsize=(6, 6))
        plt.imshow(mask_image, cmap='nipy_spectral')
        plt.axis('off')

        for _, row in features_df.iterrows():
            label = row['label']
            pred_label = row['highest_confidence_label']
            y, x = center_of_mass(mask_image == label)
            plt.text(x, y, pred_label, color='white', fontsize=16,
                     ha='center', va='center', weight='bold')

        plt.title("Predicted Labels on Cytosol Masks")
        predicted_label_filename = os.path.join(output_directory, f'predicted_label_{image_name}.png')
        plt.savefig(predicted_label_filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    # --- Save outputs ---
    features_df_output = os.path.join(output_directory, f'features_df_{image_name}.csv')
    features_df.to_csv(features_df_output, index=False)

    if verbose:
        print(features_df.tail())
        
        # --- Plot cell centroid position ---
        plt.figure(figsize=(6,6))
        for label, group in features_df.groupby('highest_confidence_label'):
            plt.scatter(group['centroid_x'], group['centroid_y'], 
                        s=group['area']/100, label=label)

        plt.gca().invert_yaxis()  # Match image coordinates
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.title("Cell positions and sizes")

        # Move legend outside the plot
        plt.legend(title="Cell Type", loc='center left', bbox_to_anchor=(1, 0.7))

        # Save figure
        centroid_position_plot_filename = os.path.join(output_directory, f'centroid_position_plot_{image_name}.png')
        plt.savefig(centroid_position_plot_filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        # --- Plot prediction confidence ---
        conf_cols = ['AB_conf', 'P1_conf']
        conf_df = features_df.melt(id_vars='label', value_vars=conf_cols,
                                   var_name='class', value_name='confidence')
        conf_df['class'] = conf_df['class'].str.replace('_conf','')

        plt.figure(figsize=(8,4))
        sns.barplot(data=conf_df, x='label', y='confidence', hue='class')
        plt.ylabel("Classifier Confidence")
        plt.xlabel("Cell Mask Label")
        plt.title("Confidence Scores per Cell and Class")
        plt.legend(title="Class")
        cell_confidence_plot_filename = os.path.join(output_directory, f'cell_confidence_plot_{image_name}.png')
        plt.savefig(cell_confidence_plot_filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    return features_df





# def classify_4cell(masks_cytosol, bf, image_name, output_directory,
#                    model_path="models/4-cell_classification_RFmodel.joblib",
#                    verbose=True):
#     """
#     Classify 4-cell embryo into ABa, ABp, EMS, and P2 cells using Random Forest classifier.
    
#     Parameters:
#     -----------
#     masks_cytosol : ndarray
#         Cytosol segmentation masks
#     bf : ndarray
#         Brightfield image
#     image_name : str
#         Image identifier for saving outputs
#     output_directory : str or Path
#         Directory to save output files
#     model_path : str
#         Path to trained Random Forest model (default: "models/4-cell_classification_RFmodel.joblib")
#     verbose : bool
#         Whether to print statements and show plots (default: True)
    
#     Returns:
#     --------
#     features_df : pd.DataFrame
#         DataFrame with cell features, predictions, and confidence scores
#     """
    
#     # --- Extract base features ---
#     props_unseen = regionprops_table(
#         masks_cytosol,
#         intensity_image=bf,
#         properties=[
#             'label', 'area', 'perimeter', 'eccentricity', 'solidity', 'extent',
#             'major_axis_length', 'minor_axis_length', 'mean_intensity',
#             'bbox', 'centroid', 'orientation'
#         ]
#     )
#     features_df = pd.DataFrame(props_unseen)
    
#     # --- define centroid_x / centroid_y ---
#     features_df['centroid_y'] = features_df['centroid-0']
#     features_df['centroid_x'] = features_df['centroid-1']

#     # --- Apply filters once to whole BF image ---
#     bf_float = img_as_float(bf)
#     smooth = filters.gaussian(bf_float, sigma=1)
#     sobel_edges = filters.sobel(smooth)
#     median_filtered = filters.rank.median(
#         (bf / bf.max() * 255).astype(np.uint8),
#         disk(3)
#     )

#     # --- Add per-cell filtered stats ---
#     extra_features = []
#     for lbl in features_df['label']:
#         mask = (masks_cytosol == lbl)
#         extra_features.append({
#             "smooth_mean": np.mean(smooth[mask]),
#             "smooth_std": np.std(smooth[mask]),
#             "smooth_median": np.median(smooth[mask]),
#             "sobel_mean": np.mean(sobel_edges[mask]),
#             "sobel_std": np.std(sobel_edges[mask]),
#             "sobel_median": np.median(sobel_edges[mask]),
#             "medianf_mean": np.mean(median_filtered[mask]),
#             "medianf_std": np.std(median_filtered[mask]),
#             "medianf_median": np.median(median_filtered[mask])
#         })
#     extra_df = pd.DataFrame(extra_features)
#     features_df = pd.concat([features_df, extra_df], axis=1)

#     # --- Load model ---
#     rf = joblib.load(model_path)

#     # --- Match training feature order ---
#     X_new = features_df[
#         [
#             'area', 'perimeter', 'eccentricity', 'solidity', 'extent',
#             'major_axis_length', 'minor_axis_length', 'mean_intensity',
#             'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3',
#             'centroid-0', 'centroid-1',
#             'orientation',
#             'smooth_mean', 'smooth_std', 'smooth_median',
#             'sobel_mean', 'sobel_std', 'sobel_median',
#             'medianf_mean', 'medianf_std', 'medianf_median'
#         ]
#     ]

#     # --- Predict ---
#     proba = rf.predict_proba(X_new)
#     classes = rf.classes_
#     initial_preds = rf.predict(X_new)

#     if verbose:
#         print(f"Model classes: {list(classes)}")

#     # --- Confidence scores ---
#     predicted_class_indices = [list(classes).index(pred) for pred in initial_preds]
#     prediction_confidence = [proba[i][idx] for i, idx in enumerate(predicted_class_indices)]
#     features_df["initial_prediction"] = initial_preds
#     features_df["prediction_confidence"] = prediction_confidence

#     # --- Map expected class names to model class names (case-insensitive) ---
#     expected_classes = ["ABa", "ABp", "EMS", "P2"]
#     class_map = {}
    
#     for expected in expected_classes:
#         for model_class in classes:
#             if expected.lower() == model_class.lower():
#                 class_map[expected] = model_class
#                 break
        
#         if expected not in class_map:
#             raise ValueError(f"Model does not contain class '{expected}'. Available classes: {list(classes)}")

#     # --- Confidence for each class ---
#     for cname in expected_classes:
#         model_class = class_map[cname]
#         features_df[f"{cname}_conf"] = proba[:, list(classes).index(model_class)]

#     # --- Fail-safe: ensure one label per class ---
#     features_df["highest_confidence_label"] = "Unassigned"
#     for cname in expected_classes:
#         features_df.loc[features_df[f"{cname}_conf"].idxmax(), "highest_confidence_label"] = cname

#     # --- Positional fail-safe using ellipse ---
#     # Fit ellipse to entire embryo
#     binary_image = (masks_cytosol > 0).astype(np.uint8)
#     contours = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

#     if contours and len(contours[0]) >= 5:
#         cnt = max(contours, key=cv2.contourArea)
#         ellipse = cv2.fitEllipse(cnt)
#         (xc, yc), (d1, d2), angle = ellipse  # center, axes, rotation
#         theta = np.deg2rad(angle)
#         minor_axis_vec = np.array([-np.sin(theta), np.cos(theta)])  # along minor axis

#         # Compute relative minor-axis positions for each cell
#         rel_positions = []
#         for idx, row in features_df.iterrows():
#             centroid = np.array([row['centroid_x'], row['centroid_y']])
#             vec_from_center = centroid - np.array([xc, yc])
#             rel_pos_minor = np.dot(vec_from_center, minor_axis_vec) / d2  # normalized
#             rel_positions.append(rel_pos_minor)
#         features_df['rel_pos_minor'] = rel_positions

#         # Assign ABa / P2 at ends
#         left_cell = features_df.loc[features_df['rel_pos_minor'].idxmin()]
#         right_cell = features_df.loc[features_df['rel_pos_minor'].idxmax()]
#         if left_cell['area'] > right_cell['area']:
#             features_df.loc[left_cell.name, 'highest_confidence_label'] = 'ABa'
#             features_df.loc[right_cell.name, 'highest_confidence_label'] = 'P2'
#         else:
#             features_df.loc[left_cell.name, 'highest_confidence_label'] = 'P2'
#             features_df.loc[right_cell.name, 'highest_confidence_label'] = 'ABa'

#         # Middle cells: EMS = smaller, ABp = larger
#         middle_cells = features_df.drop([left_cell.name, right_cell.name])
#         ems_cell = middle_cells.loc[middle_cells['area'].idxmin()]
#         abp_cell = middle_cells.loc[middle_cells['area'].idxmax()]
#         features_df.loc[ems_cell.name, 'highest_confidence_label'] = 'EMS'
#         features_df.loc[abp_cell.name, 'highest_confidence_label'] = 'ABp'

#     # --- Plot results ---
#     if verbose:
#         mask_image = np.max(masks_cytosol, axis=0) if masks_cytosol.ndim == 3 else masks_cytosol
#         plt.figure(figsize=(6, 6))
#         plt.imshow(mask_image, cmap='nipy_spectral')
#         plt.axis('off')

#         for idx, row in features_df.iterrows():
#             y, x = center_of_mass(mask_image == row['label'])
#             plt.text(x, y, row['highest_confidence_label'], color='white',
#                      fontsize=16, ha='center', va='center', weight='bold')

#         plt.title("Predicted Labels on Cytosol Masks")

#     # --- Save outputs ---
#     features_df_output = os.path.join(output_directory, f'features_df_{image_name}.csv')
#     features_df.to_csv(features_df_output, index=False)
    
#     if verbose:
#         predicted_label_filename = os.path.join(output_directory, f'predicted_label_{image_name}.png')
#         plt.savefig(predicted_label_filename, dpi=300, bbox_inches='tight')
#         plt.show()

#         print(features_df.tail())

#         ## plot cell centroid position
#         plt.figure(figsize=(6,6))
#         for label, group in features_df.groupby('highest_confidence_label'):
#             plt.scatter(group['centroid_x'], group['centroid_y'], 
#                         s=group['area']/50, label=label)

#         plt.gca().invert_yaxis()  # Match image coordinates
#         plt.xlabel("X position")
#         plt.ylabel("Y position")
#         plt.legend()
#         plt.title("Cell positions and sizes")
        
#         # Save the figure
#         centroid_position_plot_filename = os.path.join(output_directory, f'centroid_position_plot_{image_name}.png')
#         plt.savefig(centroid_position_plot_filename, dpi=300, bbox_inches='tight')
#         plt.show()

#         ### plot prediction confidence per cell
#         # Reshape dataframe for plotting
#         conf_cols = ['ABa_conf', 'ABp_conf', 'EMS_conf', 'P2_conf']
#         conf_df = features_df.melt(id_vars='label', value_vars=conf_cols,
#                                    var_name='class', value_name='confidence')

#         # Strip "_conf" from class names
#         conf_df['class'] = conf_df['class'].str.replace('_conf','')

#         plt.figure(figsize=(8,4))
#         sns.barplot(data=conf_df, x='label', y='confidence', hue='class')
#         plt.ylabel("Classifier Confidence")
#         plt.xlabel("Cell Mask Label")
#         plt.title("Confidence Scores per Cell and Class")
#         plt.legend(title="Class")
        
#         # Save the figure
#         cell_confidence_plot_filename = os.path.join(output_directory, f'cell_confidence_plot_{image_name}.png')
#         plt.savefig(cell_confidence_plot_filename, dpi=300, bbox_inches='tight')
#         plt.show()

#     return features_df




def classify_4cell(masks_cytosol, bf, image_name, output_directory,
                   model_path="models/4-cell_classification_RFmodel.joblib",
                   verbose=True):
    """
    Classify 4-cell embryo into ABa, ABp, EMS, and P2 cells using Random Forest classifier.
    """
    # --- Extract base features ---
    props_unseen = regionprops_table(
        masks_cytosol,
        intensity_image=bf,
        properties=[
            'label', 'area', 'perimeter', 'eccentricity', 'solidity', 'extent',
            'major_axis_length', 'minor_axis_length', 'mean_intensity',
            'bbox', 'centroid', 'orientation'
        ]
    )
    features_df = pd.DataFrame(props_unseen)

    # --- define centroid_x / centroid_y ---
    features_df['centroid_y'] = features_df['centroid-0']
    features_df['centroid_x'] = features_df['centroid-1']

    # --- Apply filters once to whole BF image ---
    bf_float = img_as_float(bf)
    smooth = filters.gaussian(bf_float, sigma=1)
    sobel_edges = filters.sobel(smooth)
    median_filtered = filters.rank.median(
        (bf / max(bf.max(), 1e-9) * 255).astype(np.uint8),
        disk(3)
    )

    # --- Add per-cell filtered stats ---
    extra_features = []
    for lbl in features_df['label']:
        mask = (masks_cytosol == lbl)
        extra_features.append({
            "smooth_mean": float(np.mean(smooth[mask])),
            "smooth_std": float(np.std(smooth[mask])),
            "smooth_median": float(np.median(smooth[mask])),
            "sobel_mean": float(np.mean(sobel_edges[mask])),
            "sobel_std": float(np.std(sobel_edges[mask])),
            "sobel_median": float(np.median(sobel_edges[mask])),
            "medianf_mean": float(np.mean(median_filtered[mask])),
            "medianf_std": float(np.std(median_filtered[mask])),
            "medianf_median": float(np.median(median_filtered[mask]))
        })
    extra_df = pd.DataFrame(extra_features)
    features_df = pd.concat([features_df, extra_df], axis=1)

    # --- Load model ---
    rf = joblib.load(model_path)

    # --- Match training feature order ---
    X_new = features_df[
        [
            'area', 'perimeter', 'eccentricity', 'solidity', 'extent',
            'major_axis_length', 'minor_axis_length', 'mean_intensity',
            'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3',
            'centroid-0', 'centroid-1',
            'orientation',
            'smooth_mean', 'smooth_std', 'smooth_median',
            'sobel_mean', 'sobel_std', 'sobel_median',
            'medianf_mean', 'medianf_std', 'medianf_median'
        ]
    ]

    # --- Predict ---
    proba = rf.predict_proba(X_new)
    classes = rf.classes_
    initial_preds = rf.predict(X_new)

    if verbose:
        print(f"Model classes: {list(classes)}")

    # --- Confidence scores for initial predictions ---
    predicted_class_indices = [list(classes).index(pred) for pred in initial_preds]
    prediction_confidence = [proba[i][idx] for i, idx in enumerate(predicted_class_indices)]
    features_df["initial_prediction"] = initial_preds
    features_df["prediction_confidence"] = prediction_confidence

    # --- Map expected class names to model class names (alias-aware, case-insensitive) ---
    expected_classes = ["ABa", "ABp", "EMS", "P2"]

    def _norm(s): return str(s).strip().lower()
    norm_model_classes = {_norm(mc): mc for mc in classes}

    # Allow common variants; notably map "ABb" -> ABp
    alias_table = {
        "ABa": {"aba", "ab-a", "ab_a", "ab anterior"},
        "ABp": {"abp", "ab-p", "ab_p", "ab posterior", "abb"},
        "EMS": {"ems"},
        "P2":  {"p2", "p-2", "p_2"}
    }

    class_map = {}
    for expected in expected_classes:
        want = _norm(expected)
        match_name = norm_model_classes.get(want, None)

        # Try aliases if no direct match
        if match_name is None:
            for alias in alias_table.get(expected, set()):
                if alias in norm_model_classes:
                    match_name = norm_model_classes[alias]
                    if verbose:
                        print(f"[classify_4cell] Using alias '{match_name}' for expected class '{expected}'.")
                    break

        # Heuristic fallback specifically for ABp/ABb confusion
        if match_name is None and expected == "ABp" and "abb" in norm_model_classes:
            match_name = norm_model_classes["abb"]
            if verbose:
                print("[classify_4cell] Fallback: treating 'ABb' as 'ABp'.")

        if match_name is None:
            raise ValueError(f"Model does not contain class '{expected}'. Available classes: {list(classes)}")

        class_map[expected] = match_name

    # --- Confidence for each expected class (using mapped model class) ---
    for cname in expected_classes:
        model_class = class_map[cname]
        features_df[f"{cname}_conf"] = proba[:, list(classes).index(model_class)]

    # --- Fail-safe: ensure one label per class (pick max per class) ---
    features_df["highest_confidence_label"] = "Unassigned"
    for cname in expected_classes:
        idxmax = features_df[f"{cname}_conf"].idxmax()
        features_df.loc[idxmax, "highest_confidence_label"] = cname

    # --- Positional fail-safe using ellipse ---
    # Fit ellipse to entire embryo
    binary_image = (masks_cytosol > 0).astype(np.uint8)

    # OpenCV findContours API differs by version; handle both
    contours_result = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_result[0] if isinstance(contours_result, tuple) and len(contours_result) == 2 else contours_result[1]

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (xc, yc), (d1, d2), angle = ellipse  # center, axes, rotation
            theta = np.deg2rad(angle)
            minor_axis_vec = np.array([-np.sin(theta), np.cos(theta)])  # along minor axis

            # Compute relative minor-axis positions for each cell
            rel_positions = []
            for _, row in features_df.iterrows():
                centroid = np.array([row['centroid_x'], row['centroid_y']])
                vec_from_center = centroid - np.array([xc, yc])
                rel_pos_minor = float(np.dot(vec_from_center, minor_axis_vec) / max(d2, 1e-9))  # normalized
                rel_positions.append(rel_pos_minor)
            features_df['rel_pos_minor'] = rel_positions

            # Assign ABa / P2 at ends
            left_cell = features_df.loc[features_df['rel_pos_minor'].idxmin()]
            right_cell = features_df.loc[features_df['rel_pos_minor'].idxmax()]
            if left_cell['area'] > right_cell['area']:
                features_df.loc[left_cell.name, 'highest_confidence_label'] = 'ABa'
                features_df.loc[right_cell.name, 'highest_confidence_label'] = 'P2'
            else:
                features_df.loc[left_cell.name, 'highest_confidence_label'] = 'P2'
                features_df.loc[right_cell.name, 'highest_confidence_label'] = 'ABa'

            # Middle cells: EMS = smaller, ABp = larger
            middle_cells = features_df.drop([left_cell.name, right_cell.name])
            if len(middle_cells) >= 2:
                ems_cell = middle_cells.loc[middle_cells['area'].idxmin()]
                abp_cell = middle_cells.loc[middle_cells['area'].idxmax()]
                features_df.loc[ems_cell.name, 'highest_confidence_label'] = 'EMS'
                features_df.loc[abp_cell.name, 'highest_confidence_label'] = 'ABp'

    # --- Plot results ---
    if verbose:
        os.makedirs(output_directory, exist_ok=True)
        mask_image = np.max(masks_cytosol, axis=0) if masks_cytosol.ndim == 3 else masks_cytosol
        plt.figure(figsize=(6, 6))
        plt.imshow(mask_image, cmap='nipy_spectral')
        plt.axis('off')

        for _, row in features_df.iterrows():
            y, x = center_of_mass(mask_image == row['label'])
            plt.text(x, y, row['highest_confidence_label'], color='white',
                     fontsize=16, ha='center', va='center', weight='bold')

        plt.title("Predicted Labels on Cytosol Masks")

    # --- Save outputs ---
    features_df_output = os.path.join(output_directory, f'features_df_{image_name}.csv')
    os.makedirs(output_directory, exist_ok=True)
    features_df.to_csv(features_df_output, index=False)

    if verbose:
        predicted_label_filename = os.path.join(output_directory, f'predicted_label_{image_name}.png')
        plt.savefig(predicted_label_filename, dpi=300, bbox_inches='tight')
        plt.show()

        print(features_df.tail())

        # plot cell centroid position
        plt.figure(figsize=(6, 6))
        for label, group in features_df.groupby('highest_confidence_label'):
            plt.scatter(group['centroid_x'], group['centroid_y'],
                        s=np.clip(group['area'] / 50.0, 10, None), label=label)

        plt.gca().invert_yaxis()  # Match image coordinates
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.legend()
        plt.title("Cell positions and sizes")
        centroid_position_plot_filename = os.path.join(output_directory, f'centroid_position_plot_{image_name}.png')
        plt.savefig(centroid_position_plot_filename, dpi=300, bbox_inches='tight')
        plt.show()

        # plot prediction confidence per cell
        conf_cols = ['ABa_conf', 'ABp_conf', 'EMS_conf', 'P2_conf']
        conf_df = features_df.melt(id_vars='label', value_vars=conf_cols,
                                   var_name='class', value_name='confidence')
        conf_df['class'] = conf_df['class'].str.replace('_conf', '', regex=False)

        plt.figure(figsize=(8, 4))
        sns.barplot(data=conf_df, x='label', y='confidence', hue='class')
        plt.ylabel("Classifier Confidence")
        plt.xlabel("Cell Mask Label")
        plt.title("Confidence Scores per Cell and Class")
        plt.legend(title="Class")
        cell_confidence_plot_filename = os.path.join(output_directory, f'cell_confidence_plot_{image_name}.png')
        plt.savefig(cell_confidence_plot_filename, dpi=300, bbox_inches='tight')
        plt.show()

    return features_df

# #### 3.4 Embryo Segementation

# In[11]:





## Embryo segmentation

def keep_largest_region(mask):
    labels = measure.label(mask)
    props = measure.regionprops(labels)
    if not props:
        return mask * 0  # return empty if nothing found
    largest_region = max(props, key=lambda x: x.area)
    return (labels == largest_region.label).astype(np.uint16)

def embryo_segmentation(bf, image_nuclei, image_name, output_directory):
    cytosol_image = bf[..., 0] if bf.ndim == 3 else bf
    nuclei_image = image_nuclei[..., 0] if image_nuclei.ndim == 3 else image_nuclei

    # Run Cellpose for cytosol (large object: the embryo)
    model_cyto = models.Cellpose(model_type='cyto')
    masks_cytosol, _, _, _ = model_cyto.eval(
        cytosol_image, diameter=embryo_diameter, channels=[0, 0]
    )
    masks_cytosol = keep_largest_region(masks_cytosol)

    # Run Cellpose for nuclei (smaller, multiple objects)
    model_nuclei = models.Cellpose(model_type='cyto2')
    masks_nuclei, _, _, _ = model_nuclei.eval(
        nuclei_image, diameter=nuclei_diameter, channels=[0, 0],
        cellprob_threshold=0.0, flow_threshold=0.2
    )
    
    # --- Post-processing: remove size outliers ---
    labeled_nuc = label(masks_nuclei)
    props = regionprops(labeled_nuc)
    areas = np.array([p.area for p in props])

    if len(areas) > 0:
        median_area = np.median(areas)
        min_area_threshold = 0.5 * median_area  # tune 0.4-0.6 if needed
        max_area_threshold = 2.0 * median_area  # optional: remove huge artifacts

        masks_nuclei_filtered = np.zeros_like(labeled_nuc)
        label_idx = 1
        for p in props:
            if min_area_threshold <= p.area <= max_area_threshold:
                masks_nuclei_filtered[labeled_nuc == p.label] = label_idx
                label_idx += 1

        masks_nuclei = masks_nuclei_filtered
    
    # Get outlines
    outlines_cytosol = utils.outlines_list(masks_cytosol)
    outlines_nuclei = utils.outlines_list(masks_nuclei)

    # Compute sizes
    labeled_cyto = label(masks_cytosol.astype(np.uint16))
    masks_cyto_sizes = [prop.area for prop in regionprops(labeled_cyto)]
    masks_nuc_sizes = [prop.area for prop in regionprops(masks_nuclei)]

    # Plot side-by-side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(cytosol_image, cmap='gray')
    for o in outlines_cytosol:
        ax[0].plot(o[:, 0], o[:, 1], color='lime', linewidth=1)
    ax[0].set_title('Embryo Mask')
    ax[0].axis('off')

    ax[1].imshow(nuclei_image, cmap='gray')
    for o in outlines_nuclei:
        ax[1].plot(o[:, 0], o[:, 1], color='deepskyblue', linewidth=1)
    ax[1].set_title('Nuclei Mask')
    ax[1].axis('off')

    # Create binary mask for embryo outline
    image_shape = cytosol_image.shape
    embryo_outline = np.zeros(image_shape, dtype=bool)
    for outline in outlines_cytosol:
        poly_mask = polygon2mask(image_shape, outline[:, ::-1])
        embryo_outline |= poly_mask

    # Save cyto masks as arrays
    masks_cytosol = np.array(masks_cytosol)
    tiff.imwrite(os.path.join(output_directory, "masks_cytosol.tif"), masks_cytosol.astype(masks_cytosol.dtype))
    
    # Save figure
    segmentation_filename = os.path.join(output_directory, f'embryo_segmentation_{image_name}.png')
    plt.savefig(segmentation_filename)
    plt.tight_layout()
    plt.show()
    plt.close()

    return masks_cytosol, masks_nuclei, masks_cyto_sizes, masks_nuc_sizes




# ## 4. Spot detection

# #### 4.1 Automated threshold selection and spot detection

# In[12]:


def spot_detection(rna,voxel_size,spot_radius,masks_cytosol):
    spots, threshold = detection.detect_spots(
        images= rna,
        return_threshold=True,
        voxel_size=voxel_size,
        spot_radius=spot_radius) 

    spot_radius_px = detection.get_object_radius_pixel(
        voxel_size_nm=voxel_size,
        object_radius_nm=spot_radius,
        ndim=3) 

    # LoG filter
    rna_log = stack.log_filter(rna, sigma=spot_radius_px)

    # local maximum detection
    mask = detection.local_maximum_detection(rna_log, min_distance=spot_radius_px)

    # thresholding
    threshold = detection.automated_threshold_setting(rna_log, mask)
    spots, _ = detection.spots_thresholding(rna_log, mask, threshold)


    # Decompose regions by simulating as many spots as possible until we match the original region intensity.
    #spots_post_decomposition = spots.copy()
    spots_post_decomposition, dense_regions, reference_spot = detection.decompose_dense(
        image=rna,
        spots=spots,
        voxel_size=voxel_size,
        spot_radius=spot_radius,
        alpha=0.7,  # alpha impacts the number of spots per candidate region
        beta=1,  # beta impacts the number of candidate regions to decompose
        gamma=5)  # gamma the filtering step to denoise the image

    # define clusters
    spots_post_clustering, clusters = detection.detect_clusters(
        spots=spots_post_decomposition,
        voxel_size=voxel_size,
        radius=1136, #626 #1000
        nb_min_spots=5) #10

   # plotting
    print(f"Image ID: {image_name} \n")
    print(f"{rna_channel} detection")
    print(f" threshold: {threshold}")
    print("\r spots: {0}".format(spots_post_clustering.shape[0]))
    print("\r clusters: {0}".format(clusters.shape[0]))
   
        #elbow plot
    threshold_output = os.path.join(output_directory, rna_channel + '_threshold_' + image_name)
    plot.plot_elbow(
        images=rna,
        voxel_size=voxel_size,
        spot_radius=spot_radius,
        size_axes=8,
        framesize=(5, 3),
        title=(f"{rna_channel} detection threshold"),
        size_title=12,
        path_output=threshold_output,
        show=True  # Set show to False to hide the plot
    )

      
    #    # Save the plots in the results folder
    detection_output = os.path.join(output_directory, rna_channel + '_detection_' + image_name)
    plot.plot_detection(
        image=np.max(rna, axis=0),
        spots=[spots_post_decomposition, clusters[:, :3]],
        shape=["circle", "circle"],
        radius=[1, 4],
        color=detection_color,
        linewidth=[3, 2],
        fill=[False, True],
        contrast=True,
        framesize=(10, 5),
        title=(f"{rna_channel} detection"),
        path_output= detection_output,
        show=True
    )


    #spots_no_ts, _, ts = multistack.remove_transcription_site(spotDetectionCSV, clusterDetectionCSV, mask_nuc, ndim=3)
    #spots_in_region, _ = multistack.identify_objects_in_region(mask, spots_post_clustering[:,:3], ndim=3)

    # Separating and counting the spots in each cell
    number_masks_cyto = np.max(masks_cytosol)
    list_spots_in_each_cell =[]
    list_clusters_in_each_cell =[]
    for i in range (1, number_masks_cyto+1):
        temp_cyto_mask= np.zeros_like(masks_cytosol)
        temp_cyto_mask[masks_cytosol == i] = i
        spots_in_region, _ = multistack.identify_objects_in_region(mask=temp_cyto_mask, coord=spots_post_clustering[:,:3], ndim=3)
        clusters_in_region,_ = multistack.identify_objects_in_region(mask=temp_cyto_mask, coord=clusters[:,:3], ndim=3)
        list_spots_in_each_cell.append(len(spots_in_region))
        list_clusters_in_each_cell.append(len( clusters_in_region ))
        del spots_in_region, clusters_in_region
    return spots_post_clustering, clusters, list_spots_in_each_cell, list_clusters_in_each_cell

