#!/usr/bin/env python
# coding: utf-8

# # WormLib: open source image analysis library for *C. elegans* 

# Standard library imports
import argparse
import os
import sys
import csv
import warnings
from datetime import datetime
from pathlib import Path

__version__ = "1.0.0"
# Scientific computing
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import label, center_of_mass, binary_dilation

# Image processing
import cv2
import tifffile as tiff
from skimage import measure, morphology, filters
from skimage.measure import regionprops, regionprops_table
from skimage.morphology import erosion, disk
from skimage.draw import polygon2mask
from skimage.util import img_as_float

# Machine learning
import joblib

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path as MPLPath
import seaborn as sns

# Specialized libraries
import bigfish.stack as stack
import bigfish.plot as plot
import bigfish.multistack as multistack
import bigfish.detection as detection
from cellpose import models, utils
import nd2

# Jupyter/IPython imports removed

# Reporting imports
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from PIL import Image

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



def normalize_optional_channel(value):
    if value is None:
        return None
    value = str(value).strip()
    if value.lower() in {"", "none", "nothing", "null", "false", "off", "na", "n/a"}:
        return None
    return value


def parse_optional_int(value, default=None):
    if value is None:
        return default
    value = str(value).strip()
    if value.lower() in {"", "none", "nothing", "null", "false", "off", "na", "n/a"}:
        return None
    return int(value)


def parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_int_tuple(value, default):
    if value is None:
        return tuple(default)
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return tuple(int(v.strip()) for v in str(value).split(","))


def resolve_path(value, base_dir=None):
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return str(path.resolve())


def normalize_pipeline_config(raw_config=None, base_dir=None, cli_input=None, cli_output=None):
    """Normalize YAML or legacy env settings into one semantic pipeline config."""
    raw_config = raw_config or {}
    input_config = raw_config.get("input", {})
    microscope_config = raw_config.get("microscope", {})
    channel_config = raw_config.get("channels", {})
    pipeline_config = raw_config.get("pipeline", {})
    segmentation_config = raw_config.get("segmentation", {})

    input_path = cli_input or input_config.get("path") or raw_config.get("input_path")
    output_directory = cli_output or input_config.get("output_directory") or raw_config.get("output_directory")

    nuclei_config = channel_config.get("nuclei") or {}
    brightfield_config = channel_config.get("brightfield") or {}
    rna_configs = channel_config.get("rna") or []
    nuclei_name = normalize_optional_channel(nuclei_config.get("name", "DAPI"))
    brightfield_name = normalize_optional_channel(brightfield_config.get("name", "brightfield"))

    normalized_rna = []
    for i, rna in enumerate(rna_configs):
        if not rna:
            continue
        name = normalize_optional_channel(rna.get("name") or rna.get("label") or f"RNA{i + 1}")
        if name is None:
            continue
        normalized_rna.append({
            "name": name,
            "fluorophore": normalize_optional_channel(rna.get("fluorophore")),
            "index": parse_optional_int(rna.get("index"), i),
            "spot_radius": parse_int_tuple(
                rna.get("spot_radius_nm") or rna.get("spot_radius"),
                microscope_config.get("default_spot_radius_nm", [1409, 340, 340]),
            ),
            "detection_color": rna.get("detection_color", ["red", "blue", "green", "magenta"][i % 4]),
            "colormap": rna.get("colormap", "PiYG"),
        })

    return {
        "input_path": resolve_path(input_path, base_dir),
        "output_directory": resolve_path(output_directory, base_dir),
        "voxel_size": parse_int_tuple(microscope_config.get("voxel_size_nm"), [1448, 450, 450]),
        "channels": {
            "nuclei": {
                "name": nuclei_name,
                "index": None if nuclei_name is None else parse_optional_int(nuclei_config.get("index"), 3),
            },
            "brightfield": {
                "name": brightfield_name,
                "index": None if brightfield_name is None else parse_optional_int(brightfield_config.get("index"), None),
                "source": brightfield_config.get("source", "auto"),
            },
            "rna": normalized_rna,
        },
        "segmentation": {
            "embryo_diameter": int(segmentation_config.get("embryo_diameter", 500)),
            "nuclei_diameter": int(segmentation_config.get("nuclei_diameter", 70)),
            "cell_diameter": int(segmentation_config.get("cell_diameter", 250)),
        },
        "pipeline": {
            "cell_segmentation": parse_bool(pipeline_config.get("cell_segmentation"), True),
            "cell_classification": parse_bool(pipeline_config.get("cell_classification"), True),
            "embryo_segmentation": parse_bool(pipeline_config.get("embryo_segmentation"), True),
            "spot_detection": parse_bool(pipeline_config.get("spot_detection"), True),
            "heatmaps": parse_bool(pipeline_config.get("heatmaps"), True),
            "rna_density": parse_bool(pipeline_config.get("rna_density"), True),
            "line_scan": parse_bool(pipeline_config.get("line_scan"), True),
        },
    }


def config_from_legacy_env(cli_input=None, cli_output=None):
    """Translate the historical environment-variable API into semantic config."""
    rna_channels = []
    cy5 = normalize_optional_channel(os.getenv('Cy5', 'mRNA1'))
    mcherry = normalize_optional_channel(os.getenv('mCherry', 'mRNA2'))
    if cy5 is not None:
        rna_channels.append({
            "name": cy5,
            "fluorophore": "Cy5",
            "index": parse_optional_int(os.getenv('CY5_INDEX'), 0),
            "spot_radius_nm": parse_int_tuple(os.getenv('SPOT_RADIUS_CH0'), [1409, 340, 340]),
            "detection_color": "red",
        })
    if mcherry is not None:
        rna_channels.append({
            "name": mcherry,
            "fluorophore": "mCherry",
            "index": parse_optional_int(os.getenv('MCHERRY_INDEX'), 1),
            "spot_radius_nm": parse_int_tuple(os.getenv('SPOT_RADIUS_CH1'), [1283, 310, 310]),
            "detection_color": "blue",
        })

    raw_config = {
        "input": {
            "path": cli_input or os.getenv('FOLDER_NAME'),
            "output_directory": cli_output or os.getenv('OUTPUT_DIRECTORY'),
        },
        "microscope": {
            "voxel_size_nm": parse_int_tuple(os.getenv('VOXEL_SIZE'), [1448, 450, 450]),
        },
        "channels": {
            "rna": rna_channels,
            "nuclei": {
                "name": normalize_optional_channel(os.getenv('DAPI', 'DAPI')),
                "index": parse_optional_int(os.getenv('DAPI_INDEX'), 3),
            },
            "brightfield": {
                "name": normalize_optional_channel(os.getenv('brightfield', 'brightfield')),
                "index": parse_optional_int(os.getenv('BRIGHTFIELD_INDEX'), None),
            },
        },
        "segmentation": {
            "embryo_diameter": int(os.getenv('EMBRYO_DIAMETER', '500')),
            "nuclei_diameter": int(os.getenv('NUCLEI_DIAMETER', '70')),
            "cell_diameter": int(os.getenv('CELL_DIAMETER', '250')),
        },
        "pipeline": {
            "cell_segmentation": parse_bool(os.getenv('RUN_CELL_SEGMENTATION'), True),
            "cell_classification": parse_bool(os.getenv('RUN_CELL_CLASSIFIER'), True),
            "embryo_segmentation": parse_bool(os.getenv('RUN_EMBRYO_SEGMENTATION'), True),
            "spot_detection": parse_bool(os.getenv('RUN_SPOT_DETECTION') or os.getenv('SPOT_DETECTION'), True),
            "heatmaps": parse_bool(os.getenv('RUN_mRNA_HEATMAPS'), True),
            "rna_density": parse_bool(os.getenv('RUN_RNA_DENSITY_ANALYSIS'), True),
            "line_scan": parse_bool(os.getenv('RUN_LINE_SCAN_ANALYSIS'), True),
        },
    }
    return normalize_pipeline_config(raw_config)


def load_pipeline_config(config_path=None, cli_input=None, cli_output=None):
    if config_path is None:
        return config_from_legacy_env(cli_input=cli_input, cli_output=cli_output)

    config_path = Path(config_path).expanduser().resolve()
    with open(config_path, "r") as handle:
        raw_config = yaml.safe_load(handle) or {}
    return normalize_pipeline_config(raw_config, base_dir=config_path.parent, cli_input=cli_input, cli_output=cli_output)


def _organize_dv_files(dv_files):
    """
    For DeltaVision files, intelligently separate reference (brightfield) from color channel files.
    
    DeltaVision naming convention:
    - Reference/brightfield: contains "_R3D_REF" (2D image)
    - Color channels: contains "_R3D" but NOT "_R3D_REF" (4D image stack)
    
    Returns: list of filenames ordered as [reference_file, color_file, ...]
    """
    if not dv_files:
        return []
    
    # Separate by naming pattern
    image_ref = [f for f in dv_files if '_R3D_REF' in f]
    image_color = [f for f in dv_files if '_R3D' in f and '_R3D_REF' not in f]
    other_files = [f for f in dv_files if f not in image_ref and f not in image_color]
    
    # Return ordered list: reference first, then colors, then others
    return image_ref + image_color + other_files


def load_images(image_path, output_directory, channel_names, slice_to_plot=0, channel_indices=None):
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
    Cy5 = normalize_optional_channel(channel_names.get('Cy5', None))
    mCherry = normalize_optional_channel(channel_names.get('mCherry', None))
    FITC = normalize_optional_channel(channel_names.get('FITC', None))
    DAPI = normalize_optional_channel(channel_names.get('DAPI', None))
    brightfield = normalize_optional_channel(channel_names.get('brightfield', None))
    channel_indices = channel_indices or {}

    def _channel_index(name, default):
        return parse_optional_int(channel_indices.get(name), default)
    
    # Detect image type
    dv_files = [f for f in list_filenames if f.endswith('.dv')]
    nd2_files = [f for f in list_filenames if f.endswith('.nd2')]
    tiff_files = [f for f in list_filenames if f.endswith(('.tif', '.tiff'))]
    
    if dv_files:
        image_type = 'dv'
        # For DeltaVision, intelligently order files (ref first, then colors)
        dv_files = _organize_dv_files(dv_files)
        print("Detected file type: DeltaVision (.dv)")
    elif nd2_files:
        image_type = 'nd2'
        print("Detected file type: Nikon (.nd2)")
    elif tiff_files:
        image_type = 'tiff'
        print("Detected file type: TIFF (.tif/.tiff)")
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
        
        # Process all loaded image stacks
        for img_idx, img_stack in enumerate(list_images):
            image_stack = img_stack.astype(np.uint16)
            print(f'Processing file {dv_files[img_idx]} with shape: {image_stack.shape}')
            
            # === KEY FIX: Check dimensions FIRST, then assign based on what's requested ===
            if image_stack.ndim == 2:
                # 2D image - assign to brightfield ONLY if requested
                if brightfield is not None:
                    bf = image_stack
                    print("Loaded 2D brightfield image")
                
            elif image_stack.ndim == 4:
                # 4D stack [C, Z, Y, X] - assign color channels ONLY if requested
                image_colors = image_stack
                
                cy5_index = _channel_index('Cy5', 0)
                if Cy5 is not None and cy5_index is not None and image_colors.shape[0] > cy5_index:
                    Cy5_array = image_colors[cy5_index, :, :, :]
                    image_Cy5 = np.max(Cy5_array, axis=0)
                
                mcherry_index = _channel_index('mCherry', 1)
                if mCherry is not None and mcherry_index is not None and image_colors.shape[0] > mcherry_index:
                    mCherry_array = image_colors[mcherry_index, :, :, :]
                    image_mCherry = np.max(mCherry_array, axis=0)
                
                fitc_index = _channel_index('FITC', 2)
                if FITC is not None and fitc_index is not None and image_colors.shape[0] > fitc_index:
                    FITC_array = image_colors[fitc_index, :, :, :]
                    image_FITC = np.max(FITC_array, axis=0)
                
                dapi_index = _channel_index('DAPI', 3)
                if DAPI is not None and dapi_index is not None and image_colors.shape[0] > dapi_index:
                    nuclei_array = image_colors[dapi_index, :, :, :]
                    image_nuclei = np.max(nuclei_array, axis=0)

                brightfield_index = _channel_index('brightfield', None)
                if brightfield is not None and brightfield_index is not None and image_colors.shape[0] > brightfield_index:
                    bf_stack = image_colors[brightfield_index, :, :, :]
                    bf = np.max(bf_stack, axis=0)
                
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
        cy5_index = _channel_index('Cy5', 0)
        if Cy5 is not None and cy5_index is not None and image_colors.shape[1] > cy5_index:
            Cy5_array = image_colors[:, cy5_index, :, :]
            image_Cy5 = np.max(Cy5_array, axis=0)
        
        mcherry_index = _channel_index('mCherry', 1)
        if mCherry is not None and mcherry_index is not None and image_colors.shape[1] > mcherry_index:
            mCherry_array = image_colors[:, mcherry_index, :, :]
            image_mCherry = np.max(mCherry_array, axis=0)
        
        fitc_index = _channel_index('FITC', 2)
        if FITC is not None and fitc_index is not None and image_colors.shape[1] > fitc_index:
            FITC_array = image_colors[:, fitc_index, :, :]
            image_FITC = np.max(FITC_array, axis=0)
        
        dapi_index = _channel_index('DAPI', 3)
        if DAPI is not None and dapi_index is not None and image_colors.shape[1] > dapi_index:
            nuclei_array = image_colors[:, dapi_index, :, :]
            image_nuclei = np.max(nuclei_array, axis=0)
        
        brightfield_index = _channel_index('brightfield', 4)
        if brightfield is not None and brightfield_index is not None and image_colors.shape[1] > brightfield_index:
            bf_stack = image_colors[:, brightfield_index, :, :]
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


def _image_files_for_path(image_path):
    image_path = Path(image_path)
    if image_path.is_file():
        folder_path = image_path.parent
        filenames = [image_path.name]
    elif image_path.is_dir():
        folder_path = image_path
        filenames = sorted([
            f for f in os.listdir(folder_path)
            if not f.startswith('.') and not f.startswith('_')
        ])
    else:
        print(f"Path does not exist: {image_path}")
        return None, None, []

    dv_files = [f for f in filenames if f.endswith('.dv')]
    nd2_files = [f for f in filenames if f.endswith('.nd2')]
    tiff_files = [f for f in filenames if f.endswith(('.tif', '.tiff'))]

    if dv_files:
        return "dv", folder_path, dv_files
    if nd2_files:
        return "nd2", folder_path, nd2_files
    if tiff_files:
        return "tiff", folder_path, tiff_files
    return None, folder_path, []


def _image_name_from_file(filename, image_type):
    if image_type == "dv":
        if '_R3D_REF.dv' in filename:
            return filename.replace("_R3D_REF.dv", "")
        if '_R3D.dv' in filename:
            return filename.replace("_R3D.dv", "")
        return filename.replace(".dv", "")
    if image_type == "nd2":
        return filename.replace(".nd2", "")
    return filename.replace(".tif", "").replace(".tiff", "")


def _project_channel(stack_data, channel_index, channel_axis):
    if channel_index is None:
        return None, None
    if stack_data.ndim <= channel_axis or stack_data.shape[channel_axis] <= channel_index:
        return None, None
    channel_stack = np.take(stack_data, channel_index, axis=channel_axis)
    channel_image = np.max(channel_stack, axis=0) if channel_stack.ndim == 3 else channel_stack
    return channel_stack, channel_image


def load_images_semantic(config, slice_to_plot=0):
    """Load images using semantic channel roles from a normalized config."""
    output_directory = config["output_directory"]
    image_type, folder_path, image_files = _image_files_for_path(config["input_path"])
    if not image_type:
        print("No supported image format found (.dv, .nd2, .tif, .tiff)")
        return None

    print(f"Detected {image_type.upper()} images")
    image_name = _image_name_from_file(image_files[0], image_type)
    print(f"Image ID: {image_name}\n")

    nuclei_config = config["channels"]["nuclei"]
    brightfield_config = config["channels"]["brightfield"]
    rna_configs = config["channels"]["rna"]

    bf = None
    image_nuclei = None
    nuclei_array = None
    loaded_rna = [
        {
            **rna,
            "array": None,
            "image": None,
            "spots": None,
            "clusters": None,
            "counts": None,
        }
        for rna in rna_configs
    ]
    image_FITC = None
    FITC_array = None
    grid_width, grid_height = (80, 60) if image_type == "nd2" else (80, 80)

    if image_type == "dv":
        for filename in image_files:
            image_stack = stack.read_dv(os.path.join(folder_path, filename)).astype(np.uint16)
            print(f'Processing file {filename} with shape: {image_stack.shape}')

            if image_stack.ndim == 2:
                if brightfield_config["name"] is not None:
                    bf = image_stack
                    print("Loaded 2D brightfield/reference image")
                continue

            if image_stack.ndim == 4:
                for rna in loaded_rna:
                    rna_array, rna_image = _project_channel(image_stack, rna["index"], channel_axis=0)
                    if rna_array is not None:
                        rna["array"] = rna_array
                        rna["image"] = rna_image

                if nuclei_config["name"] is not None:
                    nuclei_array, image_nuclei = _project_channel(image_stack, nuclei_config["index"], channel_axis=0)

                if brightfield_config["name"] is not None and brightfield_config["index"] is not None:
                    _, bf = _project_channel(image_stack, brightfield_config["index"], channel_axis=0)

                print("Loaded 4D channel stack")

    elif image_type == "nd2":
        image_stack = nd2.imread(os.path.join(folder_path, image_files[0]))
        print(f"Image colors shape: {image_stack.shape}\n")

        for rna in loaded_rna:
            rna_array, rna_image = _project_channel(image_stack, rna["index"], channel_axis=1)
            if rna_array is not None:
                rna["array"] = rna_array
                rna["image"] = rna_image

        if nuclei_config["name"] is not None:
            nuclei_array, image_nuclei = _project_channel(image_stack, nuclei_config["index"], channel_axis=1)

        if brightfield_config["name"] is not None:
            _, bf = _project_channel(image_stack, brightfield_config["index"], channel_axis=1)

    else:
        image_stack = tiff.imread(os.path.join(folder_path, image_files[0]))
        print(f"Image shape: {image_stack.shape}")
        if image_stack.ndim == 4:
            for rna in loaded_rna:
                rna_array, rna_image = _project_channel(image_stack, rna["index"], channel_axis=0)
                if rna_array is not None:
                    rna["array"] = rna_array
                    rna["image"] = rna_image
            if nuclei_config["name"] is not None:
                nuclei_array, image_nuclei = _project_channel(image_stack, nuclei_config["index"], channel_axis=0)
            if brightfield_config["name"] is not None:
                _, bf = _project_channel(image_stack, brightfield_config["index"], channel_axis=0)
        elif image_stack.ndim == 3 and loaded_rna:
            loaded_rna[0]["array"] = image_stack
            loaded_rna[0]["image"] = np.max(image_stack, axis=0)
        elif image_stack.ndim == 2 and brightfield_config["name"] is not None:
            bf = image_stack

        plt.figure(figsize=(4, 4))
        if image_stack.ndim == 3:
            total_z = image_stack.shape[0]
            if slice_to_plot >= total_z:
                slice_to_plot = 0
            plt.imshow(image_stack[slice_to_plot], cmap='gray')
            plt.title(f"Slice {slice_to_plot} of {total_z}")
        elif image_stack.ndim == 2:
            plt.imshow(image_stack, cmap='gray')
            plt.title("2D Image")
        plt.axis('off')
        plt.show()

    loaded_rna = [rna for rna in loaded_rna if rna["array"] is not None and rna["image"] is not None]

    overview_images = [(rna["image"], rna["name"]) for rna in loaded_rna]
    if image_nuclei is not None and nuclei_config["name"] is not None:
        overview_images.append((image_nuclei, nuclei_config["name"]))
    if bf is not None and brightfield_config["name"] is not None:
        overview_images.append((bf, brightfield_config["name"]))
    if overview_images:
        fig, ax = plt.subplots(1, len(overview_images), figsize=(4 * len(overview_images), 4))
        if len(overview_images) == 1:
            ax = [ax]
        for i, (img, title) in enumerate(overview_images):
            ax[i].imshow(img, cmap="gray")
            ax[i].set_title(title, size=20)
            ax[i].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, f'channels_{image_name}.png'))
        plt.show()
        plt.close()

    missing_rna = [rna["name"] for rna in rna_configs if rna["name"] not in {loaded["name"] for loaded in loaded_rna}]
    for name in missing_rna:
        print(f"⚠️ RNA channel '{name}' was configured but not loaded. Check its channel index.")
    if nuclei_config["name"] is not None and image_nuclei is None:
        print(f"⚠️ Nuclei channel '{nuclei_config['name']}' was configured but not loaded.")
    if brightfield_config["name"] is not None and bf is None:
        print(f"⚠️ Brightfield/reference channel '{brightfield_config['name']}' was configured but not loaded.")

    return {
        "image_type": image_type,
        "image_name": image_name,
        "bf": bf,
        "image_nuclei": image_nuclei,
        "nuclei_array": nuclei_array,
        "rna_channels": loaded_rna,
        "image_FITC": image_FITC,
        "FITC_array": FITC_array,
        "grid_width": grid_width,
        "grid_height": grid_height,
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
    labeled_mask, num_nuclei = label(masks_nuclei)
    
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
try:
    from sklearn.exceptions import InconsistentVersionWarning
except ImportError:
    class InconsistentVersionWarning(UserWarning):
        pass

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)




# 2-cell classification using Random Forest

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

    if len(features_df) != 2:
        print(f" classify_2cell: Expected 2 cytosol masks, got {len(features_df)}. Skipping classification.")
        return None

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



# 4-cell classification using Random Forest

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

    if len(features_df) != 4:
        print(f"⚠️ classify_4cell: Expected 4 cytosol masks, got {len(features_df)}. Skipping classification.")
        return None

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

def embryo_segmentation(bf, image_nuclei, image_name, output_directory,
                        embryo_diameter=500, nuclei_diameter=70):
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
    labeled_nuc, _ = label(masks_nuclei)
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
    labeled_cyto, _ = label(masks_cytosol.astype(np.uint16))
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


def spot_detection(rna, voxel_size, spot_radius, masks_cytosol,
                   image_name="", rna_channel="", detection_color="red",
                   output_directory=""):
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


def analyze_rna_density(image, masks_cytosol, colormap, mRNA_name, image_name, output_directory):
    """
    Analyze RNA intensity along the embryo AP axis defined by an ellipse.
    """

    # If the image is 3D (z, y, x), perform max projection
    if image.ndim == 3:
        image_proj = np.max(image, axis=0)  # max projection along z-axis
    else:
        image_proj = image

    binary_image = masks_cytosol.astype(np.uint8)

    # Find contours in the binary image
    contours = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    if contours:
        cnt = max(contours, key=cv2.contourArea)

        if len(cnt) >= 5:  # At least 5 points needed to fit an ellipse
            ellipse = cv2.fitEllipse(cnt)
            (xc, yc), (d1, d2), angle = ellipse  # d1 = major axis, d2 = minor axis

            fig, ax = plt.subplots()
            ax.imshow(image_proj, cmap='gray')

            ellipse_patch = patches.Ellipse(
                xy=(xc, yc), width=d1, height=d2, angle=angle,
                edgecolor='r', facecolor='none', linewidth=2
            )
            ax.add_patch(ellipse_patch)

            num_lines = 50
            line_positions = np.linspace(-d2 / 2, d2 / 2, num_lines)
            colormap_values = plt.get_cmap(colormap)(np.linspace(0, 1, num_lines))

            mean_intensities = []

            for i, y in enumerate(line_positions[:-1]):
                x1, y1 = (
                    xc + (d1 / 2) * np.cos(np.deg2rad(angle)) - (y * np.sin(np.deg2rad(angle))),
                    yc + (d1 / 2) * np.sin(np.deg2rad(angle)) + (y * np.cos(np.deg2rad(angle)))
                )
                x2, y2 = (
                    xc - (d1 / 2) * np.cos(np.deg2rad(angle)) - (y * np.sin(np.deg2rad(angle))),
                    yc - (d1 / 2) * np.sin(np.deg2rad(angle)) + (y * np.cos(np.deg2rad(angle)))
                )
                
                # Plot the line segment on the max projection image
                ax.plot([x1, x2], [y1, y2], color=colormap_values[i], linestyle='-', linewidth=0.5)

                line_coords = np.array([[int(round(yc)), int(round(xc))] for yc, xc in zip(np.linspace(y1, y2, num_lines), np.linspace(x1, x2, num_lines))])

                # Make sure line_coords are inside image bounds
                valid_mask = (
                    (line_coords[:, 0] >= 0) & (line_coords[:, 0] < image_proj.shape[0]) &
                    (line_coords[:, 1] >= 0) & (line_coords[:, 1] < image_proj.shape[1])
                )
                valid_coords = line_coords[valid_mask]

                pixel_values = image_proj[valid_coords[:, 0], valid_coords[:, 1]]

                mean_intensity = np.mean(pixel_values) if len(pixel_values) > 0 else 0
                mean_intensities.append(mean_intensity)

            mean_intensities = np.array(mean_intensities)
            max_intensity = np.max(mean_intensities)
            if max_intensity > 0:
                normalized_intensity = mean_intensities / max_intensity
            else:
                normalized_intensity = mean_intensities

            ax.scatter(xc, yc, color='red', s=50, label='Ellipse Center')

            ellipse_plot_path = os.path.join(output_directory, f'{mRNA_name}_ellipse_ROI_{image_name}.png')
            plt.title(f"Ellipse ROI for {mRNA_name}")
            plt.xlabel("X Coordinate")
            plt.ylabel("Y Coordinate")
            ax.set_axis_off()
            plt.legend()
            plt.axis('equal')
            plt.savefig(ellipse_plot_path, bbox_inches='tight', dpi=300)
            plt.close()

            # Plot normalized intensity along AP axis (flip so AB = 0 μm)
            positions = np.linspace(0, 100, len(normalized_intensity))

            fig, ax = plt.subplots()

            for i in range(len(positions)):
                ax.scatter(positions[i], normalized_intensity[i], color=colormap_values[i], s=50,
                           label=f'Grid {i}' if i == 0 else "")

            ax.plot(positions, normalized_intensity, color='gray', linestyle='-', linewidth=1)
            ax.set_xlabel('Position along Body Axis (% distance)')
            ax.set_ylabel('Normalized Mean Pixel Intensity')
            ax.set_title(f'{mRNA_name} Normalized Intensity Along Body Axis')

            # Add minor ticks for precise % counting
            ax.set_xticks(np.arange(0, 101, 10))   # major ticks every 10%
            ax.set_xticks(np.arange(0, 101, 1), minor=True)  # minor ticks every 1%
            ax.tick_params(axis='x', which='minor', length=5, color='k')  # minor tick length
            ax.tick_params(axis='x', which='major', length=10, color='k')  # major tick length

            plt.tight_layout()
            scatter_plot_path = os.path.join(output_directory, f'{mRNA_name}_AP_profile_{image_name}.png')
            plt.savefig(scatter_plot_path, bbox_inches='tight', dpi=300)
            plt.close()

            # Save the raw density data to a CSV
            density_data = pd.DataFrame({
                'Image_ID': image_name,
                'Position (μm)': positions,
                f'{mRNA_name} Normalized density': normalized_intensity
            })
            output_path = os.path.join(output_directory, f'{mRNA_name}_AP_profile_data_{image_name}.csv')
            density_data.to_csv(output_path, index=False)
        else:
            print(f"Not enough points to fit an ellipse for {mRNA_name}.")
    else:
        print(f"No contours found in the mask for {mRNA_name}.")


def line_scan(image, masks_cytosol, colormap, mRNA_name, image_name, output_directory, run_cell_classifier=False, features_df=None, df_long=None):
    """
    Analyze RNA intensity along the body axis with cell area normalized shading.
    """

    
    # If the image is 3D (z, y, x), perform max projection
    if image.ndim == 3:
        image_proj = np.max(image, axis=0)  # max projection along z-axis
    else:
        image_proj = image

    binary_image = masks_cytosol.astype(np.uint8)

    # Find contours in the binary image
    contours = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    if contours:
        cnt = max(contours, key=cv2.contourArea)

        if len(cnt) >= 5:  # At least 5 points needed to fit an ellipse
            ellipse = cv2.fitEllipse(cnt)
            (xc, yc), (d1, d2), angle = ellipse  # d1 = major axis, d2 = minor axis

            fig, ax = plt.subplots()
            ax.imshow(image_proj, cmap='gray')

            ellipse_patch = patches.Ellipse(xy=(xc, yc), width=d1, height=d2, angle=angle,
                                            edgecolor='r', facecolor='none', linewidth=2)
            ax.add_patch(ellipse_patch)

            rect_length = d1 * 0.3
            rect_width = d2 * 1

            rotated_rect = ((xc, yc), (rect_width, rect_length), angle + 90)
            box_points = cv2.boxPoints(rotated_rect)
            box_points = np.intp(box_points)

            rectangle_patch = patches.Polygon(
                box_points,
                closed=True,
                edgecolor='yellow',
                facecolor='none',
                linewidth=2,
                linestyle='--',
                label='Minor Axis ROI'
            )
            ax.add_patch(rectangle_patch)

            # Create a Path object from the rectangle to check if points lie inside
            rect_path = MPLPath(box_points)

            num_lines = 50
            line_positions = np.linspace(-d2 / 2, d2 / 2, num_lines)

            colormap_values = plt.get_cmap(colormap)(np.linspace(0, 1, num_lines))

            mean_intensities = []

            for i, y in enumerate(line_positions[:-1]):
                x1, y1 = (
                    xc + (d1 / 2) * np.cos(np.deg2rad(angle)) - (y * np.sin(np.deg2rad(angle))),
                    yc + (d1 / 2) * np.sin(np.deg2rad(angle)) + (y * np.cos(np.deg2rad(angle)))
                )
                x2, y2 = (
                    xc - (d1 / 2) * np.cos(np.deg2rad(angle)) - (y * np.sin(np.deg2rad(angle))),
                    yc - (d1 / 2) * np.sin(np.deg2rad(angle)) + (y * np.cos(np.deg2rad(angle)))
                )

                # Clip the line segment to rectangle ROI
                n_points = 100
                xs = np.linspace(x1, x2, n_points)
                ys = np.linspace(y1, y2, n_points)
                points = np.vstack((xs, ys)).T

                inside_mask = rect_path.contains_points(points)
                if not any(inside_mask):
                    continue  # Skip if no points inside ROI

                inside_points = points[inside_mask]
                clip_x1, clip_y1 = inside_points[0]
                clip_x2, clip_y2 = inside_points[-1]

                ax.plot([clip_x1, clip_x2], [clip_y1, clip_y2], color=colormap_values[i], linestyle='-', linewidth=0.5)

                line_coords = np.array([[int(round(yc)), int(round(xc))] for yc, xc in zip(np.linspace(clip_y1, clip_y2, num_lines), np.linspace(clip_x1, clip_x2, num_lines))])

                # Make sure line_coords are inside image bounds
                valid_mask = (
                    (line_coords[:, 0] >= 0) & (line_coords[:, 0] < image_proj.shape[0]) &
                    (line_coords[:, 1] >= 0) & (line_coords[:, 1] < image_proj.shape[1])
                )
                valid_coords = line_coords[valid_mask]

                pixel_values = image_proj[valid_coords[:, 0], valid_coords[:, 1]]

                mean_intensity = np.mean(pixel_values) if len(pixel_values) > 0 else 0
                mean_intensities.append(mean_intensity)

            mean_intensities = np.array(mean_intensities)
            max_intensity = np.max(mean_intensities)
            if max_intensity > 0:
                normalized_intensity = mean_intensities / max_intensity
            else:
                normalized_intensity = mean_intensities

            ax.scatter(xc, yc, color='red', s=50, label='Ellipse Center')

            ellipse_plot_path = os.path.join(output_directory, f'{mRNA_name}_line_ROI_{image_name}.png')
            plt.title(f"Line scan for {mRNA_name}")
            plt.xlabel("X Coordinate")
            plt.ylabel("Y Coordinate")
            ax.set_axis_off()
            plt.legend()
            plt.axis('equal')
            plt.savefig(ellipse_plot_path, bbox_inches='tight', dpi=300)
            plt.show()
            plt.close()

            # Dynamically set AP axis positions using rel_pos_minor (flip so AB = 0 μm)
            positions = np.linspace(0, 100, len(normalized_intensity))

            # Dynamically decide whether to flip based on AB orientation
            if run_cell_classifier and features_df is not None:
                if 'highest_confidence_label' in features_df.columns:
                    ab_row = features_df[
                        features_df['highest_confidence_label'].isin(['AB', 'ABa'])
                    ]
                    if not ab_row.empty:
                        ab_orientation = ab_row['rel_pos_minor'].values[0]
                        if ab_orientation > 0:  # AB is on left, need to flip
                            positions = -positions  # temporarily -100 → 0
                            positions = positions - positions.min()  # shift so min = 0 → 0 → 100
            
            # === Line scan plot WITHOUT cell area shading ===
            fig, ax = plt.subplots()

            # Compute AP-axis positions
            ap_positions = []
            if features_df is not None:
                total_major = features_df['area'].sum()
                current_pos = 0
                for _, row in features_df.iterrows():
                    frac = row['area'] / total_major * 100
                    start = current_pos
                    end = current_pos + frac
                    ap_positions.append((start, end, row['highest_confidence_label']))
                    current_pos = end

            # Plot normalized intensity points and line
            for i in range(len(positions)):
                ax.scatter(
                    positions[i],
                    normalized_intensity[i],
                    color=colormap_values[i],
                    s=50,
                    label=f'Grid {i}' if i == 0 else ""
                )

            ax.plot(positions, normalized_intensity, color='gray', linestyle='-', linewidth=1)
            ax.set_xlabel('Position along Body Axis (% distance)')
            ax.set_ylabel('Normalized Mean Pixel Intensity')
            ax.set_title(f'{mRNA_name} Normalized Intensity Along Body Axis')

            # Add minor ticks for precise % counting
            ax.set_xticks(np.arange(0, 101, 10))   # major ticks every 10%
            ax.set_xticks(np.arange(0, 101, 1), minor=True)  # minor ticks every 1%
            ax.tick_params(axis='x', which='minor', length=5, color='k')
            ax.tick_params(axis='x', which='major', length=10, color='k')

            plt.tight_layout()
            scatter_plot_path = os.path.join(
                output_directory, f'{mRNA_name}_line_scan_{image_name}.png'
            )
            plt.savefig(scatter_plot_path, bbox_inches='tight', dpi=300)
            plt.show()
            # plt.close()

            # #  -------- Line scan with cell area normalized shaded -------- #
            # if run_cell_classifier and features_df is not None and df_long is not None:
            #     fig, ax = plt.subplots()

            #     # Draw shaded regions for each cell
            #     for start, end, label in ap_positions:
            #         color = 'C0' if label == 'AB' or label == 'ABa' else 'C1'
            #         ax.axvspan(start, end, color=color, alpha=0.2)
                    
            #     # Annotate each cell with label
            #     for _, row in df_long.iterrows():
            #         label = row['label']
            #         start, end = None, None

            #         # Find start/end from AP-axis positions
            #         for s, e, l in ap_positions:
            #             if l == label:
            #                 start, end = s, e
            #                 break

            #         if start is not None and end is not None:
            #             mid = (start + end) / 2
            #             # Cell label on top
            #             ax.text(mid, 0.9, f"{label}", ha='center', va='bottom', fontsize=20, fontweight='bold',
            #                     color='k', transform=ax.get_xaxis_transform())

            #     # Scatter + line plot
            #     for i in range(len(positions)):
            #         ax.scatter(positions[i], normalized_intensity[i], color=colormap_values[i], s=50,
            #                    label=f'Grid {i}' if i == 0 else "")

            #     ax.plot(positions, normalized_intensity, color='gray', linestyle='-', linewidth=1)
            #     ax.set_xlabel('Position along Body Axis (% distance)')
            #     ax.set_ylabel('Normalized Mean Pixel Intensity')
            #     ax.set_title(f'{mRNA_name} Normalized Intensity Along Body Axis')

            #     # Add minor ticks for precise % counting
            #     ax.set_xticks(np.arange(0, 101, 10))   # major ticks every 10%
            #     ax.set_xticks(np.arange(0, 101, 1), minor=True)  # minor ticks every 1%
            #     ax.tick_params(axis='x', which='minor', length=5, color='k')  # minor tick length
            #     ax.tick_params(axis='x', which='major', length=10, color='k')  # major tick length

            #     plt.tight_layout()
            #     scatter_plot_path = os.path.join(output_directory, f'{mRNA_name}_line_scan_shaded_{image_name}.png')
            #     plt.savefig(scatter_plot_path, bbox_inches='tight', dpi=300)
            #     plt.close()

            # Save the raw density data to a CSV using both names
            density_data = pd.DataFrame({
                'Image_ID': image_name,
                'Position (μm)': positions,
                f'{mRNA_name} Normalized density': normalized_intensity
            })
            output_path_scan = os.path.join(output_directory, f'{mRNA_name}_line_scan_data_{image_name}.csv')
            density_data.to_csv(output_path_scan, index=False)
            output_path_density = os.path.join(output_directory, f'{mRNA_name}_line_density_data_{image_name}.csv')
            density_data.to_csv(output_path_density, index=False)

        else:
            print(f"Not enough points to fit an ellipse for {mRNA_name}.")
    else:
        print(f"No contours found in the mask for {mRNA_name}.")


def save_spot_quantification(spot_counts_dict, image_name, output_directory, 
                             features_df=None, run_cell_classifier=False):
    """
    Save mRNA spot quantification to CSV files (total and per-cell/region counts).
    
    Parameters:
    -----------
    spot_counts_dict : dict
        Dictionary mapping channel names to list of spot counts per cell/region.
        Example: {'set3_mRNA': [10, 20, 15], 'erm1_mRNA': [30, 25, 28]}
    image_name : str
        Name of the image for output filenames
    output_directory : str or Path
        Directory to save CSV files
    features_df : pd.DataFrame, optional
        DataFrame with cell classification results (has 'highest_confidence_label' and 'prediction_confidence' columns)
    run_cell_classifier : bool, optional
        Whether cell classifier was run (affects output filename prefix)
    
    Returns:
    --------
    tuple : (df_quantification, df_long)
        DataFrames for total and per-cell/region counts
    """
    if not spot_counts_dict or all(not counts for counts in spot_counts_dict.values()):
        print("No spot counts to quantify.")
        return None, None
    
    output_directory = str(output_directory)
    
    # Wide format: total abundance
    data_wide = {'Image ID': image_name}
    for channel_name, counts in spot_counts_dict.items():
        if counts:
            data_wide[f"{channel_name} total molecules"] = sum(counts)
    
    df_quantification = pd.DataFrame([data_wide])
    quantification_output = os.path.join(output_directory, f'total_mRNA_counts_{image_name}.csv')
    df_quantification.to_csv(quantification_output, index=False)
    print(f"Total mRNA counts saved to {quantification_output}")
    print(df_quantification)
    
    # Long format: per-cell/region counts
    num_regions = max((len(counts) for counts in spot_counts_dict.values() if counts), default=0)
    
    if num_regions > 0:
        rows_long = []
        for i in range(num_regions):
            row = {'Image ID': image_name, 'region_id': i + 1}
            
            # Add spot counts for each channel
            for channel_name, counts in spot_counts_dict.items():
                row[channel_name] = counts[i] if i < len(counts) else 0
            
            # Add classification info if available
            if features_df is not None and run_cell_classifier:
                row['label'] = features_df.at[i, "highest_confidence_label"]
                row['confidence'] = round(features_df.at[i, "prediction_confidence"], 3)
            
            rows_long.append(row)
        
        df_long = pd.DataFrame(rows_long)
        
        # Save with appropriate prefix
        output_prefix = "per_cell" if run_cell_classifier and features_df is not None else "per_region"
        long_output = os.path.join(output_directory, f'{output_prefix}_mRNA_counts_{image_name}.csv')
        df_long.to_csv(long_output, index=False)
        print(f"\n{output_prefix.title()} mRNA counts saved to {long_output}")
        print(df_long)
        
        return df_quantification, df_long
    
    return df_quantification, None


def heatmap(spots, max_proj, channel_name, masks_cytosol, grid_width, grid_height, 
                         image_name, output_directory, vmin=0, vmax=None, normalize_scale=False):
    """
    Create a grid-based heatmap visualization of RNA spots.
    
    Parameters:
    -----------
    spots : ndarray
        3D coordinates of detected spots (z, y, x)
    max_proj : ndarray
        2D or 3D image array to project for visualization
    channel_name : str
        Name of the RNA channel (e.g., "set3_mRNA")
    masks_cytosol : ndarray
        Segmentation mask to determine grid dimensions
    grid_width : int
        Number of grid cells in x-direction (default: 80)
    grid_height : int
        Number of grid cells in y-direction (default: 80)
    image_name : str
        Name of the image for output filename
    output_directory : str
        Directory to save heatmap PNG
    vmin : float, optional
        Minimum value for color scale (default: 0)
    vmax : float, optional
        Maximum value for color scale; if None, uses max value in grid
    normalize_scale : bool, optional
        If True, scale the heatmap values to the range 0-1 before plotting.
    
    Returns:
    --------
    None (saves PNG and displays plot)
    """
    img_width, img_height = masks_cytosol.shape[1], masks_cytosol.shape[0]
    cell_w = img_width / grid_width
    cell_h = img_height / grid_height
    grid = np.zeros((grid_height, grid_width), dtype=int)
    
    if spots is not None and len(spots) > 0:
        for spot in spots:
            z, y, x = spot[:3]
            cell_x = int(x / cell_w)
            cell_y = int(y / cell_h)
            if 0 <= cell_x < grid_width and 0 <= cell_y < grid_height:
                grid[cell_y, cell_x] += 1
    
    # Ensure max_proj is 2D
    if max_proj.ndim > 2:
        max_proj_2d = max_proj.max(axis=0)
    else:
        max_proj_2d = max_proj
    
    # Optionally normalize the heatmap scale so different channels are comparable
    if normalize_scale:
        grid_plot = grid.astype(float)
        max_value = grid_plot.max()
        if max_value > 0:
            grid_plot = grid_plot / max_value
        if vmax is None:
            vmax = 1.0
    else:
        grid_plot = grid
        if vmax is None:
            vmax = grid.max() if grid.max() > 0 else 1
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(max_proj_2d, cmap='gray')
    axs[0].set_title(f"{channel_name} Max Projection")
    axs[0].axis('off')
    
    im = axs[1].imshow(grid_plot, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
    axs[1].set_title(f"{channel_name} Heatmap")
    cbar = fig.colorbar(im, ax=axs[1], label="Spot Count")
    axs[1].axis('off')
    
    plt.tight_layout()
    heatmap_path = os.path.join(output_directory, f'{channel_name}_heatmap_{image_name}.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {heatmap_path}")

# Backwards compatibility: keep old name pointing to new function
create_local_heatmap = heatmap


def generate_pdf_report(image_name, output_directory):
    """
    Generate a PDF report from PNG/CSV outputs in a pipeline output directory.

    Parameters:
    -----------
    image_name : str
        Image identifier shown in the report header.
    output_directory : str or Path
        Directory containing analysis outputs (PNG and CSV files).

    Returns:
    --------
    str
        Path to the generated PDF report.
    """
    output_directory = str(output_directory)
    output_pdf_path = os.path.join(output_directory, "report.pdf")

    # Collect images and CSVs sorted by creation time
    output_file_paths = []
    for filename in os.listdir(output_directory):
        if filename.lower().endswith((".png", ".csv")):
            output_file_paths.append(os.path.join(output_directory, filename))
    sorted_files = sorted(output_file_paths, key=lambda f: os.path.getctime(f))

    c = canvas.Canvas(output_pdf_path, pagesize=letter)
    c.setFont("Times-Roman", 16)
    c.drawString(32, 728, f"{image_name}")
    c.setFont("Times-Roman", 14)
    c.drawString(32, 713, f"Report Generated: {datetime.now().date()}")

    def draw_csv_table(file_path, canv, margin, current_y, padding):
        with open(file_path, newline="") as csvFile:
            reader = csv.reader(csvFile)
            data = list(reader)
        
        # Format values to 3 decimal places for floating points
        formatted_data = []
        for row in data:
            formatted_row = []
            for val in row:
                try:
                    formatted_row.append(f"{float(val):.3f}")
                except ValueError:
                    formatted_row.append(val)
            formatted_data.append(formatted_row)

        table = Table(formatted_data)
        table.setStyle(TableStyle([
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Times-Roman"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black)
        ]))
        tableWidth, tableHeight = table.wrapOn(canv, 400, 600)
        current_y -= tableHeight + padding
        table.drawOn(canv, margin, current_y)
        return current_y

    def get_image_size(file_path):
        with Image.open(file_path) as img:
            w, h = img.size
            aspect = w / h
        width = 145.6 * aspect
        if width > 548:
            height = 548 / aspect
            width = 548
        else:
            height = 145.6
        return height, width

    def draw_page_number(canv):
        canv.setFont("Times-Roman", 10)
        canv.drawRightString(580, 32, f"{canv.getPageNumber()}")

    current_y = 700
    padding = 20
    
    for file_path in sorted_files:
        if file_path.endswith(".png"):
            h, w = get_image_size(file_path)
            title = os.path.basename(file_path)
            if current_y >= h + padding + 15:
                c.setFont("Times-Roman", 12)
                c.drawString(32, current_y - 15, title)
                c.drawImage(file_path, 32, current_y - h - padding, w, h)
                current_y -= h + padding + 10
            else:
                c.showPage()
                c.setFont("Times-Roman", 16)
                c.drawString(32, 728, f"{image_name}")
                current_y = 710
                c.setFont("Times-Roman", 12)
                c.drawString(32, current_y, title)
                c.drawImage(file_path, 32, current_y - h - padding, w, h)
                current_y -= h + padding + 20
                draw_page_number(c)
        elif file_path.endswith(".csv"):
            c.setFont("Times-Roman", 12)
            c.drawString(32, current_y - 15, os.path.basename(file_path))
            current_y -= 15
            current_y = draw_csv_table(file_path, c, 32, current_y, padding)

    # Dynamically find the .out file (log) and write to report
    out_file = None
    for item in os.listdir(output_directory):
        if item.lower().endswith(".out"):
            out_file = os.path.join(output_directory, item)
            break

    if out_file and os.path.isfile(out_file):
        with open(out_file, 'r') as f:
            lines = f.readlines()

        req_space = len(lines) * 12 + 40
        if current_y < req_space:
            c.showPage()
            current_y = 720
            c.setFont("Times-Roman", 16)
            c.drawString(32, 728, f"{image_name}")

        c.setFont("Times-Roman", 12)
        c.drawString(32, current_y, os.path.basename(out_file))
        current_y -= 16

        text_data = c.beginText(32, current_y)
        text_data.setFont("Times-Roman", 10)
        for line in lines:
            text_data.textLine(line.strip())
        c.drawText(text_data)

    c.save()
    print("PDF report successfully generated.")
    return output_pdf_path


if __name__ == "__main__":


    print("Running WormLib CLI pipeline...")

    parser = argparse.ArgumentParser(description="Run the WormLib image-analysis pipeline.")
    parser.add_argument("input_path", nargs="?", help="Input image file or folder. Overrides config input.path.")
    parser.add_argument("output_directory", nargs="?", help="Output directory. Overrides config input.output_directory.")
    parser.add_argument("-c", "--config", help="YAML config file with semantic channel roles.")
    args = parser.parse_args()

    config = load_pipeline_config(args.config, cli_input=args.input_path, cli_output=args.output_directory)
    folder_name = config["input_path"]
    output_directory_path = config["output_directory"]

    if not folder_name or not output_directory_path:
        print("ERROR: Provide input and output paths either as arguments, in --config, or via FOLDER_NAME/OUTPUT_DIRECTORY.")
        sys.exit(1)

    output_directory = Path(output_directory_path)
    output_directory.mkdir(parents=True, exist_ok=True)

    print(f"Input Directory: {folder_name}")
    print(f"Output Directory: {output_directory}")

    voxel_size = config["voxel_size"]
    embryo_diameter = config["segmentation"]["embryo_diameter"]
    nuclei_diameter = config["segmentation"]["nuclei_diameter"]
    cell_diameter = config["segmentation"]["cell_diameter"]
    run_embryo_segmentation = config["pipeline"]["embryo_segmentation"]
    run_cell_segmentation = config["pipeline"]["cell_segmentation"]
    run_cell_classifier = config["pipeline"]["cell_classification"]
    run_spot_detection = config["pipeline"]["spot_detection"]
    run_mRNA_heatmaps = config["pipeline"]["heatmaps"]
    run_rna_density_analysis = config["pipeline"]["rna_density"]
    run_line_scan_analysis = config["pipeline"]["line_scan"]

    # Print parsed settings
    print(f"\nMicroscope Settings:\n  Voxel Size: {voxel_size}")
    print("\nChannel Roles:")
    print(f"  nuclei: {config['channels']['nuclei']['name']} (index {config['channels']['nuclei']['index']})")
    print(f"  brightfield/reference: {config['channels']['brightfield']['name']} (index {config['channels']['brightfield']['index']})")
    for rna in config["channels"]["rna"]:
        print(f"  RNA: {rna['name']} (index {rna['index']}, spot radius {rna['spot_radius']})")

    # Determine model paths relative to src directory
    src_dir = Path(__file__).resolve().parent
    main_dir = src_dir.parent
    model_2cell_path = main_dir / "models/2-cell_classification_RFmodel.joblib"
    model_4cell_path = main_dir / "models/4-cell_classification_RFmodel.joblib"

    # 3. Load Images using semantic channel roles
    loaded_result = load_images_semantic(config, slice_to_plot=0)
    if loaded_result is None:
        print("ERROR: Image loading failed. Exiting.")
        sys.exit(1)

    image_type = loaded_result['image_type']
    image_name = loaded_result['image_name']
    bf = loaded_result['bf']
    image_FITC = loaded_result['image_FITC']
    image_nuclei = loaded_result['image_nuclei']
    FITC_array = loaded_result['FITC_array']
    nuclei_array = loaded_result['nuclei_array']
    rna_channels = loaded_result['rna_channels']
    grid_width = loaded_result['grid_width']
    grid_height = loaded_result['grid_height']

    print(f"\nSuccessfully loaded image: {image_name} (Format: {image_type.upper()})")

    def first_analysis_shape():
        for rna in rna_channels:
            if rna["array"] is not None:
                return rna["array"].shape[-2:]
        for array in (FITC_array, nuclei_array):
            if array is not None:
                return array.shape[-2:]
        for image in (image_FITC, image_nuclei, bf):
            if image is not None:
                return image.shape[-2:]
        return None

    # 4. Cell Segmentation & Classification
    masks_cytosol, masks_nuclei = None, None
    cell_stage = "no-nuclei"
    features_df = None
    fallback_to_embryo = False

    if run_cell_segmentation and bf is not None and image_nuclei is not None:
        print("\nRunning single-cell segmentation...")
        try:
            masks_cytosol, masks_nuclei, _, _ = segmentation(bf, image_nuclei, second_image_cytosol=image_nuclei, output_directory=output_directory)
            
            # Save cell segmentation plot
            segmentation_filename = os.path.join(output_directory, f'cell_segmentation_{image_name}.png')
            plt.savefig(segmentation_filename, bbox_inches='tight')
            plt.close()
            print("Cell segmentation outlines saved.")

            # Filter nuclei and determine stage
            cell_stage, nuclei_sizes, masks_filtered = get_cell_stage_and_size_filtered(masks_nuclei, voxel_size)
            print(f"Blastomere Stage: {cell_stage}")

            # Run classifier
            if run_cell_classifier:
                classifier_attempted = False
                if cell_stage == "2-cell" and model_2cell_path.exists():
                    print("Running 2-cell blatomere classifier...")
                    classifier_attempted = True
                    features_df = classify_2cell(masks_cytosol, bf, image_name, output_directory, model_path=str(model_2cell_path), verbose=True)
                elif cell_stage == "4-cell" and model_4cell_path.exists():
                    print("Running 4-cell blastomere classifier...")
                    classifier_attempted = True
                    features_df = classify_4cell(masks_cytosol, bf, image_name, output_directory, model_path=str(model_4cell_path), verbose=True)
                else:
                    print(f"Skipping classifier: Stage '{cell_stage}' is not supported or models are missing.")
                    fallback_to_embryo = True

                if classifier_attempted and features_df is None:
                    fallback_to_embryo = True

                if features_df is None:
                    run_cell_classifier = False
        except Exception as e:
            print(f"⚠️ Cell segmentation failed ({e}). Falling back to whole-embryo segmentation.")
            cell_stage = "no-nuclei"
            fallback_to_embryo = True
            masks_cytosol = None
    elif run_cell_segmentation and bf is None and image_nuclei is not None:
        print("\nRunning nuclear-only segmentation because no brightfield/reference image was loaded.")
        masks_nuclei = nuclear_segmentation(image_nuclei)
        masks_cytosol = masks_nuclei
        cell_stage = "nuclei-only"
        run_cell_classifier = False

        tiff.imwrite(os.path.join(output_directory, "masks_nuclei.tif"), masks_nuclei.astype(masks_nuclei.dtype))
        tiff.imwrite(os.path.join(output_directory, "masks_cytosol.tif"), masks_cytosol.astype(masks_cytosol.dtype))

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(image_nuclei, cmap="gray")
        ax[0].set_title("DAPI")
        ax[0].axis("off")
        ax[1].imshow(masks_nuclei, cmap="viridis")
        ax[1].set_title("Nuclear masks")
        ax[1].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, f'nuclear_segmentation_{image_name}.png'), bbox_inches='tight')
        plt.close()

    # Fallback to embryo segmentation if cell stage couldn't be determined or cell segmentation was skipped
    if cell_stage == "no-nuclei" or masks_cytosol is None or fallback_to_embryo:
        if (run_embryo_segmentation or fallback_to_embryo) and bf is not None and image_nuclei is not None:
            print("\nRunning fallback whole-embryo segmentation...")
            masks_cytosol, masks_nuclei, _, _ = embryo_segmentation(
                bf, image_nuclei, image_name, output_directory,
                embryo_diameter=embryo_diameter, nuclei_diameter=nuclei_diameter)
            run_cell_classifier = False
            features_df = None
        else:
            if fallback_to_embryo:
                masks_cytosol = None
            print("Skipping segmentation.")

    if run_spot_detection and masks_cytosol is None:
        analysis_shape = first_analysis_shape()
        if analysis_shape is not None:
            print("\nNo segmentation mask available; using a whole-image mask for spot detection.")
            masks_cytosol = np.ones(analysis_shape, dtype=np.uint16)
            run_cell_classifier = False
            features_df = None

    # 5. Spot Detection
    if run_spot_detection and masks_cytosol is not None:
        print("\nRunning smFISH spot detection...")
        for rna in rna_channels:
            print(f"Detecting {rna['name']} molecules...")
            spots, clusters, counts, cluster_counts = spot_detection(
                rna["array"], voxel_size, rna["spot_radius"], masks_cytosol,
                image_name=image_name, rna_channel=rna["name"],
                detection_color=rna["detection_color"], output_directory=output_directory)
            rna["spots"] = spots
            rna["clusters"] = clusters
            rna["counts"] = counts
            rna["cluster_counts"] = cluster_counts

        # 6. Save Spot Abundance Tables (CSVs)
        detected_rnas = [rna for rna in rna_channels if rna.get("counts") is not None]
        if detected_rnas:
            # Wide format: total abundance
            data_wide = {'Image ID': image_name}
            for rna in detected_rnas:
                data_wide[f"{rna['name']} total molecules"] = sum(rna["counts"])
            df_quantification = pd.DataFrame([data_wide])
            quantification_output = os.path.join(output_directory, f'total_mRNA_counts_{image_name}.csv')
            df_quantification.to_csv(quantification_output, index=False)
            print(f"Total abundance CSV saved at {quantification_output}")

            num_regions = max(len(rna["counts"]) for rna in detected_rnas)

            if num_regions > 0:
                rows_long = []
                for i in range(num_regions):
                    row = {'Image ID': image_name, 'region_id': i + 1}
                    for rna in detected_rnas:
                        row[rna["name"]] = rna["counts"][i] if i < len(rna["counts"]) else 0

                    if run_cell_classifier and features_df is not None:
                        row['label'] = features_df.at[i, "highest_confidence_label"]
                        row['confidence'] = round(features_df.at[i, "prediction_confidence"], 3)

                    rows_long.append(row)

                df_long = pd.DataFrame(rows_long)

                if run_cell_classifier and features_df is not None:
                    # Write both file formats to guarantee compatibility with all external scripts
                    long_output_path1 = os.path.join(output_directory, f'per_cell_mRNA_counts_{image_name}.csv')
                    long_output_path2 = os.path.join(output_directory, f'quantification_cell_{image_name}.csv')
                    df_long.to_csv(long_output_path1, index=False)
                    df_long.to_csv(long_output_path2, index=False)
                    print(f"Per-cell abundance CSVs saved at {long_output_path1} and {long_output_path2}")
                else:
                    region_output_path = os.path.join(output_directory, f'per_region_mRNA_counts_{image_name}.csv')
                    df_long.to_csv(region_output_path, index=False)
                    print(f"Per-region abundance CSV saved at {region_output_path}")

    # 7. Spatial Analysis of mRNA
    if masks_cytosol is not None:
        if run_mRNA_heatmaps:
            print("\nGenerating mRNA heatmaps...")

            
            # Helper to create heatmaps (local inline implementation)
            def heatmap(spots, max_proj, title, channel_name):
                img_width, img_height = masks_cytosol.shape[1], masks_cytosol.shape[0]
                cell_w = img_width / grid_width
                cell_h = img_height / grid_height
                grid = np.zeros((grid_height, grid_width), dtype=int)
                
                if spots is not None:
                    for spot in spots:
                        z, y, x = spot[:3]
                        cell_x = int(x / cell_w)
                        cell_y = int(y / cell_h)
                        if 0 <= cell_x < grid_width and 0 <= cell_y < grid_height:
                            grid[cell_y, cell_x] += 1

                fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                axs[0].imshow(max_proj, cmap='gray')
                axs[0].set_title(f"{channel_name} Max Projection")
                axs[0].axis('off')
                
                im = axs[1].imshow(grid, cmap='hot', interpolation='nearest')
                axs[1].set_title(f"{channel_name} Heatmap")
                fig.colorbar(im, ax=axs[1], shrink=0.7)
                axs[1].axis('off')
                
                plt.tight_layout()
                heatmap_path = os.path.join(output_directory, f'{channel_name}_heatmap_{image_name}.png')
                plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                plt.close()

            for rna in rna_channels:
                if rna.get("image") is not None:
                    heatmap(rna.get("spots"), rna["image"], f"{rna['name']} Heatmap", rna["name"])

        if run_rna_density_analysis:
            print("\nGenerating RNA density profiles along AP axis...")
            for rna in rna_channels:
                if rna.get("image") is not None:
                    analyze_rna_density(rna["image"], masks_cytosol, rna["colormap"], rna["name"], image_name, output_directory)

        if run_line_scan_analysis:
            print("\nGenerating Line Scan intensity profiles...")
            for rna in rna_channels:
                if rna.get("image") is not None:
                    # Retrieve df_long from locals safely if it exists
                    l_df = locals().get('df_long', None)
                    line_scan(rna["image"], masks_cytosol, rna["colormap"], rna["name"], image_name, output_directory,
                              run_cell_classifier=run_cell_classifier, features_df=features_df, df_long=l_df)

    # 8. Export PDF Report
    print("\nExporting final PDF data report...")
    generate_pdf_report(image_name, str(output_directory))
    print("Pipeline run completed successfully.")
