import os
import os.path as op
import nibabel as nib
import numpy as np
import subprocess
import shutil
import glob
import pandas as pd

def get_skull_stripped_anatomical(bids_root, preproc_root, subject_id, robust=False): 
    """
    Function to perform skull-stripping (removing the skull around the brain).
    This is a simple wrapper around the brain extraction tool (BET) in FSL's suite.
    It assumes data to be in the BIDS format.

    The method also saves the brain mask which was used to extract the brain.

    The brain extraction is conducted only on the T1w of the participant.

    Parameters
    ----------
    bids_root: string
        The root of the BIDS directory.
    preproc_root: string
        The root of the preprocessed data, where the result of the brain extraction will be saved.
    subject_id: string
        Subject ID, the subject on which brain extraction should be conducted.
    robust: bool
        Whether to conduct robust center estimation with BET or not. Default is False.
    """

    # Format the subject ID following the BIDS convention (e.g., 'sub-001')
    subject_dir = 'sub-{}'.format(subject_id) 
    
    # Define the path to the anatomical T1-weighted image for the subject
    anatomical_path = op.join(bids_root, subject_dir, 'anat', 'sub-{}_T1w.nii.gz'.format(subject_id)) 

    # Define the path where the skull-stripped brain will be saved
    betted_brain_path = op.join(preproc_root, subject_dir, 'anat', 'sub-{}_T1w'.format(subject_id)) 

    # Run FSL's brain extraction tool (BET) using the os.system command
    # - If 'robust' is True, the '-R' option is used for robust center estimation
    # - Otherwise, no additional flag is used
    os.system('bet {} {} -m {}'.format(anatomical_path, betted_brain_path, '-R' if robust else ''))

    # Print a message indicating the process is complete
    print("Done with BET.")

def apply_python_mask_approach(img_path, mask_path, masked_img_path):
    """
    The first approach, Pythonic way. The goal is, given a mask, to apply it to a T1 image where the brain is to be extracted.

    Parameters
    ----------
    img_path: str
        Path to the image on which we would like to apply the mask (in this case, the T1 with the skull still on). Should be a .nii.gz file.
    mask_path: str
        Path to the mask you would like to apply to your image. Should be a .nii.gz file, containing only binary values (0 or 1).
    masked_img_path: str
        Path to which the resulting masked image will be saved. 
    """

    # Load both the T1 image and the mask from disk
    img = nib.load(img_path)
    mask = nib.load(mask_path)
    
    # Extract the image data and mask data as numpy arrays. 
    img_data = img.get_fdata()
    mask_data = mask.get_fdata()

    #######################
    # Solution 1
    # Create an empty image array and fill it with values where the mask is greater than 0
    #######################
    # Initialize an empty array with the same shape as the T1 image
    saved_img_data = np.zeros(img_data.shape)
    # Fill the array with image data only where the mask is greater than 0 (brain region)
    saved_img_data[mask_data > 0] = img_data[mask_data > 0]

    # Save the masked image to disk by creating a new Nifti image and writing it out
    img_out = nib.Nifti1Image(saved_img_data, img.affine, img.header)
    nib.save(img_out, masked_img_path)

    #######################
    # Solution 2
    # An alternative approach: set all values outside the mask (where mask = 0) to zero.
    #######################
    
    # Set all image data values to 0 where the mask is equal to 0 (outside the brain region)
    img_data[mask_data == 0] = 0
    
    # Save the modified image to disk, creating a new Nifti image and writing it out
    img_out = nib.Nifti1Image(img_data, img.affine, img.header)
    nib.save(img_out, masked_img_path)
    
def apply_fsl_math_approach(img_path, mask_path, masked_img_path):
    ###########################
    # Solution
    # Based on fslmaths documentation, the -mas option is used to apply a mask to the image.
    ###########################
    os.system('fslmaths {} -mas {} {}'.format(img_path, mask_path, masked_img_path))

def launch_freeview(img_list):
    """
    Wrapper around Freeview to launch it with several images.
    This wrapper is necessary to launch Freeview in a separate thread, ensuring the notebook is free to do something else.

    Parameters
    ----------
    img_list: list of string
        List of images (files) to load. Assumed by default to be volume files.
    """
    args = []
    
    # Iterate over the list of images and prepare the command arguments for Freeview
    for i in range(len(img_list)):
        args.append("-v")        # '-v' is used in Freeview to specify the volume files
        args.append(img_list[i]) # Append each image file path to the arguments list

    # Run the Freeview command with all the prepared arguments
    subprocess.run(["freeview"] + args)

def fsl_anat_wrapped(anatomical_target, output_path):
    # Run the 'fsl_anat' command to perform automatic brain extraction and segmentation on the anatomical image.
    # - 'anatomical_target': Path to the anatomical image to process.
    # - '--clobber': Overwrite any existing output with the same name.
    # - '--nosubcortseg': Skip subcortical segmentation to speed up processing.
    # - '-o': Specifies the output directory where results will be saved.
    os.system('fsl_anat -i {} --clobber --nosubcortseg -o {}'.format(anatomical_target, output_path))
    
    # Define the path to the folder created by FSL ('output_path.anat').
    # FSL saves the output files in this directory by default.
    fsl_anat_path = output_path + '.anat'
    
    # Find all files in the 'output_path.anat' folder.
    files_to_move = glob.glob(op.join(fsl_anat_path, '*'))
    
    # Move each file from the 'output_path.anat' folder to the main 'output_path' folder.
    for f in files_to_move:
        # 'shutil.move' moves each file to the output directory.
        # 'op.split(f)[1]' gets the file name (e.g., 'T1_brain.nii.gz') from the full path.
        shutil.move(f, op.join(output_path, op.split(f)[1]))
    
    # Remove the now-empty 'output_path.anat' directory, as all files have been moved.
    os.rmdir(fsl_anat_path)

def load_mot_params_fsl_6_dof(path):
    return pd.read_csv(path, sep='  ', header=None, 
            engine='python', names=['Rotation x', 'Rotation y', 'Rotation z','Translation x', 'Translation y', 'Translation z'])

def compute_FD_power(mot_params):
    framewise_diff = mot_params.diff().iloc[1:]

    rot_params = framewise_diff[['Rotation x', 'Rotation y', 'Rotation z']]
    # Estimating displacement on a 50mm radius sphere
    # To know this one, we can remember the definition of the radian!
    # Indeed, let the radian be theta, the arc length be s and the radius be r.
    # Then theta = s / r
    # We want to determine here s, for a sphere of 50mm radius and knowing theta. Easy enough!
    
    # Another way to think about it is through the line integral along the circle.
    # Integrating from 0 to theta with radius 50 will give you, unsurprisingly, r0 theta.
    converted_rots = rot_params*50  
    trans_params = framewise_diff[['Translation x', 'Translation y', 'Translation z']]
    fd = converted_rots.abs().sum(axis=1) + trans_params.abs().sum(axis=1)
    return fd
