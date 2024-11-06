import os
import os.path as op
import nibabel as nib
import numpy as np
import subprocess
import shutil
import glob
import pandas as pd
import progressbar

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

    # Format the subject ID following the BIDS convention (e.g., 'sub-control01')
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

def combine_all_transforms(reference_volume, warp_save_name,  is_linear, epi_2_moco=None, epi_2_anat_warp=None, anat_2_standard_warp=None):
    """
    Combines transformation BEFORE motion correction all the way to standard space transformation
    The various transformation steps are optional. As such, the final warp to compute is based on 
    which transforms are provided.

    Parameters
    ----------
    reference_volume: str
        Reference volume. The end volume after all transformations have been applied, relevant for final resolution and field of view.
    warp_save_name: str
        Under which name to save the total warp
    is_linear: bool
        Whether the transformation is linear or non linear.
    epi_2_moco: str
        Transformation of the EPI volume to motion-correct it (located in the .mat/ folder of the EPI
    epi_2_anat_warp: str
        Transformation of the EPI volume to the anatomical space, typically obtained by epi_reg. Assumed to include fieldmap correction and thus be non-linear.
    anat_2_standard_warp: str
        Transformation of the anatomical volume to standard space, such as the MNI152 space. Might be linear or non linear, which affects is_linear value accordingly.
    """
    from fsl.wrappers import convertwarp
    # args_base = {
    #     'premat': epi_2_moco                 # Initial affine transformation (e.g., motion correction)
    # }
    args_base = {}
    args_base['premat'] = epi_2_anat_warp  # Affine transformation from EPI to anatomical space
    args_base['warp1'] = anat_2_standard_warp   # Non-linear warp to standard space

    # # Add either a linear postmat or non-linear warp2, depending on is_linear
    # if is_linear:
    #     # Apply two sequential linear transformations with premat and postmat
    #     args_base['midmat'] = epi_2_anat_warp  # Affine transformation from EPI to anatomical space
    #     args_base['postmat'] = anat_2_standard_warp  # Final transformation to standard space
    # else:
    #     # Apply one linear transformation and one non-linear warp
    #     args_base['postmat'] = epi_2_anat_warp      # Linear transformation from EPI to anatomical
    #     args_base['warp2'] = anat_2_standard_warp   # Non-linear warp to standard space

    args_filtered = {k: v for k, v in args_base.items() if v is not None}

    convertwarp(warp_save_name, reference_volume, **args_filtered)
    print("Done with warp conversion")

def apply_transform(reference_volume, target_volume, output_name, transform):
    """
    Applies a warp field to a target volume and resamples to the space of the reference volume.

    Parameters
    ----------
    reference_volume: str
        The reference volume for the final interpolation, resampling and POV setting
    target_volume: str
        The target volume to which the warp should be applied
    output_name: str
        The filename under which to save the new transformed image
    transform: str
        The filename of the warp (assumed to be a .nii.gz file)

    See also
    --------
    combine_all_transforms to see how to build a warp field
    """
    from fsl.wrappers import applywarp
    applywarp(target_volume,reference_volume, output_name, w=transform, rel=False)

def normalize_fMRI(source, output, threshold):
    img = nib.load(source)
    data = img.get_fdata()
    mask = data>=threshold
    masked_data = data[mask]
    mean = masked_data.mean()
    std = masked_data.std()
    normalized = (data - mean)/std
    img_out = nib.Nifti1Image(normalized,img.affine, img.header)
    nib.save(img_out, output)

def normalize_fMRI_mean_std(source, output):
    img = nib.load(source)
    data = img.get_fdata()
    mean = data.mean()
    std = data.std()
    normalized = (data - mean)/std
    img_out = nib.Nifti1Image(normalized,img.affine, img.header)
    nib.save(img_out, output)

def normalize_fMRI_minmax(source, output):
    img = nib.load(source)
    data = img.get_fdata()
    max = data.max()
    min = data.min()
    standardized = (data-min)/(max-min)
    mean = standardized.mean()
    standardized = standardized - mean
    img_out = nib.Nifti1Image(standardized,img.affine, img.header)
    nib.save(img_out, output)

def run_subprocess(preproc_root, ref, warp_name, split_vol, vol_nbr):
    """
    SAFETY GOGGLES ON
    This function launches applywarp in parallel to reach complete result quicker

    Parameters
    -----------
    split_vol: str
        Path to the volume on which to apply the transformation
    vol_nbr: str
        Number of the volume in the timeserie. Useful to reorder volumes after the fact, since parallelisation does not honour order.

    Returns
    -------
    out_vol: str
        Path to the transformed volume
    vol_nbr: str
        Number of the volume in the timeserie. Useful to reorder volumes after the fact.
    """
    try:
        split_nbr = split_vol.split('_')[-1].split('.')[0].split('split')[1]
        epi_moco = op.join(preproc_root, 'sub-control01', 'func', 'sub-control01_task-music_concat_bold_moco.mat/', 'MAT_' + split_nbr)
        out_vol = op.join(preproc_root, 'sub-control01', 'func', 'std', 'sub-control01_task-music_concat_bold_moco_std_vol' + split_nbr)
        result = subprocess.run(['applywarp', '-i', split_vol, '-r', ref, '-o', out_vol, '-w', warp_name, '--abs', '--premat={}'.format(epi_moco)], check=True)
        return out_vol, vol_nbr
    except subprocess.CalledProcessError as e:
        return f"applywarp for volume '{split_vol}' failed with error: {e.stderr.decode('utf-8')}"
    
def merge_to_mni(preproc_root, produced_vols):
    first_vol = nib.load(produced_vols[0])
    v_shape = first_vol.get_fdata().shape

    filename = op.join(preproc_root, 'sub-control01', 'func', 'sub-control01_task-music_concat_bold_moco_bbr_std.dat')
    large_array = np.memmap(filename, dtype=np.float32, mode='w+', shape=(v_shape[0],v_shape[1],v_shape[2], len(produced_vols)))
    batch_size = len(produced_vols)//4

    A = np.zeros((v_shape[0],v_shape[1],v_shape[2], batch_size))

    with progressbar.ProgressBar(max_value=len(produced_vols)) as bar:
        for batch_i in range(4):
            print('Starting for batch {}/4'.format(batch_i+1))
            start_batch = batch_size * batch_i
            end_batch = min(batch_size * (batch_i+1),len(produced_vols))
            max_len = end_batch - start_batch + 1
            for i in range(start_batch, end_batch):
                vol = nib.load(produced_vols[i])
                A[:,:,:,i-start_batch] = vol.get_fdata()
                bar.update(i)
            large_array[:,:,:, start_batch:end_batch] = A[:,:,:,:max_len]
    large_array.flush()
    return filename

def smooth_volume(file_path, output_path):
    subprocess.run(['fslmaths',file_path, '-s', str(6/2.3548), '{}_smoothed-6mm'.format(output_path)])