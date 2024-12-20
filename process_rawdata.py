import os
import json
import pandas as pd
import glob
import numpy as np
from pathlib import Path
# Define the path to your raw gaze data directory
RAW_GAZE_DIR = 'data/raw_gaze'

def read_scene_segments(json_path):
    with open(json_path, 'r') as f:
        scene_segments = json.load(f)
    # Initialize startTimeStamp for each scene
    scenes = []
    previous_end = 0.0
    for segment in scene_segments:
        start_time = previous_end
        end_time = segment['endTimeStamp']
        scenes.append({
            'name': segment['name'],
            'startTimeStamp': start_time,
            'endTimeStamp': end_time,
            'type': segment.get('type', '')
        })
        previous_end = end_time
    return scenes

def segment_csv_files():
    # Path to the scene_segments.json file
    json_path = os.path.join(RAW_GAZE_DIR, 'scene_segments.json')
    scenes = read_scene_segments(json_path)

    # Get all CSV files starting with '360VideoEyeGaze'
    csv_files = glob.glob(os.path.join(RAW_GAZE_DIR, '360VideoEyeGaze*_processed.csv'))

    for csv_file in csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)
        # Keep the original 'timestamp' as 'timestamp_old'
        df['timestamp_old'] = df['timestamp']
        # Add a 'frameindex' column
        df['frameindex'] = df.index

        # Process each scene
        for scene in scenes:
            scene_name = scene['name']
            start_time = scene['startTimeStamp']
            end_time = scene['endTimeStamp']

            # Filter the DataFrame for rows within the timestamp range
            scene_df = df[(df['timestamp_old'] >= start_time) & (df['timestamp_old'] < end_time)].copy()

            if scene_df.empty:
                continue  # Skip if no data in this scene

            # Adjust the 'timestamp' to start from 0 for each scene
            scene_df['timestamp'] = scene_df['timestamp_old'] - start_time

            # Create the scene subfolder if it doesn't exist
            scene_folder = os.path.join(RAW_GAZE_DIR, scene_name)
            os.makedirs(scene_folder, exist_ok=True)

            # Save the segmented CSV file
            csv_filename = os.path.basename(csv_file)
            scene_csv_path = os.path.join(scene_folder, csv_filename)

            # Reorder columns to place 'timestamp_old' and 'frameindex' appropriately
            cols = scene_df.columns.tolist()
            # Move 'timestamp_old' and 'frameindex' to desired positions
            cols.insert(1, cols.pop(cols.index('timestamp_old')))
            cols.insert(2, cols.pop(cols.index('frameindex')))
            scene_df = scene_df[cols]

            # Save the scene DataFrame to CSV
            scene_df.to_csv(scene_csv_path, index=False)
            print(f"Saved segmented CSV for scene '{scene_name}' to '{scene_csv_path}'") 

def dist_angle_arrays_unsigned(vecs1, vecs2):
	# Unsigned distance computed pair-wise between two arrays of unit vectors
	dot = np.einsum("ji,ji->j", vecs1, vecs2)

	return np.arccos(dot)

def getVelocity(gp, return_keep=False, outlierSigma=5):
    """Calculate velocity of gaze points."""
    diffT = gp[1:, 3] - gp[:-1, 3]

    # Handle invalid or zero time differences
    if np.any(diffT <= 0):
        print("Warning: Invalid or zero time differences found in Timestamps.")
        problematic_indices = np.where(diffT <= 0)[0]
        for idx in problematic_indices:
            print(f"Problematic Timestamp at index {idx}: current={gp[idx + 1, 3]}, previous={gp[idx, 3]}")
        diffT[diffT <= 0] = np.nan  # Replace invalid differences with NaN

    distance = dist_angle_arrays_unsigned(gp[1:, :3], gp[:-1, :3])
    velocity = distance / diffT

    # Replace NaN and Inf values in velocity
    velocity = np.nan_to_num(velocity, nan=np.inf, posinf=np.inf, neginf=np.inf)

    velocity = np.append(velocity, [velocity[0]])  # Repeat first velocity for equal length

    if return_keep:
        # Remove samples farther than X std from the mean
        keep = np.abs((velocity - np.nanmean(velocity)) / np.nanstd(velocity)) < outlierSigma
        keep &= np.logical_not(np.isnan(velocity) | np.isinf(velocity))
        return keep, velocity
    else:
        return None, velocity
      
def fix_gen(label_list):
    """Clean up fixation markers."""
    for i in range(1, label_list.shape[0] - 1):
        if label_list[i] != label_list[i - 1] and label_list[i] != label_list[i + 1]:
            label_list[i] = label_list[i - 1]
    for i in [0, label_list.shape[0] - 1]:
        if label_list[i] != label_list[i - 1 if i > 0 else 1]:
            label_list[i] = label_list[i - 1 if i > 0 else 1]


# from salient360 tool box, I-VT
def parse(Timestamp, velocity, threshold=100, minFixationTime=60):
    """Parse gaze data and label fixations.
    threshold: maximum angular velocity that is considered part of a fixation. <100 is fixation, >300 is saccade
    reference: https://www.cs.drexel.edu/~dds26/publications/Salvucci-ETRA00.pdf

    minFixationTime: fixation typically ranges from 150 to 300ms
    """
    threshold = np.deg2rad(threshold) # Eye threshold rad/ms
    minFixationTime = minFixationTime/1000

    # Label as part of fixations samples with velocity below threshold
    # 	i.e., Fixation == 1, Saccade == 0
    fixationMarkers = np.array(velocity <= threshold, dtype=bool)
    

    fix_gen(fixationMarkers)
    

    Nsamples = velocity.shape[0]

    iDisp = 1
    startMarker = 0

    # Remove short fixations
    change = False
    iDisp = 1
    fixStart = np.zeros(1)
    fixEnd = np.zeros(1)
    while iDisp < (Nsamples - 1):
        # print("  \rremove sf", iDisp, end="");
        # Point where sacc ends and fix starts
        if not fixationMarkers[iDisp] and fixationMarkers[iDisp + 1]:
            # print("  \rremove sf", iDisp, end="");
            startMarker = iDisp
            fixStart[:] = Timestamp[iDisp]

            # Loop ahead until we find the start of a new saccade
            while iDisp < (Nsamples - 1) and fixationMarkers[iDisp + 1]:
                iDisp += 1

            fixEnd[:] = Timestamp[iDisp]
            if fixEnd - fixStart < minFixationTime:
                
                fixationMarkers[startMarker : iDisp + 1] = False
                change = True
        else:
            iDisp += 1

        # Reset until no small fixations are found
        if iDisp == (Nsamples - 1) and change:
            iDisp = 1
            change = False

    return fixationMarkers
           

def process_gaze_data(input_dir,which_eye = 'left'):
    """Process all raw gaze data and save fixation CSV files."""
    assert which_eye in ['left','right']
    input_path = Path(input_dir)

    for scene_dir in input_path.iterdir():
        if scene_dir.is_dir():
            fixation_dir = scene_dir / "fixation_data"
            fixation_dir.mkdir(exist_ok=True)
            
            for csv_file in scene_dir.glob("*.csv"):
                df = pd.read_csv(csv_file)
                
                # Copy required columns
                processed_data = df[['timestamp', 'timestamp_old', 'frameindex']].copy()

                # Calculate sphere coordinates
                if which_eye == 'left': # use left eye to calculate fixation
                    gaze_dirs = df[['leftgazedirx', 'leftgazediry', 'leftgazedirz']].to_numpy()
                    processed_data['leftgaze_x'] = df['leftgaze_x']
                    processed_data['leftgaze_y'] = df['leftgaze_y']
                    
                else:
                    gaze_dirs = df[['rightgazedirx', 'rightgazediry', 'rightgazedirz']].to_numpy()
                    processed_data['rightgaze_x'] = df['rightgaze_x'] 
                    processed_data['rightgaze_y'] = df['rightgaze_y'] 
                gaze_dirs = gaze_dirs / np.linalg.norm(gaze_dirs, axis=1, keepdims=True)  # Normalize to unit vectors
                lon = np.arctan2(gaze_dirs[:, 0], gaze_dirs[:, 2])  # Longitude
                lat = np.arcsin(gaze_dirs[:, 1])  # Latitude
                lon = np.degrees(lon)  # Convert to degrees
                lat = np.degrees(lat)
                lon = (lon - 180) % 360  # horizontal shift 180 degrees and wrap around
                lon[lon > 180] -= 360 
                processed_data['lon'] = lon
                processed_data['lat'] = lat

                # Compute fixation markers
                gaze_with_time = np.hstack((gaze_dirs, df[['timestamp']].to_numpy()))
                _, velocity = getVelocity(gaze_with_time)
                print(f'Velocity Min: {np.rad2deg(np.min(velocity))}, Avg: {np.rad2deg(np.mean(velocity))},Max: {np.rad2deg(np.max(velocity))}')
                processed_data['is_fixation'] = parse(df['timestamp'].to_numpy(), velocity)

                 # Count fixation durations
                fixation_markers = processed_data['is_fixation'].to_numpy()
                fixation_durations = (fixation_markers[:-1] == 1) & (fixation_markers[1:] == 0)
                fixation_count = fixation_durations.sum()

                # Save processed data
                # Extract the timestamp part from the original filename
                timestamp_part = csv_file.stem.split("_")[1] + "_" + csv_file.stem.split("_")[2]
                output_file = fixation_dir / f"{timestamp_part}_fixation.csv"
                processed_data.to_csv(output_file, index=False)
                print(f"Processed file saved: {output_file}")
                print(f"Number of fixation durations found: {fixation_count}")

if __name__ == '__main__':
    segment_csv_files()
    process_gaze_data(RAW_GAZE_DIR)