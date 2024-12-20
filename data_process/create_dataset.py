import torch
import torch.nn.parallel
import torch.utils.data
import numpy as np
import os
import pickle
import pickle as pck
import pandas as pd
import glob
import math
import json
import cv2
import torchvision.transforms as transforms
import random
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models import resnet50
import torchvision
import clip


with open("data/imagenet_class_index.json") as f:
    class_idx = json.load(f)
IMAGENET_CLASS = [class_idx[str(k)][1] for k in range(len(class_idx))]

'''
    Data format:
    
    [data]
        ['train']
            ['image1_name']
                ['image']: Tensor[3, 128, 256]
                ['scanpaths']: Tensor[n_scanpath, n_gaze_point, 3] # (x, y, z) for the 3-th dimension
                ['frameindex']: np.ndarray [n_scanpath,n_gaze_point] # each fixation point's recorded frame index, for debugging
                ['csv_ids']: List[Str] (n_scanpath) # each scan_path testee id, for visualization
                ['image_feats']: np.ndarray (n_scanpath,n_gaze_point,2048) # extracted visual feature by ResNet 50 for each fixation frame
                ['image_labels']: np.ndarray (n_scanpath,n_gaze_point) # string label predicted by ResNet 50 for each fixation frame
                ['sem_feats']:np.ndarray (n_scanpath,n_gaze_point,512) # CLIP embedding of the string labels
            ['image2_name']
                ...
        ['test']
            ['imageN_name']
                ...
        ['info']
            ['train']: {
                'num_image': int,
                'num_scanpath': int,
                'scanpath_length': int,
                'max_scan_length': int,
            }
            ['test']: {
                ...
            }
'''

def map_labels_to_strings(all_labels):
        """
        Convert numerical labels to string labels using the ResNet class names.

        Args:
            all_labels (np.ndarray): Array of numerical labels of shape [n_scanpaths, n_gaze_points].

        Returns:
            np.ndarray: Array of string labels with the same shape as input.
        """
        # Load ImageNet class labels (1000 classes for ResNet)
        imagenet_class_index = IMAGENET_CLASS

        # Map numerical labels to their corresponding string labels
        string_labels = np.vectorize(lambda label: imagenet_class_index[int(label)])(all_labels)

        return string_labels


def embed_labels_with_clip(string_labels, model, batch_size=32):
    """
    Embed string labels into semantic feature vectors using CLIP.

    Args:
        string_labels (np.ndarray): Array of string labels (e.g., [['vault', 'stone_wall'], ...]).
        batch_size (int): Batch size for processing.

    Returns:
        np.ndarray: Array of semantic feature vectors of shape [n_scanpaths, n_gaze_points, embedding_dim].
    """
    # Flatten the label array for easier processing
    shape = string_labels.shape
    flattened_labels = string_labels.flatten()


    # Tokenize the labels
    text_tokens = clip.tokenize(list(flattened_labels)).to('cuda')

    # Batch processing
    all_embeddings = []
    for i in range(0, len(text_tokens), batch_size):
        batch_tokens = text_tokens[i : i + batch_size]
        with torch.no_grad():
            embeddings = model.encode_text(batch_tokens)
            all_embeddings.append(embeddings.cpu())

    # Concatenate all embeddings and reshape to match the original label array
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    semantic_features = all_embeddings.reshape(*shape, -1)

    return semantic_features

def process_gaze_frames(resnet, cropped_images,batch_size=32):
    """
    Process gaze frames by extracting image features and labels from cropped image patches.

    Args:
        resnet (torch.nn.Module): ResNet model for feature extraction.
        cropped_images: list of image patches
        batch_size (int): Number of images to process in a batch.

    Returns:
        tuple: Extracted image features and predicted labels.
    """
    n_scanpaths, n_gaze_points, _, crop_size, _ = cropped_images.shape
    dataset = cropped_images.view(-1, 3, crop_size, crop_size)  # Flatten to [n_total_frames, 3, crop_size, crop_size]

    # Preprocessing transformations for ResNet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Apply normalization to all cropped images
    dataset = torch.stack([normalize(image) for image in dataset])  # Apply normalization

    feature_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1]))  # All layers except FC
    classifier = list(resnet.children())[-1]  # FC layer

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_features = []
    all_labels = []

    for batch in dataloader:
        with torch.no_grad():
            features = feature_extractor(batch).squeeze()  # Extract features
            logits = classifier(features)  # Predict labels
            predicted_labels = torch.argmax(logits, dim=1)

        all_features.append(features.cpu().numpy())
        all_labels.append(predicted_labels.cpu().numpy())

    # Concatenate and reshape
    all_features = np.concatenate(all_features, axis=0).reshape(n_scanpaths, n_gaze_points, -1)
    all_labels = np.concatenate(all_labels, axis=0).reshape(n_scanpaths, n_gaze_points)

    return all_features, all_labels

def crop_image_patches(all_frame_indices, all_csv_ids, all_frame_gaze_coords, frame_data_path='data/unity_recording', crop_size=224):
    """
    Crop image patches centered around gaze coordinates.

    Args:
        all_frame_indices (np.ndarray): Frame indices of shape (n_scanpaths, n_gaze_points).
        all_csv_ids (list): List of CSV IDs for each scanpath.
        all_frame_gaze_coords (np.ndarray): Gaze coordinates normalized to [0, 1], shape (n_scanpaths, n_gaze_points, 2).
        frame_data_path (str): Root directory containing frame images.
        crop_size (int): Size of the cropped image patches.

    Returns:
        list: List of cropped image patches of shape (n_scanpaths, n_gaze_points, crop_size, crop_size, 3).
    """
    n_scanpaths, n_gaze_points = all_frame_indices.shape
    cropped_images = []
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((crop_size, crop_size)),
    ])

    for scanpath_idx in range(n_scanpaths):
        scanpath_images = []
        for point_idx in range(n_gaze_points):
            csv_id = all_csv_ids[scanpath_idx]
            frame_index = all_frame_indices[scanpath_idx, point_idx]
            gaze_coord = all_frame_gaze_coords[scanpath_idx, point_idx]  # Normalized (x, y)

            # Load the corresponding frame
            frame_path = Path(frame_data_path) / csv_id / "frames" / f"frame_{frame_index:05d}.png"
            if not frame_path.exists():
                # Placeholder gray image if the frame doesn't exist
                frame_image = Image.new("RGB", (256, 256), (200, 200, 200))
            else:
                frame_image = Image.open(frame_path).convert("RGB")

            # Calculate crop region
            width, height = frame_image.size
            x, y = gaze_coord[0] * width, gaze_coord[1] * height
            left = max(0, min(width - crop_size, x - crop_size // 2))
            upper = max(0, min(height - crop_size, y - crop_size // 2))
            if left + crop_size > width:
                left = width - crop_size
            if upper + crop_size > height:
                upper = height - crop_size
            right = left + crop_size
            lower = upper + crop_size
            # Crop the frame
            cropped_frame = frame_image.crop((left, upper, right, lower))
            tensor_frame = preprocess(cropped_frame)
            scanpath_images.append(tensor_frame)
        cropped_images.append(torch.stack(scanpath_images))
    return torch.stack(cropped_images)  # Shape [n_scanpaths, n_gaze_points, 3, crop_size, crop_size]
 



def save_file(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
    f.close()


def load_logfile(path):
    log = pck.load(open(path, 'rb'), encoding='latin1')
    return log


def twoDict(pack, key_a, key_b, data):
    if key_a in pack:
        pack[key_a].update({key_b: data})
    else:
        pack.update({key_a: {key_b: data}})
    return pack


def create_info():
    info = {
        'train': {
            'num_image': 0,
            'num_scanpath': 0,
            'scanpath_length': 0,
            'max_scan_length': 0
        },
        'test': {
            'num_image': 0,
            'num_scanpath': 0,
            'scanpath_length': 0,
            'max_scan_length': 0
        }
    }
    return info


def summary(info):
    print("\n============================================")

    print("Train_set:   {} images, {} scanpaths,  length ={}".
          format(info['train']['num_image'], info['train']['num_scanpath'], info['train']['scanpath_length']))

    print("Test_set:    {} images, {} scanpaths,  length ={}".
          format(info['test']['num_image'], info['test']['num_scanpath'], info['test']['scanpath_length']))

    print("============================================\n")

def print_data_structure(data, indent=0):
    """
    Recursively print the structure of a dictionary with type information.
    
    Args:
        data (dict): The dictionary to inspect.
        indent (int): Current indentation level for pretty printing.
    """
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"{' ' * indent}{key}: (dict)")
            print_data_structure(value, indent + 4)
        elif isinstance(value, list):
            list_type = type(value[0]).__name__ if value else "Empty"
            print(f"{' ' * indent}{key}: (list of {list_type}, length {len(value)})")
        elif isinstance(value, np.ndarray):
            print(f"{' ' * indent}{key}: (np.ndarray, shape {value.shape})")
        
        elif isinstance(value, torch.Tensor):
            print(f"{' ' * indent}{key}: (torch.tensor, shape {value.shape})")

        else:
            print(f"{' ' * indent}{key}: ({type(value).__name__}) {value}")

def forward():
    if not os.path.exists('data/datasets'):
        os.makedirs('data/datasets')
    data = EyeGazeDataset()
    dic = data.run()
    print_data_structure(dic)
    save_file('data/datasets/eyegaze360.pkl', dic)
    summary(dic['info'])

def sphere2xyz(shpere_cord):
    """ input:  (lat, lon) shape = (n, 2)
        output: (x, y, z) shape = (n, 3) """
    lat, lon = shpere_cord[:, 0], shpere_cord[:, 1]
    pi = math.pi
    lat = lat / 180 * pi
    lon = lon / 180 * pi
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    return torch.cat((x.view(-1, 1), y.view(-1, 1), z.view(-1, 1)), 1)

def image_process(path,image_size = [128, 256]):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    image = transform(image)
    return image



class EyeGazeDataset:
    def __init__(self):
        super().__init__()
        self.images_path = "data/stimuli/"
        self.gaze_path = "data/raw_gaze/"
        self.duration = 30  # Fixed scanpath duration
        self.info = create_info()
        self.image_and_scanpath_dict = {}
        # Initialize ResNet as feature extractor
        resnet = resnet50(pretrained=True)
        resnet.eval()  # Set to evaluation mode
        self.resnet = resnet
        self.clip, _ = clip.load("ViT-B/32", device='cuda')
        

    def handle_empty(self, sphere_coords):
        """Handle invalid gaze points."""
        empty_index = np.where(sphere_coords[:, 0] == -999)[0]
        for idx in empty_index:
            if idx == 0:  # First point
                sphere_coords[idx] = sphere_coords[idx + 1]
            elif idx == len(sphere_coords) - 1:  # Last point
                sphere_coords[idx] = sphere_coords[idx - 1]
            else:  # Middle points
                sphere_coords[idx] = (sphere_coords[idx - 1] + sphere_coords[idx + 1]) / 2
        return sphere_coords
    
    def sample_gaze_points(self, gaze_data):
        """
        Sample gaze points for fixed duration without iteration, with adaptive randomness.

        Args:
            gaze_data (pd.DataFrame): Original gaze data containing all points.

        Returns:
            tuple: Sampled gaze points (lon, lat) and their corresponding original indices in the gaze_data DataFrame.
        """
        # Filter fixation points and their indices
        fixation_data = gaze_data[gaze_data['is_fixation'] == 1]
        fixation_points = fixation_data[['lon', 'lat']].to_numpy()
        fixation_indices = fixation_data.index.to_numpy()

        num_points = len(fixation_points)
        if num_points < self.duration:
            raise ValueError("Insufficient fixation points for sampling.")

        # Calculate bin size and max_shift
        bin_size = num_points // self.duration
        max_shift = bin_size // 2

        # Calculate bin start indices
        bin_starts = np.arange(0, num_points, bin_size)[:self.duration]

        # Apply random shifts to the bin start indices
        random_shifts = np.random.randint(-max_shift, max_shift + 1, size=len(bin_starts))
        sampled_fixation_indices = np.clip(bin_starts + random_shifts, 0, num_points - 1)

        # Map to original DataFrame indices
        sampled_indices = fixation_indices[sampled_fixation_indices]
        sampled_points = fixation_points[sampled_fixation_indices]

        return sampled_points, sampled_indices

    

    def process_scene(self, scene_path, mode, augmentations=1):
        """
        Process gaze data from all testee CSVs for a single scene, with optional augmentations.

        Args:
            scene_path (str): Path to the folder containing testee CSVs for a scene.
            mode (str): Dataset mode ('train' or 'test').
            augmentations (int): Number of random sampling augmentations (applies to training set only).
        """
        # Collect all CSV files for the scene
        csv_files = glob.glob(os.path.join(scene_path, "*.csv"))
        scene_name = Path(scene_path).parent.name
        image_file = os.path.join(self.images_path, f"{scene_name}.png")
        image = image_process(image_file)

        # Initialize lists to aggregate scanpaths, frame indices, and csv_ids
        all_scanpaths = []
        all_frame_indices = []
        all_csv_ids = []
        all_frame_gaze_coords = []

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            csv_id = Path(csv_file).stem.split("_")[0] + "_" + Path(csv_file).stem.split("_")[1]

            for _ in range(augmentations if mode == "train" else 1):
                # Sample gaze points and get sampled indices
                fixation_coords, sampled_indices = self.sample_gaze_points(df)
                fixation_coords = self.handle_empty(fixation_coords)
                fixation_coords_xyz = sphere2xyz(torch.tensor(fixation_coords))
                frame_ids = df['frameindex'][sampled_indices]
                frame_coords = np.stack([df['leftgaze_x'][sampled_indices],df['leftgaze_y'][sampled_indices]],axis=1)

                # Reshape sampled indices to match the scanpath shape
                sampled_indices = np.array(frame_ids,dtype=np.int64).reshape(1, -1)  # Shape [1, n_gaze_points]

                # Aggregate scan paths, frame indices, and csv IDs
                all_scanpaths.append(fixation_coords_xyz.numpy())
                all_frame_gaze_coords.append(frame_coords.reshape(-1,2))
                all_frame_indices.append(sampled_indices)
                all_csv_ids.append([csv_id])  # Track the testee ID for each scanpath

        # Concatenate all testees' data for this scene
        all_scanpaths = np.stack(all_scanpaths, axis=0) # Shape [n_total_scanpaths, n_gaze_points, 3]
        all_frame_indices = np.concatenate(all_frame_indices, axis=0)  # Shape [n_total_scanpaths, n_gaze_points]
        all_csv_ids = np.concatenate(all_csv_ids, axis=0)  # Shape [n_total_scanpaths]
        all_frame_gaze_coords = np.stack(all_frame_gaze_coords,axis=0) # Shape [n_total_scanpaths, n_gaze_points, 2]
        cropped_image_patches = crop_image_patches(all_frame_indices,all_csv_ids,all_frame_gaze_coords)
        image_feats,image_labels = process_gaze_frames(self.resnet,cropped_image_patches)
        string_labels = map_labels_to_strings(image_labels)
        sem_feats = embed_labels_with_clip(string_labels,self.clip)
        
        # Store the aggregated data under a shared key
        dic = {
            "image": image,
            "scanpaths": all_scanpaths,
            "frameindex": all_frame_indices,
            "csv_ids": all_csv_ids.tolist(),  # Track testee IDs as a list
            "image_feats":image_feats,
            "image_labels":string_labels,
            "sem_feats":sem_feats
        }
        # print(all_frame_indices[0])
        # self.plot_frames(dic)
        twoDict(self.image_and_scanpath_dict, mode, scene_name, dic)

        # Update scanpath count
        self.info[mode]['num_scanpath'] += len(all_scanpaths)
        print(f'finish {scene_name}')
        
    
    def plot_frames(self,dic, frame_data_path = 'data/unity_recording', num_plot=5):
        """
        Plot frames for randomly selected scanpaths.

        Args:
            dic (dict): Data dictionary containing image, scanpaths, frame indices, and CSV IDs.
            frame_data_path (str): Path to the root directory of frame data.
            num_plot (int): Number of scanpaths to randomly plot.
        """

        all_frame_indices = dic["frameindex"]
        all_csv_ids = dic["csv_ids"]

        if len(all_csv_ids) < num_plot:
            raise ValueError(f"Not enough scanpaths to plot. Available: {len(all_csv_ids)}, Requested: {num_plot}")

        # Randomly select scanpaths
        selected_indices = random.sample(range(len(all_csv_ids)), num_plot)
        selected_frame_indices = all_frame_indices[selected_indices]
        selected_csv_ids = [all_csv_ids[i] for i in selected_indices]

        fig, axes = plt.subplots(num_plot, selected_frame_indices.shape[1], figsize=(15, num_plot * 3))

        if num_plot == 1:
            axes = [axes]  # Ensure axes is iterable if only one row is plotted

        for row, (csv_id, frame_indices) in enumerate(zip(selected_csv_ids, selected_frame_indices)):
            for col, frame_index in enumerate(frame_indices):
                # Load the corresponding frame
                frame_path = Path(frame_data_path) / csv_id / "frames" / f"frame_{frame_index:05d}.png"
                if frame_path.exists():
                    frame_image = Image.open(frame_path)
                else:
                    frame_image = Image.new("RGB", (256, 256), (200, 200, 200))  # Placeholder for missing frames

                # Plot the frame
                ax = axes[row][col] if num_plot > 1 else axes[col]
                ax.imshow(frame_image)
                ax.axis("off")

                # Add title for scanpath and frame index
                # ax.set_title(f"{csv_id} - Frame {frame_index}", fontsize=8)

        plt.tight_layout()
        plt.show()


    def get_dataset(self, train_ratio=0.8):
        """
        Split files into train and test sets and process them.

        Args:
            train_ratio (float): Ratio of files to include in the training set.
        """
        # Collect all scene paths
        all_scene_paths = glob.glob(os.path.join(self.gaze_path, "*", "fixation_data"))
        random.shuffle(all_scene_paths)

        # Split into train and test
        split_index = int(len(all_scene_paths) * train_ratio)
        train_scenes = all_scene_paths[:split_index]
        test_scenes = all_scene_paths[split_index:]

        self.info['train']['csv_paths'] = train_scenes
        self.info['test']['csv_paths'] = test_scenes

        # Process train scenes with augmentations
        print("Processing training set...")
        for scene_path in train_scenes:
            self.process_scene(scene_path, "train", augmentations=5)
 

        # Process test scenes without augmentations
        print("Processing test set...")
        for scene_path in test_scenes:
            self.process_scene(scene_path, "test", augmentations=1)
            

        # Update info
        self.info['train']['num_image'] = len(self.image_and_scanpath_dict['train'])
        self.info['test']['num_image'] = len(self.image_and_scanpath_dict['test'])
        self.info['train']['scanpath_length'] = self.duration
        self.info['test']['scanpath_length'] = self.duration
    def run(self):
        """Main processing pipeline."""

        self.get_dataset()

        self.image_and_scanpath_dict["info"] = self.info
        return self.image_and_scanpath_dict

if __name__ == "__main__":
    forward()