import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import re
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import matplotlib.pyplot as plt


class EyeGazeDataset(Dataset):
    def __init__(self, csv_file, target_frequency=60):
        # Read CSV data
        self.data = pd.read_csv(csv_file)

        # Extract frequency from file name
        frequency_match = re.search(r"(\d+)hz", csv_file)
        if frequency_match:
            self.frequency = int(frequency_match.group(1))
        else:
            self.frequency = None
            raise ValueError("Sampling frequency not found in the file name.")

        # Resample data if target_frequency is specified
        if target_frequency is not None:
            self.data = self._resample_data(self.data, self.frequency, target_frequency)
            self.frequency = target_frequency  # Update frequency after resampling

        # Preprocess data: filter out blink data
        self._filter_blink_data()

        # Compute features
        self.features = self._compute_features()

        # Compute labels
        self.labels = self._compute_labels()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def _resample_data(self, data, original_freq, target_freq):
        # Generate original time vector
        num_samples = len(data)
        original_time = np.arange(num_samples) / original_freq  # In seconds

        # Generate target time vector
        total_duration = original_time[-1]
        num_target_samples = int(np.floor(total_duration * target_freq)) + 1
        target_time = np.linspace(0, total_duration, num_target_samples)

        # Initialize a DataFrame to hold the resampled data
        resampled_data = pd.DataFrame()

        # Interpolate each column
        for column in data.columns:
            interp_func = interp1d(
                original_time, data[column], kind="linear", fill_value="extrapolate"
            )
            resampled_data[column] = interp_func(target_time)

        return resampled_data.reset_index(drop=True)

    def _filter_blink_data(self):
        # Both blink weights exceeding default threshold
        default_threshold = 0.5  # Adjust threshold as needed
        left_blink = self.data["LeftBlinkWeight"] > default_threshold
        right_blink = self.data["RightBlinkWeight"] > default_threshold
        blink_filter = ~(left_blink & right_blink)
        self.data = self.data[blink_filter].reset_index(drop=True)

    def _compute_features(self):
        # Get gaze direction vectors (normalized)
        u_vectors = self.data[
            ["LeftGazeDir_X", "LeftGazeDir_Y", "LeftGazeDir_Z"]
        ].values
        norms = np.linalg.norm(u_vectors, axis=1, keepdims=True)
        u_vectors = u_vectors / norms
        u_vectors[np.isnan(u_vectors)] = 0  # Handle division by zero

        # Compute angular displacement θ between consecutive gaze samples using dot product
        u_prev = u_vectors[:-1]
        u_next = u_vectors[1:]
        dot_products = np.sum(u_prev * u_next, axis=1)
        dot_products = np.clip(dot_products, -1.0, 1.0)  # Avoid numerical errors
        theta = np.arccos(dot_products)  # Length N - 1

        # Time difference between samples
        delta_t = 1.0 / self.frequency  # In seconds

        # Compute scalar angular velocity
        angular_velocity = theta / delta_t  # Length N - 1

        # Compute scalar angular acceleration
        angular_acceleration = np.diff(angular_velocity) / delta_t  # Length N - 2

        # Compute velocity vectors (rate of change of gaze direction)
        delta_u = u_next - u_prev  # Length N - 1 x 3
        velocity_vectors = delta_u / delta_t  # Length N - 1 x 3

        # Compute acceleration vectors
        delta_v = velocity_vectors[1:] - velocity_vectors[:-1]  # Length N - 2 x 3
        acceleration_vectors = delta_v / delta_t  # Length N - 2 x 3

        # Adjust lengths to match acceleration_vectors (length N - 2)
        theta = theta[1:]  # Now length is N - 2
        angular_velocity = angular_velocity[1:]  # Now length is N - 2
        velocity_vectors = velocity_vectors[1:]  # Now length is N - 2
        u_vectors = u_vectors[2:]  # Now length is N - 2

        # Assemble the feature vector
        features = np.column_stack(
            (
                theta[:, np.newaxis],  # Scalar angular displacement
                angular_velocity[:, np.newaxis],  # Scalar angular velocity
                angular_acceleration[:, np.newaxis],  # Scalar angular acceleration
                velocity_vectors,  # Velocity vector components (3D)
                acceleration_vectors,  # Acceleration vector components (3D)
            )
        )  # Total features per sample: 1 + 1 + 1 + 3 + 3  = 9

        return features

    def _compute_labels(self):
        N = len(self.features)
        labels = np.array(["other"] * N, dtype=object)

        # Identify saccades using IN-VT
        angular_velocity_deg = self.features[:, 1] * (180 / np.pi)

        # Use fixed threshold of 70 degrees per second
        threshold = 70  # degrees per second

        # Create saccade mask
        saccade_mask = angular_velocity_deg > threshold

        # Enforce minimum and maximum duration for saccades
        min_saccade_samples = max(1, int(np.ceil(17e-3 * self.frequency)))
        max_saccade_samples = int(np.floor(200e-3 * self.frequency))

        saccade_regions = self._find_contiguous_regions(saccade_mask)

        for start, end in saccade_regions:
            duration_samples = end - start
            if min_saccade_samples <= duration_samples <= max_saccade_samples:
                labels[start:end] = "saccade"

        # Identify fixations using I-DT
        # Fixation parameters
        min_fixation_samples = max(1, int(np.ceil(50e-3 * self.frequency)))
        max_fixation_samples = int(np.floor(1500e-3 * self.frequency))
        dispersion_threshold_rad = np.deg2rad(1)  # 1 degree in radians

        labels = self._identify_fixations(
            labels, min_fixation_samples, max_fixation_samples, dispersion_threshold_rad
        )

        return labels

    def _find_contiguous_regions(self, condition):
        """Finds contiguous True regions of the boolean array "condition". Returns
        a list of tuples where each tuple is (start, end) of a contiguous region."""
        d = np.diff(condition)
        (idx,) = d.nonzero()
        idx += 1

        if condition[0]:
            idx = np.r_[0, idx]
        if condition[-1]:
            idx = np.r_[idx, condition.size]

        idx.shape = (-1, 2)
        return idx

    def _identify_fixations(
        self, labels, min_samples, max_samples, dispersion_threshold_rad
    ):
        u_vectors = self.data[
            ["LeftGazeDir_X", "LeftGazeDir_Y", "LeftGazeDir_Z"]
        ].values
        norms = np.linalg.norm(u_vectors, axis=1, keepdims=True)
        u_vectors = u_vectors / norms
        u_vectors[np.isnan(u_vectors)] = 0  # Handle division by zero
        u_vectors = u_vectors[2:]
        N = len(u_vectors)

        i = 0
        while i < N:
            # Skip if this index is already labeled (e.g., 'saccade')
            if labels[i] != "other":
                i += 1
                continue

            window = []
            indices = []
            j = i
            while j < N and len(window) < max_samples:
                # Skip if this index is already labeled
                if labels[j] != "other":
                    break  # Cannot extend the window further

                window.append(u_vectors[j])
                indices.append(j)
                centroid = np.mean(window, axis=0)
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm == 0:
                    break  # Avoid division by zero
                centroid_unit = centroid / centroid_norm
                # Compute angular displacement from centroid
                disp = []
                for w in window:
                    w_norm = np.linalg.norm(w)
                    if w_norm == 0:
                        continue
                    w_unit = w / w_norm
                    angle = np.arccos(np.clip(np.dot(w_unit, centroid_unit), -1.0, 1.0))
                    disp.append(angle)
                if not disp:
                    break
                max_disp_rad = max(disp)
                if max_disp_rad > dispersion_threshold_rad:
                    break
                j += 1
            duration_samples = len(window)
            if min_samples <= duration_samples <= max_samples:
                labels[indices] = "fixation"
                i = j
            else:
                i += 1
        return labels


if __name__ == "__main__":
    # Create dataset
    dataset = EyeGazeDataset("data/EyeTrackingData_10hz.csv")

    # Extract features and labels
    X = dataset.features  # Shape: (N, 6)
    y_true = dataset.labels  # Shape: (N,)

    # Perform unsupervised clustering
    kmeans = KMeans(n_clusters=5, random_state=42).fit(X)
    cluster_labels = kmeans.labels_

    # Reduce dimension to 2D for visualization
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Prepare DataFrame for Plotly
    df = pd.DataFrame(X_2d, columns=["Component 1", "Component 2"])
    df["Cluster"] = cluster_labels.astype(str)
    df["Ground Truth"] = y_true

    # First plot: Unsupervised cluster labels
    fig1 = px.scatter(
        df,
        x="Component 1",
        y="Component 2",
        color="Cluster",
        title="Unsupervised Clustering of Eye Gaze Data",
    )
    fig1.show()

    # Second plot: Ground truth labels
    fig2 = px.scatter(
        df,
        x="Component 1",
        y="Component 2",
        color="Ground Truth",
        title="Ground Truth Labels of Eye Gaze Data",
    )
    fig2.show()
