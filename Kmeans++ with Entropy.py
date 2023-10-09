import os
import cv2
from sklearn.cluster import KMeans
from PIL import Image
import numpy as np
from scipy.io.wavfile import write
from scipy.stats import skew, entropy


def create_audio_from_image(image, proportions, medians, avg_distances, stats, sample_rate=48000, duration=1.875):
    image = np.array(image)
    audio_data = np.zeros(int(sample_rate * duration))
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    proportions = np.array(proportions)
    medians = np.array(medians) / 255.0
    avg_distances = np.array(avg_distances) / np.max(avg_distances)

    min_freq_log = np.log2(20)
    max_freq_log = np.log2(20000)

    modulation_signal = create_modulation_signal(stats, len(t), sample_rate)

    for i, row in enumerate(image):
        unique_labels = np.unique(row)
        freq_ranges_log = np.linspace(min_freq_log, max_freq_log, len(unique_labels) + 1)
        row_audio = np.zeros_like(t)
        label_counts = np.zeros(len(unique_labels), dtype=int)

        for pixel in row:
            label_index = np.where(unique_labels == pixel)[0][0]
            pixel_frequency_log = np.maximum((freq_ranges_log[label_index] + ((freq_ranges_log[label_index + 1] - freq_ranges_log[label_index]) / len(row)) * label_counts[label_index]) / avg_distances[label_index], np.log2(20))
            pixel_frequency = 2 ** pixel_frequency_log 

            amplitude = proportions[label_index] * medians[label_index]
            row_audio += amplitude * np.cos(2 * np.pi * pixel_frequency * t + modulation_signal)

            label_counts[label_index] += 1

        audio_data += row_audio

    audio_data /= len(image)

    audio_data *= 32767 / np.max(np.abs(audio_data))
    audio_data = audio_data.astype(np.int16)
    return audio_data

def create_modulation_signal(stats, length, sample_rate):
    t = np.linspace(0, length / sample_rate, length, False)

    # We will use variance for the amplitude of the cosine signal
    # And entropy for the frequency modulation
    # To avoid division by zero, we add a small constant (1e-3) to the denominator
    return stats['variance'] * np.cos(2 * np.pi * (1 / (length / sample_rate + stats['entropy'] + 1e-3)) * t + stats['mean'])


def crossfade(audio1, audio2, sample_rate, fade_duration):
    fade_samples = int(sample_rate * fade_duration)

    ramp = np.linspace(0, 1, fade_samples)

    if len(audio1) < fade_samples:
        audio1 = np.pad(audio1, (0, fade_samples - len(audio1)))
    if len(audio2) < fade_samples:
        audio2 = np.pad(audio2, (fade_samples - len(audio2), 0))

    audio1[-fade_samples:] = audio1[-fade_samples:] * (1 - ramp)
    audio2[:fade_samples] = audio2[:fade_samples] * ramp

    return np.concatenate([audio1, audio2])

def process_image(img, output_path, optimal_k=3):
    img = img.resize((300, 300))
    img_data = np.array(img)

    stats = {}
    stats['variance'] = np.var(img_data)
    stats['skewness'] = skew(img_data.flatten())
    stats['mean'] = np.mean(img_data)

    # Calculate entropy of the original image
    hist = np.histogram(img_data.flatten(), bins=256, range=[0,256])[0]
    hist = hist / hist.sum()
    stats['entropy'] = entropy(hist)

    img_data_reshaped = img_data.flatten().reshape(-1, 1)

    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=0).fit(img_data_reshaped)

    clusters = {i: [] for i in range(optimal_k)}
    for i in range(img_data.shape[0]):
        for j in range(img_data.shape[1]):
            clusters[kmeans.labels_[i * img_data.shape[1] + j]].append(img_data[i][j])
    
    medians = [np.median(cluster) for cluster in clusters.values()]
    
    sorted_clusters = sorted(list(clusters.items()), key=lambda x: medians[x[0]])

    new_labels = {old: new for new, (old, _) in enumerate(sorted_clusters)}

    segmented_img_data = np.array([new_labels[label] for label in kmeans.labels_])

    medians = [np.median(sorted_clusters[i][1]) for i in range(optimal_k)]
    avg_distances = [np.mean(np.abs(sorted_clusters[i][1] - np.mean(sorted_clusters[i][1]))) for i in range(optimal_k)]

    segmented_img_data = segmented_img_data.reshape(img.size[1], img.size[0])

    pixel_counts = np.bincount(segmented_img_data.flatten())
    proportions = [pixel_counts[cluster] / pixel_counts.sum() for cluster in range(optimal_k)]

    intensities = np.linspace(0, 255, optimal_k+1)
    for i in range(optimal_k):
        segmented_img_data[segmented_img_data == i] = intensities[i]

    segmented_img = Image.fromarray(segmented_img_data.astype(np.uint8))
    segmented_img.save(output_path)

    return segmented_img, proportions, medians, avg_distances, stats

def process_video(input_path, output_directory_image, output_path_audio, max_k, output_directory_image2):
    vidcap = cv2.VideoCapture(input_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_duration = 1.0 / fps
    fade_duration = frame_duration / 2.0

    success, frame = vidcap.read()
    count = 0
    total_audio = np.zeros(0)

    while success:
        frame = cv2.cvtColor(frame)
        original_frame_path = os.path.join(output_directory_image2, f"original_frame_{count}.png")
        cv2.imwrite(original_frame_path, frame)
        img = Image.fromarray(frame)
        output_path_image = os.path.join(output_directory_image, f"frame_{count}.png")
        segmented_img, proportions, medians, avg_distances, stats = process_image(img, output_path_image, max_k)
        frame_audio = create_audio_from_image(segmented_img, proportions, medians, avg_distances, stats, sample_rate = 48000, duration = frame_duration)
        
        if count != 0:
            total_audio = crossfade(total_audio, frame_audio, sample_rate=48000, fade_duration=fade_duration)
        else:
            total_audio = frame_audio

        success, frame = vidcap.read()
        count += 1

    write(output_path_audio, 48000, total_audio)
    print("Finished processing", input_path)

def process_directory(input_directory, output_directory_image, output_directory_audio, max_k, output_directory_image2):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    videos = [vid for vid in os.listdir(input_directory) if os.path.splitext(vid)[1] in video_extensions]
    for video in videos:
        input_path = os.path.join(input_directory, video)
        output_path_audio = os.path.join(output_directory_audio, os.path.splitext(video)[0] + '.wav')
        process_video(input_path, output_directory_image, output_path_audio, max_k, output_directory_image2)

if __name__ == "__main__":
    optimal_k = 3
    input_directory = '/Users/erdematbas/Desktop/Work/RA/Sonification/videos/'
    output_directory_image_k_frames = '/Users/erdematbas/Desktop/Work/RA/Sonification/k-frames'
    output_directory_audio = '/Users/erdematbas/Desktop/Work/RA/Sonification/audio'
    output_directory_image = '/Users/erdematbas/Desktop/Work/RA/Sonification/frames'
    process_directory(input_directory, output_directory_image_k_frames, output_directory_audio, optimal_k, output_directory_image)
