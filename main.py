import cv2
import torch
from fastai.vision.all import load_learner, PILImage
import os
import gc
import boto3
import csv
import shutil
import argparse
import glob
import numpy as np
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel


# Global variables

# REQUIRED: REPLACE THIS WITH YOUR S3 BUCKET NAME
S3_BUCKET = "pascomm-crowd-counting"

# Set the interval for processing frames in seconds
# To reduce the number of frames processed and speed up the classification,
# only capture and process a frame every X seconds in the long video
PROCESS_INTERVAL = 5

# Local temporary directory for processing
# This directory will be created if it doesn't exist
# and will be used to store temporary files during processing
TEMP_DIR = "temp"

# S3 folder for output
# This is where the results will be uploaded in the S3 bucket
S3_OUTPUT_FOLDER = "results"

# Do not edit this unless you want to use your own models
FRAME_RELEVANCE_MODEL_PATH = "models/frame-relevance.pkl"
CROWD_COUNTER_MODEL_PATH = "models/crowd_counter.pt"

def download_from_s3(s3_client, bucket, key, local_path):
    """Download a file from S3 to a local path"""
    print(f"Downloading s3://{bucket}/{key} to {local_path}")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3_client.download_file(bucket, key, local_path)
    return local_path

def upload_to_s3(s3_client, local_path, bucket, key):
    """Upload a file to S3"""
    print(f"Uploading {local_path} to s3://{bucket}/{key}")
    s3_client.upload_file(local_path, bucket, key)

def upload_directory_to_s3(s3_client, local_dir, bucket, prefix):
    """Upload a directory to S3"""
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            s3_key = os.path.join(prefix, relative_path)
            upload_to_s3(s3_client, local_path, bucket, s3_key)

def process_video(model_path, video_path, video_name, temp_output_dir, s3_client, s3_bucket, s3_output_folder):
    """Process video frames and classify them using the model"""
    # Load the trained model
    learn = load_learner(model_path)
    IMG_SIZE = 320

    # Set up video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    frames_to_skip = int(fps * PROCESS_INTERVAL)

    print(f"Video info: {total_frames} frames, {fps} fps, duration: {video_duration/60:.1f} minutes")
    print(f"Processing every {PROCESS_INTERVAL} seconds ({frames_to_skip} frames)")
    print(f"Total frames to process: {total_frames//frames_to_skip}")

    # Set output settings
    video_output_dir = os.path.join(temp_output_dir, video_name)

    # Get all possible classes from the model
    class_names = learn.dls.vocab
    print(f"Creating directories for classes: {class_names}")

    # Create a directory structure for all possible classes
    for class_name in class_names:
        class_dir = os.path.join(video_output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        print(f"Created directory: {class_dir}")

    # Results storage
    results = []

    # Process frames at specified intervals
    current_frame = 0
    while current_frame < total_frames:
        # Set frame position directly to the next frame to process
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        # Read the frame
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame at position {current_frame}. Exiting.")
            break

        # Get video time
        video_time = current_frame / fps
        minutes, seconds = divmod(video_time, 60)

        # Get the dimensions of the frame
        height, width = frame.shape[:2]

        # Create a square crop from the center of the frame
        if height > width:
            margin = (height - width) // 2
            square_frame = frame[margin:margin+width, 0:width]
        else:
            margin = (width - height) // 2
            square_frame = frame[0:height, margin:margin+height]

        # Resize frame to 320px
        resized_frame = cv2.resize(square_frame, (IMG_SIZE, IMG_SIZE))

        # Convert frame to RGB (OpenCV uses BGR by default)
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        img = PILImage.create(rgb_frame)

        # Make prediction
        with torch.no_grad():
            pred_class, pred_idx, outputs = learn.predict(img)
            confidence = outputs[pred_idx].item()

        # Store results
        results.append({
            'frame': current_frame,
            'time': f"{int(minutes):02d}:{int(seconds):02d}",
            'class': pred_class,
            'confidence': confidence,
        })

        if pred_class != 'irrelevant' and confidence > 0.6:
            frame_filename = os.path.join(
                video_output_dir, 
                pred_class, 
                f"conf_{confidence:.3f}frame_{int(minutes):02d}m_{int(seconds):02d}s.jpg"
            )
            cv2.imwrite(frame_filename, frame)

        print(f"Processed {current_frame}/{total_frames} - Time: {int(minutes):02d}:{int(seconds):02d} - Class: {pred_class}, Conf: {confidence:.2f}")

        # Skip to next frame to process
        current_frame += frames_to_skip

        # Force garbage collection to free memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results to CSV
    results_csv_path = os.path.join(temp_output_dir, f"{video_name}_results.csv")
    with open(results_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'time', 'class', 'confidence'])
        writer.writeheader()
        writer.writerows(results)

    # Upload results to S3
    upload_directory_to_s3(s3_client, video_output_dir, s3_bucket, f"{s3_output_folder}/{video_name}")
    upload_to_s3(s3_client, results_csv_path, s3_bucket, f"{s3_output_folder}/{video_name}/frame_relevance_results.csv")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished classifying the frames. Results are saved to S3: s3://{s3_bucket}/{s3_output_folder}/{video_name}_results.csv")
    
    return f"{video_output_dir}/relevant"


def count_people_in_folder(model_path, folder_path, video_name, temp_output_dir, s3_client, s3_bucket, s3_output_folder):
    """Process all images in a folder and count people"""
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return
    
    print(f"Processing {len(image_files)} images")
    
    # Crowd counting model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=0.3,
        device="cpu",  # or 'cuda:0' if gpu is available
    )
    
    # Process images and store results
    results = []
    for image_file in image_files:
        file_name = os.path.basename(image_file)
        try:
            result = get_sliced_prediction(
                image_file,
                detection_model,
                slice_height=256,
                slice_width=256,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
            )
            people_count = len(result.object_prediction_list)
            results.append({
                'image_file': file_name,
                'people_count': people_count
            })
            print(f"{file_name}: {people_count} people")
        except Exception as e:
            print(f"Error with {file_name}: {e}")
    
    output_csv = os.path.join(temp_output_dir, 'people_count.csv')
    
    # Save to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['image_file', 'people_count'])
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    upload_to_s3(s3_client, output_csv, s3_bucket, f"{s3_output_folder}/{video_name}/people_count.csv")
    
    # Calculate statistics
    if results:
        calculate_statistics(results, video_name, temp_output_dir, s3_client, s3_bucket, s3_output_folder)

def calculate_statistics(results, video_name, temp_output_dir, s3_client, s3_bucket, s3_output_folder):
    """Calculate and save statistics"""
    counts = [result['people_count'] for result in results]
    
    # Calculate basic statistics
    stats = {
        'Total images': len(counts),
        'Minimum': min(counts),
        'Maximum': max(counts),
        'Average': np.mean(counts),
        'Median': np.median(counts),
        'Standard Deviation': np.std(counts),
        '25th percentile': np.percentile(counts, 25),
        '50th percentile': np.percentile(counts, 50),
        '75th percentile': np.percentile(counts, 75),
        '90th percentile': np.percentile(counts, 90),
        '95th percentile': np.percentile(counts, 95)
    }
    
    # Save stats to S3 CSV
    output_csv = os.path.join(temp_output_dir, 'stats_people_count.csv')
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Statistic', 'Value'])
        for key, value in stats.items():
            writer.writerow([key, f"{value:.2f}" if isinstance(value, float) else value])
    
    # Print summary
    print("\nStatistics Summary:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
        
    upload_to_s3(s3_client, output_csv, s3_bucket, f"{s3_output_folder}/{video_name}/stats_people_count.csv")


def main():
    print("called")
    parser = argparse.ArgumentParser(description='Process video frames with a fastai model')
    # Only keep the video-key argument, use global variables for the rest
    parser.add_argument('--video-key', required=True, help='S3 key for the video file')
    
    args = parser.parse_args()
    
    # Set up S3 client
    s3_client = boto3.client('s3')
   
    # Download video from S3
    video_name_with_extension = os.path.basename(args.video_key)
    video_name = os.path.splitext(video_name_with_extension)[0]
    video_path = os.path.join(TEMP_DIR, video_name_with_extension)
    print(f"Preparing to download {video_name_with_extension} from S3 and saving it to {video_path}")
    download_from_s3(s3_client, S3_BUCKET, f'input_videos/{args.video_key}', video_path)
    print("Finished downloading video")
        
    # Create output directory
    temp_output_dir = os.path.join(TEMP_DIR, 'output')
    os.makedirs(temp_output_dir, exist_ok=True)
        
    # Process the video using global variables for S3_BUCKET and OUTPUT_PREFIX
    output_video_dir = process_video(
        model_path=FRAME_RELEVANCE_MODEL_PATH,
        video_path=video_path,
        video_name=video_name,
        temp_output_dir=temp_output_dir,
        s3_client=s3_client,
        s3_bucket=S3_BUCKET,
        s3_output_folder=S3_OUTPUT_FOLDER
    )
        
    print("Finished identifying the relevant frames in the video")
    print(f"Output saved in {output_video_dir}")
    
    count_people_in_folder(        
        model_path=CROWD_COUNTER_MODEL_PATH,
        folder_path=output_video_dir,
        video_name=video_name,
        temp_output_dir=temp_output_dir,
        s3_client=s3_client,
        s3_bucket=S3_BUCKET,
        s3_output_folder=S3_OUTPUT_FOLDER
    )
    
    print("Finished counting the number of people")
    
    # (Optional): Delete temp folder
    #if os.path.exists(TEMP_DIR):
    #    shutil.rmtree(TEMP_DIR)

if __name__ == "__main__":
    main()