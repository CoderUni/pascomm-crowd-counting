# Video Crowd Counter

An automated solution for filtering relevant frames and crowd counting in long videos. This system downloads long videos from AWS S3, identifies relevant frames, and counts the number of people present in those frames.

## Overview

This project automates the following workflow:
1. Downloads a video from an S3 bucket
2. Processes the video to identify relevant frames using the `frame-relevance` model
3. Analyzes the relevant frames to count the number of people in each frame using the `crowd_counter` model
4. Calculates crowd statistics and saves results back to S3

## Prerequisites

- Python 3.9.X (tested using v3.9.6)
- AWS account with access to S3
- AWS credentials configured on your machine
- At least 3GB of free disk space for temporary processing
- 4GB RAM minimum (8GB recommended)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/video-crowd-counter.git
   cd video-crowd-counter
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure your AWS credentials are configured and have read/write access to an S3 bucket:
   ```bash
   aws configure
   ```

## Usage

### Basic Usage

Run the script by specifying the S3 key of the video to process:

```bash
python main.py --video-key "your_video_filename.mp4"
```

Note: The script assumes the video is stored in the `input_videos/` directory in the S3 bucket.

### Example

```bash
python main.py --video-key "event_recording_2023_05_21.mp4"
```

This command will:
1. Download `event_recording_2023_05_21.mp4` from `s3://bucket-name/input_videos/`
2. Process the video to identify relevant frames
3. Count people in those frames
4. Upload results to `s3://bucket-name/results/event_recording_2023_05_21/`

## Output Structure

The script produces the following outputs in the S3 bucket:

```
results/
└── video_name/
    ├── frame_relevance_results.csv  # Frame classification results
    ├── people_count.csv             # People count for each relevant frame
    ├── stats_people_count.csv       # Statistical analysis of crowd numbers
    └── relevant/                    # Directory containing relevant frame images
```

### Output Files

- `frame_relevance_results.csv`: Contains frame number, timestamp, classification and confidence
- `people_count.csv`: Lists each image file with the corresponding count of people detected
- `stats_people_count.csv`: Statistical analysis including min/max/average/median attendance and percentile distribution
- `relevant/`: This folder contains all the images that are considered relevant by frame-relevance.pkl

## Configuration

The main configuration variables are defined at the top of the script:

```python
# REQUIRED: REPLACE THIS WITH YOUR S3 BUCKET NAME
S3_BUCKET = "pascomm-crowd-counting"

# Better to keep the default values
TEMP_DIR = "temp"  # Local temporary directory for processing
S3_OUTPUT_FOLDER = "results"  # S3 folder for output

# Do not edit this unless you want to use your own models
FRAME_RELEVANCE_MODEL_PATH = "models/frame-relevance.pkl"  # Path to frame relevance model
CROWD_COUNTER_MODEL_PATH = "models/crowd_counter.pt"  # Path to crowd counting model
```

You can modify these values directly in the script if needed.

## Advanced Configuration

### Frame Processing Interval

By default, the script processes one frame every 5 seconds. To change this, modify the `PROCESS_INTERVAL` variable at the top of the script:

```python
# Set the time interval for processing frames (in seconds)
PROCESS_INTERVAL = 5  # Change this to your desired interval
```

### Crowd Detection Confidence Threshold

To adjust the sensitivity of the crowd counter, modify the confidence threshold:

```python
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=0.3,  # Adjust this value (0.0-1.0)
    device="cpu",  # or 'cuda:0' if gpu is available
)
```

## Performance Considerations

- GPU acceleration is supported and recommended for faster processing
- For GPU support, ensure you have CUDA installed and change `device="cpu"` to `device="cuda:0"` 
- Processing time depends on video length and chosen processing interval (Usually takes a few mins)
- Temporary files are stored locally in the `temp` directory
- To automatically delete temporary files after processing, uncomment the cleanup code at the end of `main()`

## Troubleshooting

### Common Issues

1. **AWS Credentials Error**:
   ```
   botocore.exceptions.NoCredentialsError: Unable to locate credentials
   ```
   
   Solution: Configure AWS credentials using `aws configure`

2. **Model Not Found**:
   ```
   FileNotFoundError: [Errno 2] No such file or directory: 'models/frame-relevance.pkl'
   ```
   
   Solution: Ensure the model files are in the correct location

3. **Memory Error**:
   ```
   MemoryError: ...
   ```
   
   Solution: Process videos in smaller chunks or increase the frame interval