# Multi-Pipeline DeepStream Video Processor

This application spawns multiple concurrent DeepStream pipelines to process videos in parallel. Each pipeline runs in a separate process and uses the `deepstream-app` command with your custom YOLO model configuration to perform video decoding and inference.

## Features

- Process multiple videos concurrently from a folder using separate processes
- Automatic video file discovery (supports .mp4, .avi, .mov, .mkv)
- Configurable number of concurrent processes
- Configurable maximum number of videos to process
- Automatically starts processing next video when a process finishes
- Completes when all videos are processed

## Prerequisites

- DeepStream SDK 6.4+ installed with `deepstream-app` command available
- Custom YOLO model configuration (`config_infer_primary_yolo11_modified.txt`)
- Custom YOLO inference library (`nvdsinfer_custom_impl_Yolo/`)
- Model engine file (`model_b10_gpu0_fp32.engine`)
- Labels file (`labels.txt`)
- Python 3.x with multiprocessing support

## Usage

```bash
python run_multi_pipeline.py <video_folder> [--num-processes NUM] [--max-videos NUM]
```

### Arguments

- `<video_folder>`: Path to the folder containing video files (required)
- `--num-processes NUM`: Number of parallel processes to run (optional, default: 2)
- `--max-videos NUM`: Maximum number of videos to process (optional, default: all videos)

### Examples

Process all videos using 2 concurrent processes (default):
```bash
python3 run_multi_pipeline.py /path/to/videos
```

Process videos using 4 concurrent processes:
```bash
python3 run_multi_pipeline.py /path/to/videos --num-processes 4
```

Process only first 10 videos using 4 processes:
```bash
python3 run_multi_pipeline.py /path/to/videos --num-processes 4 --max-videos 10
```

Process videos with default Boreal dataset:
```bash
python3 run_multi_pipeline.py ../../datasets/Boreal-Forest-Fire/Boreal-Forest-Fire-Subset-B/Evo-Videos --num-processes 4 --max-videos 20
```

## Configuration

The script uses the following configuration files:

- `../deepstream_app_config.txt` - Main DeepStream configuration
- `../config_infer_primary_yolo11_modified.txt` - YOLO inference model configuration
- `../labels.txt` - Detection class labels
- `../nvdsinfer_custom_impl_Yolo/` - Custom YOLO inference library
- `../models/` - Model files (ONNX, etc.)
- `../model_b10_gpu0_fp32.engine` - Pre-built TensorRT engine (batch size 10)

## How It Works

1. Script reads all video files from the specified folder
2. Limits to `--max-videos` if specified
3. Creates N worker processes (specified by `--num-processes`)
4. Each worker process:
   - Takes a video from the queue
   - Creates a temporary directory with modified config files
   - Replaces the video URI in the deepstream config
   - Runs `deepstream-app -c <config>` 
   - Cleans up temporary files after completion
5. Worker continues until queue is empty

## Notes

- Display output is disabled in the default config (`type=1` fakesink)
- Each process runs in isolation with its own temporary config directory
- The engine filename is automatically adjusted to match batch size (b10)
- Relative paths in config files are handled by copying necessary files to temp directories


- `DEEPSTREAM_TEST1_EXEC`: Path to the deepstream-test1 executable
- `CONFIG_FILE`: Path to the deepstream-test1 config file
- `NUM_PIPELINES`: Default number of concurrent pipelines

## How It Works

1. The application scans the specified folder for video files
2. Creates a queue with all discovered videos
3. Spawns the specified number of worker threads
4. Each worker thread:
   - Takes a video from the queue
   - Runs deepstream-test1 on that video
   - Repeats until the queue is empty
5. The application waits for all videos to be processed before exiting

## Output

The application prints:
- When each pipeline starts processing a video
- When each pipeline finishes processing a video
- Error messages if a pipeline fails
- A completion message when all videos are processed
