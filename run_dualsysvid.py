import argparse
import cv2
import numpy as np
import torch
import os
import matplotlib
from tqdm import tqdm
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2
import pygame # Added for audio
import math # Added for audio calculations
from scipy.io import wavfile # Added to save audio file

# --- Audio Generation Function ---
def generate_audio_feedback(all_frame_detections, fps, frame_width, max_dist, sound_config):
    """
    Generates a binaural WAV audio file based on detected objects per frame.

    Args:
        all_frame_detections (list):List of detections for each frame.
                                    Each element is a list of detection dicts.
        fps (float): Frames per second of the video.
        frame_width (int): Width of the video frame for panning calculation.
        max_dist (float): Maximum distance for audio feedback generation.
        sound_config (dict): Configuration parameters for sounds.
    """
    print("\nGenerating audio feedback...")
    sample_rate = sound_config['sample_rate']
    pygame.mixer.init(frequency=sample_rate, size=-16, channels=2) # Initialize mixer

    # Sound generation parameters
    beep_duration_samples = int(sound_config['beep_duration_s'] * sample_rate)
    hum_freq = sound_config['hum_frequency']
    person_beep_freq = sound_config['person_beep_frequency']
    other_beep_freq = sound_config['other_beep_frequency']

    # Function to generate a sine wave beep
    def generate_beep(frequency, duration_samples):
        t = np.linspace(0., duration_samples / sample_rate, duration_samples, endpoint=False)
        beep = 0.5 * np.sin(2 * np.pi * frequency * t) # Amplitude 0.5 to leave headroom
        # Add a simple fade-out to avoid clicking
        fade_len = min(duration_samples // 10, int(0.01 * sample_rate)) # 10% or 10ms fade
        if fade_len > 1:
            fade = np.linspace(1.0, 0.0, fade_len)
            beep[-fade_len:] *= fade
        return beep.astype(np.float32)

    # Generate base sound samples
    person_beep_sound = generate_beep(person_beep_freq, beep_duration_samples)
    other_beep_sound = generate_beep(other_beep_freq, beep_duration_samples)
    # Generate a short segment of hum, long enough for one frame, we'll repeat/use as needed
    hum_segment_duration = 1.0 / fps # Duration for one frame
    hum_segment_samples = int(hum_segment_duration * sample_rate)
    hum_sound_segment = generate_beep(hum_freq, hum_segment_samples) # Using beep func for simplicity

    # Calculate total audio duration and buffer size
    num_frames = len(all_frame_detections)
    total_duration_s = num_frames / fps
    total_samples = int(total_duration_s * sample_rate)
    audio_buffer = np.zeros((total_samples, 2), dtype=np.float32) # Stereo float buffer

    # Calculate frame intervals for beeps
    person_beep_interval_frames = int(sound_config['person_beep_interval_s'] * fps)
    other_beep_interval_frames = int(sound_config['other_beep_interval_s'] * fps)

    # --- Process detections frame by frame to build audio ---
    pbar_audio = tqdm(total=num_frames, desc="Generating audio")
    for frame_idx, frame_detections in enumerate(all_frame_detections):
        start_sample = int((frame_idx / fps) * sample_rate)
        end_sample = start_sample + hum_segment_samples # End sample for this frame's hum

        # Ensure we don't write past the buffer end
        end_sample = min(end_sample, total_samples)
        current_hum_samples = min(hum_segment_samples, total_samples - start_sample)
        if current_hum_samples <= 0:
            continue # Skip if no samples fit

        # --- Process Contour Detections (Walls/Environment) ---
        # Simplified: Check if *any* close contour exists on left/right
        left_hum_volume = 0.0
        right_hum_volume = 0.0
        min_left_depth = float('inf')
        min_right_depth = float('inf')

        for det in frame_detections:
            if det['type'] == 'contour':
                center_x = det['center'][0]
                depth = det['depth']
                if center_x < frame_width / 2:
                    min_left_depth = min(min_left_depth, depth)
                else:
                    min_right_depth = min(min_right_depth, depth)

        # Calculate volume based on the *closest* wall on each side
        if min_left_depth <= max_dist:
            # Volume increases linearly from max_dist (0 volume) to 5m (max volume)
            volume = 1.0 if min_left_depth <= 5.0 else max(0.0, 1.0 - (min_left_depth - 5.0) / (max_dist - 5.0))
            left_hum_volume = volume * sound_config['hum_max_volume'] # Apply max vol scaling

        if min_right_depth <= max_dist:
            volume = 1.0 if min_right_depth <= 5.0 else max(0.0, 1.0 - (min_right_depth - 5.0) / (max_dist - 5.0))
            right_hum_volume = volume * sound_config['hum_max_volume'] # Apply max vol scaling

        # Add hum to buffer if volume > 0
        if left_hum_volume > 0 or right_hum_volume > 0:
            # Simple panning: full volume to the respective channel
            hum_slice = hum_sound_segment[:current_hum_samples]
            audio_buffer[start_sample:end_sample, 0] += hum_slice * left_hum_volume
            audio_buffer[start_sample:end_sample, 1] += hum_slice * right_hum_volume


        # --- Process YOLO Detections (Person, Others) ---
        for det in frame_detections:
            if det['type'] == 'yolo':
                depth = det['depth']
                center_x = det['center'][0]
                class_name = det['class']

                # Skip if beyond max distance
                if depth > max_dist:
                    continue

                # Determine beep type and interval
                is_person = (class_name == 'person')
                beep_sound = person_beep_sound if is_person else other_beep_sound
                interval_frames = person_beep_interval_frames if is_person else other_beep_interval_frames
                max_beep_vol = sound_config['person_beep_max_volume'] if is_person else sound_config['other_beep_max_volume']

                # Check if it's time for a beep for this type
                if interval_frames > 0 and frame_idx % interval_frames == 0:
                    # Calculate volume (max at 5m, decreasing to max_dist)
                    volume = 1.0 if depth <= 5.0 else max(0.0, 1.0 - (depth - 5.0) / (max_dist - 5.0))
                    volume *= max_beep_vol # Apply max vol scaling


                    # Calculate panning (0.0 = left, 1.0 = right)
                    # Clamp center_x just in case it's slightly outside bounds
                    clamped_center_x = max(0, min(frame_width - 1, center_x))
                    pan_right = clamped_center_x / frame_width
                    pan_left = 1.0 - pan_right

                    # Apply panning using constant power panning (approximated) for better feel
                    # Left Volume = Total Volume * cos(pan_angle)
                    # Right Volume = Total Volume * sin(pan_angle)
                    # Where pan_angle goes from 0 (left) to PI/2 (right)
                    pan_angle = pan_right * (math.pi / 2)
                    left_vol = volume * math.cos(pan_angle)
                    right_vol = volume * math.sin(pan_angle)

                    # Mix beep into the buffer
                    beep_start_sample = start_sample
                    beep_end_sample = beep_start_sample + len(beep_sound)

                    # Ensure beep fits within the buffer
                    if beep_start_sample < total_samples:
                        samples_to_write = min(len(beep_sound), total_samples - beep_start_sample)
                        if samples_to_write > 0:
                             audio_buffer[beep_start_sample : beep_start_sample + samples_to_write, 0] += beep_sound[:samples_to_write] * left_vol
                             audio_buffer[beep_start_sample : beep_start_sample + samples_to_write, 1] += beep_sound[:samples_to_write] * right_vol
        pbar_audio.update(1)

    pbar_audio.close()

    # --- Finalize and Save Audio ---
    # Clip buffer to prevent overload distortion
    audio_buffer = np.clip(audio_buffer, -1.0, 1.0)

    # Convert to 16-bit PCM
    audio_int16 = (audio_buffer * 32767).astype(np.int16)

    # Save as WAV file
    output_audio_path = os.path.join('output', 'feedback_audio.wav')
    try:
        wavfile.write(output_audio_path, sample_rate, audio_int16)
        print(f"Audio feedback saved to: {output_audio_path}")
    except Exception as e:
        print(f"Error saving WAV file: {e}")

    pygame.mixer.quit() # Clean up pygame mixer


# --- Main Processing ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dual Detection System with YOLO and Contour-based Detection')
    parser.add_argument('--pred-only', action='store_true', help='Only include the depth prediction in output')
    parser.add_argument('--grayscale', action='store_true', help='Display depth in grayscale')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)

    # Device selection
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Load Depth Anything V2 model
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }
    # depth anything model configs
    encoder = 'vits'          # 'vits' (small), 'vitb' (base), vitl (large)
    dataset = 'vkitti'        # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 80            # 20 for indoor model, 80 for outdoor model

    # contours configs
    depth_threshold = 5.0     # Depth threshold for nearby objects (in meters) / (perlu di test buat diluar plg bagus brp meter)
    min_area = 5000           # Minimum area for contours to be considered valid

    # YOLO resolution configs
    yolo_resolution = 'default' # Video resolution for the YOLO model to process, separated from the resolution that depth anything takes and can be set to "default" for original video resolution
    maximum_distance_detection = 15 # Maximum distance (in meters) for YOLO to detect objects in the valid_classes list AND for audio feedback

    # depth anything input/resolution configuration parameters
    depth_input_size = 336    # Input size for depth estimation (smaller = faster = less accurate), default is 518, other options are 448, 392, 336, 280, 224, 168, 112, 56
    input_resolution = 'default' # Input resolution for depth anything, width x height and can be set to "default" for original resolution

    # sound configurations
    sound_config = {
        'sample_rate': 44100,              # Audio sample rate in Hz
        
        # walls/environment hum
        'hum_frequency': 110,              # Frequency for wall/environment hum in Hz (A2 note)
        'hum_max_volume': 0.6,             # Maximum volume for hum (0.0 to 1.0)
        
        # person beep
        'person_beep_frequency': 880,      # Frequency for person beep in Hz (A5 note)
        'person_beep_interval_s': 0.3,     # Interval between person beeps in seconds
        'person_beep_max_volume': 0.9,     # Maximum volume for person beeps (0.0 to 1.0)
        
        # other beep (bicycle, motorcycle, truck, car, bus)
        'other_beep_frequency': 440,       # Frequency for other objects beep in Hz (A4 note)
        'other_beep_interval_s': 0.3,      # Interval between other object beeps in seconds
        'other_beep_max_volume': 0.7,      # Maximum volume for other beeps (0.0 to 1.0)
        
        'beep_duration_s': 0.2,            # Duration of each beep sound in seconds
        # Add other potential configs like volume falloff curve type if needed later
    }

    # Classes to detect
    valid_classes = ['person', 'bicycle', 'motorcycle', 'truck', 'car', 'bus']

    # Open input video to get original dimensions
    input_path = os.path.join('input', 'video.mp4')
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Error: Could not open video file {input_path}.")

    # Get original video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Input video: {input_path} ({original_width}x{original_height} @ {fps:.2f} FPS, {frame_count} frames)")


    # Parse input resolution
    if input_resolution == 'default':
        input_width, input_height = original_width, original_height
    else:
        input_width, input_height = map(int, input_resolution.split('x'))

    # Parse YOLO resolution
    if yolo_resolution == 'default':
        yolo_width, yolo_height = original_width, original_height
    else:
        yolo_width, yolo_height = map(int, yolo_resolution.split('x'))

    # Reset video capture to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    depth_model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    depth_model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
    depth_model.to(DEVICE)
    depth_model.eval()

    # Load YOLOv11 segmentation model
    yolo_model = YOLO("yolo11n-seg.pt") # Make sure this model exists or use a different one

    # Prepare output video writer
    output_path = os.path.join('output', 'processed_output.mp4')

    # Define the codec and create VideoWriter object
    # If pred-only, we'll just have the depth map; otherwise, we'll have side-by-side view
    if args.pred_only:
        output_vid_width, output_vid_height = input_width, input_height
    else:
        output_vid_width, output_vid_height = input_width * 2, input_height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use 'mp4v' for MP4 output
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_vid_width, output_vid_height))
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        cap.release()
        exit()
    print(f"Output video: {output_path} ({output_vid_width}x{output_vid_height} @ {fps:.2f} FPS)")


    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    # Create a progress bar
    pbar = tqdm(total=frame_count, desc="Processing video")

    # List to store detections for each frame for audio generation later
    all_frame_detections = []

    while cap.isOpened():
        ret, raw_frame = cap.read()
        if not ret:
            break

        # Resize the input frame to the desired resolution if not using default
        if input_resolution != 'default' or (original_width != input_width or original_height != input_height):
            raw_frame = cv2.resize(raw_frame, (input_width, input_height))
        # Ensure raw_frame dimensions are correct *after* potential resize
        current_frame_height, current_frame_width = raw_frame.shape[:2]


        # Create a copy specifically for YOLO if using different resolution
        if yolo_resolution != 'default' or (yolo_width != input_width or yolo_height != input_height):
            yolo_frame = cv2.resize(raw_frame, (yolo_width, yolo_height))
        else:
            yolo_frame = raw_frame

        # Run depth estimation with the smaller input size
        depth_map = depth_model.infer_image(raw_frame, depth_input_size)
        original_depth = depth_map.copy() # Keep original metric depth

        # --- Depth Visualization (same as before) ---
        min_val, max_val = np.nanmin(depth_map[depth_map > 0]), np.nanmax(depth_map) # Avoid zero depth if possible
        if max_val > min_val: # Avoid division by zero
            depth_map_normalized = ((np.clip(depth_map, min_val, max_val) - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
        else:
            depth_map_normalized = np.zeros_like(depth_map, dtype=np.uint8)

        if args.grayscale:
            depth_display = np.repeat(depth_map_normalized[..., np.newaxis], 3, axis=-1)
        else:
            depth_display = (cmap(depth_map_normalized)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            # Ensure depth_display has same dims as raw_frame for concatenation
            if depth_display.shape[0] != current_frame_height or depth_display.shape[1] != current_frame_width:
                depth_display = cv2.resize(depth_display, (current_frame_width, current_frame_height))


        # Prepare frame for visualization
        output_frame = raw_frame.copy()

        # Calculate scale factors if YOLO uses different resolution
        scale_x = current_frame_width / yolo_frame.shape[1]
        scale_y = current_frame_height / yolo_frame.shape[0]

        # Create a mask to track YOLO detections (to avoid duplicate contour detections)
        yolo_mask = np.zeros((current_frame_height, current_frame_width), dtype=np.uint8)

        # List to hold detections for the *current* frame
        current_detections = []

        # 1. PRIMARY DETECTOR: Run YOLO segmentation on the YOLO-specific frame
        yolo_detections_viz = [] # For visualization drawing
        try:
            results = yolo_model(yolo_frame, verbose=False) # Reduce console spam
        except Exception as e:
            print(f"YOLO inference error: {e}")
            results = [] # Handle potential errors gracefully

        for result in results:
            if hasattr(result, 'masks') and result.masks is not None and hasattr(result, 'boxes') and result.boxes is not None:
                for box, cls, mask_data in zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.cls.cpu().numpy(),
                    result.masks.data.cpu().numpy() # Get mask data
                ):
                    # Get class name
                    class_name = yolo_model.names[int(cls)]

                    # Skip if not in our valid_classes list
                    if class_name not in valid_classes:
                        continue

                    # Scale the bounding box if resolutions differ
                    if yolo_resolution != 'default' or (yolo_width != input_width or yolo_height != input_height):
                        x1, y1, x2, y2 = map(int, [box[0] * scale_x, box[1] * scale_y,
                                                   box[2] * scale_x, box[3] * scale_y])
                    else:
                        x1, y1, x2, y2 = map(int, box)

                    # Clamp bounding box to frame dimensions
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(current_frame_width, x2), min(current_frame_height, y2)

                    # Generate binary mask from the segmentation output
                    binary_mask = (mask_data > 0.5).astype(np.uint8)
                    if y1 < y2 and x1 < x2 and binary_mask.sum() > 0:
                        # Resize mask to match the *current* frame dimensions (output_frame)
                        resized_mask = cv2.resize(binary_mask, (current_frame_width, current_frame_height), interpolation=cv2.INTER_NEAREST)

                        # Isolate depth values for the segmented object using the original metric depth
                        masked_depth = original_depth.copy()
                        masked_depth[resized_mask == 0] = np.nan # Use NaN for non-object pixels
                        valid_depth_values = masked_depth[~np.isnan(masked_depth)]

                        if len(valid_depth_values) > 0:
                            # Compute the median depth as the object's distance
                            object_depth = np.nanmedian(valid_depth_values)

                            # Skip objects beyond the maximum detection distance (for audio consistency)
                            if object_depth > maximum_distance_detection:
                                continue

                            # Add this detection to the YOLO mask (to avoid duplicate contour detections)
                            yolo_mask = cv2.bitwise_or(yolo_mask, resized_mask)

                            # Find contours for visualization drawing
                            contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                            # Calculate center using moments for better accuracy with masks
                            moments = cv2.moments(resized_mask)
                            if moments["m00"] != 0:
                                center_x = int(moments["m10"] / moments["m00"])
                                center_y = int(moments["m01"] / moments["m00"])
                            else: # Fallback for empty moments
                                center_x = (x1 + x2) // 2
                                center_y = (y1 + y2) // 2

                            # Clamp center coordinates
                            center_x = max(0, min(current_frame_width - 1, center_x))
                            center_y = max(0, min(current_frame_height - 1, center_y))


                            # Save detection details for audio generation and visualization
                            detection_info = {
                                'type': 'yolo',
                                'class': class_name,
                                #'contours': contours, # Storing contours for potentially hundreds of frames can use a lot of memory
                                'bbox': (x1, y1, x2, y2),
                                'center': (center_x, center_y),
                                'depth': float(object_depth), # Ensure depth is float
                                'color': (0, 255, 0), # Green for YOLO detections
                                # Store contours separately only for viz if needed, to save memory in all_frame_detections
                            }
                            current_detections.append(detection_info)
                            yolo_detections_viz.append({**detection_info, 'contours': contours}) # Add contours only for viz list


        # Apply morphological operations to refine the YOLO mask
        kernel = np.ones((5, 5), np.uint8)
        yolo_mask_refined = cv2.dilate(yolo_mask, kernel, iterations=2) # Slightly larger dilation

        # 2. SECONDARY DETECTOR: Apply contour-based detection for objects YOLO might miss

        # Create binary mask of "close" areas based on depth threshold using original metric depth
        depth_binary = ((original_depth > 0) & (original_depth < depth_threshold)).astype(np.uint8) * 255

        # Apply morphological operations to reduce noise
        depth_binary = cv2.morphologyEx(depth_binary, cv2.MORPH_OPEN, kernel) # Remove small noise
        depth_binary = cv2.morphologyEx(depth_binary, cv2.MORPH_CLOSE, kernel) # Fill small holes

        # Create a new mask that excludes YOLO detections
        exclusive_depth_mask = depth_binary.copy()
        exclusive_depth_mask[yolo_mask_refined > 0] = 0 # Remove areas detected by YOLO

        # Find contours in the exclusive mask
        contours_env, _ = cv2.findContours(exclusive_depth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the center of the screen
        screen_center_x = current_frame_width // 2
        screen_center_y = current_frame_height // 2

        # Process each contour that meets minimum area requirement
        contour_detections_viz = [] # For visualization drawing
        for i, contour in enumerate(contours_env):
            area = cv2.contourArea(contour)
            if area >= min_area:
                # Find bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Extract depth values for the masked region using original metric depth
                contour_mask = np.zeros_like(depth_binary)
                cv2.drawContours(contour_mask, [contour], 0, 255, -1)

                masked_depth = original_depth.copy()
                masked_depth[contour_mask == 0] = np.nan
                valid_depth_values = masked_depth[~np.isnan(masked_depth)]

                if len(valid_depth_values) > 0:
                    # Compute median depth as the object's distance
                    object_depth = np.nanmedian(valid_depth_values)

                    # Skip if beyond max distance (should be covered by depth_threshold, but good practice)
                    # Also skip if depth is effectively zero
                    if object_depth > maximum_distance_detection or object_depth < 0.1:
                        continue

                    # Use centroid of contour for center point
                    moments = cv2.moments(contour)
                    if moments["m00"] != 0:
                        center_x = int(moments["m10"] / moments["m00"])
                        center_y = int(moments["m01"] / moments["m00"])
                    else: # Fallback
                        center_x = x + w // 2
                        center_y = y + h // 2

                    # Clamp center coordinates
                    center_x = max(0, min(current_frame_width - 1, center_x))
                    center_y = max(0, min(current_frame_height - 1, center_y))

                    # Save detection details for audio generation and visualization
                    detection_info = {
                        'type': 'contour',
                        #'contour': contour, # Avoid storing contours for memory
                        'bbox': (x, y, x+w, y+h),
                        'center': (center_x, center_y),
                        'depth': float(object_depth), # Ensure depth is float
                        'color': (255, 0, 0), # Blue for contour detections
                        'id': i+1
                    }
                    current_detections.append(detection_info)
                    contour_detections_viz.append({**detection_info, 'contour': contour}) # Add contour only for viz


        # Store the detections for this frame
        all_frame_detections.append(current_detections)

        # 3. VISUALIZATION: Draw all detections on the output frame

        # First, draw contour detections (using contour_detections_viz)
        for detection in contour_detections_viz:
            # Create a colored overlay for the contour
            overlay = output_frame.copy()
            cv2.drawContours(overlay, [detection['contour']], 0, detection['color'], -1)
            cv2.addWeighted(overlay, 0.4, output_frame, 0.6, 0, output_frame)

            # Draw contour outline and bounding box
            cv2.drawContours(output_frame, [detection['contour']], 0, detection['color'], 2)
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), detection['color'], 2)

            # Annotate with object ID and distance
            cx, cy = detection['center']
            cv2.circle(output_frame, (cx, cy), 5, (0, 0, 255), -1)
            if depth_display is not None: # Check if depth_display exists
                # Ensure cx, cy are within depth_display bounds before drawing
                dh, dw = depth_display.shape[:2]
                if 0 <= cx < dw and 0 <= cy < dh:
                    cv2.circle(depth_display, (cx, cy), 5, (0, 0, 255), -1)

            cv2.putText(output_frame, f"Env {detection['id']}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, detection['color'], 2)
            cv2.putText(output_frame, f"D: {detection['depth']:.2f}m",
                        (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Then, draw YOLO detections (using yolo_detections_viz)
        for detection in yolo_detections_viz:
            # Create a colored overlay
            overlay = output_frame.copy()
            # Ensure contours exist before drawing
            if 'contours' in detection and detection['contours']:
                cv2.drawContours(overlay, detection['contours'], -1, detection['color'], -1)
                cv2.addWeighted(overlay, 0.4, output_frame, 0.6, 0, output_frame)
                # Draw contour outline
                cv2.drawContours(output_frame, detection['contours'], -1, detection['color'], 2)

            # Draw bounding box
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), detection['color'], 2)

            # Annotate with class name and distance
            cx, cy = detection['center']
            cv2.circle(output_frame, (cx, cy), 5, (0, 0, 255), -1)
            if depth_display is not None: # Check if depth_display exists
                # Ensure cx, cy are within depth_display bounds before drawing
                dh, dw = depth_display.shape[:2]
                if 0 <= cx < dw and 0 <= cy < dh:
                    cv2.circle(depth_display, (cx, cy), 5, (0, 0, 255), -1)

            cv2.putText(output_frame, f"{detection['class']}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, detection['color'], 2)
            cv2.putText(output_frame, f"D: {detection['depth']:.2f}m",
                        (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


        # Write frame to output video
        if args.pred_only:
            # Ensure depth_display matches output dimensions if needed
            if depth_display.shape[1] != output_vid_width or depth_display.shape[0] != output_vid_height:
                depth_display_resized = cv2.resize(depth_display, (output_vid_width, output_vid_height))
                out.write(depth_display_resized)
            else:
                out.write(depth_display)
        else:
            # Ensure both frames have the same dimensions for concatenation
            h1, w1 = output_frame.shape[:2]
            h2, w2 = depth_display.shape[:2]
             # Make sure they are *exactly* the input_width/height derived earlier
            if h1 != input_height or w1 != input_width:
                output_frame = cv2.resize(output_frame, (input_width, input_height))
            if h2 != input_height or w2 != input_width:
                depth_display = cv2.resize(depth_display, (input_width, input_height))

            combined_display = cv2.hconcat([output_frame, depth_display])
            out.write(combined_display)

        # Update progress bar
        pbar.update(1)

    # Clean up
    pbar.close()
    cap.release()
    out.release()
    print(f"\nVideo processing complete. Output saved to: {output_path}")

    # --- Generate Audio Feedback ---
    # Pass the width used for visualization/panning (input_width)
    generate_audio_feedback(all_frame_detections, fps, input_width, maximum_distance_detection, sound_config)