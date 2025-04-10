import argparse
import cv2
import numpy as np
import torch
import matplotlib
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2
import pygame  # Added for audio
import math    # Added for audio calculations
import threading  # Added for parallel audio processing
import time     # Added for timing audio updates

# --- Audio Generation Function ---
def init_audio_system(sound_config):
    """
    Initialize the audio system with pygame
    """
    pygame.mixer.init(frequency=sound_config['sample_rate'], size=-16, channels=2)
    pygame.mixer.set_num_channels(8)  # Allocate enough channels for concurrent sounds
    
    # Generate the different sound samples we'll use
    sounds = {}
    
    # Function to generate a sine wave beep
    def generate_beep(frequency, duration_samples):
        sample_rate = sound_config['sample_rate']
        t = np.linspace(0., duration_samples / sample_rate, duration_samples, endpoint=False)
        beep = 0.5 * np.sin(2 * np.pi * frequency * t)  # Amplitude 0.5 to leave headroom
        # Add a simple fade-out to avoid clicking
        fade_len = min(duration_samples // 10, int(0.01 * sample_rate))  # 10% or 10ms fade
        if fade_len > 1:
            fade = np.linspace(1.0, 0.0, fade_len)
            beep[-fade_len:] *= fade
        return beep.astype(np.float32)

    # Generate sound samples
    beep_duration_samples = int(sound_config['beep_duration_s'] * sound_config['sample_rate'])
    
    # Person beep
    person_beep = generate_beep(sound_config['person_beep_frequency'], beep_duration_samples)
    # Other object beep
    other_beep = generate_beep(sound_config['other_beep_frequency'], beep_duration_samples)
    # Hum sound for environment/walls
    hum = generate_beep(sound_config['hum_frequency'], beep_duration_samples)
    
    # Convert to stereo and prepare for pygame
    def to_pygame_sound(audio_data):
        # Convert mono to stereo (duplicated)
        stereo = np.column_stack((audio_data, audio_data))
        # Convert to 16-bit integers
        audio_int16 = (stereo * 32767).astype(np.int16)
        # Create pygame Sound object
        return pygame.mixer.Sound(audio_int16)
    
    sounds['person'] = to_pygame_sound(person_beep)
    sounds['other'] = to_pygame_sound(other_beep)
    sounds['hum'] = to_pygame_sound(hum)
    
    return sounds

def play_audio_for_detections(detections, frame_width, max_dist, sounds, sound_config):
    """
    Play audio cues for the detected objects in real-time
    """
    # Keep track of unique object sounds to prevent too many overlapping sounds
    person_played = False
    other_played = False
    left_hum_played = False
    right_hum_played = False
    
    # Reset audio - stop any previous sounds
    pygame.mixer.stop()
    
    # Process environment/contour detections first
    left_hum_volume = 0.0
    right_hum_volume = 0.0
    min_left_depth = float('inf')
    min_right_depth = float('inf')
    
    # Find closest contours (walls/environment) on left/right sides
    for det in detections:
        if det['type'] == 'contour':
            center_x = det['center'][0]
            depth = det['depth']
            if center_x < frame_width / 2:
                min_left_depth = min(min_left_depth, depth)
            else:
                min_right_depth = min(min_right_depth, depth)
    
    # Play hum for left side if close enough
    if min_left_depth <= max_dist:
        # Volume increases as objects get closer
        volume = 1.0 if min_left_depth <= 5.0 else max(0.0, 1.0 - (min_left_depth - 5.0) / (max_dist - 5.0))
        left_hum_volume = volume * sound_config['hum_max_volume']
        
        if left_hum_volume > 0.05 and not left_hum_played:  # Threshold to avoid very quiet sounds
            left_hum = sounds['hum']
            left_hum.set_volume(left_hum_volume, 0.0)  # Left channel only
            left_hum.play(loops=0)
            left_hum_played = True
    
    # Play hum for right side if close enough
    if min_right_depth <= max_dist:
        volume = 1.0 if min_right_depth <= 5.0 else max(0.0, 1.0 - (min_right_depth - 5.0) / (max_dist - 5.0))
        right_hum_volume = volume * sound_config['hum_max_volume']
        
        if right_hum_volume > 0.05 and not right_hum_played:  # Threshold to avoid very quiet sounds
            right_hum = sounds['hum']
            right_hum.set_volume(0.0, right_hum_volume)  # Right channel only
            right_hum.play(loops=0)
            right_hum_played = True
    
    # Process YOLO detections (persons and other objects)
    for det in detections:
        if det['type'] == 'yolo':
            depth = det['depth']
            center_x = det['center'][0]
            class_name = det['class']
            
            # Skip if beyond max distance
            if depth > max_dist:
                continue
            
            # Determine sound type and settings
            is_person = (class_name == 'person')
            sound = sounds['person'] if is_person else sounds['other']
            max_vol = sound_config['person_beep_max_volume'] if is_person else sound_config['other_beep_max_volume']
            
            # Skip if we already played this type of sound in this frame
            if (is_person and person_played) or (not is_person and other_played):
                continue
            
            # Calculate volume based on distance
            volume = 1.0 if depth <= 5.0 else max(0.0, 1.0 - (depth - 5.0) / (max_dist - 5.0))
            volume *= max_vol
            
            # Calculate panning (left/right balance)
            clamped_center_x = max(0, min(frame_width - 1, center_x))
            pan_right = clamped_center_x / frame_width
            
            # Apply constant power panning for better audio experience
            pan_angle = pan_right * (math.pi / 2)
            left_vol = volume * math.cos(pan_angle)
            right_vol = volume * math.sin(pan_angle)
            
            # Play the sound
            sound.set_volume(left_vol, right_vol)
            sound.play(loops=0)
            
            # Mark this type as played
            if is_person:
                person_played = True
            else:
                other_played = True

# Function to handle audio processing in a separate thread
def audio_thread_function(audio_queue, frame_width, max_dist, sounds, sound_config):
    """
    Thread function to process audio feedback in parallel with video processing
    """
    last_audio_time = 0
    min_audio_interval = 0.1  # Minimum time between audio updates (seconds)
    
    while True:
        # Get latest detections from the queue
        try:
            detections = audio_queue.pop()
            current_time = time.time()
            
            # Limit audio updates to avoid overwhelming the audio system
            if current_time - last_audio_time >= min_audio_interval:
                play_audio_for_detections(detections, frame_width, max_dist, sounds, sound_config)
                last_audio_time = current_time
                
            # Small sleep to prevent thread from consuming too much CPU
            time.sleep(0.01)
        except IndexError:
            # Queue is empty, wait a bit
            time.sleep(0.01)
        except Exception as e:
            print(f"Audio thread error: {e}")
            # If queue is None, exit the thread (program is shutting down)
            if audio_queue is None:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dual Detection System with YOLO and Contour-based Detection')
    parser.add_argument('--pred-only', action='store_true', help='Only display the depth prediction')
    parser.add_argument('--grayscale', action='store_true', help='Display depth in grayscale')
    args = parser.parse_args()

    # Device selection
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Load Depth Anything V2 model
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }
    # depth anything model configs
    encoder = 'vits'            # 'vits' (small), 'vitb' (base), vitl (large)
    dataset = 'hypersim'        # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 20              # 20 for indoor model, 80 for outdoor model

    # contours configs
    depth_threshold = 5.0       # Depth threshold for nearby objects (in meters) / (perlu di test buat diluar plg bagus brp meter)
    min_area = 5000             # Minimum area for contours to be considered valid

    # YOLO resolution configs
    yolo_resolution = 'default'   # Resolution for YOLO model, separated from depth anything , can be set to "default" for original resolution
    maximum_distance_detection = 10  # Maximum distance (in meters) for YOLO to detect objects in the valid_classes list

    # depth anything input/resolution configuration parameters
    depth_input_size = 336      # Input size for depth estimation (smaller = faster = less accurate, but doesn't matter too much), default is 518, other options are 448, 392, 336, 280, 224, 168, 112, 56
    input_resolution = 'default'  # Input resolution for webcam capture width x height, can be set to "default" for original resolution
    # input_resolution = 1920x1080

    # Sound configurations (kept identical to the original code)
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
    }

    # Classes to detect
    valid_classes = ['person', 'bicycle', 'motorcycle', 'truck', 'car', 'bus']

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    
    # Get original webcam capabilities
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Parse input resolution
    if input_resolution == 'default':
        input_width, input_height = original_width, original_height
    else:
        input_width, input_height = map(int, input_resolution.split('x'))
    
    # Parse YOLO resolution
    if yolo_resolution == 'default':
        yolo_width, yolo_height = input_width, input_height  # Use input resolution as YOLO default
    else:
        yolo_width, yolo_height = map(int, yolo_resolution.split('x'))
        
    # Set webcam resolution if not using default
    if input_resolution != 'default':
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_height)
    
    # Get actual resolution after setting (webcams might not support exact requested resolution)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Update input dimensions to match actual camera capabilities
    input_width, input_height = actual_width, actual_height
    
    # Print actual resolution
    print(f"Actual webcam resolution: {input_width}x{input_height}")
    print(f"Depth model processing size: {depth_input_size}x{depth_input_size}")
    print(f"YOLO model resolution: {yolo_width}x{yolo_height}")

    depth_model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    depth_model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
    depth_model.to(DEVICE)
    depth_model.eval()
    
    # Load YOLOv11 segmentation model
    yolo_model = YOLO("yolo11n-seg.pt")
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    # Initialize the audio system
    print("Initializing audio system...")
    sounds = init_audio_system(sound_config)
    
    # Create a shared queue for audio processing
    audio_queue = []
    
    # Start audio processing thread
    audio_thread = threading.Thread(
        target=audio_thread_function,
        args=(audio_queue, input_width, maximum_distance_detection, sounds, sound_config),
        daemon=True
    )
    audio_thread.start()
    print("Audio system ready.")
    
    while cap.isOpened():
        ret, raw_frame = cap.read()
        if not ret:
            break
        
        raw_frame = cv2.flip(raw_frame, 1)  # Mirror effect
        
        # Create a copy specifically for YOLO if needed
        if yolo_width != input_width or yolo_height != input_height:
            yolo_frame = cv2.resize(raw_frame, (yolo_width, yolo_height))
        else:
            yolo_frame = raw_frame
        
        # Run depth estimation with the smaller input size
        depth_map = depth_model.infer_image(raw_frame, depth_input_size)
        original_depth = depth_map.copy()
        
        # Normalize depth map for visualization
        min_val, max_val = depth_map.min(), depth_map.max()
        depth_map_normalized = ((depth_map - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
        if args.grayscale:
            depth_display = np.repeat(depth_map_normalized[..., np.newaxis], 3, axis=-1)
        else:
            depth_display = (cmap(depth_map_normalized)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        # Prepare frame for visualization
        output_frame = raw_frame.copy()
        
        # Calculate scale factors if YOLO uses different resolution
        scale_x = raw_frame.shape[1] / yolo_frame.shape[1]
        scale_y = raw_frame.shape[0] / yolo_frame.shape[0]
        
        # Create a mask to track YOLO detections (to avoid duplicate contour detections)
        yolo_mask = np.zeros((raw_frame.shape[0], raw_frame.shape[1]), dtype=np.uint8)
        
        # List to collect all detections for audio processing
        all_detections = []
        
        # 1. PRIMARY DETECTOR: Run YOLO segmentation on the YOLO-specific frame
        yolo_detections = []
        results = yolo_model(yolo_frame)
        for result in results:
            if hasattr(result, 'masks') and result.masks is not None:
                for box, cls, mask in zip(
                    result.boxes.xyxy.cpu().numpy(), 
                    result.boxes.cls.cpu().numpy(),
                    result.masks.data.cpu().numpy()
                ):
                    # Get class name
                    class_name = yolo_model.names[int(cls)]
                    
                    # Skip if not in our valid_classes list
                    if class_name not in valid_classes:
                        continue
                        
                    # Scale the bounding box if resolutions differ
                    if yolo_width != input_width or yolo_height != input_height:
                        x1, y1, x2, y2 = map(int, [box[0] * scale_x, box[1] * scale_y, 
                                                  box[2] * scale_x, box[3] * scale_y])
                    else:
                        x1, y1, x2, y2 = map(int, box)
                    
                    # Generate binary mask for the segmentation output
                    binary_mask = (mask > 0.5).astype(np.uint8)
                    if y1 < y2 and x1 < x2 and binary_mask.sum() > 0:
                        # Resize mask to match the frame dimensions
                        resized_mask = cv2.resize(binary_mask, (raw_frame.shape[1], raw_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                        
                        # Isolate depth values for the segmented object
                        masked_depth = original_depth.copy()
                        masked_depth[resized_mask == 0] = np.nan
                        valid_depth_values = masked_depth[~np.isnan(masked_depth)]
                        
                        if len(valid_depth_values) > 0:
                            # Compute the median depth as the object's distance
                            object_depth = np.nanmedian(valid_depth_values)
                            
                            # Skip objects beyond the maximum detection distance
                            if object_depth > maximum_distance_detection:
                                continue
                                
                            # Add this detection to the YOLO mask (to avoid duplicate contour detections)
                            yolo_mask = cv2.bitwise_or(yolo_mask, resized_mask)
                            
                            # Save detection details for later visualization
                            contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            moments = cv2.moments(resized_mask)
                            if moments["m00"] != 0:
                                center_x = int(moments["m10"] / moments["m00"])
                                center_y = int(moments["m01"] / moments["m00"])
                            else:
                                center_x = (x1 + x2) // 2
                                center_y = (y1 + y2) // 2
                                
                            detection_info = {
                                'type': 'yolo',
                                'class': class_name,
                                'contours': contours,
                                'bbox': (x1, y1, x2, y2),
                                'center': (center_x, center_y),
                                'depth': object_depth,
                                'color': (0, 255, 0)  # Green for YOLO detections
                            }
                            yolo_detections.append(detection_info)
                            all_detections.append(detection_info)
        
        # Apply morphological operations to refine the YOLO mask
        kernel = np.ones((5, 5), np.uint8)
        yolo_mask_refined = cv2.dilate(yolo_mask, kernel, iterations=1)  # Slightly expand YOLO mask
        
        # 2. SECONDARY DETECTOR: Apply contour-based detection for objects YOLO might miss
                
        # Create binary mask of "close" areas based on depth threshold
        depth_binary = (depth_map < depth_threshold).astype(np.uint8) * 255
                
        # Apply morphological operations to reduce noise
        depth_binary = cv2.morphologyEx(depth_binary, cv2.MORPH_OPEN, kernel)  # Remove small noise
        depth_binary = cv2.morphologyEx(depth_binary, cv2.MORPH_CLOSE, kernel)  # Fill small holes

        # Create a new mask that excludes YOLO detections
        exclusive_depth_mask = depth_binary.copy()
        exclusive_depth_mask[yolo_mask_refined > 0] = 0  # Remove areas detected by YOLO

        # Display the exclusive mask for debugging
        cv2.imshow('Exclusive Depth Mask', exclusive_depth_mask)
                
        # Find contours in the exclusive mask
        contours, _ = cv2.findContours(exclusive_depth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
        # Find the center of the screen
        screen_center = (raw_frame.shape[1] // 2, raw_frame.shape[0] // 2)

        # Process each contour that meets minimum area requirement
        contour_detections = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area >= min_area:
                # Find bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract depth values for the masked region
                contour_mask = np.zeros_like(depth_binary)
                cv2.drawContours(contour_mask, [contour], 0, 255, -1)
                
                masked_depth = original_depth.copy()
                masked_depth[contour_mask == 0] = np.nan
                valid_depth_values = masked_depth[~np.isnan(masked_depth)]
                
                if len(valid_depth_values) > 0:
                    # Compute median depth as the object's distance
                    object_depth = np.nanmedian(valid_depth_values)
                    
                    # Find the contour point that's closest to the center of the screen
                    min_distance = float('inf')
                    closest_point = None
                    
                    # Flatten the contour to get individual points
                    contour_points = contour.reshape(-1, 2)
                    
                    for point in contour_points:
                        # Calculate Euclidean distance to screen center
                        dist = np.sqrt((point[0] - screen_center[0])**2 + (point[1] - screen_center[1])**2)
                        if dist < min_distance:
                            min_distance = dist
                            closest_point = point
                    
                    # Use the closest point as the center for visualization
                    center_x, center_y = closest_point
                    
                    detection_info = {
                        'type': 'contour',
                        'contour': contour,
                        'bbox': (x, y, x+w, y+h),
                        'center': (center_x, center_y),
                        'depth': object_depth,
                        'color': (255, 0, 0),  # Blue for contour detections
                        'id': i+1
                    }
                    contour_detections.append(detection_info)
                    all_detections.append(detection_info)
        
        # Update the audio queue with the latest detections
        audio_queue.clear()  # Remove old detections (we only care about current frame)
        audio_queue.append(all_detections)
        
        # 3. VISUALIZATION: Draw all detections on the output frame
        
        # First, draw contour detections (YOLO takes precedence visually)
        for detection in contour_detections:
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
            cv2.circle(depth_display, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(output_frame, f"Object {detection['id']}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, detection['color'], 2)
            cv2.putText(output_frame, f"Distance: {detection['depth']:.2f} m", 
                        (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Then, draw YOLO detections
        for detection in yolo_detections:
            # Create a colored overlay
            overlay = output_frame.copy()
            cv2.drawContours(overlay, detection['contours'], -1, detection['color'], -1)
            cv2.addWeighted(overlay, 0.4, output_frame, 0.6, 0, output_frame)
            
            # Draw contour outline and bounding box
            cv2.drawContours(output_frame, detection['contours'], -1, detection['color'], 2)
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), detection['color'], 2)
            
            # Annotate with class name and distance
            cx, cy = detection['center']
            cv2.circle(output_frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.circle(depth_display, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(output_frame, f"{detection['class']}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, detection['color'], 2)
            cv2.putText(output_frame, f"Distance: {detection['depth']:.2f} m", 
                        (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        # Display the YOLO and contour masks for debugging
        cv2.imshow('YOLO Mask', yolo_mask_refined * 255)
        cv2.imshow('Depth Binary Mask', depth_binary)
        
        # Display the output
        if args.pred_only:
            cv2.imshow('Depth Only', depth_display)
        else:
            # Ensure both frames have the same dimensions for concatenation
            h1, w1 = output_frame.shape[:2]
            h2, w2 = depth_display.shape[:2]
            if h1 != h2 or w1 != w2:
                depth_display = cv2.resize(depth_display, (w1, h1))
            combined_display = cv2.hconcat([output_frame, depth_display])
            cv2.imshow('Dual Detection System - YOLO + Contour', combined_display)
        
        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    # Clean up audio system
    pygame.mixer.quit()