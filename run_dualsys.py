import argparse
import cv2
import numpy as np
import torch
import matplotlib
import pygame
import time
import threading
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2

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
    dataset = 'vkitti'        # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 80              # 20 for indoor model, 80 for outdoor model

    # contours configs
    depth_threshold = 5.0       # Depth threshold for nearby objects (in meters) / (perlu di test buat diluar plg bagus brp meter)
    min_area = 5000             # Minimum area for contours to be considered valid

    # YOLO resolution configs
    yolo_resolution = 'default'   # Resolution for YOLO model, separated from depth anything , can be set to "default" for original resolution
    maximum_distance_detection = 15  # Maximum distance (in meters) for YOLO to detect objects in the valid_classes list

    # depth anything input/resolution configuration parameters
    depth_input_size = 336      # Input size for depth estimation (smaller = faster = less accurate, but doesn't matter too much), default is 518, other options are 448, 392, 336, 280, 224, 168, 112, 56
    input_resolution = 'default'  # Input resolution for webcam capture width x height, can be set to "default" for original resolution
    # input_resolution = 1920x1080

    # Classes to detect
    valid_classes = ['person', 'bicycle', 'motorcycle', 'truck', 'car', 'bus']

    # Define quadrants for vertical segments. ex: 20 quadrants, each quadrant occupy 5% of screen width
    num_quadrants = 40

    # Audio system configs
    audio_enabled = True        # Enable/disable audio feedback
    audio_volume = 0.7          # Master volume (0.0 to 1.0)
    min_distance_beep = 3       # Distance (in meters) at which beeping interval becomes smallest
    max_beep_interval = 1.0     # Maximum interval between beeps in seconds (at maximum_distance_detection)
    min_beep_interval = 0.05    # Minimum interval between beeps in seconds (at min_distance_beep)
    fixed_beep_duration = 0.1   # Fixed beep duration in seconds
    object_timeout = 1.0        # Time in seconds before an object is considered "lost" if not detected
    audio_thread_sleep = 0.01   # Sleep time for audio thread (10ms for fine timing control)
    
    # Sound characteristics
    wall_hum_freq = 100         # Frequency for wall/environment humming sound (Hz)
    person_beep_freq = 800      # Frequency for person detection beep (Hz)
    vehicle_beep_freq = 400     # Frequency for vehicle detection beep (Hz)
    
    # Wall hum specific parameters
    wall_hum_enabled = True     # Enable/disable continuous wall humming
    wall_hum_duration = 0.5     # Duration of each wall hum loop (shorter for more responsive panning)
    wall_hum_volume = 0.4       # Volume multiplier for wall humming (relative to master volume)
    
    # Audio buffer sizes and channel settings
    buffer_size = 512           # Audio buffer size (smaller = lower latency but higher CPU)
    sample_rate = 44100         # Audio sample rate
    
    # Global dictionary to track detected objects between frames
    tracked_objects = {}
    tracking_lock = threading.Lock()  # Thread-safe access
    
    # Specific tracking for wall/contour objects to manage continuous humming
    wall_objects = {}
    wall_lock = threading.Lock()  # Thread-safe access for wall objects
    
    # Stop event for audio thread
    stop_event = threading.Event()
    
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
    
    # Initialize pygame mixer for audio
    if audio_enabled:
        pygame.mixer.init(frequency=sample_rate, size=-16, channels=2, buffer=buffer_size)
        pygame.mixer.set_num_channels(50)  # Allow many simultaneous sounds
        
        # Create a separate channel for wall humming
        wall_hum_channel = pygame.mixer.Channel(0)  # Reserve channel 0 for wall humming
        
        # Generate sound samples
        def generate_sine_wave(freq, duration, volume=1.0):
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            wave = np.sin(2 * np.pi * freq * t) * volume
            # Apply fade in/out to avoid clicks
            fade_samples = int(0.005 * sample_rate)  # 5ms fade
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            if len(wave) > 2 * fade_samples:
                wave[:fade_samples] *= fade_in
                wave[-fade_samples:] *= fade_out
            return wave.astype(np.float32)

        # Create stereo sound with different volumes for left and right channels
        def create_stereo_sound(freq, duration, left_vol, right_vol):
            wave = generate_sine_wave(freq, duration)
            stereo = np.column_stack((wave * left_vol, wave * right_vol)) * 32767
            sound_buffer = pygame.sndarray.make_sound(stereo.astype(np.int16)).get_raw()
            return pygame.mixer.Sound(buffer=sound_buffer)
            
        # Sound cache for different frequencies
        sound_cache = {}
        
        # Function to get or create a stereo sound
        def get_stereo_sound(freq, duration, left_vol, right_vol):
            # Create a unique key for this sound configuration
            key = f"{freq}_{duration}_{left_vol:.2f}_{right_vol:.2f}"
            if key not in sound_cache:
                sound_cache[key] = create_stereo_sound(freq, duration, left_vol, right_vol)
            return sound_cache[key]
            
        print("Audio system initialized")
    
    # Function to determine quadrant number based on x-coordinate
    def get_quadrant(x_coord, frame_width):
        quadrant_width = frame_width / num_quadrants
        for i in range(num_quadrants):
            if x_coord >= i * quadrant_width and x_coord < (i + 1) * quadrant_width:
                return i + 1  # +1 to make it 1-indexed instead of 0-indexed
        return num_quadrants  # Default to last quadrant if outside bounds
    
    # Function to calculate volume ratio based on quadrant
    def get_stereo_volume(quadrant):
        # Calculate left and right volume percentages
        # For quadrant 1: left=100%, right=0%
        # For quadrant num_quadrants: left=0%, right=100%
        left_volume = (num_quadrants - quadrant) / (num_quadrants - 1)
        right_volume = (quadrant - 1) / (num_quadrants - 1)
        return left_volume, right_volume
    
    # Function to calculate beep interval based on distance
    def calculate_beep_interval(distance):
        if distance <= min_distance_beep:
            return min_beep_interval
        # Linear scaling from min_beep_interval at min_distance_beep to max_beep_interval at maximum_distance_detection
        interval = min_beep_interval + ((distance - min_distance_beep) / 
                   (maximum_distance_detection - min_distance_beep)) * (max_beep_interval - min_beep_interval)
        return min(max_beep_interval, interval)  # Ensure we don't go above maximum interval
    
    # Function to play sound with binaural effect for regular objects
    def play_binaural_sound(sound_type, quadrant, left_vol, right_vol):
        # Apply master volume
        left_vol *= audio_volume
        right_vol *= audio_volume
        
        # Select appropriate frequency
        if sound_type == 'person':
            freq = person_beep_freq
        elif sound_type == 'vehicle':
            freq = vehicle_beep_freq
        else:  # fallback to person frequency if unknown type
            freq = person_beep_freq
            
        # Use fixed beep duration
        beep_duration = fixed_beep_duration
        
        # Get or create the sound
        stereo_beep = get_stereo_sound(freq, beep_duration, left_vol, right_vol)
        
        # Play the sound
        stereo_beep.play(loops=0, fade_ms=5)
    
    # Function to update wall humming based on current wall objects
    def update_wall_humming():
        global wall_objects
        
        # Check if wall humming is enabled
        if not wall_hum_enabled or len(wall_objects) == 0:
            # Stop any playing wall hum if no walls detected
            if wall_hum_channel.get_busy():
                wall_hum_channel.stop()
            return
        
        # Calculate average position and distance of wall objects
        total_left_vol = 0
        total_right_vol = 0
        count = 0
        
        with wall_lock:
            # Average the stereo volumes from all wall objects
            for obj_id, obj_data in wall_objects.items():
                left_vol, right_vol = get_stereo_volume(obj_data['quadrant'])
                total_left_vol += left_vol
                total_right_vol += right_vol
                count += 1
        
        if count > 0:
            # Calculate average stereo volumes
            avg_left_vol = total_left_vol / count
            avg_right_vol = total_right_vol / count
            
            # Apply master volume and wall hum specific volume
            final_left_vol = avg_left_vol * audio_volume * wall_hum_volume
            final_right_vol = avg_right_vol * audio_volume * wall_hum_volume
            
            # Create or get the wall hum sound
            wall_hum_sound = get_stereo_sound(wall_hum_freq, wall_hum_duration, final_left_vol, final_right_vol)
            
            # Play or update the wall humming sound
            if not wall_hum_channel.get_busy():
                # Start the wall hum if not already playing
                wall_hum_channel.play(wall_hum_sound, loops=-1, fade_ms=50)  # -1 for infinite looping
            else:
                # Queue the next wall hum with updated volumes
                wall_hum_channel.queue(wall_hum_sound)
    
    # Audio feedback thread function
    def audio_feedback_thread():
        global tracked_objects, wall_objects
        
        while not stop_event.is_set():
            current_time = time.time()
            
            # Make a thread-safe copy of the current objects (non-wall objects only)
            beeping_objects = {}
            with tracking_lock:
                # Filter out wall objects and copy the rest
                beeping_objects = {obj_id: obj_data for obj_id, obj_data in tracked_objects.items() 
                                if obj_data['sound_type'] != 'wall'}
            
            # Process each tracked object that uses interval-based beeping
            for obj_id, obj_data in beeping_objects.items():
                # Check if object is still valid (not too old)
                if current_time - obj_data['last_seen'] > object_timeout:
                    continue
                    
                # Calculate time since last beep
                time_since_beep = current_time - obj_data.get('last_beep_time', 0)
                
                # Get the desired beep interval based on distance
                desired_interval = calculate_beep_interval(obj_data['depth'])
                
                # If enough time has passed, play a beep
                if time_since_beep >= desired_interval:
                    # Get stereo volume ratio based on quadrant
                    left_vol, right_vol = get_stereo_volume(obj_data['quadrant'])
                    
                    # Play the appropriate sound
                    play_binaural_sound(obj_data['sound_type'], obj_data['quadrant'], left_vol, right_vol)
                    
                    # Update last beep time in the global dictionary
                    with tracking_lock:
                        if obj_id in tracked_objects:  # Make sure object still exists
                            tracked_objects[obj_id]['last_beep_time'] = current_time
            
            # Update the wall humming sound
            update_wall_humming()
            
            # Sleep a small amount to prevent CPU overuse
            time.sleep(audio_thread_sleep)
    
    # Start audio thread if audio is enabled
    if audio_enabled:
        audio_thread = threading.Thread(target=audio_feedback_thread)
        audio_thread.daemon = True  # Thread will exit when main program exits
        audio_thread.start()
        print("Audio feedback thread started")
    
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
        
        # Current frame detections
        current_detections = set()
        current_wall_objects = {}
        current_time = time.time()
        
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
                                
                            # Determine quadrant for this detection
                            quadrant = get_quadrant(center_x, raw_frame.shape[1])
                            
                            obj_id = f"yolo_{class_name}_{len(yolo_detections)}"
                                
                            yolo_detections.append({
                                'type': 'yolo',
                                'class': class_name,
                                'contours': contours,
                                'bbox': (x1, y1, x2, y2),
                                'center': (center_x, center_y),
                                'depth': object_depth,
                                'color': (0, 255, 0),  # Green for YOLO detections
                                'quadrant': quadrant,
                                'id': obj_id
                            })
                            
                            # Add to current frame detections
                            current_detections.add(obj_id)
                            
                            # Update tracked objects for audio thread
                            with tracking_lock:
                                tracked_objects[obj_id] = {
                                    'sound_type': 'person' if class_name == 'person' else 'vehicle',
                                    'depth': object_depth,
                                    'quadrant': quadrant,
                                    'last_seen': current_time,
                                    'last_beep_time': tracked_objects.get(obj_id, {}).get('last_beep_time', 0)
                                }
        
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
                    
                    # Check if the center of the screen is inside the contour
                    center_inside_contour = cv2.pointPolygonTest(contour, screen_center, False) >= 0
                    
                    if center_inside_contour:
                        # If center is inside contour, use the screen center
                        center_x, center_y = screen_center
                    else:
                        # Otherwise find the contour point that's closest to the center of the screen
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
                    
                    # Determine quadrant for this detection
                    quadrant = get_quadrant(center_x, raw_frame.shape[1])
                    
                    obj_id = f"contour_{i+1}"
                    
                    contour_detections.append({
                        'type': 'contour',
                        'contour': contour,
                        'bbox': (x, y, x+w, y+h),
                        'center': (center_x, center_y),
                        'depth': object_depth,
                        'color': (255, 0, 0),  # Blue for contour detections
                        'id': obj_id,
                        'quadrant': quadrant
                    })
                    
                    # Add to current frame detections
                    current_detections.add(obj_id)
                    
                    # Add to wall objects for continuous humming
                    current_wall_objects[obj_id] = {
                        'quadrant': quadrant,
                        'depth': object_depth,
                        'last_seen': current_time
                    }
                    
                    # Also add to tracked objects for compatibility
                    with tracking_lock:
                        tracked_objects[obj_id] = {
                            'sound_type': 'wall',
                            'depth': object_depth,
                            'quadrant': quadrant,
                            'last_seen': current_time,
                            'last_beep_time': current_time  # Not used for walls
                        }
        
        # Update wall objects dictionary with thread safety
        with wall_lock:
            # Clear old wall objects
            wall_objects.clear()
            # Add current wall objects
            wall_objects.update(current_wall_objects)
        
        # Clean up objects that are no longer detected
        with tracking_lock:
            # Find objects that were not detected in this frame
            objects_to_remove = [obj_id for obj_id in tracked_objects if obj_id not in current_detections]
            
            # Remove objects that have not been seen for too long
            for obj_id in objects_to_remove:
                if current_time - tracked_objects[obj_id]['last_seen'] > object_timeout:
                    del tracked_objects[obj_id]
        
        # Draw quadrant lines (optional, for visualization)
        quadrant_display = output_frame.copy()
        quadrant_width = raw_frame.shape[1] / num_quadrants
        for i in range(1, num_quadrants):
            x_pos = int(i * quadrant_width)
            cv2.line(quadrant_display, (x_pos, 0), (x_pos, raw_frame.shape[0]), (100, 100, 100), 1)
            cv2.putText(quadrant_display, str(i), (x_pos - 10, 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show the quadrant display for debugging
        cv2.imshow('Quadrants', quadrant_display)
        
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
            
            # Annotate with object ID, quadrant and distance
            cx, cy = detection['center']
            q_num = detection['quadrant']
            
            cv2.circle(output_frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.circle(depth_display, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(output_frame, f"Object {detection['id']} (Q{q_num})", (x1, y1 - 10),
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
            
            # Annotate with class name, quadrant and distance
            cx, cy = detection['center']
            q_num = detection['quadrant']
            
            cv2.circle(output_frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.circle(depth_display, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(output_frame, f"{detection['class']} (Q{q_num})", (x1, y1 - 10),
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
    if audio_enabled:
        stop_event.set()  # Signal audio thread to stop
        audio_thread.join(timeout=1.0)  # Wait for thread to finish
        pygame.mixer.quit()
        
    cap.release()
    cv2.destroyAllWindows()
