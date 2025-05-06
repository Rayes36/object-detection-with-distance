import argparse
import cv2
import numpy as np
import torch
import matplotlib
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2
import pygame
import threading
import time

def init_audio_system(sound_config):
    pygame.mixer.init(frequency=sound_config['sample_rate'], size=-16, channels=2)
    pygame.mixer.set_num_channels(8)

    channels = {
        'left_hum': pygame.mixer.Channel(0),
        'right_hum': pygame.mixer.Channel(1),
        'person': pygame.mixer.Channel(2),
        'other': pygame.mixer.Channel(3)
    }

    sound_cache = {}

    def generate_beep(frequency, duration_samples):
        cache_key = (frequency, duration_samples)
        if cache_key in sound_cache:
            return sound_cache[cache_key]

        sample_rate = sound_config['sample_rate']
        t = np.linspace(0., duration_samples / sample_rate, duration_samples, endpoint=False)
        beep = 0.5 * np.sin(2 * np.pi * frequency * t)  
        fade_len = min(duration_samples // 10, int(0.01 * sample_rate))
        if fade_len > 1:
            fade = np.linspace(1.0, 0.0, fade_len)
            beep[-fade_len:] *= fade

        sound_cache[cache_key] = beep.astype(np.float32)
        return sound_cache[cache_key]

    beep_duration_samples = int(sound_config['beep_duration_s'] * sound_config['sample_rate'])

    person_beep = generate_beep(sound_config['person_beep_frequency'], beep_duration_samples)
    other_beep = generate_beep(sound_config['other_beep_frequency'], beep_duration_samples)
    hum = generate_beep(sound_config['hum_frequency'], beep_duration_samples)

    def to_pygame_sound(audio_data):
        stereo = np.column_stack((audio_data, audio_data))
        audio_int16 = (stereo * 32767).astype(np.int16)
        return pygame.mixer.Sound(audio_int16)

    sounds = {
        'person': to_pygame_sound(person_beep),
        'other': to_pygame_sound(other_beep),
        'hum': to_pygame_sound(hum)
    }

    return sounds, channels

num_quadrants = 80

def get_quadrant(x_coord, frame_width):
    quadrant_width = frame_width / num_quadrants
    for i in range(num_quadrants):
        if x_coord >= i * quadrant_width and x_coord < (i + 1) * quadrant_width:
            return i + 1
    return num_quadrants

def get_stereo_volume(quadrant, audio_volume):
    left_volume = (num_quadrants - quadrant) / (num_quadrants - 1)
    right_volume = (quadrant - 1) / (num_quadrants - 1)
    return left_volume * audio_volume, right_volume * audio_volume


def play_audio_for_detections(detections, frame_width, max_dist_primary, max_dist_secondary, sounds, channels, sound_config):
    if not hasattr(play_audio_for_detections, 'last_person_beep_time'):
        play_audio_for_detections.last_person_beep_time = 0
    if not hasattr(play_audio_for_detections, 'last_other_beep_time'):
        play_audio_for_detections.last_other_beep_time = 0

    current_time = time.time()

    left_hum_active = False
    right_hum_active = False
    min_left_depth = float('inf')
    min_right_depth = float('inf')
    
    for det in detections:
        if det['type'] == 'contour' or det['type'] == 'secondary' or det['type'] == 'custom':
            center_x = det['center'][0]
            depth = det['depth']
            if center_x < frame_width / 2:
                min_left_depth = min(min_left_depth, depth)
            else:
                min_right_depth = min(min_right_depth, depth)
    
    if min_left_depth <= max(max_dist_primary, max_dist_secondary):
        volume = 1.0 if min_left_depth <= 5.0 else max(0.0, 1.0 - (min_left_depth - 5.0) / (max(max_dist_primary, max_dist_secondary) - 5.0))
        left_hum_volume = volume * sound_config['hum_max_volume']
        
        if left_hum_volume > 0.05:
            channels['left_hum'].set_volume(left_hum_volume, 0.0)
            if not channels['left_hum'].get_busy():
                channels['left_hum'].play(sounds['hum'], loops=-1)
            left_hum_active = True
    
    if not left_hum_active and channels['left_hum'].get_busy():
        channels['left_hum'].fadeout(100)
    
    if min_right_depth <= max(max_dist_primary, max_dist_secondary):
        volume = 1.0 if min_right_depth <= 5.0 else max(0.0, 1.0 - (min_right_depth - 5.0) / (max(max_dist_primary, max_dist_secondary) - 5.0))
        right_hum_volume = volume * sound_config['hum_max_volume']
        
        if right_hum_volume > 0.05:
            channels['right_hum'].set_volume(0.0, right_hum_volume)
            if not channels['right_hum'].get_busy():
                channels['right_hum'].play(sounds['hum'], loops=-1)
            right_hum_active = True
    
    if not right_hum_active and channels['right_hum'].get_busy():
        channels['right_hum'].fadeout(100)
    
    person_detected = False
    other_detected = False
    
    for det in detections:
        if det['type'] == 'primary':
            depth = det['depth']
            if depth > max_dist_primary:
                continue

            center_x = det['center'][0]
            class_name = det['class']

            quadrant = get_quadrant(center_x, frame_width)

            left_vol, right_vol = get_stereo_volume(quadrant, sound_config['person_beep_max_volume'])

            if class_name == 'person':
                if current_time - play_audio_for_detections.last_person_beep_time >= sound_config['person_beep_interval_s']:
                    if not channels['person'].get_busy():
                        channels['person'].set_volume(left_vol, right_vol)
                        channels['person'].play(sounds['person'])
                        play_audio_for_detections.last_person_beep_time = current_time

            elif class_name != 'person':
                if current_time - play_audio_for_detections.last_other_beep_time >= sound_config['other_beep_interval_s']:
                    if not channels['other'].get_busy():
                        channels['other'].set_volume(left_vol, right_vol)
                        channels['other'].play(sounds['other'])
                        play_audio_for_detections.last_other_beep_time = current_time

def audio_thread_function(audio_queue, frame_width, max_dist_primary, max_dist_secondary, sounds, channels, sound_config):
    last_audio_time = 0
    min_audio_interval = 0.1
    
    while True:
        try:
            detections = audio_queue.pop()
            current_time = time.time()
            
            if current_time - last_audio_time >= min_audio_interval:
                play_audio_for_detections(detections, frame_width, max_dist_primary, max_dist_secondary, sounds, channels, sound_config)
                last_audio_time = current_time
                
            time.sleep(0.01)
        except IndexError:
            time.sleep(0.01)
        except Exception as e:
            print(f"Audio thread error: {e}")
            if audio_queue is None:
                break

def calculate_center_point(contour, screen_center):
    center_inside_contour = cv2.pointPolygonTest(contour, screen_center, False) >= 0

    if center_inside_contour:
        center_x, center_y = screen_center
    else:
        min_distance = float('inf')
        closest_point = None

        contour_points = contour.reshape(-1, 2)

        for point in contour_points:
            dist = np.sqrt((point[0] - screen_center[0])**2 + (point[1] - screen_center[1])**2)
            if dist < min_distance:
                min_distance = dist
                closest_point = point

        center_x, center_y = closest_point
        
    return center_x, center_y

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio-Only Detection System with YOLO and Contour-based Detection')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }
    encoder = 'vits'
    dataset = 'vkitti'
    max_depth = 80

    depth_threshold = 5
    min_area = 5000

    yolo_resolution = 'default'
    maximum_distance_detection_primary = 15
    maximum_distance_detection_secondary = 8

    depth_input_size = 336
    input_resolution = 'default'

    sound_config = {
        'sample_rate': 44100,
        'hum_frequency': 110,
        'hum_max_volume': 0.6,
        'person_beep_frequency': 880,
        'person_beep_interval_s': 0.2,
        'person_beep_max_volume': 0.9,
        'other_beep_frequency': 440,
        'other_beep_interval_s': 0.2,
        'other_beep_max_volume': 0.7,
        'beep_duration_s': 0.4,
    }

    primary_classes = ['person', 'bicycle', 'motorcycle', 'truck', 'car', 'bus']
    secondary_classes = ['fire hydrant', 'bench']
    custom_classes = ['bollard', 'electrical box', 'roadblock', 'traffic cone', 'trash can']
    
    max_people_detection = 3

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if input_resolution == 'default':
        input_width, input_height = original_width, original_height
    else:
        input_width, input_height = map(int, input_resolution.split('x'))

    if yolo_resolution == 'default':
        yolo_width, yolo_height = input_width, input_height
    else:
        yolo_width, yolo_height = map(int, yolo_resolution.split('x'))

    if input_resolution != 'default':
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_height)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    input_width, input_height = actual_width, actual_height

    print(f"Actual webcam resolution: {input_width}x{input_height}")
    print(f"Depth model processing size: {depth_input_size}x{depth_input_size}")
    print(f"YOLO model resolution: {yolo_width}x{yolo_height}")

    depth_model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    depth_model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
    depth_model.to(DEVICE)
    depth_model.eval()
    
    yolo_model = YOLO("yolo11n-seg.pt")
    custom_model = YOLO("my_model.pt")
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    print("Initializing audio system...")
    sounds, channels = init_audio_system(sound_config)
    
    audio_queue = []
    
    audio_thread = threading.Thread(
        target=audio_thread_function,
        args=(audio_queue, input_width, maximum_distance_detection_primary, maximum_distance_detection_secondary, sounds, channels, sound_config),
        daemon=True
    )
    audio_thread.start()
    print("Audio system ready.")
    print("Running audio-only detection system. Press Ctrl+C to exit.")
    
    try:
        while cap.isOpened():
            ret, raw_frame = cap.read()
            if not ret:
                break
            
            # raw_frame = cv2.flip(raw_frame, 1)
            
            if yolo_width != input_width or yolo_height != input_height:
                yolo_frame = cv2.resize(raw_frame, (yolo_width, yolo_height))
            else:
                yolo_frame = raw_frame
            
            depth_map = depth_model.infer_image(raw_frame, depth_input_size)
            original_depth = depth_map.copy()
            
            min_val, max_val = depth_map.min(), depth_map.max()
            depth_map_normalized = ((depth_map - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
            
            scale_x = raw_frame.shape[1] / yolo_frame.shape[1]
            scale_y = raw_frame.shape[0] / yolo_frame.shape[0]
            
            yolo_mask = np.zeros((raw_frame.shape[0], raw_frame.shape[1]), dtype=np.uint8)
            secondary_mask = np.zeros((raw_frame.shape[0], raw_frame.shape[1]), dtype=np.uint8)
            custom_mask = np.zeros((raw_frame.shape[0], raw_frame.shape[1]), dtype=np.uint8)
            
            all_detections = []
            all_potential_detections = []
            primary_detections = []
            secondary_detections = []
            custom_detections = []

            screen_center = (raw_frame.shape[1] // 2, raw_frame.shape[0] // 2)

            results = yolo_model(yolo_frame)
            for result in results:
                if hasattr(result, 'masks') and result.masks is not None:
                    for box, cls, mask in zip(
                        result.boxes.xyxy.cpu().numpy(), 
                        result.boxes.cls.cpu().numpy(),
                        result.masks.data.cpu().numpy()
                    ):
                        class_name = yolo_model.names[int(cls)]
                        
                        if class_name not in primary_classes and class_name not in secondary_classes:
                            continue
                            
                        if yolo_width != input_width or yolo_height != input_height:
                            x1, y1, x2, y2 = map(int, [box[0] * scale_x, box[1] * scale_y, 
                                                      box[2] * scale_x, box[3] * scale_y])
                        else:
                            x1, y1, x2, y2 = map(int, box)
                        
                        binary_mask = (mask > 0.5).astype(np.uint8)
                        if y1 < y2 and x1 < x2 and binary_mask.sum() > 0:
                            resized_mask = cv2.resize(binary_mask, (raw_frame.shape[1], raw_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                            
                            masked_depth = original_depth.copy()
                            masked_depth[resized_mask == 0] = np.nan
                            valid_depth_values = masked_depth[~np.isnan(masked_depth)]
                            
                            if len(valid_depth_values) > 0:
                                object_depth = np.nanmedian(valid_depth_values)
                                
                                if class_name in primary_classes:
                                    if object_depth > maximum_distance_detection_primary:
                                        continue
                                    yolo_mask = cv2.bitwise_or(yolo_mask, resized_mask)
                                elif class_name in secondary_classes:
                                    if object_depth > maximum_distance_detection_secondary:
                                        continue
                                    secondary_mask = cv2.bitwise_or(secondary_mask, resized_mask)
                                
                                contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                
                                if class_name in primary_classes:
                                    moments = cv2.moments(resized_mask)
                                    if moments["m00"] != 0:
                                        center_x = int(moments["m10"] / moments["m00"])
                                        center_y = int(moments["m01"] / moments["m00"])
                                    else:
                                        center_x = (x1 + x2) // 2
                                        center_y = (y1 + y2) // 2
                                    
                                    detection_info = {
                                        'type': 'primary',
                                        'class': class_name,
                                        'contours': contours,
                                        'bbox': (x1, y1, x2, y2),
                                        'center': (center_x, center_y),
                                        'depth': object_depth,
                                        'color': (0, 255, 0),
                                        'quadrant': get_quadrant(center_x, raw_frame.shape[1])
                                    }
                                    all_potential_detections.append(detection_info)
                                
                                elif class_name in secondary_classes:
                                    center_x, center_y = calculate_center_point(contours[0], screen_center)
                                    
                                    detection_info = {
                                        'type': 'secondary',
                                        'class': class_name,
                                        'contours': contours,
                                        'bbox': (x1, y1, x2, y2),
                                        'center': (center_x, center_y),
                                        'depth': object_depth,
                                        'color': (255, 0, 0),
                                        'quadrant': get_quadrant(center_x, raw_frame.shape[1])
                                    }
                                    secondary_detections.append(detection_info)
                                    all_detections.append(detection_info)
                                    
            custom_results = custom_model(yolo_frame)
            for result in custom_results:
                if hasattr(result, 'masks') and result.masks is not None:
                    for box, cls, mask in zip(
                        result.boxes.xyxy.cpu().numpy(), 
                        result.boxes.cls.cpu().numpy(),
                        result.masks.data.cpu().numpy()
                    ):
                        class_name = custom_model.names[int(cls)]
                        
                        if class_name not in custom_classes:
                            continue
                        
                        if yolo_width != input_width or yolo_height != input_height:
                            x1, y1, x2, y2 = map(int, [box[0] * scale_x, box[1] * scale_y, 
                                                      box[2] * scale_x, box[3] * scale_y])
                        else:
                            x1, y1, x2, y2 = map(int, box)
                        
                        binary_mask = (mask > 0.5).astype(np.uint8)
                        if y1 < y2 and x1 < x2 and binary_mask.sum() > 0:
                            resized_mask = cv2.resize(binary_mask, (raw_frame.shape[1], raw_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                            
                            masked_depth = original_depth.copy()
                            masked_depth[resized_mask == 0] = np.nan
                            valid_depth_values = masked_depth[~np.isnan(masked_depth)]
                            
                            if len(valid_depth_values) > 0:
                                object_depth = np.nanmedian(valid_depth_values)
                                
                                if object_depth > maximum_distance_detection_secondary:
                                    continue
                                    
                                custom_mask = cv2.bitwise_or(custom_mask, resized_mask)
                                
                                contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                center_x, center_y = calculate_center_point(contours[0], screen_center)
                                
                                detection_info = {
                                    'type': 'custom',
                                    'class': class_name,
                                    'contours': contours,
                                    'bbox': (x1, y1, x2, y2),
                                    'center': (center_x, center_y),
                                    'depth': object_depth,
                                    'color': (255, 0, 0),
                                    'quadrant': get_quadrant(center_x, raw_frame.shape[1])
                                }
                                custom_detections.append(detection_info)
                                all_detections.append(detection_info)
            
            all_masks = cv2.bitwise_or(cv2.bitwise_or(yolo_mask, secondary_mask), custom_mask)
            
            all_potential_detections.sort(key=lambda det: det['depth'])

            person_count = 0  
            for detection in all_potential_detections:
                if detection['class'] == 'person':
                    if person_count >= max_people_detection:
                        continue
                    person_count += 1
                primary_detections.append(detection)
                all_detections.append(detection)
            
            kernel = np.ones((5, 5), np.uint8)
            all_masks_refined = cv2.dilate(all_masks, kernel, iterations=1)
                    
            depth_binary = (depth_map < depth_threshold).astype(np.uint8) * 255
                    
            depth_binary = cv2.morphologyEx(depth_binary, cv2.MORPH_OPEN, kernel)
            depth_binary = cv2.morphologyEx(depth_binary, cv2.MORPH_CLOSE, kernel)

            exclusive_depth_mask = depth_binary.copy()
            exclusive_depth_mask[all_masks_refined > 0] = 0
                    
            contours, _ = cv2.findContours(exclusive_depth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
            contour_detections = []
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area >= min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    contour_mask = np.zeros_like(depth_binary)
                    cv2.drawContours(contour_mask, [contour], 0, 255, -1)
                    
                    masked_depth = original_depth.copy()
                    masked_depth[contour_mask == 0] = np.nan
                    valid_depth_values = masked_depth[~np.isnan(masked_depth)]
                    
                    if len(valid_depth_values) > 0:
                        object_depth = np.nanmedian(valid_depth_values)
                        
                        center_x, center_y = calculate_center_point(contour, screen_center)
                        quadrant = get_quadrant(center_x, raw_frame.shape[1])

                        detection_info = {
                            'type': 'contour',
                            'contour': contour,
                            'bbox': (x, y, x+w, y+h),
                            'center': (center_x, center_y),
                            'depth': object_depth,
                            'color': (255, 0, 0),
                            'id': i+1,
                            'quadrant': quadrant
                        }
                        contour_detections.append(detection_info)
                        all_detections.append(detection_info)
            
            audio_queue.clear()
            audio_queue.append(all_detections)
            
            # Check for keyboard interrupt
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Exiting...")
    
    # Clean up
    cap.release()
    
    # Clean up audio system
    pygame.mixer.quit()