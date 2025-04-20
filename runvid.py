import argparse
import cv2
import numpy as np
import torch
import matplotlib
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2
import wave
import os
import tqdm

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

def generate_beep(frequency, duration_samples, sample_rate):
    t = np.linspace(0., duration_samples / sample_rate, duration_samples, endpoint=False)
    beep = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    fade_len = min(duration_samples // 10, int(0.005 * sample_rate))
    if fade_len > 1:
        fade_in = np.linspace(0.0, 1.0, fade_len)
        fade_out = np.linspace(1.0, 0.0, fade_len)
        beep[:fade_len] *= fade_in
        beep[-fade_len:] *= fade_out
        
    return beep.astype(np.float32)

def collect_detection_events(detections, frame_width, timestamp, detection_events):
    for det in detections:
        if det['type'] == 'yolo':
            sound_type = 'person' if det['class'] == 'person' else 'vehicle'
            quadrant = get_quadrant(det['center'][0], frame_width)
            detection_events.append((timestamp, sound_type, quadrant, det['depth']))
        elif det['type'] == 'contour':
            sound_type = 'wall'
            quadrant = get_quadrant(det['center'][0], frame_width)
            detection_events.append((timestamp, sound_type, quadrant, det['depth']))

def generate_audio_file(detection_events, sound_config, duration, output_file):
    sample_rate = sound_config['sample_rate']
    audio_buffer = np.zeros((int(sample_rate * duration), 2), dtype=np.float32)

    last_beep_time = {'person': -float('inf'), 'vehicle': -float('inf')}

    for timestamp, sound_type, quadrant, distance in detection_events:
        if sound_type == 'person':
            frequency = sound_config['person_beep_frequency']
            max_volume = sound_config['person_beep_max_volume']
            duration_s = sound_config['beep_duration_s']
            interval_s = sound_config['person_beep_interval_s']
        elif sound_type == 'vehicle':
            frequency = sound_config['other_beep_frequency']
            max_volume = sound_config['other_beep_max_volume']
            duration_s = sound_config['beep_duration_s']
            interval_s = sound_config['other_beep_interval_s']
        else:  # wall
            frequency = sound_config['hum_frequency']
            max_volume = sound_config['hum_max_volume']
            duration_s = sound_config['beep_duration_s']
            interval_s = 0  # No interval for wall hum

        if timestamp - last_beep_time.get(sound_type, -float('inf')) < interval_s:
            continue

        last_beep_time[sound_type] = timestamp

        duration_samples = int(duration_s * sample_rate)
        beep = generate_beep(frequency, duration_samples, sample_rate)

        volume = max_volume
        if distance > 5.0:
            volume = max(0.0, max_volume * (1.0 - (distance - 5.0) / 5.0))

        if volume < 0.05:
            continue

        left_vol, right_vol = get_stereo_volume(quadrant, volume)

        start_sample = int(timestamp * sample_rate)
        end_sample = min(start_sample + duration_samples, len(audio_buffer))

        if start_sample < len(audio_buffer) and end_sample > start_sample:
            samples_to_add = end_sample - start_sample
            audio_buffer[start_sample:end_sample, 0] += beep[:samples_to_add] * left_vol
            audio_buffer[start_sample:end_sample, 1] += beep[:samples_to_add] * right_vol

    max_val = np.max(np.abs(audio_buffer))
    if max_val > 0:
        audio_buffer = audio_buffer * (0.9 / max_val)

    audio_int16 = (audio_buffer * 32767).astype(np.int16)

    with wave.open(output_file, 'w') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dual Detection System with YOLO and Contour-based Detection')
    parser.add_argument('--pred-only', action='store_true', help='Only include the depth prediction in output')
    parser.add_argument('--grayscale', action='store_true', help='Display depth in grayscale')
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
    depth_threshold = 5.0 # threshold untuk countour
    min_area = 5000
    yolo_resolution = 'default'
    maximum_distance_detection = 20 # maximum utk yolo
    depth_input_size = 336
    input_resolution = 'default'
    
    sound_config = {
        'sample_rate': 44100,
        'hum_frequency': 110,
        'hum_max_volume': 0.4,
        'person_beep_frequency': 880,
        'person_beep_interval_s': 0.5,
        'person_beep_max_volume': 0.9,
        'other_beep_frequency': 440,
        'other_beep_interval_s': 0.5,
        'other_beep_max_volume': 0.9,
        'beep_duration_s': 0.2,
    }

    valid_classes = ['person', 'bicycle', 'motorcycle', 'truck', 'car', 'bus']
    max_people_detection = 2 # max deteksi orang

    input_file = os.path.join('input', 'video.mp4')
    output_video = os.path.join('output', 'processed_output.mp4')
    output_audio = os.path.join('output', 'processed_audio.wav')
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        exit(1)
    
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"Error: Could not open input video: {input_file}")
        exit(1)
    
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    
    if input_resolution == 'default':
        input_width, input_height = original_width, original_height
    else:
        input_width, input_height = map(int, input_resolution.split('x'))
    
    if yolo_resolution == 'default':
        yolo_width, yolo_height = input_width, input_height
    else:
        yolo_width, yolo_height = map(int, yolo_resolution.split('x'))
    
    print(f"Video resolution: {input_width}x{input_height}, {fps} fps, {total_frames} frames")
    print(f"Depth model processing size: {depth_input_size}x{depth_input_size}")
    print(f"YOLO model resolution: {yolo_width}x{yolo_height}")

    depth_model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    depth_model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
    depth_model.to(DEVICE)
    depth_model.eval()
    
    yolo_model = YOLO("yolo11n-seg.pt")
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (input_width * 2, input_height) if not args.pred_only else (input_width, input_height))
    
    detection_events = []
    
    kernel = np.ones((5, 5), np.uint8)
    
    progress_bar = tqdm.tqdm(total=total_frames, desc="Processing frames")
    
    frame_index = 0
    
    while cap.isOpened():
        ret, raw_frame = cap.read()
        if not ret:
            break
            
        timestamp = frame_index / fps
        frame_index += 1
        
        if yolo_width != input_width or yolo_height != input_height:
            yolo_frame = cv2.resize(raw_frame, (yolo_width, yolo_height))
        else:
            yolo_frame = raw_frame
        
        depth_map = depth_model.infer_image(raw_frame, depth_input_size)
        original_depth = depth_map.copy()
        
        min_val, max_val = depth_map.min(), depth_map.max()
        depth_map_normalized = ((depth_map - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
        if args.grayscale:
            depth_display = np.repeat(depth_map_normalized[..., np.newaxis], 3, axis=-1)
        else:
            depth_display = (cmap(depth_map_normalized)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        output_frame = raw_frame.copy()
        
        scale_x = raw_frame.shape[1] / yolo_frame.shape[1]
        scale_y = raw_frame.shape[0] / yolo_frame.shape[0]
        
        yolo_mask = np.zeros((raw_frame.shape[0], raw_frame.shape[1]), dtype=np.uint8)
        
        all_detections = []
        all_potential_detections = []

        results = yolo_model(yolo_frame)
        for result in results:
            if hasattr(result, 'masks') and result.masks is not None:
                for box, cls, mask in zip(
                    result.boxes.xyxy.cpu().numpy(), 
                    result.boxes.cls.cpu().numpy(),
                    result.masks.data.cpu().numpy()
                ):
                    class_name = yolo_model.names[int(cls)]
                    
                    if class_name not in valid_classes:
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
                            
                            if object_depth > maximum_distance_detection:
                                continue
                                
                            yolo_mask = cv2.bitwise_or(yolo_mask, resized_mask)
                            
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
                                'color': (0, 255, 0),
                                'quadrant': get_quadrant(center_x, raw_frame.shape[1])
                            }
                            all_potential_detections.append(detection_info)
        
        all_potential_detections.sort(key=lambda det: det['depth'])

        person_count = 0
        for detection in all_potential_detections:
            if detection['class'] == 'person':
                if person_count >= max_people_detection:
                    continue
                person_count += 1
            all_detections.append(detection)
        
        yolo_mask_refined = cv2.dilate(yolo_mask, kernel, iterations=1)
        
        depth_binary = (depth_map < depth_threshold).astype(np.uint8) * 255
        depth_binary = cv2.morphologyEx(depth_binary, cv2.MORPH_OPEN, kernel)
        depth_binary = cv2.morphologyEx(depth_binary, cv2.MORPH_CLOSE, kernel)

        exclusive_depth_mask = depth_binary.copy()
        exclusive_depth_mask[yolo_mask_refined > 0] = 0
        
        contours, _ = cv2.findContours(exclusive_depth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        screen_center = (raw_frame.shape[1] // 2, raw_frame.shape[0] // 2)

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
                    all_detections.append(detection_info)
        
        collect_detection_events(all_detections, input_width, timestamp, detection_events)
        
        for detection in all_detections:
            if detection['type'] == 'contour':
                overlay = output_frame.copy()
                cv2.drawContours(overlay, [detection['contour']], 0, detection['color'], -1)
                cv2.addWeighted(overlay, 0.4, output_frame, 0.6, 0, output_frame)
                
                cv2.drawContours(output_frame, [detection['contour']], 0, detection['color'], 2)
                x1, y1, x2, y2 = detection['bbox']
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), detection['color'], 2)
                
                cx, cy = detection['center']
                q_num = detection['quadrant']
                cv2.circle(output_frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.circle(depth_display, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(output_frame, f"Object {detection['id']} (Q{q_num})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, detection['color'], 2)
                cv2.putText(output_frame, f"Distance: {detection['depth']:.2f} m", 
                            (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif detection['type'] == 'yolo':
                if detection['class'] == 'person':
                    detection_color = (0, 255, 0)
                else:
                    detection_color = (128, 0, 128)

                overlay = output_frame.copy()
                cv2.drawContours(overlay, detection['contours'], -1, detection_color, -1)
                cv2.addWeighted(overlay, 0.4, output_frame, 0.6, 0, output_frame)

                cv2.drawContours(output_frame, detection['contours'], -1, detection_color, 2)
                x1, y1, x2, y2 = detection['bbox']
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), detection_color, 2)

                cx, cy = detection['center']
                q_num = detection['quadrant']
                cv2.circle(output_frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.circle(depth_display, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(output_frame, f"{detection['class']} (Q{q_num})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, detection_color, 2)
                cv2.putText(output_frame, f"Distance: {detection['depth']:.2f} m", 
                            (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if args.pred_only:
            out.write(depth_display)
        else:
            h1, w1 = output_frame.shape[:2]
            h2, w2 = depth_display.shape[:2]
            if h1 != h2 or w1 != w2:
                depth_display = cv2.resize(depth_display, (w1, h1))
            combined_display = cv2.hconcat([output_frame, depth_display])
            out.write(combined_display)
        
        progress_bar.update(1)

    cap.release()
    out.release()
    progress_bar.close()
    
    print(f"Video processing complete. Generating audio file...")
    generate_audio_file(detection_events, sound_config, video_duration, output_audio)
    print(f"Processing complete.")
    print(f"Output video: {output_video}")
    print(f"Output audio: {output_audio}")