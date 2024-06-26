from pydub import AudioSegment
import os
import imageio
from ast_model import AudioTaggingModel
import json


def extract_audio_segments(audio_filename, selected_time_segments, audio_folder, output_folder):
    audio_path = os.path.join(audio_folder, audio_filename)
    output_path = os.path.join(output_folder, audio_filename)

    if not os.path.exists(output_path):
        os.makedirs(output_path)


    original_audio = AudioSegment.from_wav(audio_path)
    segment_ranges = []
    current_range = []
    for time in selected_time_segments:
        if not current_range or time == current_range[-1] + 1:
            current_range.append(time)
        else:
            segment_ranges.append(current_range)
            current_range = [time]
    if current_range:
        segment_ranges.append(current_range)

    extracted_audio_segments = []
    for segment_range in segment_ranges:
        segment_audio = AudioSegment.silent(duration=0)  
        for start_time in segment_range:
            start_ms = start_time * 1000  
            end_ms = (start_time + 1) * 1000 
            segment = original_audio[start_ms:end_ms]
            segment_audio += segment  
        extracted_audio_segments.append(segment_audio)


    for i, segment in enumerate(extracted_audio_segments):
        output_filename = os.path.join(output_path, f"extracted_segment_{i}.wav")
        segment.export(output_filename, format="wav")


def extract_visual_triples(video_name, selected_time_segments, video_dir, all_visual_triples, all_gt_triples):
    frame_list = list(all_visual_triples[video_name].keys())
    video2frames = {}
    for frame in frame_list:
        if video_name not in video2frames:
            video2frames[video_name] = []
        video2frames[video_name].append(frame)


    segment_ranges = []
    current_range = []
    for time in selected_time_segments:
        if not current_range or time == current_range[-1] + 1:
            current_range.append(time)
        else:
            segment_ranges.append(current_range)
            current_range = [time]
    if current_range:
        segment_ranges.append(current_range)


    segment_indices_list = []
    video_path = os.path.join(video_dir, video_name)
    reader = imageio.get_reader(video_path, "ffmpeg")
    fps = reader.get_meta_data()['fps']
    key_frames = []
    for keyframe_path in video2frames[video_name]:
        index = int(os.path.splitext(os.path.basename(keyframe_path))[0])
        time_str = int(index / fps)
        segment_index = -1
        for i, segment in enumerate(segment_ranges):
            if time_str in segment:
                segment_index = i
                key_frames.append(keyframe_path)
                break
        # Check if the time_str is within any of the selected_time_segments
        if segment_index != -1:
            segment_indices_list.append(segment_index)
    triples = {}
    gt_triples = {}
    for index, path in enumerate(key_frames):
        triples[index] = all_visual_triples[video_name][path]
        gt_triples[index] = all_gt_triples[video_name][path]

    return triples, gt_triples, segment_indices_list


def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


