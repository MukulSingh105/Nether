from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
from substrate import LLMClient
import librosa
import moviepy as mp
import cv2
import textwrap
import subprocess
import openai
import numpy as np
from ffmpeg import FFmpeg
import json
import math
import time
import pdb
from abc import ABC


class EditingOperator(ABC):
    def apply(video, metadata={"object_bbox":[0,0,10,10], "name":"guitar"}):
        pass


llm_client = LLMClient(True)


def download_video(url, filename):
    yt = YouTube(url, use_oauth=True, allow_oauth_cache=True)
    video = yt.streams.filter(file_extension='mp4').first()

    # Download the video
    video.download(filename=filename)


def trim_video(input_file, start_time, end_time, output_file):
    (
        FFmpeg()
        .input(input_file, ss=start_time, to=end_time)
        .output(output_file, c='copy')
        .execute()
    )

# Example usage
trim_video('input_video.mp4', '00:00:10', '00:00:20', 'output_video.mp4')



#Segment Video function
def segment_video(response):
  for i, segment in enumerate(response):
        start_time = math.floor(float(segment.get("start_time", 0)))
        end_time = math.ceil(float(segment.get("end_time", 0))) + 2
        output_file = f"output{str(i).zfill(3)}.mp4"
        trim_video("input_video.mp4", start_time, end_time, output_file)


#Face Detection function
# def detect_faces(video_file):
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     # Load the video
#     cap = cv2.VideoCapture(video_file)
#     print(video_file)

#     faces = []

#     # Detect and store unique faces
#     while len(faces) < 1:
#         ret, frame = cap.read()
#         if ret:
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#             # Iterate through the detected faces
#             for face in detected_faces:
#                 # Check if the face is already in the list of faces
#                 if not any(np.array_equal(face, f) for f in faces):
#                     faces.append(face)

#             # Print the number of unique faces detected so far
#             print(f"Number of unique faces detected: {len(faces)}")

#     # Release the video capture object
#     cap.release()

#     # If faces detected, return the list of faces
#     if len(faces) > 0:
#         return faces

#     # If no faces detected, return None
#     return None

def detect_human_faces(video_file):
    # Load pre-trained deep learning model for face detection
    net = cv2.dnn.readNetFromCaffe(
        "deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel"
    )
    
    cap = cv2.VideoCapture(video_file)
    faces = []
    
    while len(faces) < 1:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        
        detected_faces = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Adjust threshold for accuracy
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                detected_faces.append((confidence, (startX, startY, endX - startX, endY - startY)))
        
        if detected_faces:
            detected_faces.sort(reverse=True, key=lambda x: x[0])  # Sort by confidence
            faces = [face[1] for face in detected_faces]
        
        print(f"Number of unique human faces detected: {len(faces)}")
    
    cap.release()
    return faces if faces else None



def crop_video(faces, input_file, output_file):
    try:
        if len(faces) > 0:
            # Constants for cropping
            CROP_RATIO = 0.9  # Adjust the ratio to control how much of the face is visible in the cropped video
            VERTICAL_RATIO = 9 / 16  # Aspect ratio for the vertical video

            # Read the input video
            cap = cv2.VideoCapture(input_file)

            # Get the frame dimensions
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Calculate the target width and height for cropping (vertical format)
            target_height = int(frame_height * CROP_RATIO)
            target_width = int(target_height * VERTICAL_RATIO)

            # Create a VideoWriter object to save the output video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_video = cv2.VideoWriter(output_file, fourcc, 30.0, (target_width, target_height))

            # Loop through each frame of the input video
            while True:
                ret, frame = cap.read()

                # If no more frames, break out of the loop
                if not ret:
                    break

                # Iterate through each detected face
                for face in faces:
                    # Unpack the face coordinates
                    x, y, w, h = face

                    # Calculate the crop coordinates
                    crop_x = max(0, x + (w - target_width) // 2)  # Adjust the crop region to center the face
                    crop_y = max(0, y + (h - target_height) // 2)
                    crop_x2 = min(crop_x + target_width, frame_width)
                    crop_y2 = min(crop_y + target_height, frame_height)

                    # Crop the frame based on the calculated crop coordinates
                    cropped_frame = frame[crop_y:crop_y2, crop_x:crop_x2]

                    # Resize the cropped frame to the target dimensions
                    resized_frame = cv2.resize(cropped_frame, (target_width, target_height))

                    # Write the resized frame to the output video
                    output_video.write(resized_frame)

            # Release the input and output video objects
            cap.release()
            output_video.release()

            print("Video cropped successfully.")
        else:
            print("No faces detected in the video.")
    except Exception as e:
        print(f"Error during video cropping: {str(e)}")


def crop_video_to_primary_face(input_file, output_file):
    # Constants
    CROP_RATIO = 0.9  # Controls how much of the face remains visible
    VERTICAL_RATIO = 9 / 16  # Aspect ratio for vertical video
    LOCK_TIME = 30  # Minimum frames (1 second) before allowing crop to move
    SMOOTHING_ALPHA = 0.2  # Smoothing factor for gradual movement

    # Read the input video
    cap = cv2.VideoCapture(input_file)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Target width and height for vertical cropping
    target_height = int(frame_height * CROP_RATIO)
    target_width = int(target_height * VERTICAL_RATIO)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(output_file, fourcc, fps, (target_width, target_height))

    # Load face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Tracking variables
    primary_face = None
    smoothed_x, smoothed_y = 0, 0
    last_face_x, last_face_y, last_face_w, last_face_h = 0, 0, 0, 0
    frames_since_last_change = 0  # Counter to enforce 1-second lock
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Stop at the end of the video

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Face selection logic
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)  # Sort by size
            new_x, new_y, w, h = faces[0]

            if primary_face is None:
                # Initialize primary face immediately
                smoothed_x, smoothed_y = new_x, new_y
                primary_face = (smoothed_x, smoothed_y, w, h)
                frames_since_last_change = 0  # Reset lock counter
            else:
                old_x, old_y, old_w, old_h = primary_face
                distance = abs(new_x - old_x) + abs(new_y - old_y)  # Compute movement distance

                if frames_since_last_change >= LOCK_TIME:  # Only allow switching after lock time
                    primary_face = (new_x, new_y, w, h)
                    smoothed_x, smoothed_y = new_x, new_y
                    frames_since_last_change = 0  # Reset counter
                else:
                    frames_since_last_change += 1  # Continue tracking the current face

            # Apply smoothing to avoid sudden jumps
            smoothed_x = int(SMOOTHING_ALPHA * new_x + (1 - SMOOTHING_ALPHA) * smoothed_x)
            smoothed_y = int(SMOOTHING_ALPHA * new_y + (1 - SMOOTHING_ALPHA) * smoothed_y)
            primary_face = (smoothed_x, smoothed_y, w, h)

        # Use last known face position if no face is detected
        if primary_face is not None:
            x, y, w, h = primary_face

            # Compute crop area while keeping it within frame boundaries
            crop_x = max(0, min(frame_width - target_width, x + (w - target_width) // 2))
            crop_y = max(0, min(frame_height - target_height, y + (h - target_height) // 2))

            # Store last known good face position
            last_face_x, last_face_y, last_face_w, last_face_h = x, y, w, h
        else:
            # Use last known face if no detection occurs
            crop_x = max(0, min(frame_width - target_width, last_face_x + (last_face_w - target_width) // 2))
            crop_y = max(0, min(frame_height - target_height, last_face_y + (last_face_h - target_height) // 2))

        # Crop the frame based on calculated coordinates
        crop_x2 = min(crop_x + target_width, frame_width)
        crop_y2 = min(crop_y + target_height, frame_height)
        cropped_frame = frame[crop_y:crop_y2, crop_x:crop_x2]

        # Resize to ensure consistent output
        resized_frame = cv2.resize(cropped_frame, (target_width, target_height))

        # Write the processed frame to output
        output_video.write(resized_frame)
        frame_count += 1

    cap.release()
    output_video.release()
    print("Video processed successfully with stable face tracking.")



def crop_video2(faces, input_file, output_file):
    try:
        if len(faces) > 0:
            # Constants for cropping
            CROP_RATIO = 0.9  # Adjust the ratio to control how much of the face is visible in the cropped video
            VERTICAL_RATIO = 9 / 16  # Aspect ratio for the vertical video
            BATCH_DURATION = 5  # Duration of each batch in seconds

            # Read the input video
            cap = cv2.VideoCapture(input_file)

            # Get the frame dimensions
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Calculate the target width and height for cropping (vertical format)
            target_height = int(frame_height * CROP_RATIO)
            target_width = int(target_height * VERTICAL_RATIO)

            # Calculate the number of frames per batch
            frames_per_batch = int(cap.get(cv2.CAP_PROP_FPS) * BATCH_DURATION)

            # Create a VideoWriter object to save the output video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_video = cv2.VideoWriter(output_file, fourcc, 30.0, (target_width, target_height))

            # Loop through each batch of frames
            while True:
                ret, frame = cap.read()

                # If no more frames, break out of the loop
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert frame to BGR color format
                # Iterate through each detected face
                for face in faces[:1]:
                    # Unpack the face coordinates
                    x, y, w, h = face

                    # Calculate the crop coordinates
                    crop_x = max(0, x + (w - target_width) // 2)  # Adjust the crop region to center the face
                    crop_y = max(0, y + (h - target_height) // 2)
                    crop_x2 = min(crop_x + target_width, frame_width)
                    crop_y2 = min(crop_y + target_height, frame_height)

                    # Crop the frame based on the calculated crop coordinates
                    cropped_frame = frame[crop_y:crop_y2, crop_x:crop_x2]

                    # Resize the cropped frame to the target dimensions
                    resized_frame = cv2.resize(cropped_frame, (target_width, target_height))

                    # Write the resized frame to the output video
                    output_video.write(resized_frame)

                    # Check if the current frame index is divisible by frames_per_batch
                    if cap.get(cv2.CAP_PROP_POS_FRAMES) % frames_per_batch == 0:
                        # Analyze the lip movement or facial muscle activity within the batch
                        is_talking = is_talking_in_batch(resized_frame)

                        # Adjust the focus based on the speaking activity
                        adjust_focus(is_talking)

            # Release the input and output video objects
            cap.release()
            output_video.release()

            print("Video cropped successfully.")
        else:
            print("No faces detected in the video.")
    except Exception as e:
        print(f"Error during video cropping: {str(e)}")


def add_captions_to_video(input_file, output_file, transcript):
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_video = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    caption_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if caption_index < len(transcript):
            # Scale font size based on video width
            font_scale = frame_width / 300  
            thickness = max(2, int(frame_width / 400))  
            outline_thickness = thickness + 2  # Slightly thicker outline
            
            wrapped_text = textwrap.fill(transcript[caption_index]['text'], width=int(frame_width * 0.08))
            lines = wrapped_text.split('\n')
            y_offset = frame_height - 50  # Position captions near the bottom
            
            for line in lines:
                # Get text size to center it
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                text_x = (frame_width - text_size[0]) // 2  # Center horizontally
                
                position = (text_x, y_offset)
                
                # Draw outline (black border)
                cv2.putText(frame, line, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), outline_thickness, cv2.LINE_AA)
                
                # Draw main text (white)
                cv2.putText(frame, line, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                
                y_offset += int(30 * font_scale)  # Move to the next line
            
            if frame_count % (total_frames // len(transcript)) == 0:
                caption_index += 1

        output_video.write(frame)
        frame_count += 1

    cap.release()
    output_video.release()
    print("Processing complete. Saved as:", output_file)


def add_music_and_flashes(video_file, audio_file, output_file, original_audio):
    video = mp.VideoFileClip(video_file)
    audio = mp.AudioFileClip(audio_file).with_volume_scaled(0.1).subclipped(0.0, video.duration)
    if video.audio or original_audio:
        audio = mp.CompositeAudioClip([original_audio, mp.AudioFileClip(audio_file)])
    video.audio = (audio)
    
    y, sr = librosa.load(audio_file, sr=None)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
    
    frames = []
    for i, frame in enumerate(video.iter_frames()):
        if i in peaks:
            flash_frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=50)
            frames.append(flash_frame)
        else:
            frames.append(frame)
    
    output_video = mp.ImageSequenceClip(frames, fps=video.fps)
    output_video.audio = audio
    output_video.write_videofile(output_file, codec='libx264')
    print("Music and flashes added successfully.")


def is_talking_in_batch(frames):
    # Calculate the motion between consecutive frames
    motion_scores = []
    for i in range(len(frames) - 1):
        frame1 = frames[i]
        frame2 = frames[i+1]
        motion_score = calculate_motion_score(frame1, frame2)  # Replace with your motion analysis function
        motion_scores.append(motion_score)

    # Determine if talking behavior is present based on motion scores
    threshold = 0.5  # Adjust the threshold as needed
    talking = any(score > threshold for score in motion_scores)

    return talking

def calculate_motion_score(frame1, frame2):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate magnitude of optical flow vectors
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    # Calculate motion score as the average magnitude of optical flow vectors
    motion_score = np.mean(magnitude)

    return motion_score

def adjust_focus(frame, talking):
    if talking:
        # Apply visual effects or adjustments to emphasize the speaker
        # For example, you can add a bounding box or overlay text on the frame
        # indicating the speaker is talking
        # You can also experiment with resizing or positioning the frame to
        # focus on the talking person

        # Example: Draw a bounding box around the face region
        face_coordinates = get_face_coordinates(frame)  # Replace with your face detection logic

        if face_coordinates is not None:
            x, y, w, h = face_coordinates
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame


def get_face_coordinates(frame):
    # Load the pre-trained Haar cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Return the coordinates of the first detected face
        x, y, w, h = faces[0]
        return x, y, w, h

    # If no face detected, return None
    return None

def get_transcript(video_id):
    # Get the transcript for the given YouTube video ID
    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    # Format the transcript for feeding into GPT-4
    formatted_transcript = ''
    for entry in transcript:
        start_time = "{:.2f}".format(entry['start'])
        end_time = "{:.2f}".format(entry['start'] + entry['duration'])
        text = entry['text']
        formatted_transcript += f"{start_time} --> {end_time} : {text}\n"

    return transcript



#Analyze transcript with GPT-3 function
response_obj='''[
  {
    "start_time": 97.19, 
    "end_time": 127.43,
    "description": "Spoken Text here"
    "duration":36 #Length in seconds
  },
  {
    "start_time": 169.58,
    "end_time": 199.10,
    "description": "Spoken Text here"
    "duration":33 
  },
]'''
def analyze_transcript(transcript, model = "dev-gpt-4o-2024-05-13-chat-completions"):
    prompt = f"This is a transcript of a video. Please identify the 3 most viral sections from the whole, make sure they are more than 30 seconds in duration,Make Sure you provide extremely accurate timestamps respond only in this format {response_obj}  \n Here is the Transcription:\n{transcript}"
    chat_request_data = {
        "messages": [
            {"role": "system", "content": "You are a ViralGPT helpful assistant. You are master at reading youtube transcripts and identifying the most Interesting and Viral Content"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 4000
    }
    again = True
    while again:
        try:
            response = llm_client.send_request(model, chat_request_data, endpoint="chat")
            response["choices"][0]['message']['content']
            again = False
        # except KeyError as e:
        except Exception as e:
            print(f"Got an error: {e} Waiting for 5 seconds...")
            time.sleep(5)
            again = True
    return response["choices"][0]['message']

"""Main function and execution"""

interseting_seg='''[{'text': 'happiness through Curiosity on Dr', 'start': 0.0, 'duration': 4.82}, {'text': 'eclipse', 'start': 2.28, 'duration': 2.54}, {'text': 'little rookie question for you okay and', 'start': 6.899, 'duration': 4.021}, {'text': "I'm asking this on behalf of mainstream", 'start': 9.24, 'duration': 3.6}, {'text': 'media how do you feel when you see', 'start': 10.92, 'duration': 5.4}, {'text': 'movies like pathan or tiger or any', 'start': 12.84, 'duration': 5.939}, {'text': "Indian I think we haven't got the art of", 'start': 16.32, 'duration': 4.5}, {'text': 'doing those movies you think they can be', 'start': 18.779, 'duration': 4.321}, {'text': 'done better oh yes I mean they can be', 'start': 20.82, 'duration': 3.42}, {'text': 'realistic', 'start': 23.1, 'duration': 3.12}, {'text': "okay we're not realistic what you see", 'start': 24.24, 'duration': 4.32}, {'text': 'what is not realistic about them huh', 'start': 26.22, 'duration': 4.219}, {'text': "it's not realistic", 'start': 28.56, 'duration': 4.38}, {'text': "you're trying to make a James Bond movie", 'start': 30.439, 'duration': 5.741}, {'text': 'which is also not realistic okay', 'start': 32.94, 'duration': 5.88}, {'text': 'then you have this story of the isi girl', 'start': 36.18, 'duration': 4.74}, {'text': 'in the raw man', 'start': 38.82, 'duration': 4.86}, {'text': 'living happily ever after I mean', 'start': 40.92, 'duration': 4.639}, {'text': 'take a break', 'start': 43.68, 'duration': 7.08}, {'text': 'has that ever happened not really right', 'start': 45.559, 'duration': 7.48}, {'text': 'no the whole atmospherics of the whole', 'start': 50.76, 'duration': 3.54}, {'text': 'thing you know', 'start': 53.039, 'duration': 3.36}, {'text': "I haven't seen batana and I won't see it", 'start': 54.3, 'duration': 5.099}, {'text': "because I don't think it is an accurate", 'start': 56.399, 'duration': 4.98}, {'text': "depiction it's not an accurate I'm not", 'start': 59.399, 'duration': 4.941}, {'text': 'going to waste my time', 'start': 61.379, 'duration': 2.961}, {'text': 'and I laughed and I enjoyed that because', 'start': 65.18, 'duration': 6.28}, {'text': 'it was so quaint', 'start': 68.04, 'duration': 5.7}, {'text': 'not because it was defeating anything', 'start': 71.46, 'duration': 3.659}, {'text': 'yeah', 'start': 73.74, 'duration': 5.4}, {'text': 'like you had that other movie of um', 'start': 75.119, 'duration': 7.5}, {'text': 'war that they can no this was this', 'start': 79.14, 'duration': 5.82}, {'text': 'fellow Salman Khan going under a tunnel', 'start': 82.619, 'duration': 5.281}, {'text': 'into Pakistan to deliver a girl who had', 'start': 84.96, 'duration': 4.88}, {'text': 'got legendary', 'start': 87.9, 'duration': 4.14}, {'text': 'but whatever', 'start': 89.84, 'duration': 4.86}, {'text': 'I mean', 'start': 92.04, 'duration': 2.66}, {'text': 'could I exaggerated okay this is not you', 'start': 95.46, 'duration': 5.4}, {'text': 'have to have entertainment which is fun', 'start': 99.0, 'duration': 4.079}, {'text': 'and realistic you should see that movie', 'start': 100.86, 'duration': 3.36}, {'text': 'The', 'start': 103.079, 'duration': 4.86}, {'text': 'Bridge of spies hey that is a real movie', 'start': 104.22, 'duration': 6.78}, {'text': 'okay that is how real spy movies are', 'start': 107.939, 'duration': 5.521}, {'text': 'made what does a real spy movie', 'start': 111.0, 'duration': 5.46}, {'text': 'constitute it means dealing with actual', 'start': 113.46, 'duration': 5.64}, {'text': 'facts no no blonde round no nothing', 'start': 116.46, 'duration': 4.74}, {'text': "around it's okay living a lonely life", 'start': 119.1, 'duration': 4.799}, {'text': "you're living on by yourself living your", 'start': 121.2, 'duration': 6.0}, {'text': 'cover story he able uh', 'start': 123.899, 'duration': 5.821}, {'text': 'with goldfish was actually a notice so', 'start': 127.2, 'duration': 3.839}, {'text': 'he was doing paintings he used to make', 'start': 129.72, 'duration': 3.78}, {'text': 'him make money out of it and but he was', 'start': 131.039, 'duration': 5.161}, {'text': 'doing this other job also so running is', 'start': 133.5, 'duration': 5.099}, {'text': 'espionage ring', 'start': 136.2, 'duration': 4.92}, {'text': 'and they show all that how a documents', 'start': 138.599, 'duration': 5.22}, {'text': 'are exchanged or document information is', 'start': 141.12, 'duration': 4.86}, {'text': 'exchanged you have things called letter', 'start': 143.819, 'duration': 5.941}, {'text': 'dead letter boxes a dead letter box in', 'start': 145.98, 'duration': 7.2}, {'text': 'in Espionage is a place it could be a', 'start': 149.76, 'duration': 6.42}, {'text': "book let's say or or that statue I put", 'start': 153.18, 'duration': 6.48}, {'text': 'my UBS under it', 'start': 156.18, 'duration': 5.279}, {'text': 'and leave it', 'start': 159.66, 'duration': 5.46}, {'text': 'and leave a sign outside on some tree or', 'start': 161.459, 'duration': 4.801}, {'text': 'a wall', 'start': 165.12, 'duration': 5.759}, {'text': "that I've I've fed the the dead litter", 'start': 166.26, 'duration': 6.42}, {'text': 'box okay so the other chap comes and', 'start': 170.879, 'duration': 3.661}, {'text': 'picks it up and takes it away the two', 'start': 172.68, 'duration': 4.26}, {'text': 'never meet based on the true nature of', 'start': 174.54, 'duration': 3.72}, {'text': 'espionage', 'start': 176.94, 'duration': 4.2}, {'text': "which Indian actor's style would be best", 'start': 178.26, 'duration': 7.259}, {'text': 'suited to portray the character of a spy', 'start': 181.14, 'duration': 6.84}, {'text': 'you know I I saw um', 'start': 185.519, 'duration': 4.921}, {'text': 'three three three actors were three or', 'start': 187.98, 'duration': 4.679}, {'text': 'four actors were very good this kind of', 'start': 190.44, 'duration': 3.299}, {'text': 'a thing', 'start': 192.659, 'duration': 3.901}, {'text': 'who could fit into these kind Sorrows', 'start': 193.739, 'duration': 6.481}, {'text': 'not giving any order of preference but', 'start': 196.56, 'duration': 7.02}, {'text': 'I like nawazuddin Siddiqui I used to', 'start': 200.22, 'duration': 5.599}, {'text': 'like Imran Khan', 'start': 203.58, 'duration': 4.439}, {'text': 'Irfan Khan sorry', 'start': 205.819, 'duration': 6.28}, {'text': 'and he was he was a consummate actor', 'start': 208.019, 'duration': 8.821}, {'text': 'Anup anupam care and', 'start': 212.099, 'duration': 8.241}, {'text': 'these two actors', 'start': 216.84, 'duration': 3.5}, {'text': 'the one who played family man um', 'start': 220.62, 'duration': 6.96}, {'text': 'very good okay they could fit into the', 'start': 224.84, 'duration': 8.02}, {'text': 'room and Mishra pankaj Mishra foreign', 'start': 227.58, 'duration': 5.28}, {'text': '[Music]', 'start': 233.72, 'duration': 3.11}, {'text': "spy all right it's a cold war story", 'start': 259.699, 'duration': 6.461}, {'text': "about the the it's actually based on", 'start': 263.52, 'duration': 5.179}, {'text': 'this Cambridge 5.', 'start': 266.16, 'duration': 5.9}, {'text': 'you know the Cambridge five those', 'start': 268.699, 'duration': 6.341}, {'text': 'Kim philby and others who were spying', 'start': 272.06, 'duration': 6.1}, {'text': 'for who were actually with the MI6 but', 'start': 275.04, 'duration': 6.0}, {'text': 'it was actually a KGB agent okay the', 'start': 278.16, 'duration': 5.58}, {'text': 'real mole and he would have been Chief', 'start': 281.04, 'duration': 4.08}, {'text': 'maybe one day', 'start': 283.74, 'duration': 4.08}, {'text': 'at the not been caught out', 'start': 285.12, 'duration': 7.579}, {'text': 'so on that is made a novel Tinker spy', 'start': 287.82, 'duration': 7.26}, {'text': "it's beautifully done the book is", 'start': 292.699, 'duration': 6.241}, {'text': 'marvelous and the acting and the', 'start': 295.08, 'duration': 3.86}, {'text': 'you should watch it okay and watch this', 'start': 302.78, 'duration': 6.04}, {'text': 'uh Bridge of spies if you enjoyed this', 'start': 305.88, 'duration': 5.9}, {'text': 'video subscribe TRS clips for more', 'start': 308.82, 'duration': 15.86}, {'text': '[Music]', 'start': 311.78, 'duration': 15.33}, {'text': 'thank you', 'start': 324.68, 'duration': 8.55}, {'text': '[Music]', 'start': 327.11, 'duration': 6.12}]''';

def main():
    # video_id='92nse3cvG_Y'
    video_id='8XnmwzsgfVs'
    url = 'https://www.youtube.com/watch?v='+video_id  # Replace with your video's URL
    filename = 'input_video.mp4'
    # download_video(url,filename)
    
    transcript = get_transcript(video_id)
    print(transcript)
    interesting_segment = analyze_transcript(transcript)
    print(interesting_segment)
    content = interesting_segment["content"]
    parsed_content = json.loads(content)
    print(parsed_content)
    #pdb.set_trace()
    segment_video(parsed_content)
    
    # Loop through each segment
    for i in range(0, 3):  # Replace 3 with the actual number of segments
        start_time = parsed_content[i]['start_time']
        end_time = parsed_content[i]['end_time']
        transcript_subset = [entry for entry in transcript if entry['start'] >= start_time and entry['start'] + entry['duration'] <= end_time]
        input_file = f'output{str(i).zfill(3)}.mp4'
        output_file = f'output_cropped{str(i).zfill(3)}.mp4'
        # faces = detect_human_faces(input_file)
        # crop_video(faces, input_file, output_file)
        audio = mp.VideoFileClip(input_file).audio
        crop_video_to_primary_face(input_file, output_file)
        output_file = f'output_cropped_captioned{str(i).zfill(3)}.mp4'
        input_file = f'output_cropped{str(i).zfill(3)}.mp4'
        add_captions_to_video(input_file, output_file, transcript_subset)
        input_file = f'output_cropped_captioned{str(i).zfill(3)}.mp4'
        output_file = f'output_cropped_captioned_music{str(i).zfill(3)}.mp4'
        add_music_and_flashes(input_file, "motivational_background.mp3", output_file, audio)
    

# Run the main function
main()