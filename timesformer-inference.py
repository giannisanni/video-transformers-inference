import streamlit as st
import av
import torch
import numpy as np
import os
from transformers import AutoImageProcessor, AutoModelForVideoClassification

# Set a seed for reproducibility
np.random.seed(0)

# Function to read and process video frames
def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

# Function to sample frame indices
def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

# Streamlit interface
st.title('Video Classification App')

# User input for model
user_model_input = st.text_input("Enter a Hugging Face model identifier or local file name", "timesformer-base-finetuned-k400-finetuned-ucf101-subset")

# Load model and processor
try:
    model = AutoModelForVideoClassification.from_pretrained(user_model_input)
    image_processor = AutoImageProcessor.from_pretrained(user_model_input)
except Exception as e:
    st.error(f"Failed to load model: {e}")

# Choose source of video
video_source = st.radio("Choose the video source", ("Upload", "Local Directory"))

if video_source == "Upload":
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi"])
    video_file = uploaded_file
elif video_source == "Local Directory":
    # Assuming videos are stored in a folder named 'videos'
    video_files = os.listdir('videos')
    video_files = [file for file in video_files if file.endswith(('.mp4', '.avi'))]
    selected_file = st.selectbox("Select a video file", video_files)
    video_file = os.path.join('videos', selected_file)

if video_file is not None:
    # Display the uploaded or selected video
    st.video(video_file)

    # Process the video file
    with av.open(video_file) as container:
        indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        video = read_video_pyav(container, indices)

    # Prepare the video for the model
    inputs = image_processor(list(video), return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the prediction
    predicted_label = logits.argmax(-1).item()
    class_name = model.config.id2label[predicted_label]
    st.write(f'Predicted Activity: {class_name}')
