import streamlit as st
import av
import torch
import numpy as np
import os
from transformers import AutoImageProcessor, AutoModelForVideoClassification, VideoMAEForVideoClassification
from huggingface_hub import hf_hub_download

# Set a seed for reproducibility
np.random.seed(0)

# Define common functions
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

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

# Initialize Streamlit app
st.sidebar.title("Model Selection")
app_mode = st.sidebar.selectbox("Choose the model type",
                                ["Video Classification - Timesformer", "Video Classification - VideoMAE"])

if app_mode == "Video Classification - Timesformer":
    st.title('Video Classification App - Timesformer')

    # User input for model
    user_model_input = st.text_input("Enter a Hugging Face model identifier", "giannisan/timesformer-base-finetuned-k400-finetuned-ucf101-subset")

    # Load model and processor
    try:
        model = AutoModelForVideoClassification.from_pretrained(user_model_input)
        image_processor = AutoImageProcessor.from_pretrained(user_model_input)
    except Exception as e:
        st.error(f"Failed to load model: {e}")

elif app_mode == "Video Classification - VideoMAE":
    st.title('Video Classification App - VideoMAE')

    # User input for model
    user_model_input = st.text_input("Enter a Hugging Face model identifier", "MCG-NJU/videomae-base-finetuned-kinetics")

    # Load model and processor
    try:
        model = VideoMAEForVideoClassification.from_pretrained(user_model_input)
        image_processor = AutoImageProcessor.from_pretrained(user_model_input)
    except Exception as e:
        st.error(f"Failed to load model: {e}")

# Video file handling common to both models
video_source = st.radio("Choose the video source", ("Upload", "Local Directory"))
if video_source == "Upload":
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi"])
    video_file = uploaded_file
elif video_source == "Local Directory":
    video_files = os.listdir('videos')
    video_files = [file for file in video_files if file.endswith(('.mp4', '.avi'))]
    selected_file = st.selectbox("Select a video file", video_files)
    video_file = os.path.join('videos', selected_file)

if video_file is not None:
    st.video(video_file)

    # Process the video file
    with av.open(video_file) as container:
        clip_len = 8 if app_mode == "Video Classification - Timesformer" else 16
        indices = sample_frame_indices(clip_len=clip_len, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
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
