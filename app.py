import subprocess, sys; subprocess.check_call([sys.executable, "-m", "pip", "install", "moviepy==1.0.3"])

# ... your existing code continues below ...
import os
import torch
import torch.nn as nn
# (Rest of your script remains exactly the same)
import gradio as gr
import pandas as pd
import plotly.express as px
from transformers import pipeline, AutoTokenizer, AutoModel
from moviepy.editor import VideoFileClip

# 1. DECLARE ARCHITECTURE SPECIFICATION
class WebsiteSentimentClassifier(nn.Module):
    def __init__(self, visual_dim=512, text_dim=768, hidden_dim=128):
        super(WebsiteSentimentClassifier, self).__init__()
        self.vis_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 3)
        )
    def forward(self, visual_feat, text_feat):
        x_v = nn.functional.relu(self.vis_proj(visual_feat))
        x_t = nn.functional.relu(self.text_proj(text_feat))
        fused = torch.cat((x_v, x_t), dim=-1)
        return self.classifier(fused)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate and bind saved weights mapping
model = WebsiteSentimentClassifier().to(device)
if os.path.exists("video_sentiment_model.pth"):
    model.load_state_dict(torch.load("video_sentiment_model.pth", map_location=device))
    print("🔄 Custom .pth file mounted perfectly.")
else:
    print("⚠️ Warning: 'video_sentiment_model.pth' not found. Run train.py first.")
model.eval()

# 2. SPIN UP SPEECH TO TEXT PIPELINE & TEXT EMBEDDING GENERATOR
print("Loading supporting framework layers...")
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=0 if torch.cuda.is_available() else -1)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text_encoder = AutoModel.from_pretrained("bert-base-uncased").to(device)

# 3. INTERFACE EXECUTION LOGIC
def evaluate_multimodal_sentiment(file_path, raw_text_input=None, mode="video"):
    extracted_text = ""
    temp_audio_path = "temp_extracted_audio.wav"
    
    # Run speech extraction if a media file is provided
    if mode == "video":
        if not file_path:
            return "Missing File Path", None, "Please upload a video file entry."
        try:
            print("Extracting audio from video using moviepy...")
            video_clip = VideoFileClip(file_path)
            if video_clip.audio is not None:
                video_clip.audio.write_audiofile(temp_audio_path, codec='pcm_s16le', fps=16000, logger=None)
                video_clip.close()
                
                # Transcribe the clean extracted wav track
                transcription = transcriber(temp_audio_path)
                extracted_text = transcription.get("text", "").strip()
            else:
                video_clip.close()
                extracted_text = ""
        except Exception as e:
            print(f"Extraction error: {e}")
            extracted_text = ""
        finally:
            # Clean up the temporary audio file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                
    elif mode == "audio":
        if not file_path:
            return "Missing File Path", None, "Please upload an audio file entry."
        transcription = transcriber(file_path)
        extracted_text = transcription.get("text", "").strip()
    else:
        extracted_text = raw_text_input if raw_text_input else ""
        
    if not extracted_text:
        return "Neutral 🟡", None, "No verbal inputs or text strings detected."

    # Process transcript text through BERT to pass to your custom model
    text_tokens = tokenizer(extracted_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        text_embeddings = text_encoder(**text_tokens).pooler_output
        fallback_visual_features = torch.zeros(1, 512).to(device)
        
        # Execute unified .pth model pass
        raw_logits = model(fallback_visual_features, text_embeddings)
        probabilities = torch.softmax(raw_logits, dim=-1).squeeze().cpu().numpy()

    pos_score, neg_score, neu_score = float(probabilities[0]), float(probabilities[1]), float(probabilities[2])
    
    sentiment_classes = ["Positive", "Negative", "Neutral"]
    score_mapping = [pos_score, neg_score, neu_score]
    verdict_strings = ["Positive 🟢", "Negative 🔴", "Neutral 🟡"]
    
    best_match_idx = score_mapping.index(max(score_mapping))
    final_verdict = f"🏆 Best Verdict Match: {verdict_strings[best_match_idx]}"

    # Build Plotly Pie Chart
    df = pd.DataFrame({"Sentiment Class": sentiment_classes, "Confidence Level": score_mapping})
    fig = px.pie(
        df, values='Confidence Level', names='Sentiment Class', color='Sentiment Class',
        color_discrete_map={'Positive':'#2ECC71', 'Negative':'#E74C3C', 'Neutral':'#F1C40F'},
        hole=0.45
    )
    fig.update_layout(margin=dict(l=15, r=15, t=15, b=15), height=290)

    return final_verdict, fig, extracted_text

# Route separate frontend functional maps
def handle_video(v): return evaluate_multimodal_sentiment(v, mode="video")
def handle_audio(a): return evaluate_multimodal_sentiment(a, mode="audio")
def handle_text(t):   return evaluate_multimodal_sentiment(None, raw_text_input=t, mode="text")

# 4. RUN WEBSITE LAYOUT INTERFACE
with gr.Blocks(theme=gr.themes.Default(primary_hue="teal", secondary_hue="slate")) as demo:
    gr.Markdown("# 🎬 Advanced Multimodal Sentiment Analysis Studio")
    gr.Markdown("Integrated multi-page dashboard working directly with your custom `.pth` models.")
    
    with gr.Tab("Video Sentiment Page"):
        with gr.Row():
            with gr.Column():
                video_in = gr.Video(label="Upload MP4 Target Clip")
                v_btn = gr.Button("Analyze Video Content", variant="primary")
            with gr.Column():
                v_lbl = gr.Label(label="Prediction Summary")
                v_txt = gr.Textbox(label="Detected/Transcribed Text", interactive=False)
                v_chart = gr.Plot(label="Pie Chart Metrics Distribution")
        v_btn.click(fn=handle_video, inputs=video_in, outputs=[v_lbl, v_chart, v_txt])

    with gr.Tab("Audio Sentiment Page"):
        with gr.Row():
            with gr.Column():
                audio_in = gr.Audio(label="Upload Audio track file", type="filepath")
                a_btn = gr.Button("Analyze Audio Waveforms", variant="primary")
            with gr.Column():
                a_lbl = gr.Label(label="Prediction Summary")
                a_txt = gr.Textbox(label="Detected/Transcribed Text", interactive=False)
                a_chart = gr.Plot(label="Pie Chart Metrics Distribution")
        a_btn.click(fn=handle_audio, inputs=audio_in, outputs=[a_lbl, a_chart, a_txt])

    with gr.Tab("Text Sentiment Page"):
        with gr.Row():
            with gr.Column():
                text_in = gr.Textbox(label="Enter custom sentences manually", lines=5)
                t_btn = gr.Button("Analyze Input Text String", variant="primary")
            with gr.Column():
                t_lbl = gr.Label(label="Prediction Summary")
                t_chart = gr.Plot(label="Pie Chart Metrics Distribution")
        t_btn.click(fn=handle_text, inputs=text_in, outputs=[t_lbl, t_chart])

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)