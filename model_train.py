import os
from datasets import Dataset
import torchaudio
import pandas as pd
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Trainer, TrainingArguments

# Define the path to the LibriSpeech dataset
dataset_path = "./LibriSpeech/dev-clean"

def load_librispeech_data(dataset_path):
    audio_paths = []
    texts = []

    # Traverse through the directory to collect audio and transcription data
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".flac"):
                # Load audio file path
                audio_path = os.path.join(root, file)
                audio_paths.append(audio_path)

                # Load the corresponding transcription
                # Assuming transcription is in a `.txt` file with the same name as the audio file
                transcription_path = audio_path.replace(".flac", ".wav")
                with open(transcription_path, "r") as f:
                    text = f.readline().strip()
                texts.append(text)
    
    # Create a pandas DataFrame for easier analysis
    data = {
        "audio_path": audio_paths,
        "text": texts
    }
    return pd.DataFrame(data)

# Load the LibriSpeech data into a DataFrame
df = load_librispeech_data(dataset_path)

# Analyze audio durations and text
def analyze_librispeech(df):
    audio_durations = []
    
    # Extract audio durations
    for audio_path in df["audio_path"]:
        audio, sr = torchaudio.load(audio_path)
        duration = audio.shape[1] / sr
        audio_durations.append(duration)

    df["duration"] = audio_durations

    # Basic statistics
    print(df.describe())

    # Plot duration distribution
    plt.figure(figsize=(12, 6))
    plt.hist(df["duration"], bins=100, color="blue", alpha=0.7)
    plt.title("Duration Distribution of Audio Samples")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Frequency")
    plt.show()

    # Show a few examples
    print("Sample texts:")
    print(df["text"].sample(10).to_string(index=False))

# Analyze the dataset
analyze_librispeech(df)

# Load the processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# Preprocessing function
def preprocess_function(row):
    # Load the audio file
    audio, sr = torchaudio.load(row["audio_path"])
    audio = audio.squeeze().numpy()

    # Process the audio input and return the labels
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    row["input_values"] = inputs.input_values[0]
    row["labels"] = processor.tokenizer(row["text"]).input_ids
    
    return row

# Apply preprocessing to the DataFrame
processed_data = df.apply(preprocess_function, axis=1)

# Convert the processed data into a Hugging Face Dataset
dataset = Dataset.from_pandas(processed_data[["input_values", "labels"]])

# Load pre-trained Wav2Vec 2.0 model
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./wav2vec2-librispeech",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=200,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)

# Save the model and processor
model.save_pretrained("./wav2vec2-librispeech")
processor.save_pretrained("./wav2vec2-librispeech")

