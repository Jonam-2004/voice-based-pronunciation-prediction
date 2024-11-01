import os
import pandas as pd
import torchaudio
from datasets import Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

data_dir = "TIMIT/data"
csv_path = "TIMIT/preprocessed_train.csv"
metadata_df = pd.read_csv(csv_path)

# Define a function to load audio and transcription data
def prepare_dataset(row):
    audio_path = os.path.join(data_dir, row['path_from_data_dir'])
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
    
    # Extract transcription from the .WRD or .PHN file
    transcription = ""
    word_path = audio_path.replace(".WAV.wav", ".WRD")  # Replace extension to find transcription
    if os.path.exists(word_path):
        with open(word_path, 'r', encoding='utf-8', errors='ignore') as f:  # Ignore unsupported characters
            words = [line.strip().split()[2] for line in f.readlines() if len(line.split())>=3]  # Take the word column
            transcription = " ".join(words)  # Join all words in the file
    else:
        # Fall back to phonetic transcription if word transcription is not available
        phonetic_path = audio_path.replace(".WAV.wav", ".PHN")
        if os.path.exists(phonetic_path):
            with open(phonetic_path, 'r', encoding='utf-8', errors='ignore') as f:  # Ignore unsupported characters
                phonemes = [line.strip().split()[2] for line in f.readlines()]
                transcription = " ".join(phonemes)

    return {
        "audio": waveform,
        "sampling_rate": 16000,
        "transcription": transcription
    }

# Apply function to metadata DataFrame
samples = [prepare_dataset(row) for _, row in metadata_df.iterrows() if row['is_audio']]
timit_dataset = Dataset.from_pandas(pd.DataFrame(samples))


# Load Wav2Vec2 processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Preprocess function to convert audio and text to Wav2Vec2-compatible format
def preprocess(batch):
    audio = batch["audio"]
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_values
    batch["input_values"] = inputs[0]
    batch["labels"] = processor(batch["transcription"], return_tensors="pt", padding=True).input_ids[0]
    return batch

# Apply preprocessing to dataset
timit_dataset = timit_dataset.map(preprocess)

# Define training arguments and Trainer
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./wav2vec2_timit",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    evaluation_strategy="epoch",
    num_train_epochs=3,
    save_steps=400,
    logging_steps=400,
    learning_rate=1e-4,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=timit_dataset,
    tokenizer=processor.feature_extractor
)

# Train the model
trainer.train()

trainer.save_model("wav_to_vec_model")