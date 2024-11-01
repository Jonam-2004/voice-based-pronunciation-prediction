import os
import pandas as pd
import torchaudio
from datasets import Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import TrainingArguments, Trainer
import gc

data_dir = "TIMIT/data"
csv_path = "TIMIT/preprocessed_train.csv"
metadata_df = pd.read_csv(csv_path)

# Define a function to load a sample given the metadata row
def prepare_dataset(row):
    sample = {}
    
    # Load audio file path if available
    if row['is_audio']:
        audio_path = os.path.join(data_dir, row['path_from_data_dir'])
        sample['audio'] = audio_path  # Store the path instead of the actual data
    
    # Load word transcription if available
    if row['is_word_file']:
        word_path = os.path.join(data_dir, row['path_from_data_dir'].replace(".WAV", ".WRD"))
        with open(word_path, 'r') as f:
            words = [line.strip().split()[2] for line in f.readlines() if len(line.split()) >= 3]
        sample['words'] = ' '.join(words)  # Join words into a single string
    else:
        sample['words'] = ""  # Ensure 'words' key exists even if empty
    
    return sample

# Prepare dataset samples
samples = [prepare_dataset(row) for _, row in metadata_df.iterrows() if row['is_audio']]
timit_dataset = Dataset.from_pandas(pd.DataFrame(samples))

# Load Wav2Vec2 processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Function to load audio files
def load_audio_file(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    return waveform.squeeze()  # Return the waveform without extra dimension

# Preprocess function to convert audio and text to Wav2Vec2-compatible format
def preprocess(batch):
    # Process the audio
    audio_waveforms = [load_audio_file(audio_input) for audio_input in batch['audio']]
    
    # Use the correct sampling rate for the Wav2Vec2 model
    sampling_rate = 16000  # For Wav2Vec2, the typical sampling rate is 16kHz
    
    # Process the audio waveforms
    features = processor(audio_waveforms, sampling_rate=sampling_rate, return_tensors="pt", padding=False, truncation=True)

    # Prepare labels
    words_list = batch['words']  # Make sure you access words from the batch correctly
    batch["labels"] = processor(words_list, return_tensors="pt", padding=True, truncation=True).input_ids
    
    return {'input_values': features.input_values, 'labels': batch['labels']}

# Apply preprocessing to dataset in smaller batches
timit_dataset = timit_dataset.map(preprocess, batched=True, batch_size=4)

# Define training arguments and Trainer
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

# Clean up
gc.collect()  # Run garbage collector
trainer.save_model("wav_to_vec_model.pth")
