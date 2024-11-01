import os
import pandas as pd
import torchaudio
from datasets import Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import TrainingArguments, Trainer


data_dir = "TIMIT/data"
csv_path = "TIMIT/preprocessed_train.csv"
metadata_df = pd.read_csv(csv_path)

# Define a function to load a sample given the metadata row
def prepare_dataset(row):
    sample = {}
    
    # Load audio file if available
    if row['is_audio']:
        audio_path = os.path.join(data_dir, row['path_from_data_dir'])
        waveform, sample_rate = torchaudio.load(audio_path)
        sample['audio'] = waveform.squeeze().numpy().tolist()  # Convert tensor to list
        sample['sample_rate'] = sample_rate
    
    # Load word transcription if available
    if row['is_word_file']:
        word_path = os.path.join(data_dir, row['path_from_data_dir'].replace(".WAV", ".WRD"))
        with open(word_path, 'r') as f:
            words = [line.strip().split()[2] for line in f.readlines() if len(line.split()) >= 3]  # Take the word column
        sample['words'] = words
    
    return sample

# Prepare dataset samples
samples = [prepare_dataset(row) for _, row in metadata_df.iterrows() if row['is_audio']]

# Create a Dataset from the samples
timit_dataset = Dataset.from_pandas(pd.DataFrame(samples))

print(f"dataset loaded with {len(timit_dataset)}entries")

# Load Wav2Vec2 processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def preprocess(batch):
    audio_paths = batch["audio"]
    audio_samples = []
    
    for audio_path in audio_paths:
        waveform, sample_rate = torchaudio.load(audio_path)
        audio_samples.append(waveform.squeeze().numpy().tolist())  # Convert to list
    
    inputs = processor(audio_samples, sampling_rate=batch["sample_rate"], return_tensors="pt", padding=True)
    batch["input_values"] = inputs.input_values  # Keep as tensor
    batch["labels"] = processor(batch["words"], return_tensors="pt", padding=True, truncation=True).input_ids[0]
    return batch



# Make sure to preprocess your dataset accordingly
timit_dataset = timit_dataset.map(preprocess, remove_columns=["audio", "words"])

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

trainer.save_model("wav_to_vec_model.pth")