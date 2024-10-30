import os
import pandas as pd
import torchaudio

# Set paths
data_dir = "TIMIT/data"  # Update this to the actual path
csv_path = "TIMIT/preprocessed_train.csv"  # Update this to the path of the CSV file


# Load the CSV file containing file metadata
metadata_df = pd.read_csv(csv_path)

# Define a function to load a sample given the metadata row
def load_timit_sample(row):
    sample = {}
    print(row['path_from_data_dir'])
    # Load audio file if available
    if row['is_audio']:
        audio_path = os.path.join(data_dir, row['path_from_data_dir'])
        waveform, sample_rate = torchaudio.load(audio_path)
        sample['audio'] = (waveform, sample_rate)
    
    # Load phonetic transcription if available
    if row['is_phonetic_file']=='TRUE':
        phonetic_path = os.path.join(data_dir, row['path_from_data_dir'].replace(".WAV", ".PHN"))
        with open(phonetic_path, 'r') as f:
            phonetic_transcription = f.readlines()
        sample['phonetic'] = phonetic_transcription
    
    # Load word transcription if available
    if row['is_word_file']:
        word_path = os.path.join(data_dir, row['path_from_data_dir'].replace(".WAV", ".WRD"))
        with open(word_path, 'r') as f:
            word_transcription = f.readlines()
        sample['word'] = word_transcription
    
    # Load sentence transcription if available
    if row['is_sentence_file']:
        sentence_path = os.path.join(data_dir, row['path_from_data_dir'].replace(".WAV", ".TXT"))
        with open(sentence_path, 'r') as f:
            sentence_transcription = f.read().strip()
        sample['sentence'] = sentence_transcription
    
    return sample

# Iterate through the entire dataset and load samples
dataset = []
for _, row in metadata_df.iterrows():
    # Only load if it's an audio file; transcriptions will follow the audio file
    if pd.isna(row['path_from_data_dir']):
        continue
    if row['is_audio']:
        sample = load_timit_sample(row)
        dataset.append(sample)

print(f"Loaded {len(dataset)} samples from the TIMIT dataset.")
