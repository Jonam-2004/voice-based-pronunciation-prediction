import os
import torchaudio

# Set path to data directory
data_dir = "path/to/TIMIT/data/TRAIN"  # Adjust the path accordingly

def load_timit_sample(file_info):
    # Load audio
    audio_path = os.path.join(data_dir, file_info['path_from_data_dir'])
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Load transcriptions
    transcriptions = {}
    for file_type in ['is_word_file', 'is_phonetic_file', 'is_sentence_file']:
        if file_info[file_type]:
            transcription_path = os.path.join(data_dir, file_info['path_from_data_dir'].replace(".WAV", ".PHN" if file_type == 'is_phonetic_file' else ".WRD" if file_type == 'is_word_file' else ".TXT"))
            with open(transcription_path, "r") as f:
                transcriptions[file_type] = f.read().strip()
    
    return waveform, sample_rate, transcriptions

# Example to load one sample
file_info = {
    "path_from_data_dir": "DR4/MMDM0/SI681.WAV",  # Example path
    "is_audio": True,
    "is_word_file": False,
    "is_phonetic_file": False,
    "is_sentence_file": True
}
waveform, sample_rate, transcriptions = load_timit_sample(file_info)

print("Audio Waveform:", waveform)
print("Sample Rate:", sample_rate)
print("Transcriptions:", transcriptions)
