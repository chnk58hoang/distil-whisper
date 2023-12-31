from datasets import Dataset
from datasets.features import Audio
from typing import List
import librosa
import os


def get_all_audio_files(directory: str):
    all_audio_paths = []
    for root, dirs, files in os.walk(directory):
        for file in sorted(files):
            if file.endswith('.mp3') or file.endswith('.wav'):
                p = os.path.join(root, file)
                all_audio_paths.append(os.path.abspath(p))
    all_audio_paths = sorted(all_audio_paths)
    return all_audio_paths


def get_all_transcription(transcription_path: str):
    # Case 1: all text in one transcription file, each line follow this format
    # <audio_file><\t><text transcript>
    if os.path.isfile(transcription_path):
        with open(transcription_path, "r") as f:
            all_lines = f.readlines()
        f.close()
        all_lines = sorted(all_lines)
        all_transcriptions = [line.split("\t")[:-1] for line in all_lines]
        return all_transcriptions

    # Case 2: all text files and their corresponding audio files is located in one directory
    elif os.path.isdir(transcription_path):
        all_text_paths = []
        for root, dirs, files in os.walk(transcription_path):
            for file in files:
                if file.endswith('.txt'):
                    p = os.path.join(root, file)
                    all_text_paths.append(os.path.abspath(p))
        all_text_paths = sorted(all_text_paths)
        all_transcription = []
        for text_path in all_text_paths:
            with open(text_path, 'r') as f:
                all_transcription.append(f.readline())
            f.close()
        return all_transcription

    else:
        raise FileNotFoundError


def convert_mp3_to_wav(mp3_path, wav_path):
    """
    Convert an MP3 file to a WAV file.
    :param mp3_path: path to the MP3 file
    :param wav_path: path to the output WAV file
    """
    os.system("sox {} -r 16000 -b 16 -c 1 {}".format(mp3_path, wav_path))


def create_local_dataset(all_wav_paths: List[str],
                         all_transcriptions: List[str]):
    """
    Create a huggingface audio dataset from local data directory
    Args:
        all_wav_paths: (List[int]) a list contains all absolute path to wav audio files
        all_transcriptions: (List[int]) a list contains all transcription of above audio files
    Return:
        Huggingface audio dataset. Each data sample can be represented as:
        {'path': '/path/to/the/wav/file', 'audio': {'path': '/path/to/the/wav/file', 'array': numpy array of the audio,
        'sampling_rate': sampling rate}, 'sentence': the corresponding transcription}
    """
    dataset_dict = {"path": all_wav_paths, "audio": all_wav_paths,
                    "sentence": all_transcriptions}
    durations = [librosa.get_duration(path=p) for p in dataset_dict['path']]
    dataset_dict["duration"] = durations

    dataset = Dataset.from_dict(dataset_dict).cast_column("audio", Audio())
    return dataset
