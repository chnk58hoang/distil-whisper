from datasets import Dataset
from datasets.features import Audio
import os


def create_local_dataset(wav_directory, metadata_file):
    all_wav_paths = []
    for root, dirs, files in os.walk(wav_directory):
        for file in sorted(files):
            p = os.path.join(root, file)
            all_wav_paths.append(os.path.abspath(p))

    with open(metadata_file, 'r') as f:
        all_lines = f.readlines()
    f.close()
    all_wav_paths = sorted(all_wav_paths)
    all_transcriptions = [' '.join(line.split(' ')[1:])[:-1] for line in all_lines]
    dataset_dict = {"audio": all_wav_paths, "sentence": all_transcriptions}
    dataset = Dataset.from_dict(dataset_dict).cast_column("audio", Audio())
    return dataset


