from pathlib import Path
import shutil
import argparse
from tqdm import tqdm

def get_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--dataset_path', default='/home/kcchang3/data/CryCeleb2023/audio')
    parser.add_argument('--des_path', default='/home/kcchang3/data/CryCeleb2023/audio_dump')
    args = parser.parse_args()
    return args

def main(args):
    dataset_path = Path(args.dataset_path)
    des_path = Path(args.des_path)
    des_path.mkdir(parents=True, exist_ok=True)
    split_paths = [x for x in dataset_path.iterdir() if x.is_dir()]
    for split_path in split_paths:
        print(f'dumping files in split {split_path.name}')
        number_paths = [x for x in split_path.iterdir() if x.is_dir()]
        for number_path in tqdm(number_paths):
            letter_paths = [x for x in number_path.iterdir() if x.is_dir()]
            for letter_path in letter_paths:
                file_list = sorted(letter_path.glob('*.wav'))
                for f in file_list:
                    shutil.copy(f, des_path)

if __name__ == "__main__":
    args = get_arguments()
    main(args)