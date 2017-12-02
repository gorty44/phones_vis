import re
import argparse
from pydub import AudioSegment

file_title = re.compile("\"\*/.*\.lab\"")
phone_length = re.compile("\d* \d* \w*")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlf-file", dest="mlf_file")
    parser.add_argument("--audio-directory", dest="audio_dir")
    parser.add_argument("--output-directory", dest="output_dir")
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.mlf_file) as mlf:
        mlf_string = mlf.read()

    for line in mlf_string.splitlines():
        if file_title.match(line):
            title = re.split("\.lab\"", re.split("\"\*/", line)[1])[0].upper()
            audio_file = AudioSegment.from_wav("{}/{}".format(args.audio_dir, title))
        if phone_length.match(line):
            parsed_length = (re.split(" ", line))
            start = int(parsed_length[0]) / 10000
            end = int(parsed_length[1]) / 10000
            phone = parsed_length[2]
            if phone == "sil":
                continue
            segmented_fragment = audio_file[start:end]
            segmented_fragment.export("{}/{}.{}".format(args.output_dir, title, phone), format="wav")

if __name__ == "__main__":
    main()
