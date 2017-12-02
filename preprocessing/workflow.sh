#!/bin/bash
#
# 1. Remove extensions from filenames
# 2. Run parser
# 3. Convert segmented files to mp3
# 4. tSNE pipeline


while [[ ${#} -gt 0 ]]; do
    case "${1}" in
        --segmented-audio-directory)
            shift
            segmented_dir="${1}"
            if [[ ! -d "${segmented_dir}" ]]; then
                mkdir -p "${segmented_dir}"
            fi
            ;;
        --mp3-output-directory)
            shift
            mp3_dir="${1}"
            if [[ ! -d "${mp3_dir}" ]]; then
                mkdir -p "${mp3_dir}"
            fi
            ;;
        --extension)
            shift
            extension="${1}"
            ;;
        --path-to-mlf-files)
            shift
            mlf_files="${1}"
            ;;
        *)
            echo "Wrong argument"
            exit 1
            ;;
    esac
    shift
done

for mlf_file in "${mlf_files}"/*\.MLF; do
    audio_dir="$(echo ${mlf_file} | cut -f 1 -d '.')"
    for audio_file in "${audio_dir}"/*\."${extension}"; do
        mv "${audio_file}" "$(echo ${audio_file} | cut -f 1 -d '.')"
    done
    python3 parser.py --mlf-file "${mlf_file}" --audio-directory "${audio_dir}" --output-directory "${segmented_dir}"
done

for wave_file in "${segmented_dir}"/*; do
    file_name="$(basename ${wave_file})"
    lame -b 320 -h "${wave_file}" "${mp3_dir}/${file_name}.mp3";
done

(
cd ../features/
python3 collect_samples.py
python3 samples_to_audio_spritesheet.py
python3 samples_to_fingerprints.py
python3 fingerprints_to_tSNE.py
)
