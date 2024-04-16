import sys
import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from glob import glob
import subprocess
import traceback
from face_detection import FaceAlignment, LandmarksType

if sys.version_info < (3, 2):
    raise Exception("Must be using >= Python 3.2")

if not os.path.isfile('preprocessing/face_detection/detection/sfd/s3fd.pth'):
    raise FileNotFoundError('Save the s3fd model to face_detection/detection/sfd/s3fd.pth before running this script!')

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", help="Root folder of the LRS2 dataset", required=True)
parser.add_argument("--preprocessed_root", help="Root folder of the preprocessed dataset", required=True)
args = parser.parse_args()

fa = FaceAlignment(LandmarksType._2D, flip_input=False, device='cpu')
ffmpeg_template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'

def process_video_file(vfile, args):
    video_stream = cv2.VideoCapture(vfile)
    frames = []
    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        frames.append(frame)

    vidname = os.path.basename(vfile).split('.')[0]
    dirname = os.path.basename(os.path.dirname(vfile))
    fulldir = os.path.join(args.preprocessed_root, dirname, vidname)
    os.makedirs(fulldir, exist_ok=True)

    batches = [frames[i:i + 32] for i in range(0, len(frames), 32)]  # Assuming batch size of 32 for CPU processing

    for i, fb in enumerate(batches):
        preds = fa.get_detections_for_batch(np.asarray(fb))
        for j, f in enumerate(preds):
            if f is None:
                continue
            x1, y1, x2, y2 = f
            cv2.imwrite(os.path.join(fulldir, '{}.jpg'.format(i * 32 + j)), fb[j][y1:y2, x1:x2])

def process_audio_file(vfile, args):
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = os.path.basename(os.path.dirname(vfile))
    fulldir = os.path.join(args.preprocessed_root, dirname, vidname)
    os.makedirs(fulldir, exist_ok=True)
    wavpath = os.path.join(fulldir, 'audio.wav')
    command = ffmpeg_template.format(vfile, wavpath)
    subprocess.call(command, shell=True)

def main(args):
    print('Started processing for {}'.format(args.data_root))
    filelist = glob(os.path.join(args.data_root, '*/*.mp4'))

    for vfile in tqdm(filelist):
        try:
            process_video_file(vfile, args)
            process_audio_file(vfile, args)
        except Exception as e:
            print(f"Error processing {vfile}: {str(e)}")
            traceback.print_exc()

if __name__ == '__main__':
    main(args)
