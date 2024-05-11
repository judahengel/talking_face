# Talking Face Repo

## Prerequisites
- ffmpeg: `sudo apt-get install ffmpeg`
- Install necessary packages using `pip install -r requirements.txt`.
- Face detection pre-trained model should be downloaded to `face_detection/detection/sfd/s3fd.pth`.
- Install AV-Hubert by following the installation: https://github.com/facebookresearch/av_hubert

## Lip-syncing videos using the pre-trained models (Inference)

The result is saved (by default) in `results/result_voice.mp4`.

You can lip-sync any video to any audio:

```bash
python inference.py --checkpoint_path <ckpt> --face <video.mp4> --audio <an-audio-source>
```