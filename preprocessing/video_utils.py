import cv2

def get_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if len(frames) < num_frames:
        frames += [frames[-1]] * (num_frames - len(frames))
    return frames[:num_frames]