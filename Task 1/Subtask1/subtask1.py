import cv2
import numpy as np

#for docker : 
#docker run -it --rm -v "${PWD}:/mnt" task1subtask1:latest
#if image name is task1subtask1 with latest tag
#and change the output '/mnt/'

def implement_pyramidal_lk(video_path, output_path='output_sparse.mp4'):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(15 * fps))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, old_frame = cap.read()
    if not ret:
        print("Failed to read video")
        cap.release()
        return

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, maxCorners=100,
                                 qualityLevel=0.3, minDistance=7, blockSize=7)
    mask = np.zeros_like(old_frame)

    while cap.get(cv2.CAP_PROP_POS_FRAMES) < int(18 * fps):
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            for new, old in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

            img = cv2.add(frame, mask)
            out.write(img)

            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

    cap.release()
    out.release()
    print(f"Sparse LK optical flow saved to {output_path}")


def implement_dense_flow(video_path, output_path='output_dense.mp4'):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(15 * fps))

    ret, frame1 = cap.read()
    if not ret:
        print("Failed to read video")
        cap.release()
        return

    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    while cap.get(cv2.CAP_PROP_POS_FRAMES) < int(18 * fps):
        ret, frame2 = cap.read()
        if not ret:
            break

        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        out.write(bgr)

        prvs = next_frame

    cap.release()
    out.release()
    print(f"Dense optical flow saved to {output_path}")


if __name__ == "__main__":
    video = 'OPTICAL_FLOW.mp4'
    implement_pyramidal_lk(video)
    implement_dense_flow(video)