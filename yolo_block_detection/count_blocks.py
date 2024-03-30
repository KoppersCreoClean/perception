from ultralytics import YOLO
from ultralytics.solutions import object_counter
import argparse
import cv2

def count_blocks(video_path, model_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Define region points
    region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]

    # Define line points
    line_points = [(20, 400), (1080, 400)]

    # Video writer
    video_writer = cv2.VideoWriter("../../data/object_counting_output.avi",
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps,
                        (w, h))

    # Init Object Counter
    counter = object_counter.ObjectCounter()
    counter.set_args(view_img=True,
                    reg_pts=line_points,
                    view_in_counts=False,
                    classes_names=model.names,
                    draw_tracks=True)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        tracks = model.track(im0, persist=True, show=False)

        im0 = counter.start_counting(im0, tracks)
        video_writer.write(im0)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model_path', type=str, required=False, help='Path to the model', default='./runs/detect/train/weights/best.pt')
    parser.add_argument('--video_path', type=str, required=False, help='Path to the video', default='../../data/output.avi')
    args = parser.parse_args()

    try:
        if args.video_path is None:
            print("Invalid video path")
            exit(1)
    except:
        pass

    count_blocks(args.video_path, args.model_path)