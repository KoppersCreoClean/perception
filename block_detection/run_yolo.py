"""
Script Name: run_yolo.py
Description: Test file for running YOLOv8 trained on drip pan block for detection
Author: Yatharth Ahuja, David Hill, Michael Gromic, Leo Mouta, Louis Plottel
"""

from ultralytics import YOLO
import argparse
import cv2

def train(data_path, model_path, patience=1000, epochs=2000, plots=True, dropout=0.1):
    yolo = YOLO(model_path)
    yolo.train(data=data_path, patience=patience, epochs=epochs, \
               plots=plots, dropout=dropout)
    valid_results = yolo.val()
    print(valid_results)

def predict(model_path, img_path):
    if img_path is None:
        print("Invalid image path")
        return
    yolo = YOLO(model_path)
    results = yolo.predict(source=img_path, conf=0.4, show=True, \
                           save=True, imgsz=640)
    print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--data_path', type=str, required=False, help='Path to the data', default='./data.yaml')
    parser.add_argument('--model_path', type=str, required=False, help='Path to the model', default='./runs/detect/train/weights/best.pt')
    parser.add_argument('--img_path', type=str, required=False, help='Path to the image', default=None)
    parser.add_argument('--action', type=str, required=False, help='Action to perform', choices=['train', 'predict'], default='predict')
    args = parser.parse_args()

    # args.action = "train"
    # args.model_path = "./yolov8n.pt"
    if args.action == "train":
        print("Training")
        train(data_path=args.data_path, model_path=args.model_path)
    elif args.action == "predict":
        print("Predicting")
        predict(model_path=args.model_path, img_path=args.img_path)
