import cv2
import glob
import argparse
import os
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="(required) path to the dir of images")
    parser.add_argument("--vdo-name", type=str, default="output.mp4", help="output video name, default: output.mp4")
    parser.add_argument("--fps", type=int, default=10, help="output fps (default: 10)")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    # read filenames
    filenames = glob.glob(f'{args.path}/*.png')
    
    # create video writer
    border_size = 2
    img = cv2.imread(filenames[0])
    frame_size = (img.shape[1] + border_size * 2, img.shape[0] + border_size * 2)
    path_vdo = os.path.join(args.path, args.vdo_name)
    out = cv2.VideoWriter(path_vdo,
                          cv2.VideoWriter_fourcc(*'mp4v'), 
                          args.fps, 
                          frame_size)

    # write to video 
    for filename in filenames:
        print(f"writing filename: {filename}")
        img = cv2.imread(filename)
        img = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        out.write(img)

    out.release()

    print(f"Done, video location: {path_vdo} (frame_size: {frame_size})")