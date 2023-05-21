from ultralytics import YOLO

# load the trained model
model = YOLO('runs/detect/drawing_paper/weights/best.pt')


def run():
    # predict method of YOLOv8
    model.predict(
        source="./screenshot1.jpg",           # source image
        # source="video.mp4"                  # using video as input source
        # source=0,                           # using camera as input source
        conf=0.25,                            # confidence threshold for detection
        save=True,                            # save results
        show=True,                            # show results
        save_crop=True                        # save the cropped image from the results
    )


if __name__ == '__main__':
    run()
