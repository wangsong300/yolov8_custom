# import the YOLO model
from ultralytics import YOLO

# load the YOLO model
model = YOLO('yolov8n.pt')


def run():
    # training method for YOLOv8
    model.train(
        data='data_custom.yaml',  # datasets configuration file
        imgsz=640,  # input image size
        epochs=50,  # number of epoch for training
        batch=2,  # number of images per batch
        name='drawing_paper',  # name of the training
        save=True  # save the training results
    )


if __name__ == '__main__':
    run()
