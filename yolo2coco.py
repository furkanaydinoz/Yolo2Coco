import os
import cv2
import json
import math
import argparse
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from ultralytics.utils import ops


class Yolo2Coco():
    def __init__(self, _modelPath, _imagePath, _jsonSavePath, _classesFilePath, _labelsPath='') -> None:
        self.modelPath = _modelPath
        self.imagePath = _imagePath
        self.jsonSavePath = _jsonSavePath
        self.classesFilePath = _classesFilePath
        self.labelsPath = _labelsPath
        self.cocoTemplate = {"info":{"year":2023,"version":"1.0","description":"","contributor":"","url":"","date_created":""},"categories":[],"images":[],"annotations":[]}

    def readFile(self, path):
        if path.endswith('txt'):
            with open(path, 'r') as f:
                txt = f.readlines()
            f.close()
            return txt
        elif path.endswith('json'):
            jsonFile = open(path)
            return json.load(jsonFile)
        else:
            return 'Wrong file type. Please choose "txt" or "json"'
    
    def writeJson(self, jsonData):
        with open(f'{self.jsonSavePath}label.json', 'w') as j:
            json.dump(jsonData, j, indent=4, separators=(',', ':'))
        j.close()

    def loadModel(self):
        self.model = YOLO(model=self.modelPath)
    
    def addClasses2Json(self):
        labels = self.readFile(path=self.classesFilePath)
        for index, label in enumerate(labels):
            self.cocoTemplate['categories'].append({'id': index,'name': label.split('\n')[0]})
        self.writeJson(self.cocoTemplate)

    def addImages2Json(self):
        for index, imageFile in enumerate(os.listdir(self.imagePath)):
            if imageFile.endswith('jpg') or imageFile.endswith('jpeg') or imageFile.endswith('png'):
                image = cv2.imread(self.imagePath + imageFile)
                self.cocoTemplate['images'].append({
                                "id":index,
                                "license":1,
                                "file_name":imageFile,
                                "height":image.shape[0],
                                "width":image.shape[1],
                                "date_captured":str(datetime.now())
                                })
        self.writeJson(self.cocoTemplate)

    def addAnnotations2JsonFromModel(self):
        maskFlaten = []
        for index, imageFile in enumerate(self.cocoTemplate["images"]):
            results = self.model.predict(source=self.imagePath + imageFile['file_name'])
            for result in results:
                result.boxes.xyxy
                result.boxes.cls
            if result.masks:
                for index, mask in enumerate(result.masks.xy):
                    w = int(result.boxes.xywh[index][2].item())
                    h = int(result.boxes.xywh[index][3].item())
                    x = int(result.boxes.xywh[index][0].item()- w/2)
                    y = int(result.boxes.xywh[index][1].item() - h/2)
                    maskFlaten = [round(float(flatedList),3) for flatedList in mask.flatten()]
                    self.cocoTemplate["annotations"].append({
                            "id":index,
                            "image_id":imageFile['id'],
                            "category_id":int(result.boxes.cls[index].item()),
                            "bbox":[x,y,w,h],
                            "area":w * h,
                            "segmentation":[maskFlaten],
                            "iscrowd":0
                        })
            self.writeJson(self.cocoTemplate)

    def toPixelCoords(self, coordinates, size):
        flag = False
        l = len(coordinates)
        coord = []
        for i in range(l):
            if (not flag):
                #for height
                coordinates[i] = math.ceil(float(coordinates[i])*size[1])

            else:
                #for width
                coordinates[i] = math.ceil(float(coordinates[i])*size[0])
            flag = not flag
        return coordinates
        
    def addAnnotations2JsonFromTXT(self):
        labelFile = ''
        for index, imageFile in enumerate(self.cocoTemplate["images"]):
            if imageFile['file_name'].endswith('jpeg'):
                labelFile = imageFile['file_name'][:-4] + 'txt'
            elif imageFile['file_name'].endswith('jpg') or imageFile['file_name'].endswith('png'):
                labelFile = imageFile['file_name'][:-3] + 'txt'
            if labels := self.readFile(self.labelsPath + labelFile):
                img = cv2.imread(self.imagePath + imageFile['file_name'])
                for label in labels:
                    polygonPoints = label.split(" ")[1:]
                    coord = self.toPixelCoords(polygonPoints, img.shape[:2])
                    coord = np.array(polygonPoints, np.int32).reshape((-1, 1, 2))
                    bbox = cv2.boundingRect(coord)
                    self.cocoTemplate["annotations"].append({
                            "id":index,
                            "image_id":imageFile['id'],
                            "category_id":int(label[0]),
                            "bbox":[bbox[0], bbox[1], bbox[2], bbox[3]],
                            "area":bbox[2] * bbox[3],
                            "segmentation":coord.reshape(1,-1).tolist(),
                            "iscrowd":0
                        })
                self.writeJson(self.cocoTemplate)

def main(args):
    y2c = Yolo2Coco(args.model_path, args.image_path, args.json_save_path, args.classes_file_path, args.labels_path)
    y2c.addClasses2Json()
    y2c.addImages2Json()
    if args.convert_type == 'from-model':
        y2c.loadModel()
        y2c.addAnnotations2JsonFromModel()
    if args.convert_type == 'from-txt':
        y2c.addAnnotations2JsonFromTXT()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                                     prog='YOLOV8 Instance Segmentation and Object Detection Data Converter',
                                     description='This script convert Yolov8 TXT and Yolov8 model outputs to COCO format for auto labeling(LabelMe or Label-Studio).'
                                     )
    parser.add_argument('-mP','--model_path', default='yolov8s-seg.pt', type=str)
    parser.add_argument('-iP', '--image_path', default='data/images/', type=str)
    parser.add_argument('-jSP', '--json_save_path', default='output/', type=str)
    parser.add_argument('-cFP', '--classes_file_path', default='classes.txt', type=str)
    parser.add_argument('-cT', '--convert_type', default='from-model', help='from-model or from-txt')
    parser.add_argument('-lP', '--labels_path', default='data/labels/',type=str)
    args = parser.parse_args()
    main(args=args)
