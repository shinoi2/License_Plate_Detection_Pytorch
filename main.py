#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:49:57 2019

@author: xingyu
"""
from .LPRNet.LPRNet_Test import *
from .MTCNN.MTCNN import *
import numpy as np
import argparse
import torch
import time
import cv2
import os


class VehicleLicensePlate:
    path = os.path.dirname(__file__)
    LPRNet_model_path = path + '/LPRNet/weights/Final_LPRNet_model.pth'
    STNet_model_path = path + '/LPRNet/weights/Final_STN_model.pth'
    PNet_model_path = path + '/MTCNN/weights/pnet_Weights'
    ONet_model_path = path + '/MTCNN/weights/onet_Weights'

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.min_lp_size = (50, 15)

        self.lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0).to(self.device)
        self.lprnet.load_state_dict(torch.load(self.LPRNet_model_path, map_location=lambda storage, loc: storage))
        self.lprnet.eval()

        self.stnet = STNet().to(self.device)
        self.stnet.load_state_dict(torch.load(self.STNet_model_path, map_location=lambda storage, loc: storage))
        self.stnet.eval()

        self.pnet = PNet().to(self.device)
        self.pnet.load_state_dict(torch.load(self.PNet_model_path, map_location=lambda storage, loc: storage))
        self.pnet.eval()

        self.onet = ONet().to(self.device)
        self.onet.load_state_dict(torch.load(self.ONet_model_path, map_location=lambda storage, loc: storage))
        self.onet.eval()


    def plate_recognition(self, image: np.ndarray):
        def detector(image: np.ndarray):
            bboxes = detect_pnet(self.pnet, image, self.min_lp_size, self.device)
            bboxes = detect_onet(self.onet, image, bboxes, self.device)
            return bboxes

        def recognition(image: np.ndarray):
            im = cv2.resize(image, (94, 24), interpolation=cv2.INTER_CUBIC)
            im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5) * 0.0078125
            data = torch.from_numpy(im).float().unsqueeze(0).to(self.device)  # torch.Size([1, 3, 24, 94]) 
            transfer = self.stnet(data)
            preds = self.lprnet(transfer)
            preds = preds.cpu().detach().numpy()
            labels, _ = decode(preds, CHARS)
            return labels[0]

        result = []
        rects = detector(image)
        for rect in rects:
            top, left, bottom, right = rect[:4]
            score = rect[-1]
            img_box = image[int(left):int(right)+1, int(top):int(bottom)+1, :]
            license = recognition(img_box)
            result.append({
                "rect": {
                    "top": top,
                    "bottom": bottom,
                    "left": left,
                    "right": right
                },
                "license": license,
                "score": score
            })
        return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MTCNN & LPR Demo')
    parser.add_argument("-image", help='image path', default='test/8.jpg', type=str)
    parser.add_argument("--scale", dest='scale', help="scale the iamge", default=1, type=int)
    parser.add_argument('--mini_lp', dest='mini_lp', help="Minimum face to be detected", default=(50, 15), type=int)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0)
    lprnet.to(device)
    lprnet.load_state_dict(torch.load('LPRNet/weights/Final_LPRNet_model.pth', map_location=lambda storage, loc: storage))
    lprnet.eval()
    
    STN = STNet()
    STN.to(device)
    STN.load_state_dict(torch.load('LPRNet/weights/Final_STN_model.pth', map_location=lambda storage, loc: storage))
    STN.eval()
    
    print("Successful to build LPR network!")
    
    since = time.time()
    image = cv2.imread(args.image)
    image = cv2.resize(image, (0, 0), fx = args.scale, fy = args.scale, interpolation=cv2.INTER_CUBIC)
    bboxes = create_mtcnn_net(image, args.mini_lp, device, p_model_path='MTCNN/weights/pnet_Weights', o_model_path='MTCNN/weights/onet_Weights')
    
    for i in range(bboxes.shape[0]):
         
        bbox = bboxes[i, :4]
        x1, y1, x2, y2 = [int(bbox[j]) for j in range(4)]      
        w = int(x2 - x1 + 1.0)
        h = int(y2 - y1 + 1.0)
        img_box = np.zeros((h, w, 3))
        img_box = image[y1:y2+1, x1:x2+1, :]
        im = cv2.resize(img_box, (94, 24), interpolation=cv2.INTER_CUBIC)
        im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5)*0.0078125
        data = torch.from_numpy(im).float().unsqueeze(0).to(device)  # torch.Size([1, 3, 24, 94]) 
        transfer = STN(data)
        preds = lprnet(transfer)
        preds = preds.cpu().detach().numpy()  # (1, 68, 18)    
        labels, pred_labels = decode(preds, CHARS)
    
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
        image = cv2ImgAddText(image, labels[0], (x1, y1-12), textColor=(255, 255, 0), textSize=15)
    
    print("model inference in {:2.3f} seconds".format(time.time() - since))      
    image = cv2.resize(image, (0, 0), fx = 1/args.scale, fy = 1/args.scale, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()