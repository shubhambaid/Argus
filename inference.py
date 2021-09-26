import cv2
import torch
import numpy as np
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from src.comp_vis import ArgusCV
import requests


class CalamityClassification:
    def __init__(self):
        self.test_transforms = transforms.Compose([transforms.Resize(256),transforms.ToTensor()])
        self.classes = ['disaster', 'fire', 'flooded_areas', 'normal', 'disaster']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_ft = torch.load("models/efficientnetb4_AdamW_256x256.pt")
        self.model_ft = model_ft.to(self.device)
        self.model_ft.eval()
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/smoke_fire.pt')

    def getCalamity(self,img):
        img = cv2.resize(img,(256,256))
        image = Image.fromarray(img)
        image_tensor = self.test_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(self.device)
        output = self.model_ft(input)
        index = output.data.cpu().numpy().argmax()
        torch.cuda.empty_cache()
        return(self.classes[index])

    def getFireSmoke(self,img):
        img = cv2.resize(img,(1280,720))
        results = self.yolo_model(img)
        results = results.pandas().xyxy[0]
        for result in results.itertuples():
            # print(result)
            cv2.rectangle(img,(int(result[1]),int(result[2])),(int(result[3]),int(result[4])),(0,0,255),2)
        return img

    def hitAPI(self,img):
        return


if __name__=='__main__':
    a = CalamityClassification()
    vid = cv2.VideoCapture('../dataset-hackzurich/video4.mp4')
    while True:
        ret, img = vid.read()
        res = a.getCalamity(img)
        op_image = a.getFireSmoke(img)
        thermal_image = ArgusCV.create_heatmap(op_image,op_image,a1=.5,a2=.5)
        cv2.applyColorMap(op_image, cv2.COLORMAP_JET)
        ArgusCV.draw_text(op_image, res, font_scale=3, pos=(20,50), text_color_bg=(255, 0, 0))
        vis = np.concatenate((op_image, thermal_image), axis=1)
        a.hitAPI(vis)
        cv2.imshow("test",vis)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    vid.release()
