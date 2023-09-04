import cv2
import torch
from super_gradients.training  import models
import numpy as np
import math
import time
from tosendmail import *

def image_detection(image,confidence,st):

    device = torch.device("cuda:0") if torch.cuda.is_available() else  torch.cuda("cpu")
    # trained model weithts
    model = models.get('yolo_nas_l',
                            num_classes=6,
                            checkpoint_path="D://PROJECTS/ibm_hackathon/project_industry/newnew_data/checkpoints/yolo_nas/ckpt_best.pth").to(device)
    classnames = ['Fall Detected', 'Fire', 'Person', 'Smoke', 'Vest', 'hard hat']
    result= list(model.predict(image,conf=confidence))[0]
    bbox_xyxyxs = result.prediction.bboxes_xyxy.tolist()
    confidences = result.prediction.confidence
    labels = result.prediction.labels.tolist()
    for (bbox_xyxy,confidence,cls) in zip(bbox_xyxyxs,confidences,labels):
        bbox = np.array(bbox_xyxy)
        x1,y1,x2,y2 = bbox[0],bbox[1],bbox[2],bbox[3]
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        classnamee= int(cls)
        class_name = classnames[classnamee]
        confid= math.ceil((confidence*100))/100 ##Round a number upward to its nearest integer
        label = f'{class_name} {confid}'
        if class_name in ['Fall Detected', 'Fire', 'Smoke']:
            cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),3) # cv2.rectangle takes input for blue,green,red not red,green,blue
            st.text("THIS IS THE DANGER SIGNAL :")

            st.text(class_name)



            message = create_message('HEAD',"ravitejaashwala2003@gmail.com",'!!!!DANGER ALERT!!!!',"!!!!!!!!!****DANGER CLASS DETECTED, CHECK THE AFFECTED AREA *****!!!!!!!!!!")
            print(send_message(service=service, user_id='me', message=message))



        else:
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),3)
        t_size = cv2.getTextSize(label,0,fontScale=1,thickness=3)[0]
        c2 = x1+t_size[0],y1-t_size[1] - 3
        cv2.rectangle(image,(x1,y1),c2,[255,0,0],-1,cv2.LINE_AA)
        cv2.putText(image,label,(x1,y1-2),0,1,[255,255,255],thickness=1,lineType=cv2.LINE_AA)
    st.subheader('Output Image')
    st.image(image,channels = 'RGB',use_column_width=True)





def video_detection(video,kpi1_text,kpi2_text,kpi3_text,stframe,st):

    j = cv2.VideoCapture(video)

    width = int(j.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(j.get(cv2.CAP_PROP_FRAME_HEIGHT))


    frame_width = int(j.get(3))
    frame_height = int(j.get(4))
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.cuda("cpu")
    # trained model weithts
    model = models.get('yolo_nas_l',
                       num_classes=6,
                       checkpoint_path="D://PROJECTS/ibm_hackathon/project_industry/newnew_data/checkpoints/yolo_nas/ckpt_best.pth").to(device)

    count = 0
    prev_time = 0

    classnames = ['Fall Detected', 'Fire', 'Person', 'Smoke', 'Vest', 'hard hat']
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 1, (frame_width, frame_height))
    while True:
        ret, frame = j.read()
        count = +1
        if ret:
            result = list(model.predict(frame, conf=0.35))[0]
            bbox_xyxyxs = result.prediction.bboxes_xyxy.tolist()
            confidences = result.prediction.confidence
            labels = result.prediction.labels.tolist()
            for (bbox_xyxy, confidence, cls) in zip(bbox_xyxyxs, confidences, labels):
                bbox = np.array(bbox_xyxy)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                classnamee = int(cls)
                class_name = classnames[classnamee]
                confid = math.ceil((confidence * 100)) / 100  ##Round a number upward to its nearest integer
                label = f'{class_name}{confid}'
                print("FRAME COUNT ", count, x1, y1, x2, y2)

                danger_zone_detected = 0
                if class_name in ['Fall Detected', 'Fire', 'Smoke']:
                    danger_zone_detected+=1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255),3)  # cv2.rectangle takes input for blue,green,red not red,green,blue
                    if danger_zone_detected == 1:
                        st.text("THIS IS THE DANGER SIGNAL :")
                        st.text(class_name)
                        message = create_message('HEAD',"ravitejaashwala2003@gmail.com", '!!!!DANGER ALERT!!!!'," !!!****DANGER CLASS DETECTED, CHECK THE AFFECTED AREA ***!!!")
                        print(send_message(service=service, user_id='me', message=message))

                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=3)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, [255, 0, 0], -1, cv2.LINE_AA)
                cv2.putText(frame, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
            stframe.image(frame,channels = 'BGR',use_column_width = True)
            present_time = time.time()
            fps = 1/(present_time-prev_time)
            prev_time = present_time
            kpi1_text.write(f"<h1 style = text-align:center;color:red;>{'{:.1f}'.format(fps)}</h1>",unsafe_allow_html = True)
            kpi2_text.write(f"<h1 style = text-align:center;color:red;>{'{:.1f}'.format(width)}</h1>",unsafe_allow_html = True)
            kpi3_text.write(f"<h1 style = text-align:center;color:red;>{'{:.1f}'.format(height)}</h1>",unsafe_allow_html = True)
        else:
            break

