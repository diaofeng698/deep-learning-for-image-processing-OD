import os
import cv2
import re
 
pattens = ['name','xmin','ymin','xmax','ymax']
 
def get_annotations(xml_path):
    bbox = []
    with open(xml_path,'r') as f:
        text = f.read().replace('\n','return')
        p1 = re.compile(r'(?<=<object>)(.*?)(?=</object>)')
        result = p1.findall(text)
        for obj in result:
            tmp = []
            for patten in pattens:
                p = re.compile(r'(?<=<{}>)(.*?)(?=</{}>)'.format(patten,patten))
                if patten == 'name':
                    tmp.append(p.findall(obj)[0])
                else:
                    tmp.append(int(float(p.findall(obj)[0])))
            bbox.append(tmp)
    return bbox
 
def save_viz_image(image_path,xml_path,save_path):
    bbox = get_annotations(xml_path)
    image = cv2.imread(image_path)
    for info in bbox:
        cv2.rectangle(image,(info[1],info[2]),(info[3],info[4]),(255,255,255),thickness=2)
        cv2.putText(image,info[0],(info[1],info[2]),cv2.FONT_HERSHEY_PLAIN,1.2,(255,255,255),2)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    name = os.path.join(save_path,image_path.split('/')[-1])
    cv2.imwrite(name,image)

def show_viz_image(image_path,xml_path):
    bbox = get_annotations(xml_path)
    image = cv2.imread(image_path)
    for info in bbox:
        cv2.rectangle(image,(info[1],info[2]),(info[3],info[4]),(255,255,255),thickness=2)
        cv2.putText(image,info[0],(info[1],info[2]),cv2.FONT_HERSHEY_PLAIN,1.2,(255,255,255),2)
    cv2.imshow("image",image)
    cv2.waitKey(0)

def get_bbox_csv(csv_path):
    bbox = []
    with open(csv_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.split(' ')
            bbox.append([int(float(tmp[0])),int(float(tmp[1])),int(float(tmp[2])),int(float(tmp[3]))])
    return bbox

def show_from_csv_viz_image(image_path,negative_csv_path,positive_csv_path):
    negative_bbox = get_bbox_csv(negative_csv_path)
    positive_bbox = get_bbox_csv(positive_csv_path)

    image = cv2.imread(image_path)
    for info in negative_bbox[:100]:
        cv2.rectangle(image,(info[0],info[1]),(info[2],info[3]),(0,0,255),thickness=1)
    for info in positive_bbox[:100]:
        cv2.rectangle(image,(info[0],info[1]),(info[2],info[3]),(0,255,0),thickness=1)
    cv2.imshow("image",image)
    cv2.waitKey(0)

   
 
if __name__ == '__main__':
    # image_dir = '../../data/voc_car/train/JPEGImages/'
    # xml_dir = '../../data/voc_car/train/Annotations/'
    # save_dir = '../../data/voc_car/train/Viz_images/'
    # image_list = os.listdir(image_dir)
    # # image_list sorted
    # image_list.sort(key=lambda x:int(x.split('.')[0]))

    # # 遍历所有的图片
    # # for i in  image_list:
    # #     image_path = os.path.join(image_dir,i)
    # #     xml_path = os.path.join(xml_dir,i.replace('.jpg','.xml'))
    # #     save_viz_image(image_path,xml_path,save_dir)

    # image_path = os.path.join(image_dir,image_list[0])
    # xml_path = os.path.join(xml_dir,image_list[0].replace('.jpg','.xml'))
    # show_viz_image(image_path,xml_path)
    
    # prase csv
    image_dir = '../../data/classifier_car/train/JPEGImages/'
    csv_dir = '../../data/classifier_car/train/Annotations/'
    image_list = os.listdir(image_dir)
    image_list.sort(key=lambda x:int(x.split('.')[0]))
    image_path = os.path.join(image_dir,image_list[0])
    # positive
    positive_csv_path = os.path.join(csv_dir,image_list[0].replace('.jpg','_1.csv'))
    # negative
    negative_csv_path = os.path.join(csv_dir,image_list[0].replace('.jpg','_0.csv'))
    show_from_csv_viz_image(image_path,negative_csv_path,positive_csv_path)

