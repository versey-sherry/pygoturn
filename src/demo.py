import os
import argparse
import time

import torch
import cv2

from test import GOTURN

#navigate to src directory
#python demo.py -w /Users/slin/Desktop/pygoturn/src/pytorch_goturn.pth.tar -d ../data/OTB/Subway -o True
#python demo.py -w /Users/slin/Desktop/pygoturn/src/pytorch_goturn.pth.tar -d ../data/OTB/Basketball_c -o True
#python demo.py -w /Users/slin/Desktop/pygoturn/src/pytorch_goturn.pth.tar -d ../data/OTB/Bolt -o True
#python demo.py -w /Users/slin/Desktop/pygoturn/src/pytorch_goturn.pth.tar -d ../data/OTB/Man -o True
#python demo.py -w /Users/slin/Desktop/pygoturn/src/pytorch_goturn.pth.tar -d ../data/OTB/Trans -o True
#python demo.py -w /Users/slin/Desktop/pygoturn/src/pytorch_goturn.pth.tar -d ../data/OTB/basketball-16 -o True
#python demo.py -w /Users/slin/Desktop/pygoturn/src/pytorch_goturn.pth.tar -d ../data/OTB/cat-20 -o True

args = None
parser = argparse.ArgumentParser(description='GOTURN Testing')
parser.add_argument('-w', '--model-weights',
                    type=str, help='path to pretrained model')
parser.add_argument('-d', '--data-directory',
                    default='../data/OTB/Man', type=str,
                    help='path to video frames')
parser.add_argument('-s', '--save-directory',
                    default='../result',
                    type=str, help='path to save directory')
parser.add_argument('-o', '--writeout', default=False, type=bool,
                    help='write to a file if True')

def axis_aligned_iou(boxA, boxB):
    # make sure that x1,y1,x2,y2 of a box are valid
    assert(boxA[0] <= boxA[2])
    assert(boxA[1] <= boxA[3])
    assert(boxB[0] <= boxB[2])
    assert(boxB[1] <= boxB[3])

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def save(im, bb, gt_bb, idx):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    bb = [int(val) for val in bb]  # GOTURN output
    gt_bb = [int(val) for val in gt_bb]  # groundtruth box
    # plot GOTURN predictions with red rectangle
    im = cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]),
                       (0, 0, 255), 2)
    # plot annotations with white rectangle
    im = cv2.rectangle(im, (gt_bb[0], gt_bb[1]), (gt_bb[2], gt_bb[3]),
                       (255, 255, 255), 2)
    save_path = os.path.join(args.save_directory, str(idx)+'.jpg')
    cv2.imwrite(save_path, im)


def main(args):
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    tester = GOTURN(args.data_directory,
                    args.model_weights,
                    device)
    if os.path.exists(args.save_directory):
        print('Save directory %s already exists' % (args.save_directory))
    else:
        os.makedirs(args.save_directory)
    print('file name is', str(args.data_directory).split('/'))
    file_name=str(args.data_directory).split('/')[-1]
    # save initial frame with bounding box
    #save(tester.img[0][0], tester.prev_rect, tester.prev_rect, 1)
    im = cv2.cvtColor(tester.img[0][0], cv2.COLOR_RGB2BGR)
    bb = [int(val) for val in tester.prev_rect]
    gt_bb = [int(val) for val in tester.prev_rect]
    #using the first frame of gt as the initial frame, draw it with red
    im = cv2.rectangle(im, (gt_bb[0], gt_bb[1]), (gt_bb[2], gt_bb[3]),
                       (0, 0, 255), 3)

    if args.writeout:
        #write out as video
        #initiate the image array
        img_array = []
        img_array.append(im)

    else:
        #write out as image frames
        save_path = os.path.join(args.save_directory, file_name+'1.jpg')
        cv2.imwrite(save_path, im)

    tester.model.eval()

    # loop through sequence images
    for i in range(min(100,tester.len)):
    #for i in range(tester.len):
        start = time.time()
        # get torch input tensor
        sample = tester[i]
        # predict box
        bb = tester.get_rect(sample)
        gt_bb = tester.gt[i]
        tester.prev_rect = bb
        print('Frame processing time is {} second'.format(round(time.time() - start, 4)))

        # save current image with predicted rectangle and gt box
        im = tester.img[i][1]
        #save(im, bb, gt_bb, i+2)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        bb = [int(val) for val in bb]  # GOTURN output
        gt_bb = [int(val) for val in gt_bb]  # groundtruth box
        # plot GOTURN predictions with green box
        im = cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]),
                           (0, 255, 0), 3)
        # plot annotations with red rectangle
        im = cv2.rectangle(im, (gt_bb[0], gt_bb[1]), (gt_bb[2], gt_bb[3]),
                           (0,0,255), 3)
        #put IOU on the video
        text = 'Frame {}: IoU is {}%'.format(i+2, round((axis_aligned_iou(gt_bb, bb) *100),2))
        im = cv2.putText(im, text, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2, cv2.LINE_AA) 


        if args.writeout:
            #write out as a video
            img_array.append(im)
        else:
            #write out as image frames
            save_path = os.path.join(args.save_directory, file_name+str(i+2)+'.jpg')
            cv2.imwrite(save_path, im)

        # print stats
        print('frame: %d, IoU = %f' % (
            i+2, axis_aligned_iou(gt_bb, bb)))

    if len(img_array) >1:
        height, width, _ = img_array[0].shape
        size = (width, height)
        save_path = os.path.join(args.save_directory, 'output'+file_name+'.avi')
        print("Saved to", save_path)
        out = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
