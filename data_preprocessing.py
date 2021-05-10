import cv2
import pickle
import os
import numpy as np
from collections import namedtuple
import random
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import imutils
import imgaug.augmenters as iaa

def one_hot_encode(input, num_classes):
    out = np.zeros([input.shape[0], num_classes])
    input_extra = input
    for i in range(input_extra.shape[0]):
        out[i, input_extra[i]] = 1
    return out

def crop_image(img, px):
    if len(img.shape) == 3:
        return img[px:img.shape[0]-px, px:img.shape[1]-px, :]
    else:
        return img[px:img.shape[0]-px, px:img.shape[1]-px]

if __name__ == "__main__":

    project_dir = "C:\\Users\\Marcel\\FU-Berlin\\object_segmentation\\project_data\\"
    data_dir = "C:\\Users\\Marcel\\FU-Berlin\\object_segmentation\\"

    # (NOTE! this is taken from the official Cityscapes scripts:)
    Label = namedtuple( 'Label' , [

        'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                        # We use them to uniquely name a class

        'id'          , # An integer ID that is associated with this label.
                        # The IDs are used to represent the label in ground truth images
                        # An ID of -1 means that this label does not have an ID and thus
                        # is ignored when creating ground truth images (e.g. license plate).
                        # Do not modify these IDs, since exactly these IDs are expected by the
                        # evaluation server.

        'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                        # ground truth images with train IDs, using the tools provided in the
                        # 'preparation' folder. However, make sure to validate or submit results
                        # to our evaluation server using the regular IDs above!
                        # For trainIds, multiple labels might have the same ID. Then, these labels
                        # are mapped to the same class in the ground truth images. For the inverse
                        # mapping, we use the label that is defined first in the list below.
                        # For example, mapping all void-type classes to the same ID in training,
                        # might make sense for some approaches.
                        # Max value is 255!

        'category'    , # The name of the category that this label belongs to

        'categoryId'  , # The ID of this category. Used to create ground truth images
                        # on category level.

        'hasInstances', # Whether this label distinguishes between single instances or not

        'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                        # during evaluations or not

        'color'       , # The color of this label
        ] )

    # (NOTE! this is taken from the official Cityscapes scripts:)
    labels = [
        #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
        Label(  'unlabeled'            ,  0 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'ego vehicle'          ,  1 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'rectification border' ,  2 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'out of roi'           ,  3 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'static'               ,  4 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'dynamic'              ,  5 ,      19 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
        Label(  'ground'               ,  6 ,      19 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
        Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
        Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
        Label(  'parking'              ,  9 ,      19 , 'flat'            , 1       , False        , True         , (250,170,160) ),
        Label(  'rail track'           , 10 ,      19 , 'flat'            , 1       , False        , True         , (230,150,140) ),
        Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
        Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
        Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
        Label(  'guard rail'           , 14 ,      19 , 'construction'    , 2       , False        , True         , (180,165,180) ),
        Label(  'bridge'               , 15 ,      19 , 'construction'    , 2       , False        , True         , (150,100,100) ),
        Label(  'tunnel'               , 16 ,      19 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
        Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
        Label(  'polegroup'            , 18 ,      19 , 'object'          , 3       , False        , True         , (153,153,153) ),
        Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
        Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
        Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
        Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
        Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
        Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
        Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
        Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
        Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
        Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
        Label(  'caravan'              , 29 ,      19 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
        Label(  'trailer'              , 30 ,      19 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
        Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
        Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
        Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
        Label(  'license plate'        , -1 ,       20 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    ]

    id2trainId = {label.id: label.trainId for label in labels}
    id2trainId_map_func = np.vectorize(id2trainId.get)

    unique_train_labels = np.unique(np.fromiter(id2trainId.values(), dtype=int))

    num_classes = unique_train_labels.shape[0]

    unique_train_labels = unique_train_labels.reshape([num_classes, 1])

    new_img_height = 512
    new_img_width = 1024

    cityscapes_dir = data_dir

    train_imgs_dir = cityscapes_dir + "leftImg8bit_trainvaltest\\leftImg8bit\\train\\"
    train_gt_dir = cityscapes_dir + "gtFine_trainvaltest\\gtFine\\train\\"

    val_imgs_dir = cityscapes_dir + "leftImg8bit_trainvaltest\\leftImg8bit\\val\\"
    val_gt_dir = cityscapes_dir + "gtFine_trainvaltest\\gtFine\\val\\"

    train_dirs = ["jena\\", "zurich\\", "weimar\\", "ulm\\", "tubingen\\", "stuttgart\\",
                  "strasbourg\\", "monchengladbach\\", "krefeld\\", "hanover\\",
                  "hamburg\\", "erfurt\\", "dusseldorf\\", "darmstadt\\", "cologne\\",
                  "bremen\\", "bochum\\", "aachen\\"]
    val_dirs = ["frankfurt\\", "munster\\", "lindau\\"]

    training_data_frame = pd.DataFrame()

    val_data_frame = pd.DataFrame()

    training_image_list = []
    training_label_list = []

    val_image_list = []
    val_label_list = []

    trainId_to_count = {}
    for trainId in unique_train_labels:
        trainId_to_count[str(trainId)] = 0

    mean_colors = np.zeros([3, ])
    std_colors = np.zeros([3, ])

    for city in train_dirs:
        file_list_tmp = os.listdir(train_imgs_dir + city)
        for file_name in file_list_tmp:
            img_id = file_name[:-16]

            img = cv2.imread(train_imgs_dir + city + file_name, -1)

            img_resized = cv2.resize(img, (new_img_width, new_img_height),
                                     interpolation=cv2.INTER_NEAREST)

            img_mean_channels = np.mean(img_resized, axis=0)
            img_mean_channels = np.mean(img_mean_channels, axis=0)

            img_std_channels = np.std(img_resized, axis=0)
            img_std_channels = np.std(img_mean_channels, axis=0)

            mean_colors += img_mean_channels
            std_colors += img_std_channels

            img_resized_path = project_dir + "train\\x\\" + img_id + ".png"
            cv2.imwrite(img_resized_path, img_resized)
            training_image_list.append(img_resized_path)

            print(img_resized_path, 'was created!')

            training_label_path = train_gt_dir + city + img_id + "_gtFine_labelIds.png"

            label = cv2.imread(training_label_path, -1)
            label_small = cv2.resize(label, (new_img_width, new_img_height),
                                     interpolation=cv2.INTER_NEAREST)

            converted_train_id_label = id2trainId_map_func(label_small)


            for trainId in unique_train_labels:
                trainId_mask = np.equal(converted_train_id_label, trainId)
                label_trainId_count = np.sum(trainId_mask)

                trainId_to_count[str(trainId)] += label_trainId_count #+ label_trainId_count_crop #+ label_trainId_count_flip_crop

            img_resized_path = project_dir + "train\\x\\" + img_id + "_label.png"
            cv2.imwrite(img_resized_path, converted_train_id_label)
            training_label_list.append(img_resized_path)

    total_count = np.sum(np.fromiter(trainId_to_count.values(), dtype=int))

    trainprob_dict = {}

    for trainId, count in trainId_to_count.items():
        trainId_prob = float(count) / float(total_count)
        trainprob_dict[trainId] = 1 / np.log(1.02 + trainId_prob)

    training_data_frame['input'] = training_image_list
    training_data_frame['ground_truth'] = training_label_list

    training_data_frame.to_csv(project_dir + 'train\\training_df.csv')


    with open(project_dir + 'train\\train_prob_dict.pickle', 'wb') as handle:
        pickle.dump(trainprob_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    no_of_train_imgs = len(training_image_list)

    mean_colors = mean_colors / float(no_of_train_imgs)
    std_colors = std_colors / float(no_of_train_imgs)

    np.save(project_dir + "train\\mean_colors.npy", mean_colors)
    np.save(project_dir + "train\\std_colors.npy", std_colors)

    for city in val_dirs:
        file_list_tmp = os.listdir(val_imgs_dir + city)
        for file_name in file_list_tmp:
            img_id = file_name[:-16]

            img = cv2.imread(val_imgs_dir + city + file_name, -1)

            img_resized = cv2.resize(img, (new_img_width, new_img_height),
                                     interpolation=cv2.INTER_NEAREST)

            img_resized_path = project_dir + "val\\x\\" + img_id + ".png"
            cv2.imwrite(img_resized_path, img_resized)
            val_image_list.append(img_resized_path)

            print(img_resized_path, 'was created!')

            val_label_path = val_gt_dir + city + img_id + "_gtFine_labelIds.png"

            label = cv2.imread(val_label_path, -1)
            label_small = cv2.resize(label, (new_img_width, new_img_height),
                                     interpolation=cv2.INTER_NEAREST)

            converted_train_id_label = id2trainId_map_func(label_small)

            # label_small_flattened = converted_train_id_label.reshape([new_img_width * new_img_height, 1])
            #
            # label_onehot_small_flattened = one_hot_encode(label_small_flattened, num_classes)
            #
            # label_small_onehot = label_onehot_small_flattened.reshape([new_img_width, new_img_height, num_classes])

            img_resized_path = project_dir + "val\\x\\" + img_id + "_label.png"
            cv2.imwrite(img_resized_path, converted_train_id_label)
            val_label_list.append(img_resized_path)

    val_data_frame['input'] = val_image_list
    val_data_frame['ground_truth'] = val_label_list

    print('Mean Colors:', mean_colors)

    val_data_frame.to_csv(project_dir + 'val\\val_df.csv')
