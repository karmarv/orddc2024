import random
import csv
import os, json
import argparse
import funcy
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
import numpy as np
from shutil import copyfile

# Configuration and path
random.seed(0) 

def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def filter_images(images, annotations):
    annotation_ids = funcy.lmap(lambda i: int(i['image_id']), annotations)
    return funcy.lfilter(lambda a: int(a['id']) in annotation_ids, images)

"""
Write list to file or append if it exists
"""
def write_list_file(filename, rows, delimiter=','):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a+") as my_csv:
        csvw = csv.writer(my_csv,delimiter=delimiter, quoting=csv.QUOTE_NONNUMERIC)
        csvw.writerows(rows)

def save_images(images, dst_folder, src_folder):
    count=1
    img_names = []
    for img in tqdm(images):
        src = os.path.join(src_folder, img["file_name"])
        dst = os.path.join(dst_folder, img["file_name"])
        img_names.append([img["file_name"]])
        copyfile(src, dst)
        count+=1
    # Save the files that are copied
    write_list_file(os.path.join(dst_folder, "..", "images.txt"), img_names)
    return count

parser = argparse.ArgumentParser(description='Splits COCO annotations file into training and test sets.')
parser.add_argument('annotations', metavar='coco_annotations', type=str,
                    help='Path to COCO annotations file.')
parser.add_argument('train', type=str, help='Where to store COCO training annotations')
parser.add_argument('test', type=str, help='Where to store COCO test annotations')
parser.add_argument('-s', dest='split', type=float, required=True,
                    help="A percentage of a split; a number in (0, 1)")
parser.add_argument('--having-annotations', dest='having_annotations', action='store_true',
                    help='Ignore all images without annotations. Keep only these with at least one annotation')

parser.add_argument('--multi-class', dest='multi_class', action='store_true',
                    help='Split a multi-class dataset while preserving class distributions in train and test sets')

args = parser.parse_args()



def main(args):

    with open(args.annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco['info']
        licenses = coco['licenses']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        number_of_images = len(images)

        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

        if args.having_annotations:
            images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)


        if args.multi_class:

            annotation_categories = funcy.lmap(lambda a: int(a['category_id']), annotations)

            #bottle neck 1
            #remove classes that has only one sample, because it can't be split into the training and testing sets
            annotation_categories =  funcy.lremove(lambda i: annotation_categories.count(i) <=1  , annotation_categories)

            annotations =  funcy.lremove(lambda i: i['category_id'] not in annotation_categories  , annotations)
            X_train, y_train, X_test, y_test = iterative_train_test_split(np.array([annotations]).T,np.array([ annotation_categories]).T, test_size = 1-args.split)

            # Save COCO annotations            
            train_images = filter_images(images, X_train.reshape(-1))
            save_coco(args.train, info, licenses, train_images, X_train.reshape(-1).tolist(), categories)
            test_images = filter_images(images, X_test.reshape(-1))
            save_coco(args.test, info, licenses, test_images, X_test.reshape(-1).tolist(), categories)
            print("Saved {} entries in {} and {} in {}".format(len(X_train), args.train, len(X_test), args.test))

            # Copy images as per split
            train_count = save_images(train_images, "./rdd2022/coco/train/images/", "./rdd2022/images/")
            test_count = save_images(test_images, "./rdd2022/coco/val/images/", "./rdd2022/images/")
            print("Copied {} images in {} and {} in {}".format(train_count, args.train, test_count, args.test))
            
        else:

            X_train, X_test = train_test_split(images, train_size=args.split)

            anns_train = filter_annotations(annotations, X_train)
            anns_test=filter_annotations(annotations, X_test)

            save_coco(args.train, info, licenses, X_train, anns_train, categories)
            save_coco(args.test, info, licenses, X_test, anns_test, categories)

            print("Saved {} entries in {} and {} in {}".format(len(anns_train), args.train, len(anns_test), args.test))
            
"""
Usage: 
python cocosplit.py --multi-class -s 0.8 ./rdd2022/coco/annotations/rdd2022_annotations.json ./rdd2022/coco/annotations/train.json ./rdd2022/coco/annotations/val.json
"""

if __name__ == "__main__":
    main(args)