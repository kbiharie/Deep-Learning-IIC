import os
import json
import glob

def cocostuff_ids(data, mode="coco"):

    min_len = 192
    things_category = 183
    stuff_ratio = 0.75
    stuff_pixels = {}

    # Keep track of which category ids to keep
    cats_to_keep = set()
    for cat in data["categories"]:
        if cat["supercategory"] == "sky" or cat["supercategory"] == "ground" or cat["supercategory"] == "plant":
            cats_to_keep.add(cat["id"])

    # Calculate total area of images
    for img in data["images"]:
        if img["height"] >= min_len and img["width"] >= min_len:
            stuff_pixels[img["id"]] = {"total": img["height"] * img["width"], "sum": 0, "coco3": 0}

    # Calculate class area of images
    for img in data["annotations"]:
        if img["image_id"] in stuff_pixels and img["category_id"] != things_category:
            stuff_pixels[img["image_id"]]["sum"] += img["area"]
            if img["category_id"] in cats_to_keep:
                stuff_pixels[img["image_id"]]["coco3"] += img["area"]

    # Calculate and filter based on ratio of images
    valid_ids = []
    for id in stuff_pixels:
        if stuff_pixels[id]["sum"] / stuff_pixels[id]["total"] > stuff_ratio and \
                (mode == "coco" or stuff_pixels[id]["coco3"] / stuff_pixels[id]["total"] > 0.75):
            valid_ids.append(id)

    return valid_ids


def cocostuff3_ids(data):
    ids = cocostuff_ids(data, "coco3")
    ids_to_keep = set()
    cats_to_keep = set()

    for cat in data["categories"]:
        if cat["supercategory"] == "sky" or cat["supercategory"] == "ground" or cat["supercategory"] == "plant":
            cats_to_keep.add(cat["id"])

    # Collect image ids to keep
    for img in data["annotations"]:
        if img["category_id"] in cats_to_keep and img["image_id"] in ids:
            ids_to_keep.add(img["image_id"])

    return ids_to_keep


def cocostuff3_dict_ids(ids, ann, dir):
    # Create dictonary with id and filename/path
    files = []
    for img in ann["images"]:
        if img["id"] in ids:
            files.append({"img_id": img["id"], "file": dir + img["file_name"]})
    return files


def cocostuff3_write_filenames():
    # Open trainind and validation annotations
    with open("../annotations/stuff_train2017.json") as f:
         train_annotations = json.load(f)
    with open("../annotations/stuff_val2017.json") as f:
         valid_annotations = json.load(f)

    # Generate valid ids
    print("generating ids")
    train_ids = cocostuff3_ids(train_annotations)
    valid_ids = cocostuff3_ids(valid_annotations)

    # Generate valid file paths
    print("generating file locations")
    train_files = cocostuff3_dict_ids(train_ids, train_annotations, "../datasets/train2017/")
    valid_files = cocostuff3_dict_ids(valid_ids, valid_annotations, "../datasets/val2017/")

    # Write json to file
    print("writing to file")
    print(len(train_files + valid_files))
    with open("../datasets/filenamescocofew.json", "w") as w:
        json.dump(train_files + valid_files, w)


def cocostuff_clean_with_json(groundtruth=False):
    # Open filenames file
    with open("../datasets/filenamescocofew.json") as f:
        filenames = json.load(f)
    count = 0
    paths = []

    # Collect filenames to keep
    for image in filenames:
        if groundtruth:
            paths.append(image["file"]
                         .replace("train2017", "traingt2017")
                         .replace("val2017", "valgt2017")
                         .replace(".jpg", ".png"))
        else:
            paths.append(image["file"])

    # Iterate through all files in directory
    files_to_remove = []
    folders = ["../datasets/traingt2017/", "../datasets/valgt2017/"]
    for folder in folders:
        for path in glob.glob(folder + "*.*"):
            path = path.replace("\\", "/")
            # Mark file for deletion
            if path not in paths:
                files_to_remove.append(path)
            count += 1
            if count % 1000 == 0:
                print(count)

    # Remove files
    print("REMOVING")
    removed = len(files_to_remove)
    for file in files_to_remove:
        removed -= 1
        if removed % 1000 == 0:
            print(removed)
        if os.path.exists(file):
            os.remove(file)
