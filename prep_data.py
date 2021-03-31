import os
import json

def cocostuff_ids(data):

    min_len = 192
    things_category = 183
    stuff_ratio = 0.75
    stuff_pixels = {}

    for img in data["images"]:
        if img["height"] >= min_len and img["width"] >= min_len:
            stuff_pixels[img["id"]] = {"total": img["height"] * img["width"], "sum": 0}

    for img in data["annotations"]:
        if img["image_id"] in stuff_pixels and img["category_id"] != things_category:
            stuff_pixels[img["image_id"]]["sum"] += img["area"]

    valid_ids = []

    for id in stuff_pixels:
        if stuff_pixels[id]["sum"] / stuff_pixels[id]["total"] > stuff_ratio:
            valid_ids.append(id)

    return valid_ids


def cocostuff3_ids(data):
    ids = cocostuff_ids(data)
    ids_to_keep = set()
    cats_to_keep = set()
    for cat in data["categories"]:
        if cat["supercategory"] == "sky" or cat["supercategory"] == "ground" or cat["supercategory"] == "plant" and cat["id"] in ids:
            cats_to_keep.add(cat["id"])

    for img in data["annotations"]:
        if img["category_id"] in cats_to_keep and img["image_id"] in ids:
            ids_to_keep.add(img["image_id"])

    return ids_to_keep


def cocostuff3_dict_ids(ids, ann, dir):
    files = []
    for img in ann["images"]:
        if img["id"] in ids:
            files.append({"img_id": img["id"], "file": dir + img["file_name"]})
    return files

def cocostuff3_write_filenames():
    with open("../annotations/stuff_train2017.json") as f:
         train_annotations = json.load(f)
    with open("../annotations/stuff_val2017.json") as f:
         valid_annotations = json.load(f)
    print("generating ids")
    train_ids = cocostuff3_ids(train_annotations)
    valid_ids = cocostuff3_ids(valid_annotations)
    print("generating file locations")
    train_files = cocostuff3_dict_ids(train_ids, train_annotations, "../datasets/train2017/")
    valid_files = cocostuff3_dict_ids(valid_ids, valid_annotations, "../datasets/val2017/")
    print("writing to file")
    with open("../datasets/filenames.json", "w") as w:
        json.dump(train_files + valid_files, w)

def cocostuff_clean(ids, ann, img_dir):
    files_to_remove = []

    for img in ann["images"]:
        if img["id"] not in ids:
            files_to_remove.append(img["file_name"])

    for file in files_to_remove:
        file_to_remove = img_dir + "/" + file
        if os.path.exists(file_to_remove):
            os.remove(file_to_remove)


def test():
    with open("../annotations/stuff_val2017.json") as f:
        data = json.load(f)

    for x in data:
        print(x)

    image_ids = []

    for img in data["annotations"]:
        image_ids.append(img["image_id"])

    image_ids = list(set(image_ids))
    image_ids.sort()

    categories = {}

    for cat in data["categories"]:
        categories[cat["id"]] = cat["name"]

    things_category = 183

    areas = {}

    for img in data["annotations"]:
        if img["image_id"] not in areas:
            areas[img["image_id"]] = {}
            areas[img["image_id"]]["sum"] = 0
            areas[img["image_id"]]["total"] = 1
        if img["category_id"] != things_category:
            areas[img["image_id"]]["sum"] += img["area"]

    min_len = 1000
    small = 0
    for img in data["images"]:
        if img["id"] not in areas:
            areas[img["id"]] = {}
            areas[img["id"]]["sum"] = 0
        areas[img["id"]]["total"] = img["height"] * img["width"]
        min_len = min(min_len, img["height"])
        min_len = min(min_len, img["width"])
        if img["height"] < 192 or img["width"] < 192:
            small += 1
    print(small)

    print(min_len)

    count = 0
    for id in areas:
        if areas[id]["sum"] / areas[id]["total"] > 0.75:
            count += 1
    print(count)