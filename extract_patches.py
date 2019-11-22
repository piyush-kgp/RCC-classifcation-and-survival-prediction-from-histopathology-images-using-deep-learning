
import openslide
import numpy as np
import glob
import os
import argparse
from multiprocessing import Pool, Process
import multiprocessing
import itertools


parser = argparse.ArgumentParser(description='Process args for patch extraction')
parser.add_argument("--root_dir", type=str, required=True, help="Root directory")
parser.add_argument("--dest_dir", type=str, required=True, help="Destination directory")
parser.add_argument("--level", type=int, default=0, help="0 means 40x, 1 means 20x")
parser.add_argument("--patch_size", type=int, default=512, help="Patch Size")
parser.add_argument("--whiteness_limit", type=int, default=210, help="Whiteness Limit")
parser.add_argument("--blackness_limit", type=int, default=5, help="Blackness Limit")
parser.add_argument("--max_faulty_pixels", type=int, default=160000, help="Max allowed # of only B/W pixels")


def valid_slide(slide, level, blackness_limit):
    # tests if a random patch is full black (as a proxy test for full black slides)
    try:
        test_patch = slide.read_region(location=(100, 100), level=level, size=(100, 100)).convert('RGB')
    except:
        return False
    arr = np.array(test_patch)
    if np.all(arr<blackness_limit):
        return False
    return True


# def create_patches(slide_file, level, patch_size, whiteness_limit, blackness_limit, max_faulty_pixels, dest_dir):
def create_patches(args):
    slide_file, level, patch_size, whiteness_limit, blackness_limit, max_faulty_pixels, dest_dir = args
    try:
        slide = openslide.OpenSlide(slide_file)
    except:
        print("Slide {} corrupted".format(slide_file), flush=True)
        return
    if not valid_slide(slide, level, blackness_limit):
        print("Slide {} is probably full black".format(slide_file), flush=True)
        return

    filename = slide_file.split('/')[-1]
    grade = int(filename.split('-')[3][:2])
    slide_id = filename[:-4]
    if grade>=1 and grade<=9:
        label = "cancer"
    elif grade>=10 and grade<=19:
        label = "normal"
    elif grade >=20:
        label = "control"
    if not os.path.exists(os.path.join(dest_dir, label, slide_id)):
        os.makedirs(os.path.join(dest_dir, label, slide_id))

    def reject_or_save_patch(x, y):
    # def reject_or_save_patch(args):
    #     x, y = args
        patch = slide.read_region(location=(x,y), level=level, size=(patch_size, patch_size)).convert('RGB')
        arr = np.array(patch)
        is_white = np.all([arr[:,:,i]>whiteness_limit for i in range(3)], axis=0)
        is_black = np.all([arr[:,:,i]<blackness_limit for i in range(3)], axis=0)
        if np.sum(is_white)+np.sum(is_black)>max_faulty_pixels:
            return
        patch.save(os.path.join(dest_dir, label, slide_id, "{}_X_{}_Y_{}.png".format(filename[:-4], x, y)))

    width, height = slide.dimensions

    # processes = []
    # for x in range(0, width, patch_size//2):
    #     for y in range(0, height, patch_size//2):
    #         # reject_or_save_patch(x, y)
    #         p = Process(target=reject_or_save_patch, args=(x, y))
    #         processes.append(p)
    # for p in processes:
    #     p.start()
    #     p.join()

    # pool = Pool(multiprocessing.cpu_count())
    # paramlist = list(itertools.product(range(0, width, patch_size//2), \
    #             range(0, height, patch_size//2)))
    # pool.map(reject_or_save_patch, paramlist)

    for x in range(0, width, patch_size//2):
        for y in range(0, height, patch_size//2):
            reject_or_save_patch(x, y)
    slide.close()
    print("Slide {} Done.".format(slide_file), flush=True)


def main():
    args = parser.parse_args()
    root_dir = args.root_dir
    dest_dir = args.dest_dir
    level = args.level
    patch_size = args.patch_size
    whiteness_limit = args.whiteness_limit
    blackness_limit = args.blackness_limit
    max_faulty_pixels = args.max_faulty_pixels

    ## TODO: change this if your storage format is different
    slide_files = glob.glob("{}/*/*.svs".format(root_dir))

    for label in ["cancer", "normal", "control"]:
        if not os.path.exists(os.path.join(dest_dir, label)):
            os.makedirs(os.path.join(dest_dir, label))

    # processes = []
    # for slide_file in slide_files:
    #     p = Process(target=create_patches, args=(slide_file, level, patch_size, \
    #                 whiteness_limit, blackness_limit, max_faulty_pixels, dest_dir))
    #     processes.append(p)
    # for p in processes:
    #   s  p.start()
    #     p.join()

    pool = Pool(multiprocessing.cpu_count())
    paramlist = list(itertools.product(slide_files,[level],[patch_size],[whiteness_limit],\
                [blackness_limit], [max_faulty_pixels], [dest_dir]))
    pool.map(create_patches, paramlist)


if __name__=="__main__":
    main()
