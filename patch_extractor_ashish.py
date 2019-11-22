from PIL import Image, ImageChops, ImageStat
import openslide
from os.path import join, exists, isdir
from os import listdir, makedirs
import pickle
from shutil import move
from time import sleep
import argparse
import concurrent.futures
import multiprocessing

core_count = multiprocessing.cpu_count()

parser = argparse.ArgumentParser(description='Extracts patches of from the highest magnification(level 0) of the svs images in the  data_directory/slides/ containing .svs files only')
parser.add_argument('data_path', metavar='data-path', help='Path to data directory where slide images are present', type=str)
parser.add_argument('-p', '--patch-size', help='Default value is 512. Other choices are 256 and 1024. ', nargs='?', type=int, const=512, default=512, choices=[256, 512, 1024])
args = parser.parse_args()

# basic settings
main_dir = args.data_path

if not isdir(join(main_dir,'slides')):
    raise Exception('Given path to folder does not exist')

slides_dir = join(main_dir, 'slides')

if not all([x.endswith('.svs') for x in listdir(slides_dir)]):
    raise Exception('Slides folder contains non-svs files, please remove them and try again')

patches_dir = join(main_dir, 'patches')

train_dir = join(patches_dir, 'train')
val_dir = join(patches_dir, 'valid')
test_dir = join(patches_dir, 'testi')

for data_dir in [train_dir, val_dir, test_dir]:
    if not exists(join(data_dir, 'cancer')):
        makedirs(join(data_dir, 'cancer'))
    if not exists(join(data_dir, 'normal')):
        makedirs(join(data_dir, 'normal'))

progress_file = join(main_dir, 'progress.pickle')


patch_size = args.patch_size
split = False
total_files = len(listdir(slides_dir))


cancer_slides = []
normal_slides = []

file_count = 0

for file in listdir(slides_dir):

    print('segregating file', file_count, end='\r')
    sleep(0.03)
    file_count += 1

    # open svs slide image
    try:
        osr = openslide.OpenSlide(join(slides_dir, file))
    except:
        print(file)
        continue

    if osr.level_count < 2:
        continue

    blank = False

    # check if it is blank
    for level in range(osr.level_count):
        patch = osr.read_region(location=(100, 100), level=level, size=(100, 100)).convert('RGB')

        blank = patch.getbbox() is None

        if blank:
            break

    if blank:
        print('svs', file, 'has a blank layer and is thus skipped')
        continue

    if int(file.split('-')[3][:2]) <= 10:
        cancer_slides.append([file, 'cancer', len(cancer_slides) + 1])
    else:
        normal_slides.append([file, 'normal', len(normal_slides) + 1])

print(len(cancer_slides), 'cancerous,', len(normal_slides), 'normal')

slides = cancer_slides + normal_slides
total_count = {'cancer' : len(cancer_slides), 'normal': len(normal_slides)}

for slide in slides:
    file, result, number = slide
    total = total_count[result]
    slide.append(total)
    slide.append(train_dir if number <= int(total*0.70)
                 else (val_dir if number <= int(total*0.85)
                       else test_dir)
                )

def extract_patch(slide):
    # extract patches
    # whiteness limit
    whiteness_limit = (patch_size ** 2) / 2

    file, result, number, total, data_dir = slide

    # open svs slide image
    try:
        osr = openslide.OpenSlide(join(slides_dir, file))
    except:
        print(file)
        return 0

    patch = osr.read_region(location=(100, 100), level=0, size=(100, 100)).convert('RGB')

    count = 0

    # slide across slide taking patches
    for x in range(0, osr.dimensions[0]-osr.dimensions[0]%patch_size, patch_size):
        for y in range(0, osr.dimensions[1]-osr.dimensions[1]%patch_size, patch_size):

            patch = osr.read_region(location=(x, y), level=0, size=(patch_size, patch_size)).convert('RGB')

            # alternative
            # get patch image stats
            white = all([w >= whiteness_limit
                         for w in
                         ImageStat.Stat(Image.eval(patch, lambda x: 1 if x >= 210 else 0)).sum]
                       )

            if white:
                continue

            # save into respective folder
            patch.save(join(data_dir, result, file[:file.find('.svs')] + '_{:06d}'.format(count) + '.jpg'))

            count += 1
    return count

with concurrent.futures.ProcessPoolExecutor(max_workers=core_count*2) as executor:
    for slide, patch_count in zip(slides, executor.map(extract_patch, slides)):
        continue

print("\n patch extraction completed.")
