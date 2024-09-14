import io
import os
import shutil
import tarfile
import urllib.request
import zipfile
from zipfile import ZipFile, ZipInfo

from tqdm import tqdm

# https://github.com/pytorch/vision/issues/7545#issuecomment-1631441616

PATH_TO_DATASET_ZIP = ''
DEV_KIT_SOURCE = 'https://github.com/pytorch/vision/files/11644847/car_devkit.tgz'

THIS_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(THIS_PATH, '../data')
DESTINATION_ROOT = os.path.join(DATA_ROOT, 'stanford_cars')


def setup() -> None:
    if os.path.exists(DESTINATION_ROOT):
        shutil.rmtree(DESTINATION_ROOT)

    if not os.path.exists(DATA_ROOT):
        os.mkdir(DATA_ROOT)

    os.mkdir(DESTINATION_ROOT)


def unzip_dataset(path_to_dataset: str) -> None:
    os.mkdir(os.path.join(DESTINATION_ROOT, 'cars_train'))
    os.mkdir(os.path.join(DESTINATION_ROOT, 'cars_test'))

    with zipfile.ZipFile(path_to_dataset, 'r') as zip_file:
        source_files = zip_file.filelist
        with tqdm(source_files, total=len(source_files)) as file_list:
            for file in file_list:
                if file.filename.startswith('cars_train'):
                    __unzip_file(zip_file, file, 'cars_train')
                elif file.filename.startswith('cars_test'):
                    __unzip_file(zip_file, file, 'cars_test')


def __unzip_file(zip_file: ZipFile, source_file: ZipInfo, target_dir: str) -> None:
    filename = source_file.filename.split('/')[-1]
    with open(os.path.join(DESTINATION_ROOT, target_dir, filename), 'wb') as target_file:
        target_file.write(zip_file.read(source_file))


def download_devkit() -> None:
    response = urllib.request.urlopen(DEV_KIT_SOURCE)
    with tarfile.open(fileobj=io.BytesIO(response.read()), mode='r') as tar_file:
        tar_file.extractall(os.path.join(DESTINATION_ROOT))


## source: https://www.kaggle.com/code/subhangaupadhaya/pytorch-stanfordcars-classification/input?select=cars_test_annos_withlabels+%281%29.mat
def copy_annotations() -> None:
    shutil.copyfile(src=os.path.join(THIS_PATH, 'cars_test_annos_withlabels.mat'),
                    dst=os.path.join(DESTINATION_ROOT, 'cars_test_annos_withlabels.mat'))


if __name__ == '__main__':
    setup()
    unzip_dataset(PATH_TO_DATASET_ZIP)
    download_devkit()
    copy_annotations()
