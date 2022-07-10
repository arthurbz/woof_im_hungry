from pathlib import Path
import os
import shutil 
import cv2


def generate(folder, video_path, steps, step_type, output_name='frame', verbose=1):
    try:
        files_path = check_folder(folder)
    except FileNotFoundError or OSError:
        raise

    if verbose: print('Destiny folder:', files_path)
    generate_by_frame(files_path, video_path, output_name, steps, verbose)


def generate_by_frame(folder, video_path, output_name, steps, verbose):
    video = cv2.VideoCapture(video_path)
    currentframe = 0

    while (True):
        ret, frame = video.read()
        if ret:
            if currentframe % steps == 0:
                name = folder + output_name + str(currentframe) + '.jpg'
                resized_frame = cv2.resize(frame, (int(frame.shape[1] / 4), int(frame.shape[0] / 4)))
                cv2.imwrite(name, resized_frame)
                if verbose: print('Creating...' + name)
            currentframe += 1
        else:
            break
    video.release()
    cv2.destroyAllWindows()


def check_folder(folder):
    try:
        Path(folder).mkdir(parents=True, exist_ok=True)
        return str(Path(folder).absolute()) + '/'
    except FileNotFoundError or OSError:
        raise


def rename_all_files(path, name):
    for count, file in enumerate(os.listdir(path)):
        f_name, f_ext = os.path.splitext(file)
        new_name = path + name + '-' + str(count) + f_ext
        os.rename(path + file, new_name)


def split_class(src, dst, class_name, train, validation):
    listdir = os.listdir(src)

    train_qty = int(len(listdir) * train)
    validation_qty = int(len(listdir) * validation)

    train_files = listdir[0 : train_qty]
    validation_files = listdir[train_qty : train_qty + validation_qty]
    submission_files = listdir[train_qty + validation_qty : ]

    move_files(train_files, src, dst + '/train/' + class_name + '/')
    move_files(validation_files, src, dst + '/validation/' + class_name + '/')
    move_files(submission_files, src, dst + '/submission/' + class_name + '/')

    os.rmdir(src)

    print("\nClass:      ", class_name,
          "\nAll:        ", len(listdir),
          "\nTrain:      ", len(train_files),
          "\nValidation: ", len(validation_files),
          "\nSubmission: ", len(submission_files))


def move_files(files, src, dst):
    check_folder(dst)
    for file in files:
        os.rename(src + file, dst + file)


if __name__ == '__main__':
    shutil.rmtree('dataset')
    #Split video by frames and create images
    #Empty
    generate('dataset/empty', 'videos/empty/empty1.mp4', 5, 'frame', 'empty1-', 1)
    generate('dataset/empty', 'videos/empty/empty2.mp4', 5, 'frame', 'empty2-', 1)
    generate('dataset/empty', 'videos/empty/empty2.mp4', 5, 'frame', 'empty3-', 1)
    #Full
    generate('dataset/full', 'videos/full/full1.mp4', 5, 'frame', 'full1-', 1)
    generate('dataset/full', 'videos/full/full2.mp4', 5, 'frame', 'full2-', 1)
    generate('dataset/full', 'videos/full/full3.mp4', 5, 'frame', 'full3-', 1)

    #Rename all files to just dir name (full or empty) + counter
    rename_all_files('dataset/full/', 'full')
    rename_all_files('dataset/empty/', 'empty')

    #Split all images in train(70%), validation(25%) and submission(5%) 
    split_class('dataset/full/', 'dataset', 'full', 0.7, 0.25) 
    split_class('dataset/empty/', 'dataset', 'empty', 0.7, 0.25) 

