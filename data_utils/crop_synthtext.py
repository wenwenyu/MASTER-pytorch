# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Email: yuwenwen62@gmail.com
# @Created Time: 6/28/2019 4:06 PM

# Multi-process crop synthtext and save it to lmdb or images/text file

from typing import *
import sys
import os
from itertools import chain
import math
import re
import logging
from multiprocessing import Queue, Pool, Process, Manager
from pathlib import Path
import argparse

import cv2
import numpy as np
from loguru import logger
import scipy.io as sio
import lmdb

logger.remove(0)
logger.add('errors.log', level=logging.DEBUG)
logger.add(sys.stdout, level=logging.INFO)

QUEUE_SIZE = 50000
WORKERS = 8
LMDB_WRITE_BATCH = 5000


def crop_box_worker(args):
    '''
    crop synthtext by word bounding box, and put cropped data into queue
    '''
    image_name, txt, boxes, queue = args
    cropped_indx = 0

    # Get image name
    # print('IMAGE : {}'.format(image_name))

    # get transcript
    txt = [re.split(' \n|\n |\n| ', t.strip()) for t in txt]
    txt = list(chain(*txt))
    txt = [t for t in txt if len(t) > 0]

    # Open image
    # img = Image.open(image_name)
    img = cv2.imread(image_name, cv2.IMREAD_COLOR)
    img_height, img_width, _ = img.shape

    # Validation
    if len(np.shape(boxes)) == 2:
        wordBBlen = 1
    else:
        wordBBlen = boxes.shape[-1]

    if wordBBlen == len(txt):
        # Crop image and save
        for word_indx in range(len(txt)):
            if len(np.shape(boxes)) == 2:  # only one word (2,4)
                wordBB = boxes
            else:  # many words (2,4,num_words)
                wordBB = boxes[:, :, word_indx]

            if np.shape(wordBB) != (2, 4):
                err_log = 'malformed box index: {}\t{}\t{}\n'.format(image_name, txt[word_indx], wordBB)
                logger.debug(err_log)
                continue

            pts1 = np.float32([[wordBB[0][0], wordBB[1][0]],
                               [wordBB[0][3], wordBB[1][3]],
                               [wordBB[0][1], wordBB[1][1]],
                               [wordBB[0][2], wordBB[1][2]]])
            height = math.sqrt((wordBB[0][0] - wordBB[0][3]) ** 2 + (wordBB[1][0] - wordBB[1][3]) ** 2)
            width = math.sqrt((wordBB[0][0] - wordBB[0][1]) ** 2 + (wordBB[1][0] - wordBB[1][1]) ** 2)

            # Coord validation check
            if (height * width) <= 0:
                err_log = 'empty file : {}\t{}\t{}\n'.format(image_name, txt[word_indx], wordBB)
                logger.debug(err_log)
                continue
            elif (height * width) > (img_height * img_width):
                err_log = 'too big box : {}\t{}\t{}\n'.format(image_name, txt[word_indx], wordBB)
                logger.debug(err_log)
                continue
            else:
                valid = True
                for i in range(2):
                    for j in range(4):
                        if wordBB[i][j] < 0 or wordBB[i][j] > img.shape[1 - i]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    err_log = 'invalid coord : {}\t{}\t{}\t{}\t{}\n'.format(
                        image_name, txt[word_indx], wordBB, (width, height), (img_width, img_height))
                    logger.debug(err_log)
                    continue

            pts2 = np.float32([[0, 0],
                               [0, height],
                               [width, 0],
                               [width, height]])

            M = cv2.getPerspectiveTransform(pts1, pts2)
            img_cropped = cv2.warpPerspective(img, M, (int(width), int(height)))

            cropped_dir_name = image_name.split('/')[-2]
            cropped_file_name = "{}_{}_{}.jpg".format(cropped_indx,
                                                      image_name.split('/')[-1][:-len('.jpg')], word_indx)
            cropped_indx += 1
            data = dict(cropped_dir_name=cropped_dir_name,
                        filename=cropped_file_name,
                        transcript=txt[word_indx],
                        image=img_cropped)
            queue.put(data)

    else:
        err_log = 'word_box_mismatch : {}\t{}\t{}\n'.format(image_name,
                                                            txt,
                                                            boxes)
        logger.write(err_log)


def writeCache(env, cache: dict):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)


def lmdb_writer(lmdb_path: str, queue: Queue):
    env = lmdb.open(lmdb_path, map_size=1099511627776)
    buffer = {}
    counter = 0
    while True:
        data = queue.get()
        if data != 'Done':
            counter += 1
            img_cropped = data['image']
            img_cropped = cv2.imencode('.jpg', img_cropped)[1]
            buffer['image-{}'.format(counter)] = img_cropped.tobytes()
            buffer['transcript-{}'.format(counter)] = data['transcript'].encode()

            if counter % LMDB_WRITE_BATCH == 0 and counter != 0:
                writeCache(env, buffer)
                logger.info('{} done.'.format(counter))
                buffer = {}
        else:
            buffer['nSamples'] = str(counter).encode()
            writeCache(env, buffer)
            logger.info('Finished. Total {}'.format(counter))
            break


def images_with_gt_file_writer(images_path: str, gt_file: str, queue: Queue):
    gtfile = os.path.join(images_path, gt_file)
    counter = 0
    while True:
        data = queue.get()
        if data != 'Done':
            cropped_dir_name = data['cropped_dir_name']
            filename = data['filename']
            transcript = data['transcript']
            img_cropped = data['image']
            cropped_dir = os.path.join(images_path, cropped_dir_name)
            if not os.path.exists(cropped_dir):
                os.mkdir(cropped_dir)
            cropped_file_name = os.path.join(cropped_dir, filename)
            cv2.imwrite(cropped_file_name, img_cropped)
            with open(gtfile, 'a+', encoding='utf-8', ) as gt_f:
                gt_f.write('%s,%s\n' % (os.path.join(cropped_dir_name, filename), transcript))

            counter += 1

            if counter % LMDB_WRITE_BATCH == 0 and counter != 0:
                logger.info('{} done.'.format(counter))
        else:
            logger.info('Finished. Total {}'.format(counter))
            break


def lmdb_and_images_with_gt_file_writer(lmdb_path: str, images_path: str, gt_file: str, queue: Queue):
    env = lmdb.open(lmdb_path, map_size=1099511627776)
    gtfile = os.path.join(images_path, gt_file)

    buffer = {}
    counter = 0
    while True:
        data = queue.get()
        if data != 'Done':
            counter += 1
            img_cropped = data['image']
            transcript = data['transcript']
            img_cropped_buf = cv2.imencode('.jpg', img_cropped)[1]
            buffer['image-{}'.format(counter)] = img_cropped_buf.tobytes()
            buffer['transcript-{}'.format(counter)] = transcript.encode()

            # write to images and gt file
            cropped_dir_name = data['cropped_dir_name']
            filename = data['filename']
            cropped_dir = os.path.join(images_path, cropped_dir_name)
            if not os.path.exists(cropped_dir):
                os.mkdir(cropped_dir)
            cropped_file_name = os.path.join(cropped_dir, filename)
            cv2.imwrite(cropped_file_name, img_cropped)
            with open(gtfile, 'a+', encoding='utf-8', ) as gt_f:
                gt_f.write('%s,%s\n' % (os.path.join(cropped_dir_name, filename), transcript))

            # write to lmdb
            if counter % LMDB_WRITE_BATCH == 0 and counter != 0:
                writeCache(env, buffer)
                logger.info('{} done.'.format(counter))
                buffer = {}
        else:
            buffer['nSamples'] = str(counter).encode()
            writeCache(env, buffer)
            logger.info('Finished. Total {}'.format(counter))
            break


def synthtext_reader(synthtext_folder: str, queue: Queue, pool: Pool):
    synthtext_folder = Path(synthtext_folder)
    logger.info('Loading gt.mat ...')
    mat_contents = sio.loadmat(synthtext_folder.joinpath('gt.mat'))
    logger.info('Loading finish.')

    image_names = mat_contents['imnames'][0]
    # crop synthtext for every image, and put it into queue
    pool.map(crop_box_worker, iter([(synthtext_folder.joinpath(item[0]).absolute().as_posix(),
                                     mat_contents['txt'][0][index],
                                     mat_contents['wordBB'][0][index],
                                     queue)
                                    for index, item in enumerate(image_names[:])]))

    # for index, item in enumerate(image_names):
    #     crop_box_worker((synthtext_folder.joinpath('imgs/{}'.format(item[0])).absolute(),
    #                           mat_contents['txt'][0][index],
    #                           mat_contents['wordBB'][0][index],
    #                           queue))


def main(args):
    if not Path(args.synthtext_folder).exists():
        logger.error('synthtext_folder does not exist!')
        raise FileNotFoundError

    manager = Manager()
    queue = manager.Queue(maxsize=QUEUE_SIZE)

    # config data writer parallel process, read cropped data from queue, then save it to lmdb or images/txt file
    if args.data_format == 'lmdb':
        writer_process = Process(target=lmdb_writer, name='lmdb writer', args=(args.lmdb_path, queue), daemon=True)
    elif args.data_format == 'images_with_gt_file':
        Path(args.images_folder).mkdir(parents=True, exist_ok=True)
        writer_process = Process(target=images_with_gt_file_writer, name='images_with_gt_file writer',
                                 args=(args.images_folder, args.gt_file, queue), daemon=True)
    else:
        Path(args.images_folder).mkdir(parents=True, exist_ok=True)
        writer_process = Process(target=lmdb_and_images_with_gt_file_writer,
                                 name='lmdb_and_images_with_gt_file_writer writer',
                                 args=(args.lmdb_path, args.images_folder, args.gt_file, queue), daemon=True)
    writer_process.start()

    logger.info('{} writer is started with PID: {}'.format(args.data_format, writer_process.pid))

    # config synthtext data reader jobs
    pool = Pool(processes=WORKERS, maxtasksperchild=10000)
    try:
        logger.info('Start cropping...')
        # crop synthtext, and put cropped data into queue
        synthtext_reader(args.synthtext_folder, queue, pool)
        queue.put('Done')
        pool.close()
        pool.join()

        writer_process.join()
        writer_process.close()
        logger.info('End cropping.')
    except KeyboardInterrupt:
        logger.info('Terminated by Ctrl+C.')
        pool.terminate()
        pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-process crop synthtext and save it to lmdb or images/text file')
    parser.add_argument('--synthtext_folder', default=None, type=str, required=True,
                        help='synthtext root folder including gt.mat file, (default: None)')
    parser.add_argument('--data_format', choices=['lmdb', 'images_with_gt_file', 'both'], default='images_with_gt_file',
                        type=str, required=True, help='output data format (default: images_with_gt_file)')
    parser.add_argument('--lmdb_path', default=None, type=str,
                        help='output lmdb path, if data_format is lmdb, this arg must be set.  (default: None)')
    parser.add_argument('--images_folder', default=None, type=str,
                        help='output cropped images root folder, '
                             'if data_format is not lmdb, this arg must be set. (default: None)')
    parser.add_argument('--gt_file', default='gt.txt', type=str,
                        help='output gt txt file, output at images_folder/gt_file, '
                             'if data_format is not lmdb, this arg must be set. (default: gt.txt)')
    args = parser.parse_args()
    main(args)
