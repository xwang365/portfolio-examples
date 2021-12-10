# Copyright (c) 2021 Graphcore Ltd.
#
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file has been created by Graphcore Ltd.
# It has been created to run the application on IPU hardware.

import multiprocessing
import random
import numpy as np
import paddle
import threading
from collections import deque

try:
    from torch_xla.utils.tf_record_reader import TfRecordReader
except ImportError:
    raise ImportError("""Torch-xla required for TFRecord dataset.
                      Please install torch 1.7.0 & torch-xla using
                     `pip install torch==1.7.0 torch-xla@https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.7-cp37-cp37m-linux_x86_64.whl`"""
                      )

KEYS = ('masked_lm_ids', 'masked_lm_weights', 'segment_ids', 'input_ids',
        'input_mask', 'next_sentence_labels', 'masked_lm_positions')


def create_data_holder(args):
    if args.device == "ipu":
        bs = args.micro_batch_size
    else:
        bs = args.batch_size
    input_ids = paddle.static.data(
        name="input_ids", shape=[bs, args.seq_len], dtype="int64")
    segment_ids = paddle.static.data(
        name="segment_ids", shape=[bs, args.seq_len], dtype="int64")
    input_mask = paddle.static.data(
        name="input_mask", shape=[bs, 1, 1, args.seq_len], dtype="float32")
    masked_lm_positions = paddle.static.data(
        name="masked_lm_positions",
        shape=[bs * args.max_predictions_per_seq],
        dtype="int32")
    masked_lm_labels = paddle.static.data(
        name="masked_lm_labels",
        shape=[bs * args.max_predictions_per_seq],
        dtype="int64")
    next_sentence_labels = paddle.static.data(
        name="next_sentence_labels", shape=[bs], dtype="int64")
    masked_lm_scale = paddle.static.data(
        name="masked_lm_scale", shape=[bs, 1], dtype="float32")
    position_ids = paddle.static.data(
        name="position_ids", shape=[bs, args.seq_len], dtype="int32")
    return [
        input_ids, segment_ids, input_mask, masked_lm_positions,
        masked_lm_labels, next_sentence_labels, masked_lm_scale, position_ids
    ]


class PretrainingTfRecordDataLoader:
    def __init__(self,
                 input_files,
                 max_seq_length=128,
                 max_mask_tokens=20,
                 batch_size=1,
                 micro_batch_size=1,
                 dtype=np.int32,
                 shuffle=False,
                 pad_position_value=384,
                 prefetch=1,
                 drop_remainder=False,
                 enable_fp16=False,
                 enable_ipu=False,
                 enable_check_data=False,
                 ignore_index=-1):
        self.files = input_files
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.max_seq_length = max_seq_length
        self.max_mask_tokens = max_mask_tokens
        self.dtype = dtype
        self.file_index = 0
        self.data_index = 0
        self.shuffle = shuffle
        self.len = None
        self.pad_position_value = pad_position_value
        self.drop_remainder = drop_remainder
        self.enable_fp16 = enable_fp16
        self.enable_ipu = enable_ipu
        self.enable_check_data = enable_check_data
        self.ignore_index = ignore_index
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        num_samples = pool.map(self.samples_in_file, self.files)
        pool.close()
        pool.join()
        self.total_samples = sum(num_samples)
        self.len = self.total_samples // (self.batch_size)
        self.num_prefetch_batches = prefetch
        self.prefetch_buffer = deque()
        if self.len < 1:
            raise ValueError(f"""Batch size {self.batch_size} larger than
                                number of samples in the TFRecord files {self.total_samples}."""
                             )

        if self.len < self.num_prefetch_batches:
            raise ValueError(
                f"""Not enough samples to prefetch: (length = {self.len},
                            num_to_prefech = {self.num_prefetch_batches}),
                            lower the number of prefetch batches.""")
        self.samples_per_file = {
            f: n
            for (f, n) in zip(self.files, num_samples)
        }
        self.data = None
        self.counter = 0
        self.con = threading.Condition()
        self.thread = threading.Thread(target=self.fill_buffer)
        self.thread_stop = False

    def samples_in_file(self, filename):
        reader = TfRecordReader(
            filename, transforms={k: lambda x: x.numpy()
                                  for k in KEYS})
        count = 0
        while reader.read_example():
            count += 1
        return count

    def __iter__(self):
        self.file_index = 0
        self.data_index = 0
        self.counter = 0
        self.data = None
        if self.shuffle:
            random.shuffle(self.files)
        self.thread.start()
        return self

    def release(self):
        self.thread_stop = True
        self.con.acquire()
        self.con.notify()
        self.con.release()

    def check_data_value(self, data, low, high):
        # Check data range, low and high include
        isnan = np.isnan(data.flatten()).any()
        isfinite = np.isfinite(data.flatten()).all()
        min_val = np.min(data.flatten())
        max_val = np.max(data.flatten())

        assert isnan!=True and isfinite==True, \
            "isnan:[%d], isfinite:[%d]" % (isnan, isfinite)
        assert low <= min_val and max_val <= high, \
            "low-high:[%d, %d], min_val-max_val:[%d, %d]" % (low, high, min_val, max_val)

    def check_data_shape(self, data, shape):
        assert data.shape == shape, data.shape

    def post_process(self, samples):
        batch_size, seq_len = samples['input_ids'].shape
        # input_ids
        input_ids = samples['input_ids']
        # segment_ids
        segment_ids = samples['segment_ids']
        # input_mask
        input_mask = (1 - np.reshape(samples['input_mask'].astype(np.float32),
                                     [batch_size, 1, 1, seq_len])) * -1e3
        # masked_lm_positions
        masked_lm_positions = np.reshape(samples['masked_lm_positions'],
                                         (-1)).astype(np.int32)
        masked_lm_positions_bias = np.array(
            range(masked_lm_positions.shape[0]), dtype=np.int32)
        masked_lm_positions_bias //= self.max_mask_tokens
        masked_lm_positions_bias %= self.micro_batch_size
        masked_lm_positions_bias *= seq_len
        masked_lm_positions = masked_lm_positions + masked_lm_positions_bias
        # masked_lm_labels
        masked_lm_labels = np.where(
            samples['masked_lm_positions'].flatten() == 0, self.ignore_index,
            np.reshape(samples['masked_lm_ids'], (-1)))
        masked_lm_labels = np.reshape(masked_lm_labels, (-1))
        # next_sentence_labels
        next_sentence_labels = np.reshape(samples['next_sentence_labels'], (-1))

        # mask_token_num, + 1.0 for avoiding div 0
        masked_lm_scale = np.sum(samples['masked_lm_positions'] != 0,
                                 axis=-1,
                                 keepdims=True,
                                 dtype=np.float32)
        masked_lm_scale = np.where(masked_lm_scale == 0, 1.0, masked_lm_scale)

        position_ids = np.tile(
            np.arange(seq_len).astype(np.int32), (batch_size, 1))

        if self.enable_check_data:
            self.check_data_value(input_ids, 0, 30521)
            self.check_data_value(segment_ids, 0, 1)
            self.check_data_value(masked_lm_positions, 0,
                                  self.batch_size * self.max_seq_length - 1)
            self.check_data_value(masked_lm_labels, -1, 30521)
            self.check_data_value(next_sentence_labels, 0, 1)

            self.check_data_shape(input_ids,
                                  (self.batch_size, self.max_seq_length))
            self.check_data_shape(segment_ids,
                                  (self.batch_size, self.max_seq_length))
            self.check_data_shape(input_mask,
                                  (self.batch_size, 1, 1, self.max_seq_length))
            self.check_data_shape(masked_lm_positions,
                                  (self.batch_size * self.max_mask_tokens, ))
            self.check_data_shape(masked_lm_labels,
                                  (self.batch_size * self.max_mask_tokens, ))
            self.check_data_shape(next_sentence_labels, (self.batch_size, ))
            self.check_data_shape(masked_lm_scale, (self.batch_size, 1))
            self.check_data_shape(position_ids,
                                  (self.batch_size, self.max_seq_length))

        if self.enable_ipu and self.enable_fp16:
            input_mask = input_mask.astype(np.float16)
            masked_lm_scale = masked_lm_scale.astype(np.float16)

        if self.enable_ipu:
            input_ids = input_ids.astype(np.int32)
            segment_ids = segment_ids.astype(np.int32)
            masked_lm_positions = masked_lm_positions.astype(np.int32)
            masked_lm_labels = masked_lm_labels.astype(np.int32)
            next_sentence_labels = next_sentence_labels.astype(np.int32)

        return [
            input_ids, segment_ids, input_mask, masked_lm_positions,
            masked_lm_labels, next_sentence_labels, masked_lm_scale,
            position_ids
        ]

    def __next__(self):
        if self.drop_remainder:
            if self.counter == self.len:
                raise StopIteration

        # if len(self.prefetch_buffer) == 0 and self.counter >= self.len:
        #     raise StopIteration

        self.con.acquire()
        if len(self.prefetch_buffer) == 0:
            self.con.wait()
        result = self.prefetch_buffer.popleft()
        self.con.notify()
        self.con.release()
        self.counter += 1
        return result

    def fill_buffer(self):
        if self.data is None:
            self.load_data()
        while True:
            if self.thread_stop:
                return
            curr_batch = []
            still_required = self.batch_size
            while still_required > 0:
                data = self.data[self.data_index:self.data_index +
                                 still_required]
                self.data_index += len(data)
                curr_batch += data
                still_required = self.batch_size - len(curr_batch)
                if still_required > 0:
                    if self.file_index < len(self.files):
                        self.load_data()
                    else:
                        # break
                        self.file_index = 0
                        self.load_data()
            if len(curr_batch) == self.batch_size:
                result = {}
                for k in KEYS:
                    result[k] = np.vstack([item[k] for item in curr_batch])
                self.con.acquire()
                if len(self.prefetch_buffer) == 100:
                    self.con.wait()
                self.prefetch_buffer.append(self.post_process(result))
                self.con.notify()
                self.con.release()

    def load_data(self):
        if self.file_index >= len(self.files):
            raise ValueError('No more files to load.')
        self.data = self.load_file(self.files[self.file_index])
        self.file_index += 1
        self.data_index = 0
        if self.shuffle:
            np.random.shuffle(self.data)

    def load_file(self, filename):
        reader = TfRecordReader(
            filename,
            transforms={k: lambda x: x.numpy().astype(np.int64)
                        for k in KEYS})
        data = []
        ex = reader.read_example()
        while ex:
            data.append(ex)
            ex = reader.read_example()
        return data
