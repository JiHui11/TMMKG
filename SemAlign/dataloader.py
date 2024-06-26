import numpy as np
import pickle
import random


class Dataset(object):
    def __init__(self, video_dir, audio_dir, label_dir, order_dir, batch_size, status):
        self.batch_size = batch_size
        self.status = status
        self.order_dir = order_dir

        with open(video_dir, 'rb') as f:
            self.video_features = pickle.load(f)
        f.close()

        with open(audio_dir, 'rb') as f:
            self.audio_features = pickle.load(f)
        f.close()
        with open(label_dir, 'rb') as f:
            self.labels = pickle.load(f)
        f.close()

        with open(order_dir, 'rb') as f:
            self.lis = pickle.load(f)
        f.close()
        print(">> the length of visual feature", len(self.video_features))
        print(">> the length of audio feature", len(self.audio_features))

        self.video_batch = {}
        self.audio_batch = {}
        self.label_batch = {}
        self.segment_label_batch = {}
        self.segment_avps_gt_batch = {}
        self.segment_bg_batch = {}
        self.list_copy = self.lis.copy().copy()

    def get_segment_wise_relation(self, batch_labels):
        # batch_labels: [bs, t, 37]
        bs = len(batch_labels) 
        for i in range(bs):
            col_sum = np.sum(batch_labels[i].T, axis=1)
            category_bg_cols = col_sum.nonzero()[0].tolist()  
            category_bg_cols.sort()  
            if category_bg_cols == [36]:
                self.segment_avps_gt_batch[i] = np.zeros(batch_labels[i].shape[0])
                self.segment_avps_gt_batch[i][:] = 1e-7
            else:
                category_col_idx = [col for col in category_bg_cols if col != 36]
                all_same_category_row_idx = []
                for idx in category_col_idx:
                    category_col = batch_labels[i][:, idx]  
                    same_category_row_idx = category_col.nonzero()[0].tolist()  
                    new_row_idx = [idx for idx in same_category_row_idx if idx not in all_same_category_row_idx]
                    all_same_category_row_idx.extend(new_row_idx)

                if len(all_same_category_row_idx) != 0:
                    self.segment_avps_gt_batch[i] = np.zeros(batch_labels[i].shape[0])
                    self.segment_avps_gt_batch[i][all_same_category_row_idx] = 1 / (len(all_same_category_row_idx))
        for i in range(bs):
            col_idx = np.zeros(batch_labels[i].shape[0])
            bg_idx = np.zeros(batch_labels[i].shape[0])
            for row_idx, row in enumerate(batch_labels[i]):
                col_idx1 = np.where(row == 1)[0]
                if 36 not in col_idx1:
                    bg_idx[row_idx] = 1
                if len(col_idx1) > 0: 
                    col_idx[row_idx] = col_idx1[0]
            self.segment_label_batch[i] = col_idx
            self.segment_bg_batch[i] = bg_idx

    def __len__(self):
        return len(self.lis)

    def get_batch(self, idx, shuffle_samples=False):  
        if shuffle_samples:
            random.shuffle(self.list_copy)
        self.video_batch.clear() 
        self.audio_batch.clear()  

        select_ids = self.list_copy[idx * self.batch_size: (idx + 1) * self.batch_size]  

        for i in range(self.batch_size):
            id = select_ids[i]
            v_id = id + '.mp4'
            a_id = id + '.wav'
            self.video_batch[v_id] = self.video_features[v_id]  
            self.audio_batch[a_id] = self.audio_features[a_id]  
            self.label_batch[i] = self.labels[v_id]  

        self.get_segment_wise_relation(self.label_batch)

        return self.audio_batch, \
               self.video_batch, \
               self.label_batch, \
               self.segment_label_batch, \
               self.segment_avps_gt_batch,\
               self.segment_bg_batch,
