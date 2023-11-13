import numpy as np

def shuffle(X):
    m = X.shape[0]
    permutation = list(np.random.permutation(m))
    X_shuffle = X[permutation]
    return X_shuffle


# class myPairedDataStop(object):
#     def __init__(self, dataset_A, label_A, dataset_B, label_B, batch_size, drop_last=False):
#         num_A = dataset_A.shape[0]
#         num_B = data
#
#
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):


class myPairedData(object):
    def __init__(self,dataset_A, label_A, dataset_B, label_B, batch_size, continue_new=True):
        self.num_A = dataset_A.shape[0]
        self.num_B = dataset_B.shape[0]
        self.dataset_A = dataset_A
        self.dataset_B = dataset_B
        self.label_A = label_A
        self.label_B = label_B
        self.dim = dataset_B.shape[1]
        self.start_A = 0
        self.end_A = batch_size
        self.start_B = 0
        self.end_B = batch_size
        self.A_reach_end = False
        self.B_reach_end = False
        self.permutation_A = list(np.random.permutation(self.num_A))
        self.permutation_B = list(np.random.permutation(self.num_B))
        self.batch_size = batch_size
        self.continue_new = continue_new
        self.end_point_A = -1
        self.end_point_B = -1

    def __iter__(self):
        return self

    def __next__(self):
        if self.end_A >= self.num_A:
            self.A_reach_end = True
            batch_perm_A = self.permutation_A[self.start_A:]
            self.permutation_A = list(np.random.permutation(self.num_A))

            if self.continue_new:
                cha = self.end_A - self.num_A
                perm_new = self.permutation_A[0:cha]
                batch_perm_A.extend(perm_new)
                self.start_A = cha
                self.end_A = cha + self.batch_size
                self.end_point_A = self.batch_size - cha
            else:
                self.start_A = 0
                self.end_A = self.batch_size
            batch_data_A = self.dataset_A[batch_perm_A]
            batch_label_A = self.label_A[batch_perm_A]

        else:
            self.A_reach_end = False
            self.end_point_A = -1
            batch_perm_A = self.permutation_A[self.start_A:self.end_A]
            batch_data_A = self.dataset_A[batch_perm_A]
            batch_label_A = self.label_A[batch_perm_A]
            self.start_A = self.end_A
            self.end_A += self.batch_size

        if self.end_B >= self.num_B:
            self.B_reach_end = True
            batch_perm_B = self.permutation_B[self.start_B:]
            self.permutation_B = list(np.random.permutation(self.num_B))

            if self.continue_new:
                cha = self.end_B - self.num_B
                perm_new = self.permutation_B[0:cha]
                batch_perm_B.extend(perm_new)
                self.start_B = cha
                self.end_B = cha + self.batch_size
                self.end_point_B = self.batch_size - cha
            else:
                self.start_B = 0
                self.end_B = self.batch_size
            batch_data_B = self.dataset_B[batch_perm_B]
            batch_label_B = self.label_B[batch_perm_B]

        else:
            self.B_reach_end = False
            self.end_point_B = -1
            batch_perm_B = self.permutation_B[self.start_B:self.end_B]
            batch_data_B = self.dataset_B[batch_perm_B]
            batch_label_B = self.label_B[batch_perm_B]
            self.start_B = self.end_B
            self.end_B += self.batch_size


        return {'S':batch_data_A,'S_label':batch_label_A,
                'T':batch_data_B,'T_label':batch_label_B}

class myPairedData_back(object):
    def __init__(self,dataset_A, label_A, dataset_B, label_B, batch_size, continue_new=True):
        self.num_A = dataset_A.shape[0]
        self.num_B = dataset_B.shape[0]
        self.dataset_A = dataset_A
        self.dataset_B = dataset_B
        self.label_A = label_A
        self.label_B = label_B
        self.dim = dataset_B.shape[1]
        self.start_A = 0
        self.end_A = batch_size
        self.start_B = 0
        self.end_B = batch_size
        self.A_reach_end = False
        self.B_reach_end = False
        self.permutation_A = list(np.random.permutation(self.num_A))
        self.permutation_B = list(np.random.permutation(self.num_B))
        self.batch_size = batch_size
        self.continue_new = continue_new

    def __iter__(self):
        return self
    def __next__(self):
        if self.continue_new:
            if self.end_A >= self.num_A:
                self.A_reach_end = True
                batch_perm_A = self.permutation_A[self.start_A:]
                self.permutation_A = list(np.random.permutation(self.num_A))
                cha = self.end_A - self.num_A
                perm_new = self.permutation_A[0:cha]
                batch_perm_A.extend(perm_new)
                batch_data_A = self.dataset_A[batch_perm_A]
                batch_label_A = self.label_A[batch_perm_A]
                self.start_A = cha
                self.end_A = cha + self.batch_size
            else:
                self.A_reach_end = False
                batch_perm_A = self.permutation_A[self.start_A:self.end_A]
                batch_data_A = self.dataset_A[batch_perm_A]
                batch_label_A = self.label_A[batch_perm_A]
                self.start_A = self.end_A
                self.end_A += self.batch_size

            if self.end_B >= self.num_B:
                self.B_reach_end = True
                batch_perm_B = self.permutation_B[self.start_B:]
                self.permutation_B = list(np.random.permutation(self.num_B))
                cha = self.end_B - self.num_B
                perm_new = self.permutation_B[0:cha]
                batch_perm_B.extend(perm_new)
                batch_data_B = self.dataset_B[batch_perm_B]
                batch_label_B = self.label_B[batch_perm_B]
                self.start_B = cha
                self.end_B = cha + self.batch_size
            else:
                self.B_reach_end = False
                batch_perm_B = self.permutation_B[self.start_B:self.end_B]
                batch_data_B = self.dataset_B[batch_perm_B]
                batch_label_B = self.label_B[batch_perm_B]
                self.start_B = self.end_B
                self.end_B += self.batch_size
        else:
            batch_perm_A = self.permutation_A[self.start_A:self.end_A]
            batch_data_A = self.dataset_A[batch_perm_A]
            batch_label_A = self.label_A[batch_perm_A]
            self.start_A = self.end_A
            self.end_A += self.batch_size

            batch_perm_B = self.permutation_B[self.start_B:self.end_B]
            batch_data_B = self.dataset_B[batch_perm_B]
            batch_label_B = self.label_B[batch_perm_B]
            self.start_B = self.end_B
            self.end_B += self.batch_size

            if self.end_A > self.num_A:
                self.permutation_A = list(np.random.permutation(self.num_A))
                self.permutation_B = list(np.random.permutation(self.num_B))
                self.start_A = 0
                self.start_B = 0
                self.end_A = self.batch_size
                self.end_B = self.batch_size

        return {'S':batch_data_A,'S_label':batch_label_A,
                'T':batch_data_B,'T_label':batch_label_B}


# class PairedData(object):
#     def __init__(self, data_loader_A, data_loader_B, max_dataset_size, flip):
#         self.data_loader_A = data_loader_A
#         self.data_loader_B = data_loader_B
#         self.stop_A = False
#         self.stop_B = False
#         self.max_dataset_size = max_dataset_size
#         self.flip = flip
#
#     def __iter__(self):
#         self.stop_A = False
#         self.stop_B = False
#         self.data_loader_A_iter = iter(self.data_loader_A)
#         self.data_loader_B_iter = iter(self.data_loader_B)
#         self.iter = 0
#         return self
#
#     def __next__(self):
#         A_data, A_label, A_paths = None, None, None
#         B_data, B_label, B_paths = None, None, None
#         try:
#             A_data, A_label, A_paths = next(self.data_loader_A_iter)
#         except StopIteration:
#             if A_data is None or A_paths is None or A_label is None:
#                 self.stop_A = True
#                 self.data_loader_A_iter = iter(self.data_loader_A)
#                 A_data, A_label, A_paths = next(self.data_loader_A_iter)
#
#         try:
#             B_data, B_label, B_paths = next(self.data_loader_B_iter)
#         except StopIteration:
#             if B_data is None or B_paths is None or B_label is None:
#                 self.stop_B = True
#                 self.data_loader_B_iter = iter(self.data_loader_B)
#                 B_data, B_label, B_paths = next(self.data_loader_B_iter)
#
#         if (self.stop_A and self.stop_B) or self.iter > self.max_dataset_size:
#             self.stop_A = False
#             self.stop_B = False
#             raise StopIteration()
#         else:
#             self.iter += 1
#             if self.flip and random.random() < 0.5:
#                 idx = [i for i in range(A_data.size(3) - 1, -1, -1)]
#                 idx = torch.LongTensor(idx)
#                 A_data = A_data.index_select(3, idx)
#                 A_data = A_data.index_select(3, idx)
#             return {'S': A_data, 'S_label': A_label, 'S_path': A_paths,
#                     'T': B_data, 'T_label': B_label, 'T_path': B_paths}
