from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch.optim as opt
from basenet import *
import torch
import gc
from copy import copy, deepcopy
import os
# os.system('file ./libmr.so')
import libmr
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform
from utils import angular_dist

def get_model(net, num_class=13, feat_size=100, d_hid_size=2048):

    if net == 'vgg1':
        model_g = VGGBase(feat_size=feat_size)
        model_c1 = Classifier(num_classes=num_class,feat_size=feat_size)
        model_d = AdversarialNetwork(in_feature=feat_size, hidden_size=d_hid_size)   

    if net == 'vgg2':
        model_g = VGGFc(vgg_name='VGG19BN',bottleneck_dim=feat_size)
        model_c1 = Classifier(num_classes=num_class,feat_size=feat_size)
        model_d = AdversarialNetwork(in_feature=feat_size, hidden_size=d_hid_size) 

    if net == 'vgg3':
        model_g = VGGFc(vgg_name='VGG19',bottleneck_dim=feat_size)
        model_c1 = Classifier(num_classes=num_class,feat_size=feat_size)
        model_d = AdversarialNetwork(in_feature=feat_size, hidden_size=d_hid_size)  

    if net == 'resnet1':
        model_g = ResBase(option='resnet50', pret=True, feat_size=feat_size)
        model_c1 = ResClassifier(num_classes=num_class, feat_size=feat_size)
        model_d = AdversarialNetwork(in_feature=feat_size, hidden_size=d_hid_size)

    if net == 'resnet2':
        model_g = ResNetFc(resnet_name='ResNet50', bottleneck_dim=feat_size)
        model_c1 = ResClassifier(num_classes=num_class, feat_size=feat_size)
        model_d = AdversarialNetwork(in_feature=feat_size, hidden_size=d_hid_size)

    return model_g, model_c1, model_d


def get_optimizer_visda(args, G, C1, C2, D):
    update_lower=args.update_lower    
    if not update_lower:
        print('NOT update lower!')
        params = list(list(G.linear1.parameters()) + list(G.linear2.parameters()) + list(
            G.bn1.parameters()) + list(G.bn2.parameters())) #+ list(G.bn4.parameters()) + list(
            #G.bn3.parameters()) + list(G.linear3.parameters()) + list(G.linear4.parameters()))
    else:
        print('update lower!')
        params = G.parameters()
    optimizer_g = opt.SGD(params, lr=args.lr_g, momentum=0.9, weight_decay=0.0005,nesterov=True)
    optimizer_c1 = opt.SGD(list(C1.parameters()), momentum=0.9, lr=args.lr_c1,weight_decay=0.0005, nesterov=True)
    optimizer_c2 = opt.SGD(list(C2.parameters()), momentum=0.9, lr=args.lr_c2,weight_decay=0.0005, nesterov=True)

    optimizerD = opt.Adam(D.parameters(), lr=args.lr_d)

    return optimizer_g, optimizer_c1, optimizer_c2, optimizerD


def bce_loss(output, target):
    output_neg = 1 - output
    target_neg = 1 - target
    result = torch.mean(target * torch.log(output + 1e-6))
    result += torch.mean(target_neg * torch.log(output_neg + 1e-6))
    return -torch.mean(result)

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def DiscrepancyLoss(input_1, input_2, m = 2.0):
    soft_1 = nn.functional.softmax(input_1, dim=1)
    soft_2 = nn.functional.softmax(input_2, dim=1)
    entropy_1 = - soft_1 * nn.functional.log_softmax(input_1, dim=1)
    entropy_2 = - soft_2 * nn.functional.log_softmax(input_2, dim=1)
    entropy_1 = torch.sum(entropy_1, dim=1)
    entropy_2 = torch.sum(entropy_2, dim=1)

    loss = torch.nn.ReLU()(m - torch.mean(entropy_1 - entropy_2))
    return loss
    
def EntropyLoss(input_1):
    soft_1 = nn.functional.softmax(input_1, dim=1)
    entropy_1 = - soft_1 * nn.functional.log_softmax(input_1, dim=1)
    entropy_1 = torch.sum(entropy_1, dim=1)
    # loss = torch.nn.ReLU()(m - torch.mean(entropy_1))
    loss = -torch.mean(entropy_1)
    return loss

def calc_entropy(input_1):
    soft_1 = nn.functional.softmax(input_1, dim=1)
    entropy_1 = - soft_1 * nn.functional.log_softmax(input_1, dim=1)
    entropy_1 = torch.sum(entropy_1, dim=1)
    return entropy_1



def save_model(encoder, classifier, centroid_distance, save_path):
    save_dic = {
        'encoder': encoder.state_dict(),
        'classifier': classifier.state_dict(),
        'centroid_distance':centroid_distance
    }
    torch.save(save_dic, save_path)


def load_model(encoder, classifier, load_path):
    checkpoint = torch.load(load_path)
    encoder.load_state_dict(checkpoint['encoder'])
    classifier.load_state_dict(checkpoint['classifier'])
    centroid_distance = checkpoint['centroid_distance']
    return encoder, classifier,centroid_distance


def adjust_learning_rate(optimizer, lr, batch_id, max_id, epoch, max_epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    beta = 0.75
    alpha = 10
    p = min(1, (batch_id + max_id * epoch) / float(max_id * max_epoch))
    lr = lr / (1 + alpha * p) ** (beta)  # min(1, 2 - epoch/float(20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)


def segment_sum(data, segment_ids):
    """
    Analogous to tf.segment_sum (https://www.tensorflow.org/api_docs/python/tf/math/segment_sum).

    :param data: A pytorch tensor of the data for segmented summation.
    :param segment_ids: A 1-D tensor containing the indices for the segmentation.
    :return: a tensor of the same type as data containing the results of the segmented summation.
    """
    if not all(segment_ids[i] <= segment_ids[i + 1] for i in range(len(segment_ids) - 1)):
        raise AssertionError("elements of segment_ids must be sorted")

    if len(segment_ids.shape) != 1:
        raise AssertionError("segment_ids have be a 1-D tensor")

    if data.shape[0] != segment_ids.shape[0]:
        raise AssertionError("segment_ids should be the same size as dimension 0 of input.")

    num_segments = len(torch.unique(segment_ids))
    return unsorted_segment_sum(data, segment_ids, num_segments)


def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """
    assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

    # segment_ids is a 1-D tensor repeat it to have the same shape as data
    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:])).long().cuda()
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

    assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    centroid = torch.zeros(*shape).cuda().scatter_add(0, segment_ids, data.float())
    centroid = centroid.type(data.dtype)
    return centroid

def unsorted_segment_sum_cpu(data, segment_ids, num_segments):
    """
    Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """
    assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

    # segment_ids is a 1-D tensor repeat it to have the same shape as data
    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:])).long()
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

    assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    centroid = torch.zeros(*shape).scatter_add(0, segment_ids, data.float())
    centroid = centroid.type(data.dtype)
    return centroid    

def get_means(tensors_list):
    """
    Calculate the mean of a list of tensors for each tensor in the list. In our case the list typically contains
    a tensor for each class, such as the per class z values.

    Parameters:
        tensors_list (list): List of Tensors

    Returns:
        list: List of Tensors containing mean vectors
    """
    means = []
    for i in range(len(tensors_list)):
        if isinstance(tensors_list[i], torch.Tensor):
            means.append(torch.mean(tensors_list[i], dim=0))
        else:
            means.append([])

    return means

def calc_distances_to_means(means, tensors, distance_function='angular'):
    """
    Function to calculate distances between tensors, in our case the mean zs per class and z for each input.
    Wrapper around torch.nn.functonal distances with specification of which distance function to choose.

    Parameters:
        means (list): List of length corresponding to number of classes containing torch tensors (typically mean zs).
        tensors (list): List of length corresponding to number of classes containing tensors (typically zs).
        distance_function (str): Specification of distance function. Choice of cosine|euclidean|mix.

    Returns:
        list: List of length corresponding to number of classes containing tensors with distance values
    """

    def distance_func(a, b, distance_function):
        if distance_function == 'euclidean':
            d = torch.nn.functional.pairwise_distance(a.view(1, -1), b, p=2)
        elif distance_function == 'cosine':
            d = (1 - torch.nn.functional.cosine_similarity(a.view(1, -1), b))
        elif distance_function == 'angular':
            eps = 1e-6
            d = angular_dist(b,a.unsqueeze(0)).squeeze()            
            # a = F.normalize(a.unsqueeze(0))
            # b = F.normalize(a.unsqueeze(0))
            # d = torch.acos(torch.clamp(torch.matmul(a,b.transpose(0,1)), -1.+eps, 1-eps))            
        return d

    distances = []

    # loop through each class in means and calculate the distances with the respective tensor.
    for i in range(len(means)):
        # check for tensor type, e.g. list could be empty
        if isinstance(tensors[i], torch.Tensor) and isinstance(means[i], torch.Tensor):
            dist_tensor = distance_func(means[i], tensors[i], distance_function)
            if torch.numel(dist_tensor) == 1:
                dist_tensor = dist_tensor.unsqueeze(0)
            distances.append(dist_tensor)
        else:
            distances.append([])

    return distances


def fit_weibull_models(distribution_values, tailsizes, num_max_fits=5):
    """
    Function to fit weibull models on distribution values per class. The distribution values in our case are the
    distances of an inputs approximate posterior value to the per class mean latent z, i.e. The Weibull model fits
    regions of high density and gives credible intervals.
    The tailsize specifies how many outliers are expected in the dataset for which the model has been trained.
    We use libmr https://github.com/Vastlab/libMR (installable through e.g. pip) for the Weibull model fitting.

    Parameters:
        distribution_values (list): Values on which the fit is conducted. In our case latent space distances.
        tailsizes (list): List of integers, specifying tailsizes per class. For a balanced dataset typically the same.
        num_max_fits (int): Number of attempts to fit the Weibull models before timing out and returning unsuccessfully.

    Returns:
        list: List of Weibull models with their respective parameters (stored in libmr class instances).
    """

    weibull_models = []

    # loop through the list containing distance values per class
    for i in range(len(distribution_values)):
        # for each class set the initial success to False and number of attempts to 0
        is_valid = False
        count = 0

        # If the list contains distance values conduct a fit. If it is empty, e.g. because there is not a single
        # prediction for the corresponding class, continue with the next class. Note that the latter isn't expected for
        # a model that has been trained for even just a short while.
        if isinstance(distribution_values[i], torch.Tensor):
            distribution_values[i] = distribution_values[i].cpu().numpy().astype(np.double)
            # weibull model per class
            weibull_models.append(libmr.MR(verbose=False,alpha=10.0))
            # attempt num_max_fits many fits before aborting
            while is_valid is False and count < num_max_fits:
                # conduct the fit with libmr
                weibull_models[i].fit_high(distribution_values[i], tailsizes[i])
                is_valid = weibull_models[i].is_valid
                count += 1
            if not is_valid:
                # print("Weibull fit for class " + str(i) + " not successful after " + str(num_max_fits) + " attempts")
                weibull_models[i] = []
        else:
            weibull_models.append([])

    return weibull_models, True

def calc_outlier_probs(weibull_models, distances):
    """
    Calculates statistical outlier probability using the weibull models' CDF.

    Note that we have coded this function to loop over each class because we have previously categorized the distances
    into their respective classes already.

    Parameters:
        weibull_models (list): List of libmr class instances containing the Weibull model parameters and functions.
        distances (list): List of per class torch tensors or numpy arrays with latent space distance values.

    Returns:
        list: List of length corresponding to number of classes with outlier probabilities for each respective input.
    """

    outlier_probs = []
    # loop through all classes, i.e. all available weibull models as there is one weibull model per class.
    for i in range(len(weibull_models)):
        # optionally convert the type of the distance vectors
        if isinstance(weibull_models[i],list):
            outlier_probs.append([])
            continue
        if isinstance(distances[i], torch.Tensor):
            distances[i] = distances[i].cpu().numpy().astype(np.double)
        elif isinstance(distances[i], list):
            # empty list
            outlier_probs.append([])
            continue
        else:
            distances[i] = distances[i].astype(np.double)

        # use the Weibull models' CDF to evaluate statistical outlier rejection probabilities.
        outlier_probs.append(weibull_models[i].w_score_vector(distances[i]))

    return outlier_probs

def calc_openset_classification(data_outlier_probs, num_classes, num_outlier_threshs=50):
    """
    Calculates the percentage of dataset outliers given a set of outlier probabilities over a range of rejection priors.

    Parameters:
         data_outlier_probs (list): List of outlier probabilities for an entire dataset, categorized by class.
         num_classes (int): Number of classes.
         num_outlier_threshs (int): Number of outlier rejection priors (evenly spread over the interval (0,1)).

    Returns:
        dict: Dictionary containing outlier percentages and corresponding rejection prior values.
    """

    dataset_outliers = []
    threshs = []

    # loop through each rejection prior value and evaluate the percentage of the dataset being considered as
    # statistical outliers, i.e. each data point's outlier probability > rejection prior.
    for i in range(num_outlier_threshs - 1):
        outlier_threshold = (i + 1) * (1.0 / num_outlier_threshs)
        threshs.append(outlier_threshold)

        dataset_outliers.append(0)
        total_dataset = 0

        for j in range(num_classes):
            total_dataset += len(data_outlier_probs[j])

            for k in range(len(data_outlier_probs[j])):
                if data_outlier_probs[j][k] > outlier_threshold:
                    dataset_outliers[i] += 1

        dataset_outliers[i] = dataset_outliers[i] / float(total_dataset)

    return {"thresholds": threshs, "outlier_percentage": dataset_outliers}

def calc_mean_class_acc(data_outlier_probs,label_pred_class_list_t, num_class):
    threshs = [0.98]
    num_outlier_threshs = len(threshs)
    label_pred_class_list_t = np.array(label_pred_class_list_t,dtype=object)
    
    best_OS_star_acc = 0
    best_unk = 0
    best_H = 0

    for i in range(num_outlier_threshs):        
        total_dataset = 0
        label_pred_class_list_t_copy = deepcopy(label_pred_class_list_t)
        for j in range(num_class-1):
            total_dataset += len(data_outlier_probs[j])

            for k in range(len(data_outlier_probs[j])):
                if data_outlier_probs[j][k] > threshs[i]:
                    label_pred_class_list_t_copy[1][j][k] = num_class-1                    

        all_pred = np.concatenate(np.array(label_pred_class_list_t_copy[1]),axis=0)
        all_label = np.concatenate(np.array(label_pred_class_list_t_copy[0]),axis=0)
        per_class_num = np.zeros((num_class))
        per_class_correct1 = np.zeros((num_class)).astype(np.float32)        
        for t in range(num_class):
            ind = np.where(all_label==t)[0]
            if len(ind) == 0:
                continue
            correct_ind = np.where(all_pred[ind] == t)[0]
            per_class_correct1[t] += float(len(correct_ind))
            per_class_num[t] += float(len(ind))

        per_class_acc1 = per_class_correct1 / per_class_num
        OS_acc1 = float(per_class_acc1.mean())
        OS_star_acc1 = float(per_class_acc1[:-1].mean())
        unk_acc1 = float(per_class_acc1[-1])
        H_acc = 0
        if OS_star_acc1 > 0 and unk_acc1 > 0:
            H_acc = 2*OS_star_acc1*unk_acc1/(OS_star_acc1+unk_acc1)
       
        if H_acc > best_H:
            best_H = H_acc
            best_OS_star_acc = OS_star_acc1
            best_unk = unk_acc1

    return best_OS_star_acc, best_unk, best_H

def correct_dist(distances_to_z_means_threshset,centroid_distance):
    num_class = centroid_distance.shape[0]
    for i in range(num_class-1):
        len_class = len(distances_to_z_means_threshset[i])
        if len_class>0:
            distances_to_z_means_threshset[i] = torch.clamp(distances_to_z_means_threshset[i] - 1.0*centroid_distance[i].expand(len_class),min=0.0)
    return distances_to_z_means_threshset

def inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    return initial_lr * ((1 + gamma * min(1.0, step / float(max_iter))) ** (- power))


class OptimWithSheduler:
    def __init__(self, optimizer, scheduler_func):
        self.optimizer = optimizer
        self.scheduler_func = scheduler_func
        self.global_step = 0.0
        for g in self.optimizer.param_groups:
            g['initial_lr'] = g['lr']
    def zero_grad(self):
        self.optimizer.zero_grad()
    def step(self):
        for g in self.optimizer.param_groups:
            g['lr'] = self.scheduler_func(step=self.global_step, initial_lr = g['initial_lr'])
        self.optimizer.step()
        self.global_step += 1


def reparameterize(z_mean, z_var, distribution='vmf'):
    if distribution == 'normal':
        q_z = torch.distributions.normal.Normal(z_mean, z_var)
    elif distribution == 'vmf':
        q_z = VonMisesFisher(z_mean, z_var)
    else:
        raise NotImplemented
    return q_z

def reparameterize2(z_mean, z_var, z_dim, distribution='vmf'):
    if distribution == 'normal':
        q_z = torch.distributions.normal.Normal(z_mean, z_var)
        p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))
    elif distribution == 'vmf':
        q_z = VonMisesFisher(z_mean, z_var)
        p_z = HypersphericalUniform(z_dim - 1)
    else:
        raise NotImplemented

    return q_z, p_z
