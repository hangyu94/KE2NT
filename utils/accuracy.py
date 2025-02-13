import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix


def compute_accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for
    the specified values of k.

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

    Returns:
        list: accuracy at top-k.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    if isinstance(output, (tuple, list)):
        output = output[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res


def compute_class_accuracy(all_predicted, all_label):
    
    matrix = confusion_matrix(all_label.data.cpu().numpy(), all_predicted.data.cpu().numpy())
    label_list_num = Counter(all_label.data.cpu().numpy())
    averData = np.empty(len(label_list_num))

    for i in range(len(label_list_num)):
        averData[i] = 100.0 * float(matrix[i, i]) / label_list_num[i]
    averAcc = np.mean(averData)

    return averAcc, averData