import torch


def accuracy(output, target, topk=(1,)):
    if output.size(1) > 1:
        """
        Computes the accuracy over the k top 
        predictions for the specified values of k
        """
        maxk = min(max(topk), output.size()[1])
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]
    else:
        pred = torch.sigmoid(output)
        pred = torch.where(pred > 0.5, 1, 0)
        correct = pred.eq(target.reshape(-1, 1)).sum() / target.size(0) * 100
        return [correct]