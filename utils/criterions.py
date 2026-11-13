import torch.nn.functional as F
import torch
import logging
import torch.nn as nn


__all__ = [
    "sigmoid_dice_loss",
    "softmax_dice_loss",
    "GeneralizedDiceLoss",
    "FocalLoss",
    "dice_loss",
    "ContrastiveLoss",
    "CosineSimilarityLoss",
    "softmax_weighted_loss",
    "softmax_loss",
]

cross_entropy = F.cross_entropy


def mse_loss(output, target):
    MSE = nn.MSELoss()
    return MSE(output, target)


def cross_entropy_loss(output, target):
    CEL = nn.CrossEntropyLoss()
    return CEL(output, target)


def l2_regularization_loss(w):
    return torch.square(w).sum()


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction="sum") * (self.T**2) / y_s.shape[0]
        return loss


def MMD_loss(x, y, width=1):
    x_n = x.shape[0]
    y_n = y.shape[0]

    x = x.reshape(x.size(0), x.size(1) * x.size(2) * x.size(3) * x.size(4)).contiguous()
    y = y.reshape(y.size(0), y.size(1) * y.size(2) * y.size(3) * y.size(4)).contiguous()

    x_square = torch.sum(x * x, 1)
    y_square = torch.sum(y * y, 1)

    kxy = torch.matmul(x, y.t())
    kxy = kxy - 0.5 * x_square.unsqueeze(1).expand(x_n, y_n)
    kxy = kxy - 0.5 * y_square.expand(x_n, y_n)
    kxy = torch.exp(width * kxy).sum() / x_n / y_n

    kxx = torch.matmul(x, x.t())
    kxx = kxx - 0.5 * x_square.expand(x_n, x_n)
    kxx = kxx - 0.5 * x_square.expand(x_n, x_n)
    kxx = torch.exp(width * kxx).sum() / x_n / x_n

    kyy = torch.matmul(y, y.t())
    kyy = kyy - 0.5 * y_square.expand(y_n, y_n)
    kyy = kyy - 0.5 * y_square.expand(y_n, y_n)
    kyy = torch.exp(width * kyy).sum() / y_n / y_n

    return kxx + kyy - 2 * kxy


class ContrastiveLoss(nn.Module):
    """Contrastive loss with dynamic batch size B = features.shape[0] // 2 and negative mask (2B, 2B)."""

    def __init__(self, batch_size=None, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = float(temperature)

    def forward(self, emb_i: torch.Tensor, emb_j: torch.Tensor | None = None) -> torch.Tensor:
        if emb_j is None:
            B = emb_i.shape[0] // 2
            z_i = emb_i[:B]
            z_j = emb_i[B:]
        else:
            B = emb_i.shape[0]
            z_i = emb_i
            z_j = emb_j

        if B == 0:
            return torch.tensor(0.0, device=emb_i.device, dtype=emb_i.dtype, requires_grad=True)

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)  # (2*B, dim)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )  # (2*B, 2*B)

        sim_ij = torch.diag(similarity_matrix, B)
        sim_ji = torch.diag(similarity_matrix, -B)
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*B

        negatives_mask = (~torch.eye(2 * B, 2 * B, dtype=torch.bool, device=emb_i.device)).float()

        nominator = torch.exp(positives / self.temperature)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / (torch.sum(denominator, dim=1) + 1e-8) + 1e-8)
        loss = torch.sum(loss_partial) / (2 * B)
        return loss


class CosineSimilarityLoss(nn.Module):
    """Cosine similarity loss for multimodal feature alignment across M=3 modalities."""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            if features.ndim >= 3 and features.shape[1] == 3:
                feats = [features[:, m, ...] for m in range(features.shape[1])]
            else:
                feats = [features]
        else:
            feats = list(features)

        M = len(feats)
        if M < 2:
            return torch.tensor(0.0, device=feats[0].device, dtype=feats[0].dtype)

        flat_feats = [f.flatten(1) for f in feats]
        total_loss = 0.0
        num_pairs = 0
        for i in range(M):
            for j in range(i + 1, M):
                cos_sim = F.cosine_similarity(flat_feats[i], flat_feats[j], dim=1, eps=self.eps)
                total_loss = total_loss + (1.0 - cos_sim.mean())
                num_pairs += 1

        return total_loss / max(num_pairs, 1)


def Dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss


def dice_loss(output, target, num_cls=4, eps=1e-7):
    """Dice loss supporting both 2D (B, C, H, W) and 3D (B, C, H, W, Z), integer or one-hot targets."""
    if target.ndim == output.ndim - 1:
        target = F.one_hot(target.long(), num_classes=output.shape[1])
        perm = (0, target.ndim - 1) + tuple(range(1, target.ndim - 1))
        target = target.permute(perm).float()
    else:
        target = target.float()

    C = output.shape[1]
    dice = 0.0
    for i in range(C):
        num = torch.sum(output[:, i, ...] * target[:, i, ...])
        l = torch.sum(output[:, i, ...])
        r = torch.sum(target[:, i, ...])
        dice += 2.0 * num / (l + r + eps)
    return 1.0 - 1.0 * dice / C


def softmax_weighted_loss(output, target, num_cls=4):
    """Weighted cross-entropy loss supporting both 2D and 3D tensors."""
    if target.ndim == output.ndim - 1:
        target = F.one_hot(target.long(), num_classes=output.shape[1])
        perm = (0, target.ndim - 1) + tuple(range(1, target.ndim - 1))
        target = target.permute(perm).float()
    else:
        target = target.float()

    C = output.shape[1]
    total_axes = tuple(range(1, target.ndim))
    total_target_sum = torch.sum(target, dim=total_axes) + 1e-8

    cross_loss = 0.0
    for i in range(C):
        outputi = output[:, i, ...]
        targeti = target[:, i, ...]
        spatial_dims = tuple(range(1, targeti.ndim))
        class_sum = torch.sum(targeti, dim=spatial_dims)
        weighted = 1.0 - (class_sum / total_target_sum)
        view_shape = (targeti.shape[0],) + (1,) * len(spatial_dims)
        weighted = weighted.view(view_shape)
        cross_loss += -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1.0)).float()

    return torch.mean(cross_loss)


def softmax_loss(output, target, num_cls=4):
    """Standard cross-entropy loss on softmax probabilities."""
    if target.ndim == output.ndim - 1:
        target = F.one_hot(target.long(), num_classes=output.shape[1])
        perm = (0, target.ndim - 1) + tuple(range(1, target.ndim - 1))
        target = target.permute(perm).float()
    else:
        target = target.float()

    C = output.shape[1]
    cross_loss = 0.0
    for i in range(C):
        outputi = output[:, i, ...]
        targeti = target[:, i, ...]
        cross_loss += -1.0 * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1.0)).float()
    return torch.mean(cross_loss)


def FocalLoss(output, target, alpha=0.25, gamma=2.0):
    target[target == 4] = 3  # label [4] -> [3]
    # target = expand_target(target, n_class=output.size()[1]) # [N,H,W,D] -> [N,4,H,W,D]
    if output.dim() > 2:
        output = output.view(
            output.size(0), output.size(1), -1
        )  # N,C,H,W,D => N,C,H*W*D
        output = output.transpose(1, 2)  # N,C,H*W*D => N,H*W*D,C
        output = output.contiguous().view(-1, output.size(2))  # N,H*W*D,C => N*H*W*D,C
    if target.dim() == 5:
        target = target.contiguous().view(target.size(0), target.size(1), -1)
        target = target.transpose(1, 2)
        target = target.contiguous().view(-1, target.size(2))
    if target.dim() == 4:
        target = target.view(-1)  # N*H*W*D
    # compute the negative likelyhood
    logpt = -F.cross_entropy(output, target)
    pt = torch.exp(logpt)
    # compute the loss
    loss = -((1 - pt) ** gamma) * logpt
    # return loss.sum()
    return loss.mean()


def dice(output, target, eps=1e-5):  # soft dice loss
    target = target.float()
    num = 2 * (output * target).sum()
    den = output.sum() + target.sum() + eps
    return 1.0 - num / den


def sigmoid_dice_loss(output, target, alpha=1e-5):
    # output: [-1,3,H,W,T]
    # target: [-1,H,W,T] noted that it includes 0,1,2,4 here
    loss1 = dice(output[:, 0, ...], (target == 1).float(), eps=alpha)
    loss2 = dice(output[:, 1, ...], (target == 2).float(), eps=alpha)
    loss3 = dice(output[:, 2, ...], (target == 4).float(), eps=alpha)
    logging.info(
        "1:{:.4f} | 2:{:.4f} | 4:{:.4f}".format(
            1 - loss1.data, 1 - loss2.data, 1 - loss3.data
        )
    )
    return loss1 + loss2 + loss3


def softmax_dice_loss(output, target):
    # output : [bsize,c,H,W,D]
    # target : [bsize,H,W,D]
    loss1 = dice(output[:, 1, ...], (target == 1).float())
    loss2 = dice(output[:, 2, ...], (target == 2).float())
    loss3 = dice(output[:, 3, ...], (target == 4).float())
    logging.info(
        "1:{:.4f} | 2:{:.4f} | 4:{:.4f}".format(
            1 - loss1.data, 1 - loss2.data, 1 - loss3.data
        )
    )

    return loss1 + loss2 + loss3


def GeneralizedDiceLoss(
    output, target, eps=1e-5, weight_type="square"
):  # Generalized dice loss
    """
    Generalised Dice : 'Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations'
    """

    if target.dim() == 4:
        target[target == 4] = 3  # label [4] -> [3]
        target = expand_target(
            target, n_class=output.size()[1]
        )  # [N,H,W,D] -> [N,4，H,W,D]

    output = flatten(output)[
        1:, ...
    ]  # transpose [N,4，H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
    target = flatten(target)[1:, ...]  # [class, N*H*W*D]

    target_sum = target.sum(-1)  # sub_class_voxels [3,1] -> 3个voxels
    if weight_type == "square":
        class_weights = 1.0 / (target_sum * target_sum + eps)
    elif weight_type == "identity":
        class_weights = 1.0 / (target_sum + eps)
    elif weight_type == "sqrt":
        class_weights = 1.0 / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError("Check out the weight_type :", weight_type)

    # print(class_weights)
    intersect = (output * target).sum(-1)
    intersect_sum = (intersect * class_weights).sum()
    denominator = (output + target).sum(-1)
    denominator_sum = (denominator * class_weights).sum() + eps

    loss1 = 2 * intersect[0] / (denominator[0] + eps)
    loss2 = 2 * intersect[1] / (denominator[1] + eps)
    loss3 = 2 * intersect[2] / (denominator[2] + eps)

    return 1 - 2.0 * intersect_sum / denominator_sum, [
        loss1.data,
        loss2.data,
        loss3.data,
    ]


def expand_target(x, n_class, mode="softmax"):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label is stored in a separate channel
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :return: 5D output image (NxCxDxHxW)
    """
    assert x.dim() == 4
    shape = list(x.size())
    shape.insert(1, n_class)
    shape = tuple(shape)
    xx = torch.zeros(shape)
    if mode.lower() == "softmax":
        xx[:, 1, :, :, :] = x == 1
        xx[:, 2, :, :, :] = x == 2
        xx[:, 3, :, :, :] = x == 3
    if mode.lower() == "sigmoid":
        xx[:, 0, :, :, :] = x == 1
        xx[:, 1, :, :, :] = x == 2
        xx[:, 2, :, :, :] = x == 3
    return xx.to(x.device)


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, -1)
