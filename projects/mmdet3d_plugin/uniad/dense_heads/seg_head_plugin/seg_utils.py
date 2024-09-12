
print("추적 UniAD/projects/mmdet3d_plugin/uniad/dense_heads/seg_head_plugin/seg_utils.py 지나감")

def IOU(intputs, targets):
    numerator = (intputs * targets).sum(dim=1)
    denominator = intputs.sum(dim=1) + targets.sum(dim=1) - numerator
    loss = numerator / (denominator + 0.0000000000001)
    return loss.cpu(), numerator.cpu(), denominator.cpu()
