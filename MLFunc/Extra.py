import torch.optim as optim

def get_lr(optimizer):
    """Get the learning rate

    Args:
        optimizer (torch.optim): optimizer

    Returns:
        _type_: _description_
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']