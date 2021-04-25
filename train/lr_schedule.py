def lr_scheduler(optimizer, iter_num, gamma, power, lr=0.001, weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = lr * ((1 + gamma * iter_num) ** (-power))
    optimizer.param_groups[0]['lr'] = lr

    return optimizer
