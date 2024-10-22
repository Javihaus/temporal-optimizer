import torch.nn as nn


def hamiltonian_loss(outputs, labels, model, reg_coeff=0.01):
    loss_fct = nn.CrossEntropyLoss()
    base_loss = loss_fct(outputs, labels)
    # Add regularization based on Hamiltonian principles
    param_norm = sum(p.norm().item() for p in model.parameters())
    reg_term = reg_coeff * param_norm
    return base_loss + reg_term
