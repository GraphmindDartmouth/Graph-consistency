import torch
from torch_geometric.nn import global_add_pool
import torch.nn.functional as F
from types import MethodType
cls_criterion = torch.nn.BCEWithLogitsLoss
reg_criterion = torch.nn.MSELoss
multicls_criterion = torch.nn.CrossEntropyLoss


def loss(self, pred, y, pooled_outputs, task_type=None):
    # classification_loss = F.cross_entropy(pred, y)
    classification_loss = None
    if task_type is None:
        classification_loss = F.cross_entropy(pred, y)
    elif task_type == 'binary classification':
        y = y.to(torch.float32).squeeze()
        pred = pred.squeeze()
        is_labeled = y==y
        classification_loss = cls_criterion()(pred[is_labeled], y[is_labeled])
    elif task_type == 'multiclass classification':
        y = y.to(torch.int64).squeeze()
        pred = pred.squeeze()
        is_labeled = y==y
        classification_loss = multicls_criterion()(pred[is_labeled], y[is_labeled])    
    else:
        raise 
    middle_loss = compute_middle_loss(pooled_outputs, self.loss_module)


    # Total loss is the sum of classification loss and the regularization term
    total_loss = classification_loss + self.reg_term * middle_loss
    return total_loss, middle_loss


def model_decorator(model, reg_term, loss_func):
    model.reg_term=reg_term
    model.loss_module=loss_func
    
    assert not hasattr(model, 'loss'), 'Method Conflict!'
    model.loss = MethodType(loss, model)


def compute_middle_loss(pooled_outputs, loss_module):
    middle_loss = 0.0

    for index in range(1, len(pooled_outputs)):
        previous_layer_output = pooled_outputs[index - 1].detach()
        current_layer_output = pooled_outputs[index]

        # Compute similarity matrices for both layers
        previous_similarity =compute_cosine_similarity(previous_layer_output)
        current_similarity = compute_cosine_similarity(current_layer_output)

        # Compute the difference and retain only the positive values
        loss_component = loss_module(previous_similarity, current_similarity)
        middle_loss += loss_component

    return middle_loss

def compute_cosine_similarity(graph_representation):
    # Normalize the graph representations to unit vectors
    graph_representation_norm = F.normalize(graph_representation, p=2, dim=1)
    cosine_sim_matrix = torch.mm(graph_representation_norm, graph_representation_norm.t())
    identity = torch.eye(cosine_sim_matrix.size(0), device=cosine_sim_matrix.device)
    
    # Use the identity matrix to zero out diagonal elements
    cosine_sim_matrix = cosine_sim_matrix * (1 - identity)
    
    return cosine_sim_matrix

