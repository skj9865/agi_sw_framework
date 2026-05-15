import torch

def get_pos_neg_batch_imgcats_new(batch_pos1, batch_pos2, p = 1):

    batch_size = len(batch_pos1)

    batch_pos =torch.cat((batch_pos1, batch_pos2), dim = 1)

    #create negative samples
    # random_indices = (torch.randperm(batch_size - 1) + 1)[:min(p,batch_size - 1)]
    random_indices = torch.randint(1, batch_size, (1,))
    labeles = torch.arange(batch_size)
    # print(random_indices)
    batch_negs = []
    for i in random_indices:
        batch_neg = batch_pos2[(labeles+i)%batch_size]
        batch_neg = torch.cat((batch_pos1, batch_neg), dim = 1)
        batch_negs.append(batch_neg)
    
    return batch_pos, torch.cat(batch_negs)


def get_pos_neg_batch_imgcats(batch_pos1, batch_pos2, p = 1):
    """
    Generates positive and negative inputs for SCFF.

    Args:
        batch_pos1 (torch.Tensor): First set of samples of shape (batch_size, ...).
        batch_pos2 (torch.Tensor): Second set of samples, typically an augmented version 
                                   of batch_pos1 with the same shape or the same with batch_pos1.
        p (int, optional): Number of negative samples per positive sample. Default is 1.

    Returns:
        tuple: 
            - batch_pos (torch.Tensor): Concatenated positive samples of shape (batch_size, 2 * feature_dim).
            - batch_negs (torch.Tensor): Concatenated negative samples of shape (batch_size * p, 2 * feature_dim).
    """

    batch_size = len(batch_pos1)

    batch_pos =torch.cat((batch_pos1, batch_pos2), dim = 1)

    #create negative samples
    random_indices = (torch.randperm(batch_size - 1) + 1)[:min(p,batch_size - 1)]
    labeles = torch.arange(batch_size)

    batch_negs = []
    for i in random_indices:
        batch_neg = batch_pos2[(labeles+i)%batch_size]
        batch_neg = torch.cat((batch_pos1, batch_neg), dim = 1)
        batch_negs.append(batch_neg)
    
    return batch_pos, torch.cat(batch_negs)


def add_outputs(cv_out):

    batch_size = len(cv_out)

    # batch_pos = torch.cat((cv_out, cv_out), dim = 1)
    batch_pos = torch.mul(cv_out, 2)

    #create negative samples
    random_indices = (torch.randperm(batch_size - 1) + 1)[:min(1,batch_size - 1)]
    # random_indices = torch.randint(1, batch_size, (1,))
    labeles = torch.arange(batch_size)
    # print(random_indices)

    batch_neg = cv_out[(labeles+random_indices[0])%batch_size]
    # batch_neg = torch.cat((cv_out, batch_neg), dim = 1)
    batch_neg = torch.add(cv_out, batch_neg)
    
    return batch_pos, batch_neg