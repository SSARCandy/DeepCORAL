import torch
from torch.autograd import Variable


fixtures = {
    'forbenius_norm': ([
        torch.FloatTensor([[1, 2, 3], [3, 2, 1]]),
        torch.FloatTensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        torch.FloatTensor([[2, 0, -2], [0, 0, 0], [-2, 0, 2]]),
    ], [
        5.291502622129181,
        3,
        4
    ]),


    'feature_covariance_mat': ([
        (2, torch.FloatTensor([[1, 2, 3], [3, 2, 1]])),
        (2, torch.FloatTensor([[1, 1], [2, 2]])),
        (2, torch.FloatTensor([[2, 1], [2, 1]])),
    ], [
        torch.FloatTensor([[2, 0, -2], [0, 0, 0], [-2, 0, 2]]),
        torch.FloatTensor([[.5, .5], [.5, .5]]),
        torch.FloatTensor([[0, 0], [0, 0]]),
    ]),


    'CORAL_forward': ([
        (
          torch.FloatTensor([[2, 0, -2], [0, 0, 0], [-2, 0, 2]]),
          torch.FloatTensor([[2, 0, -2], [0, 0, 0], [-2, 0, 2]])
        ),
    ], [
        torch.FloatTensor([0])
    ]),


    'CORAL_backward': ([
        (
            Variable(torch.FloatTensor([[1, 1], [2, 2]]), requires_grad=True),
            Variable(torch.FloatTensor([[2, 1], [2, 1]]), requires_grad=True),
        ),
    ], [
        (
            torch.FloatTensor([[-0.125, -0.125], [0.125, 0.125]]),
            torch.FloatTensor([[0, 0], [0, 0]]),
        )
    ]),
}