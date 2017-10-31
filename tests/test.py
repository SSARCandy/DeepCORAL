import torch
import numpy as np
import unittest

from DeepCORAL.models import *
from DeepCORAL.tests.fixtures import fixtures

class TestCORAL(unittest.TestCase):
    def test_forbenius_norm(self):
        test, ans = fixtures['forbenius_norm']

        for x in range(len(test)):
            res = forbenius_norm(test[x])
            self.assertEqual(res, ans[x])

    def test_feature_covariance_mat(self):
        test, ans = fixtures['feature_covariance_mat']

        for x in range(len(test)):
            res = feature_covariance_mat(*test[x])
            res = np.array_equal(res.numpy(), ans[x].numpy())
            self.assertTrue(res)

    def test_CORAL_forward(self):
        test, ans = fixtures['CORAL_forward']

        for x in range(len(test)):
            coral = CORAL()
            res = coral.forward(*test[x])
            self.assertEqual(res.numpy(), ans[x].numpy())

    def test_CORAL_backward(self):
        test, ans = fixtures['CORAL_backward']

        for x in range(len(test)):
            coral = CORAL()
            valid = torch.autograd.gradcheck(coral, test[x], eps=1e-3)
            self.assertTrue(valid)

            coral(*test[x])  # forward
            out1, out2 = coral.backward(1)
            ans1, ans2 = ans[x]
            self.assertTrue(np.array_equal(out1.numpy(), ans1.numpy()))
            self.assertTrue(np.array_equal(out2.numpy(), ans2.numpy()))


if __name__ == '__main__':
    unittest.main()