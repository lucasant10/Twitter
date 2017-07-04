from unittest import TestCase
class TestMetrics(TestCase):
    Metrics = metrics=
    def test_purity(self):
        data_a = [0, 0, 0, 0, 1, 1, 1, 1]
        data_b = [2, 2, 2, 1, 2, 1, 1, 1]
        self.assertEquals(metrics.purity_score(data_a, data_b), 1)
