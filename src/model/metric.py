import torch
import torch.nn as nn


class BaseMetric(nn.Module):
    def __init__(self, nickname, output_key, target_key, **kwargs):
        super().__init__()
        self.__name__ = nickname
        self.output_key = output_key
        self.target_key = target_key
        self.metric_fn = None

    def _preprocess(self, logits, target):
        return logits, target

    def forward(self, data, output):
        logits = output[self.output_key]
        target = data[self.target_key]
        logits, target = self._preprocess(logits, target)

        return self.metric_fn(logits, target)


class TopKAcc():
    def __init__(self, k, nickname="", output_key='verb_logits', target_key='verb_class'):
        self.k = k
        self.__name__ = f'top{self.k}_acc_{target_key}' if nickname == "" else nickname
        self.output_key = output_key
        self.target_key = target_key

    def __call__(self, data, output):
        with torch.no_grad():
            logits = output[self.output_key]
            target = data[self.target_key]
            pred = torch.topk(logits, self.k, dim=1)[1]
            assert pred.shape[0] == len(target)
            correct = 0
            for i in range(self.k):
                correct += torch.sum(pred[:, i] == target).item()
        return correct / len(target)


class MSEMetric(BaseMetric):
    def __init__(self, nickname, output_key, target_key, **kwargs):
        super().__init__(nickname, output_key, target_key, **kwargs)
        self.metric_fn = nn.MSELoss()
