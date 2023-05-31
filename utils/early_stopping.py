import torch
import wandb
from tqdm import tqdm
from copy import deepcopy
from typing import Callable



class EarlyStopping:
    """Early stopping module."""
    def __init__(
        self,
        path_prefix: str,
        model: Callable,
        patience: int = 8,
        low_is_good: bool = True,
        hyperopt: bool = False,
        verbose: bool = False,
        model_name: str = "",
    ) -> None:
        """
        Early stopping module to identify when a training loop can exit because a local optima is found.
        :path_prefix (str): Path to store file.
        :model (base.ModelType): The model to store.
        :patience (int, default = 8): The number of epochs to allow the model to get out of local optima.
        :low_is_good (bool, default = False): Lower scores indicate better performance.
        :hyperopt (bool, False): Under hyper-optimisation.
        :verbose (bool, False): Stop if the current epoch has a worse score then the best epoch so far.
        """
        self.patience = patience
        self.best_model = None
        self.best_score = None
        self.best_epoch = 0
        self.epoch = 0
        self.low_is_good = low_is_good
        self.path_prefix = f"{path_prefix}_{model_name}.pkl"
        self.hyperopt = hyperopt
        self.verbose = verbose
        self.model = model
    def __call__(self, model: Callable, score: float) -> bool:
        """
        Perform check to see if training can be stoppped.
        :model (base.ModelType): The model being trained.
        :score (float): The score achieved in the current epoch.
        """
        self.epoch += 1
        if self.best_score is None:
            self.best_score = score
        if self.new_best(score):
            self.best_state = model
            self.best_score = score
            self.best_epoch = self.epoch
            return False
        elif self.epoch > self.best_epoch + self.patience:  # Best score achieved
            tqdm.write("Early stopping: Terminate")
            return True
        if self.verbose:
            tqdm.write("Early stopping: Worse epoch")
        return False
    def new_best(self, score: float) -> bool:
        """
        Identiy if the current score is better than previous scores.
        :score (float): Score for the current epoch.
        :returns (bool): True if the current score is better than the previous best.
        """
        if self.low_is_good:
            return score <= self.best_score
        else:
            return score >= self.best_score
    @property
    def best_state(self):
        """Load/save the best model state prior to early stopping being activated."""
        tqdm.write("Loading weights from epoch {0}".format(self.best_epoch))
        try:
            self.model.load_state_dict(self.best_state_dict["model_state_dict"])
            torch.save({"model_state_dict": self.best_state_dict}, self.path_prefix)
            if self.hyperopt:
                wandb.save(self.path_prefix)
            # self.model.load_state_dict(torch.load(self.path_prefix)['model_state_dict'])
        except Exception as e:
            tqdm.write(
                f"Exception occurred loading the model after early termination. {e}"
            )
            raise e
        return self.model
    @best_state.setter
    def best_state(self, model: Callable) -> None:
        """
        Save best model thus far.
        :model (base.ModelType): Model being trained.
        """
        self.best_state_dict = deepcopy({"model_state_dict": model.state_dict()})
        torch.save({'model_state_dict': model.state_dict()}, self.path_prefix)

