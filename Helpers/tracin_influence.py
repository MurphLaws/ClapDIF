from typing import List
import numpy as np
import torch
from captum.influence import TracInCPFast, TracInCP


def checkpoints_load_func(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint["learning_rate"]


def compute_train_to_test_influence(
        model,
        train_set,
        test_set,
        checkpoints_files: List[str],
        layers: List[str],
        batch_size: int = 3500,
        fast_cp=False,
        loss_fn=torch.nn.CrossEntropyLoss
) -> np.ndarray:
    # TODO check the documentation of TracIn of Captum to see the meaning of each parameter
    # https://captum.ai/api/influence.html#tracincp
    # CAREFUL: the layers parameter must contain the layer names that you did not freeze during the training
    # fast_cp parameter indicates that the tracin function will use just the last layer to compute the influence
    if fast_cp:
        if isinstance(layers, List):
            assert len(layers) == 1
        tracin_cp = TracInCPFast(
            model=model,
            train_dataset=train_set,
            final_fc_layer=layers[0],
            checkpoints=checkpoints_files,
            checkpoints_load_func=checkpoints_load_func,
            loss_fn=loss_fn,
            batch_size=batch_size,
        )
    else:
        tracin_cp = TracInCP(
            model,
            train_set,
            checkpoints_files,
            checkpoints_load_func=checkpoints_load_func,
            loss_fn=loss_fn,
            batch_size=batch_size,
            sample_wise_grads_per_batch=True,
            layers=layers,
        )

    test_examples_features = torch.stack(
        [test_set[i][0] for i in range(len(test_set))]
    )
    test_examples_true_labels = torch.Tensor(
        [test_set[i][1] for i in range(len(test_set))]
    ).long()

    train_to_test_influence = tracin_cp.influence(
        (test_examples_features, test_examples_true_labels), show_progress=True
    )

    # The Influence Matrix is returned. The dimenisons are NxM where N are the train samples and M are the test samples

    return np.array(train_to_test_influence).transpose()


def compute_self_influence(
        model,
        dataset,
        checkpoints_files: List[str],
        layers: List[str],
        batch_size: int = 3500,
        fast_cp=False,
        loss_fn=torch.nn.CrossEntropyLoss
) -> np.ndarray:
    # TODO check the documentation of TracIn of Captum to see the meaning of each parameter
    # https://captum.ai/api/influence.html#tracincp
    # CAREFUL: the layers parameter must contain the layer names that you did not freeze during the training
    # fast_cp parameter indicates that the tracin function will use just the last layer to compute the influence
    if fast_cp:
        if isinstance(layers, List):
            assert len(layers) == 1
        tracin_cp = TracInCPFast(
            model=model,
            train_dataset=dataset,
            final_fc_layer=layers[0],
            checkpoints=checkpoints_files,
            checkpoints_load_func=checkpoints_load_func,
            loss_fn=loss_fn,
            batch_size=batch_size,
        )
    else:
        tracin_cp = TracInCP(
            model,
            dataset,
            checkpoints_files,
            checkpoints_load_func=checkpoints_load_func,
            loss_fn=loss_fn,
            batch_size=batch_size,
            sample_wise_grads_per_batch=True,
            layers=layers,
        )

    # This function returns an array of N elements, each one indicating the

    self_influence = tracin_cp.self_influence(show_progress=True)
    return self_influence


if __name__ == '__main__':

    pass

    

    # AFTER that you have injected the poisons

    # read the poisoned checkpoints (.pt files) in a list ckpt_list.
    # For each cp in ckpt_list init the model's parameters with the load function (check the checkpoints_load_func
    # if you have questions) and compute self and train-to-test influence.
    # In each epoch, for the poisonous samples and their targets, see how the targets are influenced, maybe this
    # is different for different epochs. If it takes too much time, just start compute the influence to the last epochs
    # where the label of the test examples is changed

    # the questions to answer are:
    # 1. Are the poison examples affect the most negatively their target samples?
    # 2. What is the self influence in terms of ranking for the poisonous examples?
