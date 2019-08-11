from numpy import add, argmax as np_argmax, array, concatenate
from pandas import DataFrame
from torch import argmax, no_grad
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def train(model, num_epochs, train_dl, optimizer, scheduler=None, criterion=CrossEntropyLoss(),
          test_dl=None, device="cuda", hide_progress=False):
    """
    Trains a Pytorch model on the data provided by a DataLoader.
    :param model: (torch.nn.module) the model to train
    :param num_epochs: (int) the number of epochs to train
    :param train_dl: (torch.utils.dataset.DataLoader) the data-loader which provides the training data
    :param optimizer: (Pytorch optimizer) the optimizer to use for training
    :param scheduler: (optional, Pytorch scheduler) an optional learning rate scheduler
    :param criterion: (optional, Pytorch loss function, default=CrossEntropyLoss())
                      the loss function to use
    :param test_dl: (optional, torch.utils.dataset.DataLoader) if not None, the model will be
                    evaluated on the data provided by this data-loader after each training epoch,
                    and the results will be printed and included in the returned results DataFrame
    :param device: (optional, "cpu" or "cuda" etc., default="cuda") which device to use for the training
    :param hide_progress: (optional, bool, default=False) if True no progress-bars will be shown during training
    :return: (pandas.DataFrame) contains the loss and accuracy scores for all epochs for the training
             dataset and, if a test_dl was provided, also the scores for this underlying dataset.
    """
    results = []

    for epoch in tqdm(range(num_epochs), "Epochs", disable=hide_progress):
        model.train()

        running_train_loss = 0.0
        total_nr_train_samples = 0
        nr_correct_train_predictions = 0

        for inputs, labels in tqdm(train_dl, desc=f"Epoch {epoch}",
                                   leave=False, disable=hide_progress):
            # Move inputs and labels to correct device:
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass through the model:
            outputs = model(inputs)
            # Calculate loss:
            loss = criterion(outputs, labels)
            # Update running metrics:
            running_train_loss += loss.item()
            total_nr_train_samples += labels.size(0)
            train_predicted = argmax(outputs.data, 1)
            nr_correct_train_predictions += (train_predicted == labels).double().sum().item()
            # Backward pass through the model (calculate gradients):
            loss.backward()

            # Update weights:
            optimizer.step()

        # If necessary, update learning rate through scheduler:
        if scheduler is not None:
            scheduler.step(loss) if isinstance(scheduler, ReduceLROnPlateau) \
                else scheduler.step()

        # Record train metrics:
        results.append({
            "epoch": epoch,
            "train_loss": running_train_loss / total_nr_train_samples,
            "train_accuracy": nr_correct_train_predictions / total_nr_train_samples
        })

        epoch_message = "Epoch {0}: Train: loss: {1:.6f}, accuracy: {2:.4f}".format(*results[-1].values())

        if test_dl is not None:
            model.eval()

            running_test_loss = 0.0
            total_nr_test_samples = 0
            nr_correct_test_predictions = 0

            with no_grad():
                for inputs, labels in test_dl:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    running_test_loss += loss.item()
                    predicted = argmax(outputs, 1)
                    total_nr_test_samples += labels.size(0)
                    nr_correct_test_predictions += (predicted == labels).double().sum().item()

            results[-1]["test_loss"] = running_test_loss / total_nr_test_samples
            results[-1]["test_accuracy"] = nr_correct_test_predictions / total_nr_test_samples

            epoch_message += ", Test: loss: {0:.6f}, accuracy: {1:.4f}".format(results[-1]["test_loss"],
                                                                               results[-1]["test_accuracy"])
        print(epoch_message)

    print('Finished Training!')

    return DataFrame(results).set_index("epoch")


def predict(model, test_dl, device="cuda"):
    """
    Uses a provided Pytorch model to calculate predictions for the samples in a passed-in
    torch.utils.dataset.DataLoader.
    :param model: (torch.nn.module) the model to be used to create the predictions
    :param test_dl: (torch.utils.dataset.DataLoader) provides the samples whose classes should be
                    predicted
    :param device: (optional, "cpu" or "cuda" etc., default="cuda") which device to use for the training
    :return: (list) the predictions
    """
    model.eval()
    with no_grad():
        if test_dl.dataset.train:
            predictions = [argmax(model(inputs.to(device)).data, 1).cpu().numpy()
                           for inputs, _ in test_dl]
        else:
            predictions = [argmax(model(inputs.to(device)).data, 1).cpu().numpy()
                           for inputs in test_dl]
    return concatenate(predictions)


def predict_with_augmentations(model, test_dataset, transform, nr_augments=4, batch_size=64, device="cuda"):
    """
    Uses a provided Pytorch model to calculate predictions for the samples in a passed-in
    torch.utils.dataset.DataLoader using test-time-augmentation. With test-time-augmentation, random
    transformations are applied to a predefined number (=nr_augments) of copies of each sample before making
    the predictions. Then the obtained class probabilities are summed and the final predicted class is calculated
    by taking the argmax of these summed probabilities.
    :param model: (torch.nn.module) the model to be used to create the predictions
    :param test_dataset: (torch.utils.dataset.DataSet) provides the data whose classes should be
                         predicted
    :param transform: (Pytorch transform) the augmentations to apply to the samples before predicting occurs
    :param nr_augments: (int, default=4) the number of times each single sample should be augmented
    :param batch_size: (int, default=64) the batch size
    :param device: (optional, "cpu" or "cuda" etc., default="cuda") which device to use for the training
    :return: (list) the predictions
    """
    if nr_augments < 1:
        raise ValueError("nr_augments must be larger than 1.")

    old_transform = test_dataset.transform
    test_dataset.transform = transform

    augmented_dl = DataLoader(test_dataset, batch_size=batch_size)

    prediction_sum = 0

    for i in range(nr_augments):
        if test_dataset.train:
            predictions = array([model(inputs.to(device)).data.cpu().numpy() for inputs, _ in augmented_dl])
        else:
            predictions = array([model(inputs.to(device)).data.cpu().numpy() for inputs in augmented_dl])

        if i == 0:
            prediction_sum = predictions
        else:
            prediction_sum = add(prediction_sum, predictions)

    test_dataset.transform = old_transform

    prediction_result = concatenate([[np_argmax(prediction) for prediction in batch_predictions]
                                     for batch_predictions in prediction_sum])

    return prediction_result
