import logging
import argparse
import numpy as np

from pyjet.callbacks import Plotter, ModelCheckpoint
from pyjet.data import NpDataset
from pyjet.models import SLModel

import kaggleutils as utils
from data_utils import SaltData
import models
from metrics import mean_iou

parser = argparse.ArgumentParser(description='Run the models.')
# parser.add_argument('train_id', help='ID of the train configuration')
parser.add_argument('--train', action="store_true", help="Whether to run this script to train a model")
parser.add_argument('--test', action="store_true", help="Whether to run this script to generate submissions")
parser.add_argument('--plot', action="store_true", help="Whether to plot the training loss")
# parser.add_argument('--num_completed', type=int, default=0, help="How many completed folds")
# parser.add_argument('--reload_model', action='store_true', help="Continues training from a saved model")
# parser.add_argument('--initial_epoch', type=int, default=0, help="Continues training from specified epoch")
# parser.add_argument('--cutoff', type=float, default=0.5, help="Cutoff to use for producing submission")
# parser.add_argument('--use_iou', action='store_true', help="creates test predictions with iou checkpointed model.")
# parser.add_argument('--test_debug', action='store_true', help="debugs the test output.")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

MODEL = models.unet
RUN_ID = "unet-test-downsampler"
SEED = 42
BATCH_SIZE = 32
EPOCHS = 1
utils.set_random_seed(SEED)
SPLIT_SEED = utils.get_random_seed()


def train_model(model: SLModel,
                trainset: NpDataset,
                valset: NpDataset,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                plot=True):

    # Create the generators
    logging.info("Training model for {epochs} epochs and {batch_size} batch "
                 "size".format(**locals()))
    logging.info("Flowing the train and validation sets")
    traingen = trainset.flow(
        batch_size=batch_size, shuffle=True, seed=utils.get_random_seed())
    valgen = valset.flow(batch_size=batch_size, shuffle=False)

    # Create the callbacks
    logging.info("Creating the callbacks")
    callbacks = [
        ModelCheckpoint(
            utils.get_model_path(RUN_ID),
            "val_loss",
            verbose=1,
            save_best_only=True),
        Plotter(
            "bce",
            scale='log',
            plot_during_train=plot,
            save_to_file=utils.get_plot_path(RUN_ID+"_bce"),
            block_on_end=False),
        Plotter(
            "dice",
            plot_during_train=plot,
            save_to_file=utils.get_plot_path(RUN_ID+"_dice"),
            block_on_end=False),
        Plotter(
            "iou",
            plot_during_train=plot,
            save_to_file=utils.get_plot_path(RUN_ID + "_iou"),
            block_on_end=False),
    ]

    # Train the model
    logs = model.fit_generator(
        traingen,
        traingen.steps_per_epoch,
        epochs=epochs,
        validation_data=valgen,
        validation_steps=valgen.steps_per_epoch,
        callbacks=callbacks,
        metrics=["iou", mean_iou],
        verbose=1)

    return logs


def test_model(model: SLModel, test_data: NpDataset, batch_size=BATCH_SIZE):
    logging.info("Testing model with batch size of {batch_size}".format(**locals()))
    logging.info("Flowing the test set")
    test_data.output_labels = False
    testgen = test_data.flow(batch_size=batch_size, shuffle=False)
    test_preds = model.predict_generator(
        testgen, testgen.steps_per_epoch, verbose=1)
    return test_preds.squeeze(-1)


def train(data: SaltData, plot=True):
    train_data = data.load_train()
    model = MODEL()
    train_data, val_data = train_data.validation_split(
        split=0.1, shuffle=True, seed=SPLIT_SEED, stratified=True,
        stratify_by=np.all(train_data.y == 0., axis=(1, 2, 3)))
    train_model(model, train_data, val_data, plot=plot)
    # Load the model and score it
    model.load_state(utils.get_model_path(RUN_ID))
    return model


def test(data: SaltData, model=None):
    test_data = data.load_test()
    if model is None:
        logging.info("No model provided, constructing one.")
        model = MODEL()
    # Load the model and score it
    model.load_state(utils.get_model_path(RUN_ID))
    test_preds = test_model(model, test_data)
    # Save the submission
    data.save_submission(
        utils.get_submission_path(RUN_ID),
        test_preds,
        test_data.ids)


if __name__ == "__main__":
    args = parser.parse_args()
    data = SaltData()
    model = None
    if args.train:
        model = train(data, plot=args.plot)
    if args.test:
        test(data, model=model)
