# Regular python libraries
import sys
import numpy as np

# from pathlib import Path
import random

# import scrapbook as sb

# fast.ai
import fastai
from fastai.vision import (
    accuracy,
    cnn_learner,
    DatasetType,
    ImageList,
    imagenet_stats,
    models,
    partial,
)

# Computer Vision repository
sys.path.extend([".", "../.."])  # to access the utils_cv library
from utils_cv.classification.data import Urls
from utils_cv.classification.model import TrainMetricsRecorder
from utils_cv.common.data import unzip_url

# from utils_cv.common.gpu import which_processor
from utils_cv.similarity.data import comparative_set_builder
from utils_cv.similarity.metrics import (
    positive_image_ranks,
)  # compute_distances, recall_at_k
from utils_cv.similarity.model import (
    compute_features_learner,
)  # compute_features,

# from utils_cv.similarity.plot import (
#     plot_comparative_set,
#     plot_distances,
#     plot_ranks_distribution,
#     plot_recalls,
# )

# Param sweeper imports
from utils_cv.classification.parameter_sweeper import (
    ParameterSweeper,
    clean_sweeper_df,
)


def similarity_accuracy(learn):
    data = learn.data

    # Build multiple sets of comparative images from the validation images
    comparative_sets = comparative_set_builder(
        data.valid_ds, num_sets=1000, num_negatives=99
    )

    # Compute DNN features for all validation images
    embedding_layer = learn.model[1][6]
    valid_features = compute_features_learner(
        data, DatasetType.Valid, learn, embedding_layer
    )

    # For each comparative set compute the distances between the query image and all reference images
    for cs in comparative_sets:
        cs.compute_distances(valid_features)

    # Compute the median rank of the positive example over all comparative sets
    ranks = positive_image_ranks(comparative_sets)
    median_rank = np.median(ranks)
    return median_rank


if __name__ == "__main__":
    print(f"Fast.ai version = {fastai.__version__}")

    # -------------------------------------------------------
    #  IMAGE SIMILARITY CODE
    # -------------------------------------------------------
    if False:
        # Set dataset, model and evaluation parameters
        DATA_PATH = (
            "C:/Users/pabuehle/Desktop/ComputerVision/data/tiny"
        )  # unzip_url(Urls.fridge_objects_tiny_path, exist_ok=True)

        # DNN configuration and learning parameters
        EPOCHS_HEAD = 0
        EPOCHS_BODY = 0
        LEARNING_RATE = 1e-4
        DROPOUT_RATE = 0.5
        BATCH_SIZE = (
            4
        )  # 16   #batch size has to be lower than nr of training ex
        ARCHITECTURE = models.resnet18
        IM_SIZE = 30  # 300

        # Load images into fast.ai's ImageDataBunch object
        random.seed(642)
        data = (
            ImageList.from_folder(DATA_PATH)
            .split_by_rand_pct(valid_pct=0.5, seed=20)
            .label_from_folder()
            .transform(size=IM_SIZE)
            .databunch(bs=BATCH_SIZE)
            .normalize(imagenet_stats)
        )
        print(
            f"""Training set: {len(data.train_ds.x)} images\nValidation set: {len(data.valid_ds.x)} images"""
        )

        # Init learner
        learn = cnn_learner(
            data,
            ARCHITECTURE,
            metrics=[accuracy],
            callback_fns=[partial(TrainMetricsRecorder, show_graph=True)],
            ps=DROPOUT_RATE,
        )

        # Train the last layer
        learn.fit_one_cycle(EPOCHS_HEAD, LEARNING_RATE)
        learn.unfreeze()
        learn.fit_one_cycle(EPOCHS_BODY, LEARNING_RATE)

        # Build multiple sets of comparative images from the validation images
        comparative_sets = comparative_set_builder(
            data.valid_ds, num_sets=1000, num_negatives=99
        )
        print(f"Generated {len(comparative_sets)} comparative image sets.")

        median_rank = similarity_accuracy(data, learn)
        print(f"The positive example ranks {median_rank}")

    # -------------------------------------------------------
    #  PARAM SWEEPER CODE
    # -------------------------------------------------------
    else:
        DATA = [
            unzip_url(Urls.fridge_objects_tiny_path, exist_ok=True)
        ]  # , unzip_url(Urls.fridge_objects_watermark_path, exist_ok=True)]
        REPS = 1  # 3
        LEARNING_RATES = [1e-5]  # [1e-3, 1e-4, 1e-5]
        IM_SIZES = [30]  # , 30]
        EPOCHS = [0]

        # Init sweeper
        sweeper = ParameterSweeper()
        sweeper.update_parameters(
            learning_rate=LEARNING_RATES, im_size=IM_SIZES, epochs=EPOCHS
        )

        # Run sweeper
        df = sweeper.run(
            datasets=DATA, reps=REPS, accuracy_hook=similarity_accuracy
        )
        print(df)

        # Show results
        df = clean_sweeper_df(df)
        acc1 = df.mean(level=(1, 2)).T
        print(acc1)
