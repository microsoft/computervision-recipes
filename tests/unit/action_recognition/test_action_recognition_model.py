# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This test is based on the test suite implemented for Recommenders project
# https://github.com/Microsoft/Recommenders/tree/master/tests


from utils_cv.action_recognition.model import VideoLearner


def test_VideoLearner(ar_milk_bottle_dataset) -> None:
    """ Test VideoLearner Initialization. """
    learner = VideoLearner(ar_milk_bottle_dataset, num_classes=2)
    learner.fit(lr=0.001, epochs=1)
    learner.evaluate()


def test_VideoLearner_using_split_file(
    ar_milk_bottle_dataset_with_split_file,
) -> None:
    """ Test VideoLearner Initialization. """
    learner = VideoLearner(
        ar_milk_bottle_dataset_with_split_file, num_classes=2
    )
    learner.fit(lr=0.001, epochs=1)
    learner.evaluate()
