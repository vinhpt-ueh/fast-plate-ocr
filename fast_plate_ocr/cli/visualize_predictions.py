"""
Script for comparing OCR model predictions with ground truth for license plate images.
"""

import logging
import pathlib

import click
import cv2
import keras
import numpy as np

from fast_plate_ocr.train.model.config import load_config_from_yaml
from fast_plate_ocr.train.utilities import utils
from fast_plate_ocr.train.utilities.utils import postprocess_model_output

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


@click.command(context_settings={"max_content_width": 120})
@click.option(
    "-m",
    "--model",
    "model_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path),
    help="Path to the saved .keras model.",
)
@click.option(
    "--config-file",
    required=True,
    type=click.Path(exists=True, file_okay=True, path_type=pathlib.Path),
    help="Path pointing to the model license plate OCR config.",
)
@click.option(
    "-d",
    "--img-dir",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=pathlib.Path),
    help="Directory (e.g., 'train') containing sub-folders with images to make predictions from.",
)
# @click.option(
#     "-l",
#     "--low-conf-thresh",
#     type=float,
#     default=0.35,
#     show_default=True,
#     help="Threshold for displaying low confidence characters.",
# )
# @click.option(
#     "-f",
#     "--filter-conf",
#     type=float,
#     help="Display plates that any of the plate characters are below this number.",
# )
@click.option(
    "--ground-truth-file",
    required=True,
    type=click.Path(exists=True, file_okay=True, path_type=pathlib.Path),
    help="Path to the ground truth file (e.g., 'test_results.txt').",
)
def compare_with_ground_truth(
    model_path: pathlib.Path,
    config_file: pathlib.Path,
    img_dir: pathlib.Path,
    ground_truth_file: pathlib.Path,
):
    """
    Compare OCR model predictions with ground truth for the first image in each sub-folder.
    
    The script processes the 'img_dir' directory (e.g., 'train'), which contains sub-folders
    (e.g., 'track0091'), each with multiple PNG images. It predicts the license plate from
    the first image of each sub-folder and compares it with the corresponding ground truth
    from 'ground_truth_file', where each line is a license plate (e.g., 'MLS5511').
    """
    # Load configuration and model
    config = load_config_from_yaml(config_file)
    model = utils.load_keras_model(
        model_path, vocab_size=config.vocabulary_size, max_plate_slots=config.max_plate_slots
    )

    # Read ground truths from the file
    with open(ground_truth_file, 'r') as f:
        ground_truths = [line.strip() for line in f]

    # Get sorted list of sub-folders
    subfolders = sorted([f for f in img_dir.iterdir() if f.is_dir()])

    # Validate that the number of sub-folders matches the number of ground truths
    if len(subfolders) != len(ground_truths):
        logging.error("Number of sub-folders does not match number of ground truths")
        return

    # Track correct predictions
    correct = 0

    # Process each sub-folder
    for i, subfolder in enumerate(subfolders):
        # Get the first PNG image in the sub-folder
        image_files = list(subfolder.glob('*.png'))
        if not image_files:
            logging.warning(f"No PNG images found in {subfolder}")
            continue

        first_image_path = image_files[0]
        img = cv2.imread(str(first_image_path),cv2.IMREAD_GRAYSCALE)
        if img is None:
            logging.error(f"Failed to load image {first_image_path}")
            continue

        def adjust_gamma(image, gamma=1.5):
            invGamma = 1.0 / gamma
            # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, table)
        
        gamma_corrected = adjust_gamma(img, gamma=1.5)

        # cv2.imshow('Gamma Corrected', gamma_corrected)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Preprocess the image
        img = cv2.resize(gamma_corrected, (config.img_width, config.img_height), interpolation=cv2.INTER_LINEAR)
        img = np.expand_dims(img, -1)
        # image = cv2.resize(image, (config.img_width, config.img_height))
        x = np.expand_dims(img, 0)

        # Make prediction
        prediction = model(x, training=False)
        prediction = keras.ops.stop_gradient(prediction).numpy()
        plate, probs = postprocess_model_output(
            prediction=prediction,
            alphabet=config.alphabet,
            max_plate_slots=config.max_plate_slots,
            vocab_size=config.vocabulary_size,
        )

        # Compare with ground truth
        gt = ground_truths[i]
        if plate == gt:
            # logging.info(f"Correct prediction for {subfolder}: {plate}")
            correct += 1
        else:
            logging.info(f"Incorrect prediction for {subfolder}: predicted {plate}, ground truth {gt}")

    # Log accuracy
    accuracy = correct / len(subfolders) if subfolders else 0
    logging.info(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    compare_with_ground_truth()