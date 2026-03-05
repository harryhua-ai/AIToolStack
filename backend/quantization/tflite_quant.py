"""
NE301 device quantization script (from STM32AI example, comments preserved)
"""

import os
import sys
import logging
import warnings
import random
import pathlib

import hydra
import tqdm
import cv2
import tensorflow as tf
import numpy as np
from munch import DefaultMunch
from omegaconf import OmegaConf, DictConfig

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def setup_seed(seed: int):
    """Fix random seed for reproducibility"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)  # tf cpu fix seed


def get_config(cfg: DictConfig):
    config_dict = OmegaConf.to_container(cfg)
    configs = DefaultMunch.fromDict(config_dict)
    return configs


@hydra.main(version_base=None, config_path="", config_name="user_config_quant")
def main(cfg: DictConfig) -> None:
    """Quantization entry point"""

    def representative_data_gen():
        if cfg.quantization.fake is True:
            for _ in tqdm.tqdm(range(5)):
                data = np.random.rand(
                    1,
                    cfg.model.input_shape[0],
                    cfg.model.input_shape[1],
                    cfg.model.input_shape[2],
                )
                yield [data.astype(np.float32)]
        else:
            representative_ds_path = cfg.quantization.calib_dataset_path
            # Get max calibration images from config, default to 200 to prevent memory issues
            max_calib_images = cfg.quantization.get("max_calib_images", 200)
            
            # Get all image files
            image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
            list_of_files = [
                f for f in os.listdir(representative_ds_path)
                if f.lower().endswith(image_extensions)
            ]
            
            total_files = len(list_of_files)
            
            # Limit the number of images to prevent memory issues
            if total_files > max_calib_images:
                # Randomly sample to get diverse calibration data
                # Note: setup_seed(42) is already called in main(), but we ensure reproducibility here
                random.seed(42)  # Use fixed seed for reproducibility
                list_of_files = random.sample(list_of_files, max_calib_images)
                print(f"Using {max_calib_images} randomly sampled images from {total_files} total image files for calibration")
            else:
                print(f"Using all {total_files} images for calibration")
            
            # Process images one by one to minimize memory usage
            for image_file in tqdm.tqdm(list_of_files, desc="Processing calibration images"):
                try:
                    image_path = os.path.join(representative_ds_path, image_file)
                    image = cv2.imread(image_path)
                    
                    if image is None:
                        print(f"Warning: Failed to load image {image_file}, skipping")
                        continue
                    
                    # Handle grayscale images
                    if len(image.shape) != 3:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    
                    # Convert BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Resize image
                    resized_image = cv2.resize(
                        image,
                        (
                            int(cfg.model.input_shape[0]),
                            int(cfg.model.input_shape[1]),
                        ),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    
                    # Normalize image
                    image_data = (
                        resized_image / cfg.pre_processing.rescaling.scale
                        + cfg.pre_processing.rescaling.offset
                    )
                    img = image_data.astype(np.float32)
                    image_processed = np.expand_dims(img, 0)
                    
                    # Yield processed image and clear references to free memory
                    yield [image_processed]
                    
                    # Explicitly delete large arrays to free memory immediately
                    del image, resized_image, image_data, img, image_processed
                    
                except Exception as e:
                    print(f"Warning: Error processing image {image_file}: {e}, skipping")
                    continue

    configs = get_config(cfg)

    setup_seed(42)

    name = cfg.model.name
    print(f"Quantization of {name}")
    uc = cfg.model.uc
    if cfg.quantization.fake is True:
        uc = "fake"
    quant_tag = "quant_pc"
    input_tag = "f"
    output_tag = "f"

    converter = tf.lite.TFLiteConverter.from_saved_model(cfg.model.model_path)

    tflite_models_dir = pathlib.Path(cfg.quantization.export_path)
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    # Configure input/output types
    if cfg.quantization.quantization_input_type == "int8":
        converter.inference_input_type = tf.int8
        input_tag = "i"
    elif cfg.quantization.quantization_input_type == "uint8":
        converter.inference_input_type = tf.uint8
        input_tag = "u"

    if cfg.quantization.quantization_output_type == "int8":
        converter.inference_output_type = tf.int8
        output_tag = "i"
    elif cfg.quantization.quantization_output_type == "uint8":
        converter.inference_output_type = tf.uint8
        output_tag = "u"

    if cfg.quantization.quantization_type == "per_tensor":
        converter._experimental_disable_per_channel = True
        quant_tag = "quant_pt"

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    
    # Enforce full integer quantization
    print(f"DEBUG: Setting up TFLITE_BUILTINS_INT8")
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.int8]
    converter.experimental_new_converter = True
    
    # Quantize and save
    print(f"DEBUG: Starting converter.convert()... This may take a few minutes.")
    try:
        tflite_model_quantio = converter.convert()
        print(f"DEBUG: converter.convert() successful. Size: {len(tflite_model_quantio)} bytes")
    except Exception as e:
        print(f"DEBUG: converter.convert() FAILED: {e}")
        raise e
    tflite_model_quantio_file = (
        tflite_models_dir
        / f"{name}_{quant_tag}_{input_tag}{output_tag}_{uc}.tflite"
    )
    tflite_model_quantio_file.write_bytes(tflite_model_quantio)

    print(
        f"Quantized model generated: {tflite_model_quantio_file.name}"
    )


if __name__ == "__main__":
    main()
