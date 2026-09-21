"""Reproducible training-only augmentations; validation never calls this module."""
import cv2
import numpy as np


def apply(image, mask, rng, config):
    if any(not 0 <= config.get(k, 0) <= 1 for k in ["weather_p", "equipment_p"]):
        raise ValueError("Augmentation probabilities must be between 0 and 1")
    image, mask = np.array(image, copy=True), np.array(mask, copy=True)
    if config.get("flip", False) and rng.random() < .5:
        image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()
    # Independent Bernoulli groups; at most one transform per group.
    if rng.random() < config.get("weather_p", 0):
        method = rng.choice(["fog", "rain", "snow"])
        x = image.astype(float)
        if method == "fog":
            alpha = rng.uniform(.1, .4)
            x = (1-alpha)*x + alpha*230
        elif method == "rain":
            overlay = np.zeros_like(image)
            for _ in range(max(1, image.shape[0]*image.shape[1]//500)):
                y, z = rng.integers(0, image.shape[0]), rng.integers(0, image.shape[1])
                cv2.line(overlay, (int(z), int(y)), (int(z)+2, int(y)+10), (120,)*3, 1)
            x += overlay
        else:
            snow = rng.random(image.shape[:2]) < .015
            x[snow] = 245
        image = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < config.get("equipment_p", 0):
        method = rng.choice(["noise", "blur", "brightness"])
        if method == "noise":
            image = np.clip(image.astype(float)+rng.normal(0, 8, image.shape), 0, 255).astype(np.uint8)
        elif method == "blur":
            image = cv2.GaussianBlur(image, (5, 5), 1)
        else:
            image = np.clip(image.astype(float)*rng.uniform(.65, 1.35), 0, 255).astype(np.uint8)
    return image, mask
