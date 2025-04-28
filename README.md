# Trajectory Analysis of Pedestrian Activities in University Campus

Group project developed for the Image Processing and Computer Vision course at Instituto Superior Técnico in 2024/25.

## Introduction
This project focuses on the analysis of pedestrian trajectories in a university campus setting, using computer vision techniques. Trajectory analysis enables the understanding of motion patterns and supports applications such as anomaly detection, traffic management, human-robot interaction, and advanced driver assistance systems.

We developed a pedestrian detection and tracking system based on conventional feature-based approaches, aiming to collect rich visual and statistical information from video sequences.

The project addressed key challenges such as occlusions, background clutter, appearance variability, and environmental conditions.

## Dataset
- **Dataset Used:** PETS 2009 S2.L1 dataset (View001 sequence).
- **Details:**
  - Sparse crowd scenarios (isolated pedestrians).
  - 795 frames available for analysis.
- **Ground Truth:**
  Provided via bounding boxes in the `gt.txt` file.

## Methodology
The project was divided into several technical objectives:

1. **Bounding Box Visualization:**
   - Reading ground truth data and drawing pedestrian bounding boxes for each frame.

2. **Detection and Tracking:**
   - Implementing a pedestrian detector and tracking moving individuals across frames.
   - Initial tracking allows label switching; later stages enforce consistent IDs.

3. **Trajectory Visualization:**
   - Plotting pedestrian movement dynamically over time.

4. **Trajectory Mapping:**
   - Generating static and dynamic heatmaps to represent occupancy and movement density.

5. **Clustering of Trajectories:**
   - Using the Expectation-Maximization (EM) algorithm for statistical clustering of movement patterns.

6. **Performance Evaluation:**
   - Measuring algorithm performance with:
     - Success plots based on Intersection over Union (IoU) scores.
     - Analysis of False Positives (FP) and False Negatives (FN).

7. **Comparison with Deep Learning:**
   - Comparing the traditional method with state-of-the-art pedestrian detection using a deep neural network baseline.

## Evaluation Metrics
- **Success Plot:** Based on IoU scores across thresholds.
- **Error Analysis:** Identification and illustration of FP and FN cases.

## Team
- Beatriz Paulo - [@beatrizp03](https://github.com/beatrizp03)
- Leonor Fortes - [@leonorf03](https://github.com/leonorf03)
- Diogo Costa - [@dcaoc03](https://github.com/dcaoc03)
