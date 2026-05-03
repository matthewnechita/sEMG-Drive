# EMG Driving Control Capstone

This repository contains the maintained codebase for a capstone project on EMG-driven vehicle control. The project uses Delsys surface EMG sensors, CNN-based gesture recognition, and CARLA driving scenarios to test whether upper-limb muscle activity can serve as a practical control interface for simple driving tasks.

The repo is organized around the full system. It covers data collection, preprocessing, training, live inference, CARLA control, and evaluation.

## Why This Project Was Built

Traditional driving controls assume that the user can reliably operate a steering wheel, pedals, and other physical interfaces. This project explores a different control pathway: using electromyography (EMG) from the forearms to recognize intentional gestures and convert them into vehicle commands in simulation.

The goal is to evaluate whether a small, well-defined EMG gesture set can support usable vehicle control in a constrained driving environment, and to measure the limits of that approach with both model metrics and driving metrics.

## System

The maintained project state uses a fixed 4-gesture contract:

- `left_turn`
- `right_turn`
- `neutral`
- `horn`

In the current CARLA client, `horn` is repurposed as the reverse request when both arms publish it together.

The active stack is:

- EMG collection with fixed sensor placement
- resampling and filtering
- CNN training with `GestureCNNv2`
- dual-arm realtime inference
- CARLA free-roam and scenario evaluation
- offline, latency, and driving-metric analysis


## Technical Reference

The canonical technical reference for the maintained repo state is:

- [technical_reference.md](/C:/Users/matth/Desktop/capstone/capstone-emg/project_notes/technical_reference.md)

That file is the source of truth for the current sensor placement contract, active training and realtime setup, CARLA integration details, and evaluation structure.
