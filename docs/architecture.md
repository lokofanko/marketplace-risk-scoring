# Architecture

This document will describe the planned marketplace risk scoring architecture.

## Planned System Boundary

- Marketplace backend: listing workflow, user state, moderation queue, and persistence.
- ML risk service: feature handling, model inference, threshold policy, and prediction metadata.
- Offline ML workflows: synthetic data generation, training, evaluation, and model artifact production.

Implementation details will be added in later milestones.
