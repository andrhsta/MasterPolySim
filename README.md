# MasterPolySim

This repository contains code developed for my Master's thesis at the Norwegian University of Science and Technology (NTNU) in Trondheim, marking the conclusion of a five-year integrated Master's program in Nanotechnology. 

The work was performed under the supervision of Rita de Sousa Dias at the Department of Physics and co-supervision by Kai Sandvold Beckwith at the Department of Biomedical Laboratory Science. 

---

A central feature of nuclear architecture is the segregation of chromatin into transcriptionally active (A) and inactive (B) compartments, as observed in both Hi-C and chromatin tracing experiments. This project explores whether such large-scale organization can emerge from a simplified coarse-grained polymer model with minimal interaction rules.

Chromosome 2 in IMR90 cells is represented as a heteropolymer of A and B monomers, defined from eigenvector decomposition of Hi-C data. The model includes a general attractive interaction between monomers, supplemented by an enhanced B–B interaction to capture preferential clustering of inactive chromatin regions.

The objective is to assess whether these minimal physical assumptions are sufficient to reproduce experimentally observed chromatin compartmentalization.

---

The repository contains _GPUPolymerSimPipeline_, a framework consisting of five modules that together form the polymer simulation workflow from initialization to analysis and visualization. The work builds on the _[polychrom](https://github.com/open2c/polychrom)_ framework developed by the Mirny Group, which is used in the _Trajectories_ module for polymer simulation setup and initialization. The _ExperimentalData_ module includes code provided by the co-supervisor for handling and processing experimental datasets. The remaining  modules, _LoadingSequence, Analysis_ and _Visualization_, are developed specifically for this thesis. 

In addition to the pipeline, the repository includes two example scripts that demonstrate how previously generated simulation trajectories can be analyzed and visualized. These examples provide a streamlined and efficient workflow for exploring simulation results.
