# GenMol: AI-Driven Molecular Generation for JAK2 Inhibitor Discovery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![AMBER](https://img.shields.io/badge/AMBER-22-red.svg)](https://ambermd.org/)

## Overview

This repository contains the complete computational workflow and data for our comparative study evaluating AI-driven molecular generation against classical fingerprint-based virtual screening for discovering novel JAK2 inhibitors. The study directly compares **GenMol** (discrete diffusion-based de novo generation) with **infiniSee Analog Hunter** (ECFP4/Tanimoto screening of 4.7 million compounds), using baricitinib (BARI) as a reference scaffold.

### Key Findings

- AI-generated molecules show superior drug-likeness (QED: 0.521-0.876)
- AI dominates MACCS-based ranking (11 of 13 top compounds)
- Top AI candidate (GM1) achieves better docking (−8.2 kcal/mol) than best classical hit (−6.3 kcal/mol)
- MD simulations reveal weaker thermodynamic stability for AI-generated candidates vs. reference
- Generative AI excels at novel scaffold proposal; rigorous biophysical validation remains essential

## Repository Structure
