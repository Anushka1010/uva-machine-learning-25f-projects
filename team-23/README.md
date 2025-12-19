# Engine Degradation Classification (NASA N-CMAPSS)

## Overview
This project implements a **binary classification pipeline** to detect **imminent turbofan engine failure** using the NASA **N-CMAPSS** dataset.  
Instead of predicting exact Remaining Useful Life (RUL), the model classifies each engine cycle as:

- **Healthy**: RUL > 5  
- **Near Failure**: RUL ≤ 5  

This framing focuses on **early failure detection**, which is more actionable in real-world maintenance scenarios.

---

## Video Submission
📺 **Project Walkthrough (YouTube):**  
https://www.youtube.com/watch?v=jBvGLsg9_vc

---

## Dataset
- **Source:** NASA N-CMAPSS (DS01-005)
- **Format:** HDF5 (`.h5`)
- **Scale:** ~5 million rows (engine-cycle time series)
- **Features:**
  - Engine metadata (unit, cycle)
  - Operating conditions
  - Physical sensor measurements
  - Virtual sensor signals
- **Label:** Remaining Useful Life (RUL)

