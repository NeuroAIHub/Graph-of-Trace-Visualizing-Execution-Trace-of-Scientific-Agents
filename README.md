# Graph of Trace

<p align="center">
  <img src="./figures/Framework.png" alt="Framework of Graph of Trace">
  <br>
  <em>Framework of Graph of Trace</em>
</p>

## Introduction

**Graph of Trace** is a monitoring and visualization framework that records fine-grained execution events and organizes them into a directed graph, making agent workflows explicit as they proceed.

You can watch our **demo video** (`demo.mp4`) to get a quick overview of Graph of Trace.

### ✨ Features
- Real-time rendering
- MCP-enabled
- Agent trajectory level

This repo contains the complete **Graph of Trace** source code, along with the packaged MCP tool `build_trace` and MCP server.

## Quick Start
**Try the online demo here →**  
👉 [Graph of Trace Online Demo](http://8.145.42.208:4500)

1. Open the demo link above — you’ll see a modified [OpenHands](https://docs.openhands.dev/overview/introduction) interface:

   <p align="center">
     <img src="./figures/openhands_init.jpeg" alt="OpenHands initial interface">
   </p>

2. Start a new conversation and wait for the agent to become ready.

3. You can find Graph of Trace on the right side.

   <p align="center">
     <img src="./figures/openhands_got.jpg" alt="Graph of Trace in OpenHands">
   </p>

You can copy-paste the example prompt below to interact with the agent.  
**Graph of Trace** will automatically refresh as the agent executes the task.

```
Please base your analysis on the SEED dataset and complete the following research tasks under a consistent experimental and evaluation framework:
### Background and motivation：
1.Deep Learning method may behave better than traditional methods.
2.DE feature may carry more information in EEG decoding task.
3.How to select the critical channels and frequency bands and how to evaluate selected pools of electrodes have not been fully investigated yet.
### Research Goals：
1.Identify which EEG frequency bands carry the most discriminative information for three-class emotion recognition.
 Assess and compare emotion-classification performance using features extracted from individual frequency bands (delta, theta, alpha, beta, gamma), selected band combinations, and the full-band representation.

2.Determine whether a deep belief network provides a measurable advantage over shallow learning methods for EEG-based emotion recognition.
Compare the performance of a deep belief network with linear SVM, L2-regularised logistic regression, and k-nearest neighbors under identical data preprocessing, feature extraction, and evaluation protocols.

3.Identify which EEG feature set conveys the most discriminative information for emotion classification.
Evaluate and compare power spectral density, differential entropy, and multiple asymmetry-based feature sets using session-level emotion-classification accuracy and variability across repeated experimental sessions.

4.Identify the minimal electrode montage that retains emotion-discrimination capability equivalent to the full 62-channel setup.
First, rank all EEG channels in descending order according to their information content. Then, progressively select the Top-4, Top-6, Top-9, Top-12, and the full 62-channel sets for experimental evaluation and comparative analysis.
```

## Notes
- You can adjust the layout​ by **dragging**​ and **zooming** on the canvas.
- Click on a **node** to view its detailed information.


