## Future Roadmap & Evolution Scenarios

The following features and architectural enhancements are planned for future releases to transition the project from an MVP into an enterprise-grade Earth Observation (EO) platform:

### 1. Physics-Informed Fuel Moisture Modeling
* **Explicit Evapotranspiration Integration:** Implement the Penman-Monteith equation to dynamically calculate the hourly drying rate of forest fuels between satellite passes, replacing implicit time-lag features with an explicit synthetic moisture index.
* **Live Micro-Climate Simulation:** Adjust regional meteorological data based on local canopy coverage and terrain shadow models.

### 2. Deep Tech & Edge AI Adaptation (Orbital Computing)
* **Model Quantization & ONNX Conversion:** Optimize the machine learning inference engine into a lightweight ONNX runtime package, reducing its memory footprint by up to 75%.
* **Edge Deployment Simulation:** Architect a decoupled, specialized inference microservice designed to be uploaded directly onto payload computers of next-gen smart satellites (e.g., Intel Myriad or FPGA hardware) for near-real-time onboard wildfire detection.

### 3. Transition to Multimodal Computer Vision
* **Patch-Based Semantic Segmentation:** Transition from pixel-by-pixel tabular processing to a fully convolutional architecture (e.g., U-Net or SegNet).
* **Spatial Context Awareness:** Train the neural network on multi-channel raster patches (64x64 or 128x128 pixels) to let the AI analyze spatial patterns, boundaries, and adjacent high-risk zones instead of isolated pixels.

### 4. Post-Ignition Fire Propagation Simulation
* **Vector Field Routing:** Implement cellular automata or dynamic vector models to simulate wildfire spreading paths based on real-time wind vectors, terrain slopes, and fuel connectivity once an active fire spot is confirmed.