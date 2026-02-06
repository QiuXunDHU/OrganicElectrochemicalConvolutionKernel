## 📊 Performance Benchmarking

To validate the efficacy of the **Organic Electrochemical Convolutional Kernel (OECK)**, we conducted extensive comparative experiments against standard digital operators using the UC Merced Land Use Dataset.

### OECK vs. Standard Digital Operators

We compared the performance of neural networks initialized with **OECK layers** (fixed weights derived from OECT physics) against those using **Standard Learnable Convolutions** (randomly initialized and digitally trained). The comparison covers multiple backbone architectures, including **ResNet** and **Swin Transformer**.

**Key Comparative Analysis:**

* **Comparable Accuracy**: Our experiments demonstrate that replacing the first learnable convolutional layer with a fixed OECK layer yields classification accuracy comparable to fully digital baselines. This indicates that the physical transfer characteristics of OECTs are naturally effective for visual feature extraction.
* **Hardware-Intrinsic Efficiency**: Unlike standard kernels that require computational resources for weight updates during training, the OECK layer utilizes fixed physical parameters. This simulates a "sensor-compute" synergy where initial processing is handled by the device physics.
* **Robustness Across Architectures**: The OECK approach maintains performance stability across both CNN-based (e.g., ResNet-18) and Transformer-based (e.g., Swin-T) architectures, proving the universality of OECT-based feature extraction.
