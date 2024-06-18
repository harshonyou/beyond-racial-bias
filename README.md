# Beyond Racial Bias: Advancing Fairness in In-The-Wild Facial Reconstruction

This repository hosts the source code and accompanying documentation for the dissertation project titled "Beyond Racial Bias: Advancing Fairness in In-The-Wild Facial Reconstruction" by Harsh Mohan, under the supervision of Dr. Patrik Huber at the University of York.

## Project Overview

This project addresses the critical challenge of racial bias in 3D face reconstruction, which is an issue of both technical and ethical significance. Our work aims to develop a more equitable and accurate method for 3D facial reconstruction by integrating a learned illumination prior, which is trained independently from any face model. This approach seeks to mitigate the biases inherent in traditional 3D Morphable Models (3DMMs) that often fail to represent diverse facial features accurately under various lighting conditions.

## Motivation

The motivation for this project stems from the increasing integration of 3D face reconstruction technologies in everyday applications, including facial recognition, virtual insurance, and digital media. These technologies are becoming ubiquitous, and it is critical to ensure they are inclusive and do not perpetuate systemic biases.

## Installation

To set up the project environment to run the code and experiments:

```bash
git clone https://github.com/harshonyou/dissertation
cd dissertation
conda create --name your-env-name --file requirements.txt
conda activate your-env-name
```

## Usage

```bash
python demos/custom.py /path/to/image.jpg cuda
```

This command will perform the 3D face reconstruction on the specified input image and output the reconstructed model.

## Key Components

- **Data Preprocessing:** Normalization, landmark detection, and segmentation of high-resolution images.
- **Model Development:** Integration of FLAME model for dynamic facial structuring, BalancedAlb model for diverse albedo texturing, and RENI++ model for handling complex lighting interactions.
- **Rendering and Optimization:** Initial rendering using Lambertian reflectance and iterative refinement of model parameters through "Analysis by Synthesis".

## Results

The results demonstrate significant improvements in the accuracy and fairness of 3D facial reconstructions. The learned illumination model successfully mitigates racial bias, providing more consistent and equitable representations across different skin tones.

## Contributions

- Harsh Mohan - Project conception, research, and software development.
- Dr. Patrik Huber - Supervision, domain expertise, and guidance.

## Acknowledgements

Special thanks to Dr. Will Smith, James Gardner, and Stephen Da’Prato-Shepar for their insights and feedback on this project, and to the faculty and staff at the University of York's Department of Computer Science for their support.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
