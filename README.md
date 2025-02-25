# Road Marking Detection

This project implements a deep learning pipeline for detecting road markings (e.g., dashed lines, solid lines, triangles, blocks) using aerial imagery and geospatial data. It leverages a U-Net model with a ResNet50 backbone, trained on preprocessed raster and vector data.

## Features

- **Data Preprocessing**:
  - Generate 224x224 tiles from raster data.
  - Mask road images using shapefiles.
  - Convert vector annotations to raster masks.

- **Class Labeling**:
  - Assign class labels to road markings:
    - Block: 1
    - Dash: 2
    - Solid: 3
    - Triangle: 4
  - Create masks
- **Model Training**:
  - Utilize PyTorch and segmentation-models-pytorch for training a segmentation model.

- **Performance Evaluation**:
  - Assess model using confusion matrices, precision, recall, and F1 scores.

- **Prediction Generation**:
  - Produce predictions on the entire dataset.

## Installation


   1. **Clone the Repository:**
   ```bash
   git clone https://github.com/xxxx/road-marking-detection.git
   cd road-marking-detection
   ```
   2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```  
   3. **Prepare Data**
   
      Before running the preprocessing scripts, ensure you have the following files and directories set up in the `data/` directory. Each file and directory is linked below for clarity.
         Files and Directories

        1. **Raster File**:
           - Place the aerial imagery file in the `data(/data/D028_lufo_2022_Almere.ecw)` directory:
             - **[D028_lufo_2022_Almere.ecw](/data/D028_lufo_2022_Almere.ecw):** 2022 aerial imagery of Almere in ecw.
        2. **Shapefiles**:
             - [Apeldoorn_grids.shp]((/data/Apeldoorn_grids.shp)): Grid polygons for tiling.
             - [Roads_Almere.shp]((/data/Roads_Almere.shp): Road outlines.

        3. **Annotation Files**:
           - Place the annotation files for road markings in the `data/annotation/` directory:
             - [DashedLines.shp]((/data/DashedLines.shp): Dashed line annotations.
             - [SolidLines.shp]((/data/SolidLines.shp): Solid line annotations.
             - [TriangleMark.shp]((/data/TriangleMark.sh): Triangle marking annotations.
             - [BlockMark.shp]((/data/BlockMark.shp): Block marking annotations.
             
   Ensure [QGIS](https://qgis.org/en/site/) is installed and that `qgis_process` is available in your command line for preprocessing.




Usage

1. Preprocessing
Convert raster data into tiles, mask with road shapefiles, and rasterize markings:

   ```bash
   python src/preprocess.py
   ```  



Output:
    Tiled images in data/tiles

2. Dataset Preparation
Assign class values and combine images:

   ```bash
   python src/dataset.py
   ```  

Output:
Class-assigned masks in data/training/masks
Combined images in data/training/images


3. Training
Train the U-Net model:

   ```bash
   python src/model.py
   ```  
Output: Trained model saved as bestUNET_model.pth in the root directory.

4. Evaluation
Evaluate the model and visualize results:


   ```bash
   python src/evaluate.py
   ```  
Output: Prints accuracy, precision, recall, F1 scores, and displays a confusion matrix. Visualizes a batch of predictions.


5. Prediction
Generate predictions on new images:


   ```bash
   python src/predict.py
   ```  
Input: Place new .tif images in data/clipped_Almere or data/clipped_Apeldoorn.
Output: Predictions saved in data/predictions/Almere or data/predictions/Apeldoorn.

Dataset
The code uses:

Raster: D028_lufo_2022_Almere.ecw (2022 aerial imagery of Almere).
Shapefiles:
Apeldoorn_grids.shp (grid polygons)
Roads_Almere.shp (road outlines)
DashedLines.shp, SolidLines.shp, TriangleMark.shp, BlockMark.shp (road markings)
Adjust file paths in scripts if your data has different names or locations.
Model Details
Architecture: U-Net with ResNet50 encoder (pretrained on ImageNet).
Classes: 5 (background + 4 road marking types: Block, Dash, Solid, Triangle).
Input Size: 224x224 pixels.
Training: Adam optimizer, CrossEntropyLoss, early stopping with patience of 10 epochs.
Results
Metrics: Accuracy, Precision, Recall, F1 Score (see evaluate.py output).
Visualizations: Ground truth vs. predicted masks with color-coded classes.

License
This project is licensed under the MIT License.

