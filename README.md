
# Vehicle damage detection with Mask R-CNN
We are asked to create a jupyter notebook to train a segmentation model for damage detection based on the given training data.
The following repository provides an implementation of the Mask R-CNN[Mask R-CNN](https://github.com/matterport/Mask_RCNN), training and evaluating a semantic segmenation model for damage detection on vehicles.

## Approach

## Requirements
The necessary requirements are specified in [requirements.txt](https://github.com/lucabnf/damage-detection/blob/master/requirements.txt). To run the following program on your own machine: 
1. Clone this repository
2. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
3. Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ``` 
4. Download pre-trained weights on COCO dataset
   ```bash
   wget "https://github.com/matterport/Mask_RCNN/releases/download/v1.0/mask_rcnn_coco.h5"
   ```

The training and the evaluation of the model is presented in the jupyter notebook [train_and_evaluate.ipynb](https://github.com/lucabnf/damage-detection/blob/master/train_and_evaluate.ipynb).

## Results

## Insights

## Citation
```
@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}
```