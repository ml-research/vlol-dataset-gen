# The Three Dimensional Michalski Train Generator

Repository for the three-dimensional Michalski train generator introduced in
Bringing the third dimension into the Michalski train problem. 


The Michalski train problem [[1]](#1) proposed by Ryszard S. Michalski in 1980 represents
one of the most prominent challenges in the domain of relational learning. The problem
constitutes a set of trains, which are composed of a wide variety of properties and labelled
into the two categories eastbound and westbound trains. Now it is up to the viewer
to ascertain a classification hypothesis governing what kinds of trains head eastbound
and what kinds head westbound. The conjectured hypothesis should accomplish both
assigning correct labels while retaining the highest possible degree of simplicity. It is
certainly easy to find a classification hypothesis satisfying a given set of Michalski trains,
e.g. learning the trains by heart. However, due to the great number of train properties, it
is a major challenge to find the most simple and general decision hypothesis.

This study takes the Michalski train problem even a step further by introducing a three-dimensional
Michalski train generator. An image generator that allows to establish versatile datasets for deep image understanding,
relational and analogical (grounded) visual reasoning.
This research aims to contribute an ILP dataset which allows to incorporate the complex rule-based logic of the Michalski
train problem and establish a modern three-dimensional dataset creating a problem of inductive inference.
The resulting datasets allow for diagnostic insights into the method’s decision-making process as well as its
capabilities of rule-based learning.

## Instructions setting up the docker container

This is a very brief list of instructions on how to generate a three-dimensional Michalski train dataset.
CUDA 11.3+ must be installed.

For easier handling:
create a screen: screen -S train_generator

Then:
1. cd to TrainGenerator folder
2. docker build -t blender_train_generator -f Dockerfile .
3. docker run --gpus device=0 --shm-size='20gb' -v $(pwd):/home/workdir blender_train_generator python3 main.py


## Generating Images

First we generate train descriptions for the whole dataset, 
subsequently we render images for the individual train descriptions and generate their ground truth information.
The train generator provides a wide range of settings allowing to adapt to the given requirements.
The default output location is TrainGenerator/output/.


### Settings

The following settings are available, the input typ and default settings are noted in parentheses:
- dataset_size (int, 10,000) -> Size of the dataset we want to create
- index_start (int, 0) -> start rendering images at index (index_start)
- index_end (int, None) -> stop rendering images at index (does not render index_end).
If None the train generator stops rendering at dataset_size.
The start and stop indices allow for parallel execution of the code thus parallel rendering of images of the same dataset.

- train_type (str, MichalskiTrains) -> The train type we want to generate. Either 'MichalskiTrains' or 'RandomTrains'
- background_scene (str, base_scene) -> Scene in which the trains are set: 'base_scene', 'desert_scene', 'sky_scene' or 'fisheye_scene'

- with_occlusion (bool, False) -> Whether to include train angles which might lead to occlusion of the individual train attributes
- save_blender (bool, False) -> Whether the blender scene is saved
- high_res (bool, False) -> whether to render the images in high resolution (1920x1080) or standard resolution (480x270)
- gen_depth (bool, False) -> Whether to generate the depth information of the individual scenes
- replace_raw (bool, False) -> If the train descriptions for the dataset are already generated shall they be replaced?
- replace_existing_img (bool, False) -> Check if the image is already rendered for the individual indices.
If there is already an image generated for a specific index shall do you want to replace it?


### Classification rule
If Michalski trains are generated, the train generator allows the creation of a labeled train dataset.
Therefore, the labels are derived from the prolog classification rule noted in TrainGenerator/classification_rule.pl.
By default, we resort to the classification rule known as 'Theory X' which is defined as follows:

    There is either a short, closed car, or a car with a circular load somewhere behind a car with a triangular load.

It Prolog the rule can be expressed as follows:

    eastbound([Car│Cars]):-
    (short(Car), closed(Car));
    (has_load0(Car,triangle), has_load1(Cars,circle));
    eastbound(Cars).

In FOL it can be noted as follows:

eastbound(Train) &vDash;
&exist; Car_1, Car_2 has-car(Train, Car_1) &and; has-car(Train, Car_2) &and;
((short(Car_1) &and; closed(Car_1)) &or;
(has-load(Car_1,golden-vase) &and; has-load(Car_2,barrel) &and; 
 somewhere-behind(Train, Car_2, Car_1)))

The classification rule noted in classification_rule.pl can be adjusted according to the requirements.
This allows us to increase or decrease the complexity of the rule-based problem incorporated into the generated dataset.
Herby the classification rule must be expressed in the Prolog description language using the provided descriptors.
Furthermore, by resorting the defined descriptors, it is also possible to define and apply new descriptors.

### Dataset structure
Once the dataset is generated we can find it in the folder TrainGenerator/output/. The dataset is structured as follows:
```
output
└───MichalskiTrains
│   │
│   └───base_scene
│   │   │
│   │   └───blendfiles
│   │   │     │0_m_train.blend
│   │   │     │...
│   │   │
│   │   └───depths
│   │   │     │0_m_train.png
│   │   │     │...
│   │   │
│   │   └───images
│   │   │     │0_m_train.png
│   │   │     │...
│   │   │
│   │   └───scenes
│   │         │0_m_train.json
│   │         │...
│   │
│   └───desert_scene
│   │   │...
│   │
│   │...
│
└───RandomTrains
    │   ...
```



## References
<a id="1">[1]</a> 
Ryszard S. Michalski. “Pattern Recognition as Rule-Guided Inductive Inference”. In:
IEEE Transactions on Pattern Analysis and Machine Intelligence PAMI-2.4 (1980),
pp. 349–361. doi: 10.1109/TPAMI.1980.4767034.
## Citation
If you find this dataset useful in your research, please consider citing:
> @unpublished{HelffTrains,
    title={Bringing the third dimension into the Michalski train problem},
    author={Helff, Lukas and Stammer, Wolfgang and Kersting, Kristian},
    note= {unpublished},
    year={2022}
    }
> [//]: # (    journal={arXiv preprint arXiv:2011.12854},)