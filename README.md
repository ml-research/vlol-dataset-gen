# V-LoL<img src="example_images/train.png" width="45"> Dataset Generation


Repository for V-LoL<img src="example_images/train.png" width="20"> generator,
a versatile framework for visual logical learning introduced in [V-LoL: A Diagnostic Dataset for Visual Logical Learning](https://arxiv.org/abs/2306.07743).
V-LoL<img src="example_images/train.png" width="20"> is specifically designed to evaluate the visual logical learning capabilities of machine learning models.
The core idea behind V-LoL is to integrate explicit logical learning tasks from classic symbolic AI benchmarks into visually complex scenes.
By doing so, it creates a unique visual input that retains the challenges and versatility of explicit logic.
This approach aims to bridge the gap between symbolic AI challenges and contemporary deep learning datasets.

Within this repository, you will find the code to generate new V-LoL<img src="example_images/train.png" width="20"> datasets.
The generator allows researchers to easily exchange or modify the logical rules,
thereby enabling the creation of new datasets incorporating novel logical learning challenges.
For more information about V-LoL, please visit our  [Homepage](https://sites.google.com/view/v-lol/home).
Additionally, example datasets are available at [Hugging Face](https://huggingface.co/datasets/AIML-TUDA/v-lol-trains),
allowing you to explore and experiment with pre-existing V-LoL datasets. The accompanying images showcase examples generated by
the V-LoL<img src="example_images/train.png" width="20"> generator, and detailed explanations on how to create new
datasets are provided below.
<div align="center">
  <img src="example_images/michalski_3D.png" width="400"  alt="">
  <img src="example_images/simple_scene.png" width="400"  alt="">
</div>

If you find this dataset useful in your research, please consider citing:

```
@misc{helff2023vlol,
title={V-LoL: A Diagnostic Dataset for Visual Logical Learning},
author={Lukas Helff and Wolfgang Stammer and Hikaru Shindo and Devendra Singh Dhami and Kristian Kersting},
journal={Dataset available from https://sites.google.com/view/v-lol},
year={2023},
eprint={2306.07743},
archivePrefix={arXiv},
primaryClass={cs.AI}
}
```

## Overview

This is a brief guide to the V-LoL<img src="example_images/train.png" width="20"> generation process.
The generation process involves sampling symbolic train representations, deriving the class affiliation using logical class rules, and rendering 3D visual images.
The code allows flexible adjustments of the visual and logical complexity of the generated datasets.
See script parameters (visual and symbolic) for more details. 
Predefined decision rules are available in the `example_rules` folder.
Users can define their own custom decision rule in the `custom_rule.pl` file.
The default output location is `TrainGenerator/output/image_generator/`.
The following image provides a concise overview of V-LoL<img src="example_images/train.png" width="20">.

<div align="center">
  <img src="example_images/main_figure.png" width="600"  alt="">
</div>

## Instructions for setting up the docker container

A docker container can be used to set up the required environment.
Additionally, CUDA 11.3+, docker and nvidia-container-toolkit must be installed to allow the
usage of the docker container and enable image rendering.

For easier handling we recommend to create a screen: `screen -S vlol`

Then:

```bash
cd vlol-dataset-gen
docker build -t vlol .
docker run --gpus device=0 -v $(pwd):/home/workdir vlol python3 main.py
```

## Script parameters

The following settings are available, the corresponding input types and default settings are noted in parentheses:

General:

- `dataset_size` (int, 10,000) -> Size of the dataset you want to create.
- `output_path` (str, 'output/image_generator') -> path to the output directory in which the generated datasets are
  saved.
- `cuda` (int, 0) -> Which GPU to use for rendering. If -1, CPU is used.

Symbolic:

- `classification` (str, 'theoryx') -> The classification rule used for generating the labels of the dataset.
  The following rules are available: 'theoryx', 'easy', 'color', 'numerical', 'multi', 'complex', 'custom'.
  For more detailed information about the rules refer to the rule section.
  Select custom rule to use your own personal rule. Therefore, define your own rule in \'example_rules/custom_rule.pl\'.
- `distribution` (str, 'MichalskiTrains') -> The distribution we want to sample from. Either 'MichalskiTrains' or '
  RandomTrains'. The 'RandomTrains' are sampled from a uniform distribution of the attributes.
  'MichalskiTrains' are sampled according to distributional assumptions defined by Muggleton [[2]](#2).
- `max_train_length` (int, 4) -> The maximum number of cars a train can have.
- `min_train_length` (int, 2) -> The minimum number of cars a train can have.
- `replace_symbolics` (bool, False) -> If the symbolic trains for the dataset are already generated shall they be replaced?
  If false, it allows to use same trains for multiple generation runs.

Visual:

- `visualization` (str, 'Trains') ->  V-LoL offers two distinct visual representations.
  Either as 'Trains' or 'SimpleObjects'. For comparison see images above.
- `background` (str, 'base_scene') -> Scene in which the trains are set: 'base_scene', 'desert_scene', 'sky_scene'
  or 'fisheye_scene'.
- `occlusion` (bool, False) -> Whether to include rotation angles of the train which might lead to occlusion of
  individual parts of the train.

Parallelization:

- `index_start` (int, 0) -> Start rendering images at index (index_start).
- `index_end` (int, None) -> Stop rendering images at index (does not include rendering image[index_end]).
  If None the train generator stops rendering at dataset_size.
- `continue` (bool, True) -> Enables parallel generation of one dataset. Uncompleted/aborted runs will be continued.
  If set to False we start a new run and the images generated in tmp folder from previously uncompleted runs
  (of the same settings) will be deleted.

Rendering settings:

- `save_blender` (bool, False) -> Whether the blender scene is saved.
  Only recommended for small image counts as the blend files can be quite large.
- `high_res` (bool, False) -> Whether to render the images in high resolution (1920x1080) or standard resolution (
  480x270)
- `depth` (bool, False) -> Whether to save depth information of the individual scenes.

The following shows example images of the four background scenes 'base_scene', 'desert_scene', 'sky_scene' and '
fisheye_scene':

<div align="center">
  <img src="example_images/background/base_scene.png" width="400"  alt="">
  <img src="example_images/background/desert_scene.png" width="400"  alt="">
  <img src="example_images/background/sky_scene.png" width="400"  alt="">
  <img src="example_images/background/fisheye_scene.png" width="400"  alt="">
</div>

The start and stop indices parameters allow for parallelization if large datasets need to be generated.
Therefore, you can start multiple docker containers each generation images at different indices of the dataset.
Keep in mind to use different docker containers as the blender engine has problems to render parallel.

## Decision rule

After sampling the symbolic trains we now need to define a classification rule from which we can derive
labels. The different rules allow us to increase or decrease the complexity of the logical learning
challenge that is incorporated into our dataset.
In the `example_rules` folder we provide a selection of predefined rules with varying degrees of complexity.
The rules are expressed in the Prolog description language. In the case you want to define your own personal rules
select and adjust the `example_rules/custom_rule.pl` according to your requirements.
For this you can refer to the predefined set of predicates described in the section below or define und use your own
predicates.
Be aware that defining a very specific decision rules can have a strong influence on the distribution of train
attributes, which in turn can lead to similar images being generated as it might become difficult to create random
variations based on a very specific rule.

By default, we resort to the classification rule known as **'Theory X'** which is defined as follows:

    There is either a short, closed car, or a car with a circular load somewhere behind a car with a triangular load.

In Prolog the rule can be expressed as follows:

    eastbound([Car│Cars]):-
    (short(Car), closed(Car));
    (has_load0(Car,triangle), has_load1(Cars,circle));
    eastbound(Cars).


The other classification are denoted as follows:

- **easy rule:** The train has a short and a long car with the same colour.
- **colour rule:** The train has three differently coloured cars.
- **easy rule:** The train has a short and a long car with the same colour.
- **multi case rule:** The train has either a car with braced walls and 2 loads, or a blue car with 3 loads or a blue
  car with brace walls.
- **numerical rule:** The train has a car where its car position equals its number of payloads which equals its number
  of wheel axis.
- **complex rule:** Either there is a car with a car number which is smaller as its wheel count and smaller as the
  number of loads, or there is a short and a long car with the same colour where the position number of the short car is
  smaller as the wheel count of the long car, or the train has three differently coloured cars.

## Symbolic train representation

The definition of the original Michalski trains heavily relies on their two-dimensional delineation and does not
meet the requirements of a vivid three-dimensional visualization.
Accordingly, we have transformed the original train representation to use more appropriate predicates for our
visualizations (see tables below). In this context, we have exchanged the original predicates in a one-to-one manner.
This facilitates the definition of new classification rules as we can refer to the predicates interchangeably, i.e.
we can resort to the predicates of the original Michalski trains and/or the new predicates for the different
visualizations.
Furthermore, it is not necessary to define new classification rules for the classification of trains depicted in the
different visualizations. However, if you want to create new predicates, you can have a look at our predicate
definitions in `raw/train_generator.pl` and define new predicates accordingly.

Below you will find an overview of the original Michalski train representation
which is expressed by the following Prolog predicates.
While T refers to the whole train as an input, C, C1, C2 refer to a single car.

- Car descriptors
    - has_car(T,C)
    - infront(T,C1,C2)

- Car shape descriptors
    - ellipse(C)
    - hexagon(C)
    - rectangle(C)
    - u_shaped(C)
    - bucket(C)

- Car length descriptors
    - long(C)
    - short(C)

- Car wall descriptor
    - double(C)

- Car roof descriptors (R: roof shape, N: car number)
    - has_roof(C,r(R,N))
    - open(C)
    - closed(C)

- Car wheel count descriptor (W: wheel count, NC: car number)
    - has_wheel(C,w(NC,W))

- Car payload descriptor (Shape: roof shape, NLoad: number of loads)
    - has_load(C,l(Shape,NLoad))
    - has_load0(C,Shape)
    - has_load1(T,Shape)

You can use these values defined below for the predicates defined above:

#### Original Michalski train representation

| Car Number | Car Shape | Car Length | Car Wall | Car Roof | Wheels Num. | Load Shape | Number of loads |
|:----------:|:---------:|:----------:|:--------:|:--------:|:-----------:|:----------:|:---------------:|
|     1      | rectangle |   short    |  single  |   none   |      2      | rectangle  |        0        |
|     2      |  bucket   |    long    |  double  |   arc    |      3      |  triangle  |        1        |
|     3      |  ellipse  |     	      |    		    |   flat   |             |   circle   |        2        |
|     4      |  hexagon  |     		     |   			    |  jagged  |             |  diamond   |        3        |
|            | u_shaped  |     		     |    		    |  peaked  |     		      |  hexagon   |
|    			     |    		     |     		     |   			    |    		    |             | utriangle  |

#### Symbolic V-LoL<img src="example_images/train.png" width="20"> representation

| Car Position | Car Colour | Car Length | Car Wall | Car Roof | Car Axles | Load Number | Load Shape  |
|:------------:|:----------:|:----------:|:--------:|:--------:|:---------:|:-----------:|:-----------:|
|      1       |   Yellow   |   Short    |   Full   |   None   |     2     |      0      |  Blue Box   |
|      2       |   Green    |    Long    | Railing  |  Frame   |     3     |      1      | Golden Vase |
|      3       |    Grey    |     	      |    		    |   Flat   |           |      2      |   Barrel    |
|      4       |    Red     |     		     |   			    |   Bars   |           |      3      |   Diamond   |
|              |    Blue    |     		     |    		    |  Peaked  |    		     |             |  Metal Pot  |
|     			      |     		     |     		     |   			    |    		    |           |             |  Oval Vase  |
|      		      |     		     |     		     |   			    |    		    |           |             |    None     |

Overview of our three-dimensional train representation.
The following image illustrates the above described predicates.

<div align="center">
  <img src="example_images/descriptor_overview/overview.png" height="350px"  alt="">
</div>

#### Symbolic V-LoL<img src="example_images/square.png" width="11"> representation

| Car Position | Car Colour | Car Length | Black Top |    Car Shape    | Black Bottom | Load Number | Load Shape |
|:------------:|:----------:|:----------:|:---------:|:---------------:|:------------:|:-----------:|:----------:|
|      1       |   Yellow   |   Short    |   True    |      Cube       |     True     |      0      |   Sphere   |
|      2       |   Green    |    Long    |   False   |    Cylinder     |    False     |      1      |  Pyramid   |
|      3       |    Grey    |     	      |    		     |   Hemisphere    |              |      2      |    Cube    |
|      4       |    Red     |     		     |    			    |     Frustum     |              |      3      |  Cylinder  |
|              |    Blue    |     		     |    		     | hexagonal Prism |      		      |             |    Cone    |
|     			      |     		     |     		     |    			    |       		        |              |             |   Torus    |
|      		      |     		     |     		     |    			    |       		        |              |             |    None    |

The Train generator also allows for a simpler visualization relying on less complex objects.
The following image illustrates the above described predicates.

<div align="center">
  <img src="example_images/descriptor_overview/overview_simple_objs.png" height="350px"  alt="">
</div>

## Ground truth scene information

For each image the ground truth information of the scene is saved as a Json file inside the 'scenes' folder.
For each car we save the binary mask in form of an encoded RLE file.
The binary masks of the car wall, car wheels and the individual payloads are also saved inside the Json.
For each scene the Json file contains the following information:

```
m_train.json
│
└base_scene
└train_type
└image_index
└image_filename
└blender_filename
└depth_map_filename
└m_train
└angle
└───car_masks: {
    "car_1": {
      "mask": {rle},
      "world_cord": [ x, y, z ],
      "roof": {
        "label": roof type },
      "wall": {
        "label": wall type,
        "mask": {rle file},
        "world_cord": [x,y,z]
      },
      "wheels": {
        "label": number of wheels,
        "mask": {rle},
        "world_cord": [x,y,z]
      },
      "color": {
        "label": color
      },
      "length": {
        "label": legth
      },
      "payload_0": {
        "label": "oval_vase",
        "mask": {rle},
        "world_cord": [x,y,z]
      },
      "payload_1": {
        ...
      },
     ...
    },
    "car_2": {
      ...
    },
    ...
   }
```

The following shows an overview of the ground truth information described above:

<div align="center">
  <img src="example_images/scene_representation/original.png" width="400"  alt="">
  <img src="example_images/scene_representation/depth.png" width="400"  alt="">
  <img src="example_images/scene_representation/box.png" width="400"  alt="">
  <img src="example_images/scene_representation/mask.png" width="400"  alt="">
</div>

## Dataset structure

Once the dataset is generated we can find it in the folder TrainGenerator/output/. The dataset is structured as follows:

```
output
│
└───MichalskiTrains
│   │
│   └───SimpleObjects
│   │   │
│   │   └───base_scene
│   │   │   │
│   │   │   └───blendfiles
│   │   │   │     │0_m_train.blend
│   │   │   │     │...
│   │   │   │
│   │   │   └───depths
│   │   │   │     │0_m_train.png
│   │   │   │     │...
│   │   │   │
│   │   │   └───images
│   │   │   │     │0_m_train.png
│   │   │   │     │...
│   │   │   │
│   │   │   └───scenes
│   │   │         │0_m_train.json
│   │   │         │...
│   │   │
│   │   └───desert_scene
│   │       │...
│   │
│   └───Trains
│       │...
│
└───RandomTrains
│   │...
│
│...
```
The images rendered can be found in the images' folder.
The corresponding ground truth information is located in the 'scenes' folder.
The depth information of the individual images is located in the 'depths' folder (if depth_gen is opted).
The blender scene which is used to render the individual images is located in the 'blendfiles' folder (if save_blend is
opted).

## The Michalski Train Problem
<div align="center">
  <img src="example_images/michalski_original.png" alt="" width="700">
</div>


The Michalski train problem proposed by Ryszard S. Michalski [[1]](#1) in 1980 represents
one of the most prominent challenges in the domain of relational learning. The problem
constitutes a set of trains, which are composed of a wide variety of properties and labeled
into the two categories `Eastbound` and `Westbound` trains. It is up to the viewer
to ascertain a classification hypothesis governing what trains are eastbound
and what are westbound. The conjectured hypothesis should accomplish both
assigning the correct labels while retaining a degree of generality. It is
certainly easy to find a classification hypothesis satisfying a given set of Michalski trains,
e.g. learning the trains by heart. However, due to the great number of train properties, it
is a major challenge to find the most general decision hypothesis.

## References

<a id="1">[1]</a>
Ryszard S. Michalski. “Pattern Recognition as Rule-Guided Inductive Inference”. In:
IEEE Transactions on Pattern Analysis and Machine Intelligence PAMI-2.4 (1980),
pp. 349–361. doi: 10.1109/TPAMI.1980.4767034.

<a id="1">[2]</a>
Stephen Muggleton. Random train generator. 1998. url: https://www.doc.ic.ac.uk/~shm/Software/GenerateTrains/.
