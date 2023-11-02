<img id="logo" class="center" src="feddrive_logo.png" alt="drawing"/>
<img class="sub-title" src="subtitle.png" alt="drawing"/>

**Official repository** of:
- [E. Fanì](https://scholar.google.com/citations?user=rwto7AgAAAAJ&hl=it), [M. Ciccone](https://scholar.google.com/citations?user=hOQjblcAAAAJ&hl=it), [B. Caputo](https://scholar.google.com/citations?user=mHbdIAwAAAAJ&hl=it). [**FedDrive v2: an Analysis of the Impact of Label Skewness in 
  Federated Semantic Segmentation for Autonomous Driving**](https://arxiv.org/abs/2309.13336). _5th Italian Conference on Robotics and Intelligent
  Machines (I-RIM)_, 2023.
- L. Fantauzzo<sup>\*</sup>, [E. Fanì](https://scholar.google.com/citations?user=rwto7AgAAAAJ&hl=it)<sup>\*</sup>, [D. Caldarola](https://scholar.google.com/citations?user=rX-VwlcAAAAJ&hl=it), [A. Tavera](https://scholar.google.com/citations?user=oQfTuXMAAAAJ&hl=it),
  [F. Cermelli](https://scholar.google.com/citations?user=-fEOFbMAAAAJ&hl=it)<sup>1</sup>, [M. Ciccone](https://scholar.google.com/citations?user=hOQjblcAAAAJ&hl=it), [B. Caputo](https://scholar.google.com/citations?user=mHbdIAwAAAAJ&hl=it). [**FedDrive: Generalizing Federated Learning to 
  Semantic Segmentation in Autonomous Driving**](https://arxiv.org/abs/2202.13670), _IEEE/RSJ International
  Conference on Intelligent Robots and Systems_, 2022.

**Corresponding author:** eros.fani@polito.it.

All the authors are supported by Politecnico di Torino, Turin, Italy. 

<sup>\*</sup>Equal contribution.
<sup>1</sup>Fabio Cermelli is with Italian Institute of Technology, Genoa, Italy.

**Official website:** https://feddrive.github.io/

## Citation

If you find our work relevant to your research or use our code, please cite our papers:

```
@inproceedings{feddrive2023,
  title={FedDrive v2: an Analysis of the Impact of Label Skewness in Federated Semantic Segmentation for Autonomous Driving},
  author={Fanì, Eros and Ciccone, Marco and Caputo, Barbara},
  journal={5th Italian Conference on Robotics and Intelligent Machines (I-RIM)},
  year={2023}
}

@inproceedings{feddrive2022,
  title={FedDrive: Generalizing Federated Learning to Semantic Segmentation in Autonomous Driving},
  author={Fantauzzo, Lidia and Fanì, Eros and Caldarola, Debora and Tavera, Antonio and Cermelli, Fabio and Ciccone, Marco and Caputo, Barbara},
  booktitle={Proceedings of the 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems},
  year={2022}
}
```

## Summary

**FedDrive** is a new benchmark for the **Semantic Segmentation** task in a **Federated Learning** scenario for
**autonomous driving**.

It consists of <u>12 distinct scenarios</u>, incorporating the real-world challenges of <u>statistical heterogeneity</u>
and <u>domain generalization</u>. FedDrive incorporates algorithms and style transfer methods from Federated Learning,
Domain  Generalization, and Domain Adaptation literature. Its main goal is to enhance model generalization and
robustness against statistical heterogeneity.

We show the importance of using the correct clients’ statistics when dealing with different domains and label skewness
and how  style transfer techniques can improve the performance on unseen domains, proving FedDrive to be a solid
baseline for future research in federated semantic segmentation.

<table class="table_max">
  <caption>Summary of the FedDrive scenarios.</caption>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Setting</th>
      <th>Distribution</th>
      <th># Clients</th>
      <th># img/cl</th>
      <th>Test clients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td> Cityscapes </td>
      <td> - </td>
      <td> <span class="uniform">Uniform</span>, <span class="heterogeneous">Heterogeneous</span>, 
        <span class="imbalance">Class Imbalance</span> </td>
      <td> 146 </td>
      <td> 10-45 </td>
      <td> unseen cities </td>
    </tr>
    <tr>
      <td rowspan="3"> IDDA </td>
      <td> <span class="country">Country</span> </td>
      <td> <span class="uniform">Uniform</span>, <span class="heterogeneous">Heterogeneous</span>, 
        <span class="imbalance">Class Imbalance</span> </td>
      <td> 90 </td>
      <td> 48 </td>
      <td> seen + unseen (country) domains </td>
    </tr>
    <tr>
      <td> <span class="rainy">Rainy</span> </td>
      <td> <span class="uniform">Uniform</span>, <span class="heterogeneous">Heterogeneous</span>, 
        <span class="imbalance">Class Imbalance</span> </td>
      <td> 69 </td>
      <td> 48 </td>
      <td> seen + unseen (rainy) domains </td>
    </tr>
    <tr>
      <td> <span class="bus">Bus</span> </td>
      <td> <span class="uniform">Uniform</span>, <span class="heterogeneous">Heterogeneous</span>, 
        <span class="imbalance">Class Imbalance</span> </td>
      <td> 83 </td>
      <td> 48 </td>
      <td> seen + unseen (bus) domains </td>
    </tr>
  </tbody>
</table>

## Results

Please visit the [FedDrive official website](https://feddrive.github.io/) for the results.

## Setup

1) Clone this repository

2) Move to the root path of your local copy of the repository

3) Create the ```feddrive``` new conda virtual environment and activate it:
```
conda env create -f environment.yml
conda activate feddrive
```

4) Download the Cityscapes dataset from [here](https://www.cityscapes-dataset.com/downloads/).
You may need a new account if you do not have one yet. Download the ```gtFine_trainvaltest.zip``` and ```leftImg8bit_trainvaltest.zip```archives

5) Extract the archives and move the ```gtFine``` and ```leftImg8bit``` folders in ```[local_repo_path]/data/cityscapes/data/```

6) Ask for the ```IDDA V3``` version of IDDA, available [here](https://idda-dataset.github.io/home/download/)

7) Extract the archive and move the ```IDDAsmall``` folder in ```[local_repo_path]/data/idda/data/```

8) Make a new [wandb](https://wandb.ai/) account if you do not have one yet, and create a new wandb project.

9) In the ```configs``` folder, it is possible to find examples of config files for some of the experiments to replicate the results of the paper.
Run one of the exemplar configs or a custom one:
```
./run.sh [path/to/config]
```
N.B. change the ```wandb_entity``` argument with the entity name of your wandb project.

N.B. always leave a blank new line at the end of the config. Otherwise, your last argument will be ignored.

## How to visualize model predictions, LAB and CFSI images

The script ```plot_samples.py``` is designed to save and eventually visualize
sets of ```(image, CFSI(image), LAB(image), target, model(image))```
from samples in the test set(s) associated with a dataset, given a checkpoint
and the indices of the images to show.

To use this script:

1) Download the checkpoint of the desired run from WandB
2) Copy the ```[run_args]``` from the info of the same run on wandb
3) Customize the load_path, indices, path_to_save_folder and plot variables options
4) Modify the ```CUDA_VISIBLE_DEVICES``` environment variable to select one single desired GPU 
5) Move to the root directory of this repository and run the following command:
```
python src/plot_samples.py [run_args]
```
