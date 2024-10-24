# VCap_CapsuleVision2024

## Dataset 
The training and validation dataset has been developed using
three publicly available (SEE-AI project dataset, KID,
and Kvasir-Capsule dataset) and one private dataset (AIIMS) VCE datasets. The training and validation dataset
consist of 37,607 and 16,132 VCE frames respectively mapped to 10 class labels namely angioectasia, bleeding, erosion, erythema, foreign body, lymphangiectasia, polyp, ulcer, worms,
and normal.
| Type of Data | Source Dataset | Angioectasia | Bleeding | Erosion | Erythema | Foreign Body | Lymphangiectasia | Normal | Polyp | Ulcer | Worms |
|--------------|----------------|--------------|----------|---------|----------|---------------|------------------|--------|-------|-------|-------|
| Training     | KID            | 18           | 3        | 0       | 0        | 0             | 6                | 315    | 34    | 0     | 0     |
|              | KVASIR         | 606          | 312      | 354     | 111      | 543           | 414              | 24036  | 38    | 597   | 0     |
|              | SEE-AI         | 530          | 519      | 2340    | 580      | 249           | 376              | 4312   | 1090  | 0     | 0     |
|              | AIIMS          | 0            | 0        | 0       | 0        | 0             | 0                | 0      | 0     | 66    | 158   |
| **Total Frames** |                | **1154**     | **834**  | **2694**| **691**  | **792**       | **796**          | **28663**| **1162**| **663**| **158** |
| Validation   | KID            | 9            | 2        | 0       | 0        | 0             | 3                | 136    | 15    | 0     | 0     |
|              | KVASIR         | 260          | 134      | 152     | 48       | 233           | 178              | 10302  | 17    | 257   | 0     |
|              | SEE-AI         | 228          | 223      | 1003    | 249      | 107           | 162              | 1849   | 468   | 0     | 0     |
|              | AIIMS          | 0            | 0        | 0       | 0        | 0             | 0                | 0      | 0     | 29    | 68    |
| **Total Frames** |                | **497**      | **359**  | **1155**| **297**  | **340**       | **343**          | **12287**| **500** | **286** | **68** |

## Citation

- Challenge ArXiv
  
@article{handa2024capsule,
  title={Capsule Vision 2024 Challenge: Multi-Class Abnormality Classification for Video Capsule Endoscopy},
  author={Handa, Palak and Mahbod, Amirreza and Schwarzhans, Florian and Woitek, Ramona and Goel, Nidhi and Chhabra, Deepti and Jha, Shreshtha and Dhir, Manas and Gunjan, Deepak and Kakarla, Jagadeesh and others},
  journal={arXiv preprint arXiv:2408.04940},
  year={2024}}
  
- Training and Validation Datasets
  
@article{Handa2024,
author = "Palak Handa and Amirreza Mahbod and Florian Schwarzhans and Ramona Woitek and Nidhi Goel and Deepti Chhabra and Shreshtha Jha and Manas Dhir and Deepak Gunjan and Jagadeesh Kakarla and Balasubramanian Raman",
title = "{Training and Validation Dataset of Capsule Vision 2024 Challenge}",
year = "2024",
month = "7",
url = "https://figshare.com/articles/dataset/Training_and_Validation_Dataset_of_Capsule_Vision_2024_Challenge/26403469",
doi = "10.6084/m9.figshare.26403469.v1",
journal={Fishare}}

- Testing Datasets
  
@article{Handa2024,
author = "Palak Handa and Amirreza Mahbod and Florian Schwarzhans and Ramona Woitek and Nidhi Goel and Deepti Chhabra and Shreshtha Jha and Manas Dhir and Pallavi Sharma and Dr. Deepak Gunjan and Jagadeesh Kakarla and Balasubramanian Ramanathan",
title = "{Testing Dataset of Capsule Vision 2024 Challenge}",
year = "2024",
month = "10",
url = "https://figshare.com/articles/dataset/Testing_Dataset_of_Capsule_Vision_2024_Challenge/27200664",
doi = "10.6084/m9.figshare.27200664.v1"
}
