## 📁 Data Description

Due to the large size of the dataset, the demo data is hosted on Zenodo:  
📦 [Download from Zenodo](https://zenodo.org/records/15878397)

You can download the archive from the link above and extract it into the `data/` directory of the project root.

The extracted directory structure is as follows:

```
data/
└── MouseBrain-HD2V/
    ├── SourceDomain/
    └── TargetDomain/
```

Each of the `SourceDomain/` and `TargetDomain/` folders contains the following files:

- `re_image.tif` — Raw histology image  
- `pseudo_st.csv` — Gene count matrix  
- `pseudo_locs.csv` — Spatial coordinates for each spot  
- `mask.png` — Tissue segmentation mask  
- `gene_names.txt` — List of gene names  
- `pixel-size-raw.txt` — Physical pixel size (in micrometers)  
- `radius-raw.txt` — Spot radius in pixels

