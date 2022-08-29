# CCF_Registration_Pipeline

## Using Elastix
- Profiling:
  - Environment: AMD Ryzen Threadripper 1950X 16-Core Processor<span>@</span>2.2GHZ
  - System: Ubuntu 21.04
  - Peak Memory Usage: 70000MB (~69GB)
  - Total Time: 3014s (~1h)
  - Experiment Brian:
      - Experiment brain id: 192341_SLA
      - Physical spacing: 0.01mmx0.01mmx0.01mm
      - Image size: 1128x1029x590 (ZYX)
      - Numerical dtype: uint8
  - CCF template:
      - Physical spacing: 0.01mmx0.01mmx0.01mm
      - Image size: 1140x800x1320 (ZYX)
      - Numerical dtype: uint16 for average template, uint32 for annotation
- To run the program:
   - Please use absolute path
   - `python pipeline_elastix.py --ref_img_path /path/to/ccf_template.tif --ref_ann_path /path/to/ccf_template_annotation.tif --sub_img_path /path/to/experimental_brain.tif--result_path /path/to/save_results/ --suffix .tif`

## Using AntsPy
- Profiling:
  - Environment: AMD Ryzen Threadripper 1950X 16-Core Processor<span>@</span>2.2GHZ
  - System: Ubuntu 21.04
  - Peak Memory Usage: 36578MB (~36GB)
  - Total Time: 10902s (~3h)
  - Experiment Brian:
      - Experiment brain id: 192341_SLA
      - Physical spacing: 0.01mmx0.01mmx0.01mm
      - Image size: 1128x1029x590 (ZYX)
      - Numerical dtype: uint8
  - CCF template:
      - Physical spacing: 0.025mmx0.025mmx0.025mm
      - Image size: 456x320x528 (ZYX)
      - Numerical dtype: uint16 for average template, uint32 for annotation
- To run the program:
   - Please use absolute path
   - `python pipeline.py --ref_img_path /path/to/ccf_template.nii.gz --ref_ann_path /path/to/ccf_template_annotation.nii.gz --sub_img_path /path/to/experimental_brain.nii.gz --result_path /path/to/save_results/ --suffix .nii.gz`

**[Results]**: Warped ccf_template and ccf_template_annotation images are stored in `/path/to/save_results/subject/ants/ccf_affined/`

**[Important Note]**: If you want faster (actually much faster) registration (~20-25mins), you can reduce the iterations of non-linear deformation. This although may suffer from inferior registration performance. Unfortunately, AntsPy doesn't provide an api to modify these paprameters (it has been hardcoded), what you can do is following:
  1. first indentify this piece of code `elif type_of_transform == "SyNCC":` (it's in **Line 816** currently for **antspyx 0.3.3**) in `ants.registration.interface.py`. You will need to look up from site-packages if you are using anaconda.
  2. replace `synits = "2100x1200x1200x20" smoothingsigmas = "3x2x1x0" shrinkfactors = "4x3x2x1"` with `synits = "50x50x0x0" smoothingsigmas = "4x2x1x0" shrinkfactors = "8x4x2x1"`, or simply replace the file with `misc/interface.py` (**This may subject to version update, be careful**).
