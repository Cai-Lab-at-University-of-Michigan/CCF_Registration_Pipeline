# CCF_Registration_Pipeline

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
   - `python pipeline.py --ref_img_path /path/to/ccf_template.nii.gz --ref_ann_path /path/to/ccf_template_annotation.nii.gz --sub_img_path /path/to/experimental_brain.nii.gz --result_path /path/to/save_results/`

