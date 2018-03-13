# Week 3 Foreground segmentation, area filter, hole filling, shadow removal

Tasks:

- [x] Task 1: Hole filling
- [x] Task 2: Area filtering
  - [x] Task 2.1: Plot AUC vs P
  - [x] Task 2.2: Estimate the best P
- [x] Task 3: Aditional morphological operations
- [x] Task 4: Shadow removal
- [x] Task 5.1: Update Precision-Recall(PR) Curve
- [x] Task 5.2: Update Area Under the Curve (AUC)

To run each experiment type:

    python taskX.py 
  
where X is the task id.


To run task1 or task2 with shadow removal run as previously but set to one this variable in :

    shadow_removal = 1
  
In gaussian_color.py and train_color.py
