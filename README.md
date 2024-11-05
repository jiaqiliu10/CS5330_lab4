# CS5330 Lab 4

### Gradio Demo

Hugging Face Demo Link:  https://huggingface.co/spaces/jiaqiliuu/CS5330_lab4

### File Placement


### Running the Code
Follow these steps to run the code in the correct order:

1. **Step 1**
    - **Run python3 image_segmentation.py**
    Use the get_segmentation_mask function to generate the segmentation mask, then call the save_segmented_person function to save the segmentation result as an image with a transparent background. The output file will be **segmented_person.png**.

2. **Step 2**
    - **Run python3 insert_stereoscopic.py**
    First, use the split_stereo_image function to split the side-by-side image into left and right views. Then, call the insert_person_with_depth function to overlay the segmented image onto the left and right views. The output files will be **output_left.png** and **output_right.png**.

3. **Step 3**
    - **Run python3 create_anaglyph.py**
    It will combine output_left.png and output_right.png into a red-cyan anaglyph image. The output file will be **anaglyph_output.png**.

4. **Step 4**
    - **Run python3 app.py**
    To launch the Gradio interface. In the interface, users can upload a person and a side-by-side stereo image, set the depth level, adjust position and scaling, and generate the final anaglyph image.
