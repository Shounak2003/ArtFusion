# ArtFusion 🎨✨

**An AI-powered art generator that transforms your photos into masterpieces by applying the styles of famous artists using neural networks and style transfer techniques.**

## 📖 Project Overview

ArtFusion allows you to take any content image and apply the artistic style of famous painters, such as Van Gogh, Picasso, or Monet, to create unique artwork. It uses advanced neural style transfer techniques with a pre-trained VGG19 model to blend the content of your image with the style of a selected artwork, generating stunning visual results.

## 🚀 How It Works

1. **Content Image**: The image you want to transform (e.g., a photo of yourself).
2. **Style Image**: The artwork style you want to apply to the content image (e.g., a painting by Van Gogh).
3. **Neural Style Transfer**: Using deep learning, ArtFusion combines both images to generate a new image that contains the content of your photo but with the artistic style of the painting.

### 📂 Repository Structure

```
/ArtFusion
│
├── style_transfer.py          # Main script to perform style transfer
├── ArtFusion.ipynb            # Jupyter notebook for interactive experiments
└── README.md                 # Project documentation
```

### 🔧 Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Shounak2003/ArtFusion.git
   cd ArtFusion
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) If you want to use the Jupyter notebook, launch it using:
   ```bash
   jupyter notebook ArtFusion.ipynb
   ```

### 📸 Example Usage

1. **Running the Script**: You can run the `style_transfer.py` script directly to perform the style transfer.

   - Make sure to replace the paths of the content and style images in the script.
   - Run the following command:
   
   ```bash
   python style_transfer.py
   ```

   - The script will process the images and display the resulting artwork at the end of the transfer.

2. **Jupyter Notebook**: The `ArtFusion.ipynb` file allows you to experiment with different content and style images interactively. You can load any image, change styles, and visualize the output on the fly.

### 🖼️ Example Image Outputs

Here are some example transformations you can experiment with:

- **Content Image**: A photo of yourself or a landscape.
- **Style Image**: Famous artworks like those from Van Gogh, Picasso, or Monet.

Once the style transfer is completed, the result will look like a blend of your content with the artist's style.

### 🧑‍💻 Code Explanation

- **`style_transfer.py`**: This Python script performs the style transfer. It loads the content and style images, processes them, and then uses a pre-trained VGG19 model to compute the loss between content and style features. The optimization process is run to minimize the loss and generate the stylized image.

- **`ArtFusion.ipynb`**: The notebook allows for easy experimentation with different images and visualizes the process step by step.

### 🔨 Functions in `style_transfer.py`

- **`load_and_process_img`**: Loads and preprocesses images for the VGG19 model.
- **`deprocess_img`**: Converts processed images back to a viewable format.
- **`content_loss`**: Computes the difference between content features of the content image and generated image.
- **`style_loss`**: Measures the difference in style features using the Gram matrix.
- **`total_variation_loss`**: Ensures the output image is smooth and free of noise.
- **`style_transfer`**: The main function that runs the optimization to apply the style transfer.

### 📊 Example Results

Below are a few examples of the style transfer results. You can experiment with your own content and style images to see the results.

| Content Image  | Style Image   | Resulting Image |
|----------------|---------------|-----------------|
| ![Content](assets/content_example.jpg) | ![Style](assets/style_example.jpg) | ![Result](assets/result_example.jpg) |

*(Feel free to replace these images with your own)*

### ⚙️ Customization

You can easily customize the script by:

- **Changing the target size**: Modify the target size in the `load_and_process_img` function to scale images.
- **Adjusting the optimization process**: Modify the number of epochs and steps per epoch for different results.
- **Changing the loss weights**: If you want to tweak the importance of content loss, style loss, or smoothness loss, adjust them in the loss computation section of the code.

### 🧑‍🎨 Experiment With Famous Artists' Styles

By using different style images from famous artists, you can generate works of art in the styles of:

- **Vincent van Gogh**
- **Pablo Picasso**
- **Claude Monet**
- **Salvador Dalí**
- **And many more!**

Simply choose an artwork image as the style image and enjoy experimenting!

### 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 💬 Questions or Feedback?

Feel free to raise an issue on GitHub if you encounter any problems or have suggestions for improving this project. I’m happy to help!

