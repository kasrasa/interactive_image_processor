# Interactive Image Processor

An interactive computer-vision learning project by Kasra Sadatsharifi.

[Try the live application](https://interactiveimageprocessing.streamlit.app/).

An interactive learning tool for classical computer vision. Upload an image, adjust an
OpenCV parameter, and see the effect immediately beside the original. Every technique
includes a plain-language explanation, a parameter sweep, a luminance histogram, and a
copy-ready Python example matching the selected settings. A Grayscale toggle lets you use
image luminance as the working original for every technique and restore the untouched color
image at any time.

The Interactive Image Processor turns a collection of image-processing experiments into a
structured, browser-based playground. It reflects a research mindset, production habits,
and clear technical communication.

Visitors can send anonymous feedback from inside the app with a quick face rating and
selectable reasons. A written note is optional, and no visitor account, sign-in, or email is
required. The notification recipient is configured outside the repository and is never
displayed publicly.

## What you can explore

| Family | Techniques |
| --- | --- |
| Color spaces | HSV controls, LAB controls |
| Transform domains | Fourier magnitude/phase, DCT coefficients/reconstruction, multi-level Haar wavelets, Radon sinograms |
| Smoothing | Box blur, Gaussian blur, bilateral filtering |
| Edges | Sobel gradients, Canny edges |
| Thresholding | Manual binary threshold, Otsu threshold |
| Contrast | Global histogram equalization, CLAHE |
| Shapes | Canny-based contour detection and area filtering |
| Geometry | Resize/interpolation, rotation, flip, shear |
| Augmentation | Gaussian noise, cutout/random erasing, unsharp masking |

## Run the preview locally

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local address, normally
[`http://localhost:8501`](http://localhost:8501). Nothing is deployed by this command.

## How to use it

1. Start with the built-in test card or upload a JPG, PNG, or WebP image.
2. Use the **Appearance** buttons at the top of the sidebar to choose Light or Dark mode.
3. Turn on **Grayscale** when you want the luminance image to become the original input for
   every technique; turn it off to restore the source colors.
4. Choose a technique family and operation in the sidebar.
5. Move the controls and compare the original with the live result.
6. Open **Learn** for parameter guidance, **Compare settings** for a three-value sweep,
   and **Copy the code** to reuse the current OpenCV operation.
7. Download the processed result as a PNG if you want to inspect it elsewhere.
8. Use the feedback form to suggest another technique, report something unclear, or share
   what worked well.

Uploaded images are processed in the running Streamlit session. Files are not stored by
the app. Inputs larger than 1,400 pixels on their longest side are reduced for a responsive
preview. Transform-domain outputs are normalized into an 8-bit image for display; the
copy-ready code shows the corresponding coefficient or projection calculation.

## Verify the processing layer

```bash
python -m unittest discover -v
```

The test suite renders the Streamlit interface, exercises every registered operation with
its default parameters, and checks feedback privacy and numeric safeguards.

## About the author

I am a computer-vision and machine-learning engineer interested in bridging the gap between research and production, turning experimental ideas into practical tools. More projects are available on
[GitHub](https://github.com/kasrasa).
