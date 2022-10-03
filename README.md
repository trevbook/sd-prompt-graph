# **Stable Diffusion Prompt Graph**
This is a React-based curve editor GUI for prompt interpolation animations made with Stable Diffusion! 

Since that sentence is a bit of a mouthful, how about I just show you how it works? 

![Alt text](stable-diffusion/assets/prompt-graph-demo.png?raw=true "Screenshot of the SD Prompt Graph user interface")
*Users populate a "prompt graph", where the X-axis represents the progression of an animation, and the Y-axis
 represents the relative "strength" of a prompt. Each curve in the graph corresponds with one of the user's prompts.*

After crafting a Prompt Graph, a series of Stable Diffusion generations can be launched. The resulting images will 
correspond with the interpolation you defined in your graph: 

![Alt text](stable-diffusion/assets/demo-generation-progression.png?raw=true "Results of the previously-shown Prompt Graph generation")

This repo is a clone of v0.9 of [basujindal's stable-diffusion fork](https://github.com/basujindal/stable-diffusion)!
I picked this fork as a basis for my Prompt Graph app, since it contains a ton of optimizations that allow Stable Diffusion 
to be run on my local GPU (NVIDIA 2070 Super). In the future, I could easily swap out the "engine" Prompt Graph 
is using - by the time of publishing this, there'll probably be more insane optimizations to make things even faster. 😉

---

### **Installation**

In order to install this app, you'll need to install a couple of pre-requesites: 

- [**node.js**](https://nodejs.org/en/) - Since this is a React app, you'll need to download `node.js` in order to have access to `npm`.


- [**conda**](https://docs.conda.io/en/latest/) - Conda is a package management system that you'll use to install the right Python packages.

- [**ffmpeg**](https://ffmpeg.org/) - FFMPEG is a command-line tool that enables a lot of video-related functionality. It's what I use to create the .mp4 files of the animations


Once you've downloaded both of these, you should move into setting up [**Stable Diffusion**](https://github.com/CompVis/stable-diffusion#requirements). The first thing you ought to do is actually download the weights for the model. I've used [Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) successfully, but there might be newer models by the time you're reading this. 

After the model is downloaded, you ought to rename the `.ckpt` file to `model.ckpt`, and then place it into the `stable-diffusion/models/ldm/stable-diffusion-v1` folder. 

Once you've placed the model into the correct folder, you ought to set up the Stable Diffusion conda environment. To set up this environment, run these commands from the root of the repo: 

```shell
cd stable-diffusion/
conda env create -f environment.yaml
conda activate ldm
pip install -r ./requirements.txt --exists-action=w
```

Once this is finished, you'll need to install all of the dependencies for the React app. To do that, run these commands from the repo root: 

```shell
cd prompt-graph/
npm install 
```

After this point, you should be all set up! Check out the **`Usage`** section below to learn more about how to actually run the app.

---

### **Usage**

In order to run the React app, you can run the following commands from the root of the repo: 

```
cd prompt-graph/
npm start
```

This will start a server running the React app at [http://localhost:3000](http://localhost:3000/). 

From there, you can specify the Prompt Graph that you want to create. (More detailed documentation / a video tutorial on how to use the app is to come!)

Once you're finished defining your prompt graph, click on the **Generate Configuration File** button. This will download a `.json` file containing all of the prompts you'd specified. 

Finally, you should be able to run 

---

### **Future Development**
I'm actively working on this repo, so I'd expect that things ought to change pretty radically in the future. 
As follows is a list of some of the things I'm actively working on: 

- Simplifying installation process
- CLIP-guided recursive latent interpolation (to smooth transitions between images) 
- Allowing users to load and edit already-created Prompt Graph configuration files
- Adding additional polish to the user interface & fixing bugs
- Allowing initial images to be used within the Prompt Graph
- Development of a [Dash version](https://dash.plotly.com/) of the React application (which'll allow for in-browser use via Google Colab)
- Better curve-editing capabilities / controls (i.e., non-linear curves)

---