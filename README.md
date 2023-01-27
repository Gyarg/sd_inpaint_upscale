# SD Inpaint Upscale

A script for [AUTOMATIC1111's Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) that works in a similar way to the built-in SD Upscale script, except it supports tiling and is made to work with and only with inpainting.  
This script always treats the image as tileable but it seems to handle non-tiling images just fine. That said, it will have a hard time making a non-tiling image tileable.

## Tiling Example vs SD Upscale Script
Generated test image:  
![Original Comparison Image](https://github.com/Gyarg/sd_inpaint_upscale/blob/main/images/01205-4174452299-a%20painting%20of%20a%20group%20of%20multicolored%20fish%2C%20ultrasharp%2C%20high%20contrast.jpeg)  
>"a painting of a group of multicolored fish, ultrasharp, high contrast", DPM++ 2M Karras, 512 x 512, cfg 3, 10 steps, tiling enabled, sd-v1-5-inpainting.ckpt

The images below all show the image shifted so that the corners meet in the middle.  
The following two images use the settings:  
>640 x 640, cfg 5, denoising .2, 4x_foolhardy_Remacri upscaler.

The rest are their defaults.  
SD Upscale  
![Original Script, Denoising .2](https://github.com/Gyarg/sd_inpaint_upscale/blob/main/images/original_2.jpeg)
SD Inpaint Upscale  
![Inpaint Script, Denoising .2](https://github.com/Gyarg/sd_inpaint_upscale/blob/main/images/inpaint_2.jpeg)
The final two have the denoising cranked up to .9 to exacerbate the differences.  
SD Upscale  
![Original Script, Denoising .9](https://github.com/Gyarg/sd_inpaint_upscale/blob/main/images/original_9.jpeg)
SD Inpaint Upscale  
![Inpaint Script, Denoising .9](https://github.com/Gyarg/sd_inpaint_upscale/blob/main/images/inpaint_9.jpeg)

## How It Works
This script will upscale the image then process smaller parts of the image one at a time, similar to the original upscale script. In addition, it will extend the edges with the opposite side of the image for context, cropping when finished.  
The pattern in which the tiles are processed is different. Since inpainting depends on what is around it, changing a piece that contains already changed regions can lead to over-changing in certain areas.  
This script uses the following pattern to help mitigate that:  
![Tile Processing Pattern](https://github.com/Gyarg/sd_inpaint_upscale/blob/main/images/tile_pattern_80.png)  
In the image above, red is first, blue is second, and the greens are last. The brighter parts represent what is inpainted while the darker parts represent the borders of the tiles. This is an example of 80px borders with a 512 x 512 tile size on a 1024 x 1024 image, resulting in 16px of overlap for the inpainted areas.

## Usage

A couple of options external to the script that need to be set:
* Select an inpainting model
* Set "Inpaint area" to "Whole picture"

No attempt was made to incorporate support for batch size or batch count.  
"Tiling" doesn't need to be on.  
It is probably a good idea to have "Inpaint masked" turned on.  
High denoising values will work and the processed tiles should blend seamlessly.  

### Upscaler
If none is selected, it is possible to do an inpainting-only pass.  
Use an upscaler that doesn't remove too much fine detail.  
Upscaler Resources:
&emsp;[](https://phhofm.github.io/upscale/favorites.html)
&emsp;[](https://upscale.wiki/wiki/Model_Database#Universal_Models)

### Minimum Border
This is the border of the tiles that will be processed.  
![80px Border Example](https://github.com/Gyarg/sd_inpaint_upscale/blob/main/images/border_80.png)  
A higher minimum value will make the inpainted area look more like the rest of the image. The higher it is, the more tiles that will potentially need to be processed, and the longer it will take if that is the case. The minimum should probably be at least 16.

### Maximum Inpaint Overlap
Inpainting over the edges of already inpainted areas can help when the processed area will differ significantly from its border. A higher overlap value will push the calculated border more towards the minimum. This will neither increase nor decrease the amount of tiles to process.

### Inpaint Over Edges
With this off, the sides of a tiled image will very slightly not match for some reason. This option both attempts to blend discrepancies in the sides from the upscaler and does another SD inpaint pass along the edges. The amount of predicted processing steps in the console can be higher than actuality when enabled, but the progress bar in the GUI remains correct.

### Edge Overlap
Edge overlap is the width or height of the tile being processed when "Inpaint Over Edges" is on. The other dimension will be what was set in the normal settings. Smaller values will be quicker, but the overall size should be one that the model handles well.

### Add Noise
This was included as a potential way to add detail. It isn't thoroughly tested. 

### Sharpen
Implements the PIL ImageEnhance Sharpen method where 0 is blurry, 1 is unaltered, and 2 is sharp. It allows higher and lower values. The image is sharpened after it is upscaled but before the inpainting pass.
---
### Hidden Options
Usually undesirable. To unhide, open sd_inpaint_upscale_tile.py and remove the `visible=False` from their respective parts in the `def_ui` section near the top. There is also a part near the top of the `def_run` section that adds metadata. Remove the `#` before it to allow it to be written.

#### Choose Tile Pattern
The "Diagonal" pattern, which is the default, is described above in the "How It Works" section. The other, "Adjacent", may cause over-changing in parts of the image. In the example image, its pattern starts with red, then does the greens, and finishes with blue. 

#### Increment Seed
With this set to false and a high denoising value, the tiles on the image can end up looking similar.

#### Noise Application
Two options are available, only one of which, "Global", might be good enough. It applies noise to the entire image before processing the tiles. The other option, "Per Tile", applies noise to the processing area of the tile before it is processed. The "Global" option is used by default.

#### Saturation, Contrast
These were there to correct the noise algorithm making the image greyer back when it did so.

#### Upscaling Iterations
Upscales the image the amount of times specified here before doing an SD inpainting pass. The SD inpainting pass is generally much slower than the upscalers, so a higher value can increase speed. A higher value might also lead to worse quality, however.

#### SD Inpaint Iterations
This was made to automate the successive upscale of an image.
The value is the amount of times the script will do the upscales then SD inpainting. Therefore, the final image size (not counting the final upscales or downscaling) will be:  
`Original width or height *2^(upscaling iterations * SD inpaint iterations)`
Example of starting with a 768x512 image and setting upscaling iterations to 2 and SD inpaint iterations to 3:
* 768 x 512    base
* 1,536 x 1,024    upscaled
* 3,072 x 2,048    upscaled
* SD inpaint
* 6,144 x 4,096    upscaled
* 12,288 x 8,192    upscaled
* SD inpaint
* 24,576 x 16,384    upscaled
* 49,152 x 32,768    upscaled
* SD inpaint

#### Final Upscale Iterations
Perform a final upscale after everything else.

#### Final Saturation, Contrast, Sharpen
These are best done in an image editing software afterwards.

#### Downscale Final Image
Downscaling an upscaled image can make it look sharper when comparing 100% crops, depending on the upscaler.
The downscaling percent refers to the perimeter, so a 1024 x 1024 image with the value at 75% would be 768 x 768.

## Some things I think might be true, but haven't thoroughly tested:
The inpainting upscale script is more sensitive to the quality of the upscaler used vs the SD upscale script due to inpainting using data from the unmasked areas.
Considering the above, the original upscale script will usually create better details.
A decently high overlap value won't cause a style drift to occur sufficiently due to the tile pattern.

## Acknowledgements
This script was originally based on the SD upscale script, though not much of it remains.

