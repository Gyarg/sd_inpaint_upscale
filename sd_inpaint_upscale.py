import modules.scripts as scripts
import gradio as gr
from PIL import Image, ImageEnhance

from modules import processing, shared, sd_samplers, images, devices
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state

import numpy as np
import random
from random import randint, seed
import math

class Script(scripts.Script):
    def title(self):
        return "SD Inpaint Upscale"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        with gr.Column(variant='panel'):
            info = gr.HTML("<p style=\"margin-bottom:0.75em\">Make sure that an inpainting model is selected.</p>")
            
            upscaler_index = gr.Radio(label='Upscaler', choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name, type="index")
        
            with gr.Group():
                with gr.Box():
                    ai_upscale_info = gr.HTML("<p>For low vram.</p>")
                tiled_ai_upscale = gr.Checkbox(label="Tiled Upscaler Upscale, Noise",value=False)
                tiled_ai_upscale_threshold = gr.Slider(minimum=0, maximum=8, step=1, label='Threshold', value=3)
                ai_upscale_info2 = gr.Textbox(value="Will tile image after a side reaches: 4096px", interactive=False, label_visible=False, label="")
            
            with gr.Group():
                with gr.Box():
                    tile_info = gr.HTML("<p>SD inpaint tile settings.</p>")
                min_border = gr.Slider(minimum=0, maximum=240, step=1, label='Minimum Border', value=48)
                inpaint_overlap = gr.Slider(minimum=0, maximum=240, step=1, label='Maximum Inpaint Overlap', value=64)
            
            with gr.Group():
                with gr.Box():
                    inpaint_edge_info = gr.HTML("<p>Should only be used for seamless images. Takes longer. Edge overlap changes tile size across edge for speed.</p>")
                blend_seams = gr.Checkbox(label="Inpaint Over Edges",value=True)
                edge_overlap = gr.Slider(minimum=64, maximum=2048, step=64, label='Edge Overlap', value=512)
                edge_denoising = gr.Slider(minimum=0, maximum=1, step=.01, label='Edge Denoising', value=.2)
            
            with gr.Group():
                with gr.Group():
                    with gr.Box():
                        upscale_iter_info = gr.HTML("<p>Use the selected upscaler this many times before doing each SD inpaint.</p>")
                    upscale_iters = gr.Slider(minimum=0, maximum=8, step=1, label='Upscale Iterations Per SD Inpaint', value=1)
                    
                space = gr.HTML("<p style=\"margin:2px\"></p>")
                
                with gr.Group():
                    with gr.Box():
                        noise_info = gr.HTML("<p>Experimental, may add detail. Applied before SD inpaint.</p>")
                    add_noise = gr.Checkbox(label="Add Noise",value=False)
                    noise_strength = gr.Slider(minimum=1, maximum=128, step=1, label='Noise Strength', value=24)
                    
                space2 = gr.HTML("<p style=\"margin:2px\"></p>")
                
                with gr.Group():
                    with gr.Box():
                        sharpen_info = gr.HTML("<p>1 = unchanged. 2 is the full effect. Applied before SD inpaint.</p>")
                    sharpen_amount = gr.Slider(minimum=1, maximum=10, step=.1, label='Sharpening', value=1)
                    
                space3 = gr.HTML("<p style=\"margin:2px\"></p>")
                    
                with gr.Group():
                    sd_upscale_iters = gr.Slider(minimum=0, maximum=8, step=1, label='SD Inpaint Iterations', value=1)
            
            with gr.Group():
                return_partials = gr.Checkbox(label="Return Intermediate Images",value=False)
            
            with gr.Accordion("Hidden Options (not recommended)", open=False):
                
                with gr.Group():
                    with gr.Box():
                        return_whole_info = gr.HTML("<p>Prevents returning a partial image on interrupt. Irrelevant when intermediate images is true.</p>")
                    return_whole = gr.Checkbox(label="Return Last Completed Image",value=True)
            
                with gr.Group():
                    with gr.Box():
                        pattern_info = gr.HTML("<p>Adjacent may blend better but drift more.</p>")
                    pattern = gr.Radio(label="Pattern", choices=['Diagonal','Adjacent'], value='Diagonal', type="index")
                
                with gr.Group():
                    with gr.Box():
                        seed_info = gr.HTML("<p>Disable to create similar patterns within the image.</p>")
                    increment_seed = gr.Checkbox(label="Increment Seed",value=True)
                    
                with gr.Group():
                    noise_application = gr.Radio(label="Noise Application", choices=['Global','Per Tile'], value='Global', type="index")
                
                with gr.Group():
                    with gr.Box():
                        image_preprocess_info = gr.HTML("<p>1 = unchanged. 2 is the full effect. Applied after upscale but before SD inpaint.</p>")
                    saturation_amount = gr.Slider(minimum=1, maximum=2, step=.01, label='Saturation', value=1)
                    contrast_amount = gr.Slider(minimum=1, maximum=2, step=.01, label='Contrast', value=1)
                    
                with gr.Group():
                    with gr.Box():
                        final_upscale_iter_info = gr.HTML("<p>Use the selected upscaler this many times after completing the SD inpaint upscales.</p>")
                    final_upscale_iters = gr.Slider(minimum=0, maximum=4, step=1, label='Final Upscale Iterations', value=0)
                
                with gr.Group():
                    with gr.Box():
                        image_postprocess_info = gr.HTML("<p>1 = unchanged. 2 is the full effect. Applied after final upscale.</p>")
                    final_saturation_amount = gr.Slider(minimum=1, maximum=4, step=.1, label='Final Saturation', value=1)
                    final_contrast_amount = gr.Slider(minimum=1, maximum=4, step=.1, label='Final Contrast', value=1)
                    final_sharpen_amount = gr.Slider(minimum=1, maximum=10, step=.1, label='Final Sharpening', value=1)
                
                with gr.Group():
                    with gr.Box():
                        downscale_info = gr.HTML("<p>Percent refers to the final perimiter. 50% is the inverse of an upscale.</p>")
                    downscale = gr.Checkbox(label="Downscale Final Image",value=False)
                    downscale_type = gr.Radio(label='Downscaler', choices=["Nearest","Box","Bilinear","Hamming","Bicubic","Lanczos"], value="Lanczos", type="value")
                    downscale_amount = gr.Slider(minimum=1, maximum=99, step=1, label='Percent', value=50)
        
        def update_ai_upscale_info(value):
            return gr.update(value=f"Will tile image after a side reaches: {512<<value}px")
        tiled_ai_upscale_threshold.change(fn = update_ai_upscale_info, inputs=tiled_ai_upscale_threshold, outputs=ai_upscale_info2)
        
        return [info, ai_upscale_info, ai_upscale_info2, tile_info, inpaint_edge_info, noise_info, sharpen_info, return_whole_info, pattern_info, seed_info, image_preprocess_info, upscale_iter_info, final_upscale_iter_info, image_postprocess_info, downscale_info, space, space2, space3, upscaler_index, tiled_ai_upscale, tiled_ai_upscale_threshold, min_border, inpaint_overlap, pattern, increment_seed, blend_seams, edge_overlap, edge_denoising, add_noise, noise_strength, noise_application, saturation_amount, contrast_amount, sharpen_amount, return_partials, return_whole, upscale_iters, sd_upscale_iters, final_upscale_iters, final_saturation_amount, final_contrast_amount, final_sharpen_amount, downscale, downscale_type, downscale_amount]

    def run(self, p, _, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, upscaler_index, tiled_ai_upscale, tiled_ai_upscale_threshold, min_border, inpaint_overlap, pattern, increment_seed, blend_seams, edge_overlap, edge_denoising, add_noise, noise_strength, noise_application, saturation_amount, contrast_amount, sharpen_amount, return_partials, return_whole, upscale_iters, sd_upscale_iters, final_upscale_iters, final_saturation_amount, final_contrast_amount, final_sharpen_amount, downscale, downscale_type, downscale_amount):
    
        processing.fix_seed(p)
        original_seed = p.seed
        original_prompt = p.prompt[0] if type(p.prompt) == list else p.prompt
        upscaler = shared.sd_upscalers[upscaler_index]
        initial_info = [None]
        img = images.flatten(p.init_images[0], opts.img2img_background_color)
        expanded_img = None
        tile_w = p.width
        tile_h = p.height
        w = img.width
        h = img.height
        max_border = (min(tile_w,tile_h)>>1)-16
        partials = []
        all_seeds = []
        all_prompts = []
        infotexts = []
        last_whole_image = None
        last_image_text = None
        if(return_whole and not return_partials):
            last_whole_image = img.copy()
            
        if(upscaler.name != "None"): 
            p.extra_generation_params["SDIU Upscaler"] = upscaler.name
        if(tiled_ai_upscale):
            p.extra_generation_params["SDIU Tiled Upscaler Threshold"] = tiled_ai_upscale_threshold
        if(sd_upscale_iters>0):
            if(min_border != 48):
                p.extra_generation_params["SDIU Border Min"] = min_border
            if(inpaint_overlap != 64):
                p.extra_generation_params["SDIU Inpaint Overlap"] = inpaint_overlap
            if(pattern == 1):
                p.extra_generation_params["SDIU Pattern"] = 'Adjacent'
        if(upscale_iters!=1):
            p.extra_generation_params["SDIU Upscaler Iterations"] = upscale_iters
        if(sd_upscale_iters!=1):
            p.extra_generation_params["SDIU SD Upscaler Iterations"] = sd_upscale_iters
        if not (increment_seed):
            p.extra_generation_params["SDIU Increment Seed"] = "false"
        if(blend_seams):
            p.extra_generation_params["SDIU Inpaint Over Edges"] = "true"
            if(edge_overlap != 512):
                p.extra_generation_params["SDIU Edge Overlap"] = edge_overlap
            if(edge_denoising != .2):
                p.extra_generation_params["SDIU Edge Denoising"] = edge_denoising
        if(add_noise):
            p.extra_generation_params["SDIU Noise"] = noise_strength
            if(noise_application == 1):
                p.extra_generation_params["SDIU Noise Application"] = 'Per Tile'
        if(saturation_amount != 1):
            p.extra_generation_params["SDIU Saturation"] = saturation_amount
        if(contrast_amount != 1):
            p.extra_generation_params["SDIU Contrast"] = contrast_amount
        if(sharpen_amount != 1):
            p.extra_generation_params["SDIU Sharpen"] = sharpen_amount
        if(final_upscale_iters>0):
            p.extra_generation_params["SDIU Final Upscaler Iterations"] = final_upscale_iters
        if(final_saturation_amount != 1):
            p.extra_generation_params["SDIU Final Saturation"] = final_saturation_amount
        if(final_contrast_amount != 1):
            p.extra_generation_params["SDIU Final Contrast"] = final_contrast_amount
        if(final_sharpen_amount != 1):
            p.extra_generation_params["SDIU Final Sharpen"] = final_sharpen_amount
        if(downscale):
            p.extra_generation_params["SDIU Downscaler"] = downscale_type
            p.extra_generation_params["SDIU Downscaler Percent"] = downscale_amount
        if(return_partials):
            p.extra_generation_params["SDIU Intermediate Images"] = "true"
        if not (return_whole):
            p.extra_generation_params["SDIU Completed Image"] = "false"
        
        def determine_border_and_tiles(w,h):
                
            def get_optimal_border_and_overlap(w_h,rows_cols,tile):
                for i in range(min_border,max_border,1):
                    non_border = tile - (i*2)
                    if(math.ceil(w_h/non_border)>rows_cols):
                        i -= 1
                        optimal_border = i-inpaint_overlap
                        if(optimal_border<min_border):
                            optimal_border = min_border
                        return [optimal_border,i]
                return [min_border,0]
                        
            border = min_border
            rows = math.ceil(w/(tile_w - (border*2)))
            cols = math.ceil(h/(tile_h - (border*2)))
            
            if(max_border>min_border):
                non_border_width = tile_w - (min_border*2)
                rows = math.ceil(w/non_border_width)
                border_and_overlap = get_optimal_border_and_overlap(w,rows,tile_w)
                border_w = border_and_overlap[0]
                overlap_w = border_and_overlap[1]
                
                non_border_height = tile_h - (min_border*2)
                cols = math.ceil(h/non_border_height)
                border_and_overlap = get_optimal_border_and_overlap(h,cols,tile_h)
                border_h = border_and_overlap[0]
                overlap_h = border_and_overlap[1]
                
                border = max(border_w,border_h)
                if(border>tile_w>>1):
                    border = (tile_w>>1) - 16
                if(border>tile_h>>1):
                    border = (tile_h>>1) - 16
                    
                non_border_width = tile_w - (border*2)
                rows = math.ceil(w/non_border_width)
                overlap_w -= border
                
                non_border_height = tile_h - (border*2)
                cols = math.ceil(h/non_border_height)
                overlap_h -= border
            
            return [border,rows,cols,overlap_w,overlap_h]
            
        job_count = 0
        edge_job_count = 0
        if(sd_upscale_iters>0):
            if(upscaler.name != "None" and upscale_iters>0): 
                for i in range(upscale_iters,(sd_upscale_iters*upscale_iters)+1,upscale_iters):
                    border_and_tiles = determine_border_and_tiles(w*math.pow(2,i),h*math.pow(2,i))
                    job_count += (border_and_tiles[1]*border_and_tiles[2])
                    edge_job_count += (border_and_tiles[1]+border_and_tiles[2])
            else:
                border_and_tiles = determine_border_and_tiles(w,h)
                job_count += (border_and_tiles[1]*border_and_tiles[2])*sd_upscale_iters
                edge_job_count += (border_and_tiles[1]+border_and_tiles[2])*sd_upscale_iters
        else:
            job_count = 1
        if(blend_seams):
            state.job_count = job_count+edge_job_count
        else:
            state.job_count = job_count
        
        upscale_count = 0
        if(sd_upscale_iters>0):
            upscale_count = upscale_iters*sd_upscale_iters
        upscale_count += final_upscale_iters
        final_width = w*math.pow(2,upscale_count)
        final_height = h*math.pow(2,upscale_count)
        if(upscaler.name == "None"):
            final_width = w
            final_height = h
        if(downscale):
            final_width *= downscale_amount
            final_height *= downscale_amount
        
        if(blend_seams):
            print(f"Final image size: {int(final_width)},{int(final_height)}. Need to process {job_count} normal tiles and {edge_job_count} edge tiles, {state.job_count} in total.")
        else:
            print(f"Final image size: {int(final_width)},{int(final_height)}. Need to process {job_count} tiles.")
            
        def create_borders(original_image,new_image,new_border,width,height):
            #corners
            new_image.paste(original_image,(-width+new_border,-height+new_border))
            new_image.paste(original_image,(-width+new_border,height+new_border))
            new_image.paste(original_image,(width+new_border,-height+new_border))
            new_image.paste(original_image,(width+new_border,height+new_border))
            #sides
            new_image.paste(original_image,(-width+new_border,new_border))
            new_image.paste(original_image,(width+new_border,new_border))
            new_image.paste(original_image,(new_border,-height+new_border))
            new_image.paste(original_image,(new_border,height+new_border))
            #center
            new_image.paste(original_image,(new_border,new_border))
            
        def expand_for_tiling(original_image,new_image,new_border,width,height):
            new_image.paste(original_image,(width,height))
            new_image.paste(original_image,(width,0))
            new_image.paste(original_image,(0,height))
            new_image.paste(original_image,(0,0))
            
        def perform_tiled_upscale(original_image,new_image,upscale_border,ai_rows,ai_cols,ai_threshold,w,h):
            print(f"Upscaler Tiles: {ai_rows*ai_cols}")
            img_parts = []
            ai_threshold >>= 1
            for x in range(ai_rows):
                img_parts.append([])
                if state.interrupted:
                    break
                for y in range(ai_cols):
                    if state.interrupted:
                        break
                    img_parts[x].append(upscaler.scaler.upscale(original_image.crop(((ai_threshold*x)-upscale_border, (ai_threshold*y)-upscale_border, min(ai_threshold*(x+1),original_image.width)+upscale_border, min(ai_threshold*(y+1),original_image.height)+upscale_border)), 2, upscaler.data_path))
            ai_threshold <<= 1
            upscale_border <<= 1
            if(len(img_parts)==ai_rows and len(img_parts[ai_rows-1])==ai_cols):
                #non-blended parts
                #centers
                for x in range(ai_rows):
                    if state.interrupted:
                        break
                    for y in range(ai_cols):
                        if state.interrupted:
                           break
                        new_image.paste(img_parts[x][y].crop((upscale_border*2,upscale_border*2,ai_threshold,ai_threshold)),((ai_threshold*x)+upscale_border,(ai_threshold*y)+upscale_border))
                #top and bottom
                for x in range(ai_rows):
                    if state.interrupted:
                        break
                    new_image.paste(img_parts[x][0].crop((upscale_border*2,upscale_border,ai_threshold,upscale_border*2)),((ai_threshold*x)+upscale_border,0))
                    new_image.paste(img_parts[x][ai_cols-1].crop((upscale_border*2,ai_threshold,ai_threshold,ai_threshold+upscale_border)),((ai_threshold*x)+upscale_border,(ai_threshold*ai_cols)-upscale_border))
                #left and right
                for y in range(ai_cols):
                    if state.interrupted:
                        break
                    new_image.paste(img_parts[0][y].crop((upscale_border,upscale_border*2,upscale_border*2,ai_threshold)),(0,(ai_threshold*y)+upscale_border))
                    new_image.paste(img_parts[ai_rows-1][y].crop((ai_threshold,upscale_border*2,ai_threshold+upscale_border,ai_threshold)),((ai_threshold*ai_rows)-upscale_border,(ai_threshold*y)+upscale_border))
                #corners
                new_image.paste(img_parts[0][0].crop((upscale_border,upscale_border,upscale_border*2,upscale_border*2)),(0,0))
                new_image.paste(img_parts[0][ai_cols-1].crop((upscale_border,ai_threshold,upscale_border*2,ai_threshold+upscale_border)),(0,(ai_threshold*ai_cols)-upscale_border))
                new_image.paste(img_parts[ai_rows-1][0].crop((ai_threshold,upscale_border,ai_threshold+upscale_border,upscale_border*2)),((ai_threshold*ai_rows)-upscale_border,0))
                new_image.paste(img_parts[ai_rows-1][ai_cols-1].crop((ai_threshold,ai_threshold,ai_threshold+upscale_border,ai_threshold+upscale_border)),((ai_threshold*ai_rows)-upscale_border,(ai_threshold*ai_cols)-upscale_border))
                
                blend_mask = Image.new("RGB", (upscale_border, ai_threshold),color=0)
                horizontal_mask_data = np.array(blend_mask).astype(np.float64)
                blend_mask.close()
                blend_mask = Image.new("RGB", (ai_threshold, upscale_border),color=0)
                vertical_mask_data = np.array(blend_mask).astype(np.float64)
                blend_mask.close()
                for thresh_i in range(ai_threshold):
                    for border_i in range(upscale_border):
                        for i in range(3):
                            horizontal_mask_data[thresh_i][border_i][i] = border_i*.5/upscale_border
                            vertical_mask_data[border_i][thresh_i][i] = border_i*.5/upscale_border
                #blended parts
                for x in range(ai_rows):
                    if state.interrupted:
                        break
                    for y in range(ai_cols):
                        if state.interrupted:
                            break
                        if(x<ai_rows-1):
                            main_right_pixels = np.array(img_parts[x][y].crop((ai_threshold,upscale_border,ai_threshold+upscale_border,ai_threshold+upscale_border)))
                            main_right_border_pixels = np.array(img_parts[x][y].crop((ai_threshold+upscale_border,upscale_border,ai_threshold+(upscale_border*2),ai_threshold+upscale_border)))
                            right_left_pixels = np.array(img_parts[x+1][y].crop((upscale_border,upscale_border,upscale_border*2,ai_threshold+upscale_border)))
                            right_left_border_pixels = np.array(img_parts[x+1][y].crop((0,upscale_border,upscale_border,ai_threshold+upscale_border)))
                            
                            new_image.paste(Image.fromarray(((main_right_pixels*(1-horizontal_mask_data))+(right_left_border_pixels*horizontal_mask_data)).astype(np.uint8)),((ai_threshold*(x+1))-upscale_border,ai_threshold*y))
                            new_image.paste(Image.fromarray(((main_right_border_pixels*(.5-horizontal_mask_data))+(right_left_pixels*(.5+horizontal_mask_data))).astype(np.uint8)),(ai_threshold*(x+1),ai_threshold*y))
                        if(y<ai_cols-1):
                            main_bottom_pixels = np.array(img_parts[x][y].crop((upscale_border,ai_threshold,ai_threshold+upscale_border,ai_threshold+upscale_border)))
                            main_bottom_border_pixels = np.array(img_parts[x][y].crop((upscale_border,ai_threshold+upscale_border,ai_threshold+upscale_border,ai_threshold+(upscale_border*2))))
                            bottom_top_pixels = np.array(img_parts[x][y+1].crop((upscale_border,upscale_border,ai_threshold+upscale_border,upscale_border*2)))
                            bottom_top_border_pixels = np.array(img_parts[x][y+1].crop((upscale_border,0,ai_threshold+upscale_border,upscale_border)))
                            
                            new_image.paste(Image.fromarray(((main_bottom_pixels*(1-vertical_mask_data))+(bottom_top_border_pixels*vertical_mask_data)).astype(np.uint8)),(ai_threshold*x,(ai_threshold*(y+1))-upscale_border))
                            new_image.paste(Image.fromarray(((main_bottom_border_pixels*(.5-vertical_mask_data))+(bottom_top_pixels*(.5+vertical_mask_data))).astype(np.uint8)),(ai_threshold*x,ai_threshold*(y+1)))
                for x in range(ai_rows):
                    for y in range(ai_cols):
                        img_parts[x][y].close()
            
        def perform_upscale(original_image):
            print(f"Upscaling using {upscaler.name}")
            upscale_border_power = 3
            upscale_border = 2<<upscale_border_power
            ai_threshold = 512<<tiled_ai_upscale_threshold
            w = original_image.width
            h = original_image.height
            expanded_img = None
            
            if(blend_seams):
                expanded_img = Image.new("RGB", (w+(upscale_border*2), h+(upscale_border*2)))
                create_borders(original_image,expanded_img,upscale_border,w,h)
                w = expanded_img.width
                h = expanded_img.height
                ai_rows = math.ceil(w*2.0/ai_threshold)
                ai_cols = math.ceil(h*2.0/ai_threshold)
                if(tiled_ai_upscale and (ai_rows>1 or ai_cols>1)):
                    img = Image.new("RGB", (w*2, h*2))
                    perform_tiled_upscale(expanded_img,img,upscale_border,ai_rows,ai_cols,ai_threshold,w,h)
                else:
                    img = upscaler.scaler.upscale(expanded_img, 2, upscaler.data_path)
                expanded_img.close()
                
                upscale_border_power += 1
                upscale_border *= 2
                w = img.width-upscale_border*2
                h = img.height-upscale_border*2
                
                left_border_pixels = np.array(img.crop((0,upscale_border,upscale_border,h+upscale_border)))
                right_border_pixels = np.array(img.crop((w+upscale_border,upscale_border,w+(upscale_border*2),h+upscale_border)))
                up_border_pixels = np.array(img.crop((upscale_border,0,w+upscale_border,upscale_border)))
                down_border_pixels = np.array(img.crop((upscale_border,h+upscale_border,w+upscale_border,h+(upscale_border*2))))

                left_pixels = np.array(img.crop((upscale_border,upscale_border,upscale_border*2,h+upscale_border)))
                right_pixels = np.array(img.crop((w,upscale_border,w+upscale_border,h+upscale_border)))
                up_pixels = np.array(img.crop((upscale_border,upscale_border,w+upscale_border,upscale_border*2)))
                down_pixels = np.array(img.crop((upscale_border,h,w+upscale_border,h+upscale_border)))

                up_left_border_pixels = np.array(img.crop((0,0,upscale_border,upscale_border)))
                up_right_border_pixels = np.array(img.crop((w+upscale_border,0,w+(upscale_border*2),upscale_border)))
                down_left_border_pixels = np.array(img.crop((0,h+upscale_border,upscale_border,h+(upscale_border*2))))
                down_right_border_pixels = np.array(img.crop((w+upscale_border,h+upscale_border,w+(upscale_border*2),h+(upscale_border*2))))

                up_left_pixels = np.array(img.crop((upscale_border,upscale_border,upscale_border*2,upscale_border*2)))
                up_right_pixels = np.array(img.crop((w,upscale_border,w+upscale_border,upscale_border*2)))
                down_left_pixels = np.array(img.crop((upscale_border,h,upscale_border*2,h+upscale_border)))
                down_right_pixels = np.array(img.crop((w,h,w+upscale_border,h+upscale_border)))
                img = img.crop((upscale_border,upscale_border,w+upscale_border,h+upscale_border))
                
                blend_mask = Image.new("RGB", (upscale_border, h),color=0)
                horizontal_mask_data = np.array(blend_mask).astype(np.float64)
                blend_mask.close()
                blend_mask = Image.new("RGB", (w, upscale_border),color=0)
                vertical_mask_data = np.array(blend_mask).astype(np.float64)
                blend_mask.close()
                for border_i in range(upscale_border):
                    for i in range(3):
                        for y in range(h):
                            horizontal_mask_data[y][border_i][i] = border_i*.5/upscale_border
                        for x in range(w):
                            vertical_mask_data[border_i][x][i] = border_i*.5/upscale_border
                blend_mask = Image.new("RGB", (upscale_border, upscale_border),color=0)
                ul_corner_mask_data = np.array(blend_mask).astype(np.float64)
                blend_mask.close()
                for y in range(upscale_border):
                    for x in range(upscale_border):
                        for i in range(3):
                            ul_corner_mask_data[y][x][i] = max(x,y)*.5/upscale_border
                ur_corner_mask_data = np.fliplr(ul_corner_mask_data)
                bl_corner_mask_data = np.flipud(ul_corner_mask_data)
                br_corner_mask_data = np.fliplr(bl_corner_mask_data)
                
                left_pixels = (right_border_pixels*(.5-horizontal_mask_data))+(left_pixels*(.5+horizontal_mask_data))
                right_pixels = (right_pixels*(1-horizontal_mask_data))+(left_border_pixels*horizontal_mask_data)
                up_pixels = (down_border_pixels*(.5-vertical_mask_data))+(up_pixels*(.5+vertical_mask_data))
                down_pixels = (down_pixels*(1-vertical_mask_data))+(up_border_pixels*vertical_mask_data)

                up_left_pixels = (up_left_pixels*(1-br_corner_mask_data))+(down_right_border_pixels*(br_corner_mask_data))
                up_right_pixels = (up_right_pixels*(1-bl_corner_mask_data))+(down_left_border_pixels*(bl_corner_mask_data))
                down_left_pixels = (down_left_pixels*(1-ur_corner_mask_data))+(up_right_border_pixels*(ur_corner_mask_data))
                down_right_pixels = (down_right_pixels*(1-ul_corner_mask_data))+(up_left_border_pixels*(ul_corner_mask_data))

                img.paste(Image.fromarray(left_pixels.astype(np.uint8)),(0,0))
                img.paste(Image.fromarray(right_pixels.astype(np.uint8)),(w-upscale_border,0))
                img.paste(Image.fromarray(up_pixels.astype(np.uint8)),(0,0))
                img.paste(Image.fromarray(down_pixels.astype(np.uint8)),(0,h-upscale_border))
                img.paste(Image.fromarray(up_left_pixels.astype(np.uint8)),(0,0))
                img.paste(Image.fromarray(up_right_pixels.astype(np.uint8)),(w-upscale_border,0))
                img.paste(Image.fromarray(down_left_pixels.astype(np.uint8)),(0,h-upscale_border))
                img.paste(Image.fromarray(down_right_pixels.astype(np.uint8)),(w-upscale_border,h-upscale_border))
                
                original_image.close()
                print(f"Current size: {w}, {h}")
                return img
            else:
                ai_rows = math.ceil(w*2.0/ai_threshold)
                ai_cols = math.ceil(h*2.0/ai_threshold)
                if(tiled_ai_upscale and (ai_rows>1 or ai_cols>1)):
                    img = Image.new("RGB", (w*2, h*2))
                    perform_tiled_upscale(original_image,img,upscale_border,ai_rows,ai_cols,ai_threshold,w,h)
                else:
                    img = upscaler.scaler.upscale(original_image, 2, upscaler.data_path)
                original_image.close()
                print(f"Current size: {w*2}, {h*2}")
                return img
                
        for sd_upscale_i in range(sd_upscale_iters):
            if state.interrupted:
                break
        
            if(expanded_img != None):
                img.close()
                img = expanded_img
            
            if(upscaler.name != "None"): 
                for i in range(upscale_iters):
                    if state.interrupted:
                        break
                    img = perform_upscale(img)
                    if not state.interrupted:
                        if(return_partials):
                            partials.append(img.copy())
                            all_seeds.append(original_seed)
                            all_prompts.append(original_prompt)
                            if initial_info[0] is None:
                                infotexts.append("")
                            else:
                                infotexts.append(initial_info[0])
                        elif(return_whole):
                            last_whole_image.close()
                            last_whole_image = img.copy()
                            last_image_text = f"Interrupted - Last Operation: Upscale, Size: {last_whole_image.width},{last_whole_image.height}"

            w = img.width
            h = img.height
                
            border_and_tiles = determine_border_and_tiles(w,h)
            border = border_and_tiles[0]
            rows = border_and_tiles[1]
            cols = border_and_tiles[2]
            non_border_w = tile_w - (border*2)
            non_border_h = tile_h - (border*2)
            
            print(f"Border: {border}; Center: {non_border_w}, {non_border_h}; Overlap: {border_and_tiles[3]}, {border_and_tiles[4]}")

            offset_x = int((w+(border*2)-tile_w)/(rows-1))
            offset_y = int((h+(border*2)-tile_h)/(cols-1))
            
            def add_pixel_channel_noise(pixel_channel):
                return pixel_channel+math.copysign(randint(0,int(math.ceil(p.denoising_strength*noise_strength*abs((pixel_channel-128)/128.0)))),-(pixel_channel-128))
            np_add_pixel_channel_noise = np.vectorize(add_pixel_channel_noise)
            
            noise_grey_inv = 1
            if(add_noise and noise_application==0):
                if state.interrupted:
                    break
                seed(p.seed)
                ai_threshold = 512<<tiled_ai_upscale_threshold
                ai_rows = math.ceil(w*1.0/ai_threshold)
                ai_cols = math.ceil(h*1.0/ai_threshold)
                if(tiled_ai_upscale and (ai_rows>1 or ai_cols>1)):
                    noise_tiles = ai_rows*ai_cols
                    for x in range(ai_rows):
                        if state.interrupted:
                            break
                        for y in range(ai_cols):
                            if state.interrupted:
                                break
                            print(f"Adding Noise ({1+y+(x*ai_cols)}/{noise_tiles})")
                            img_crop = img.crop((x*ai_threshold,y*ai_threshold,(x+1)*ai_threshold,(y+1)*ai_threshold))
                            noise_crop = Image.fromarray(np_add_pixel_channel_noise(np.array(img_crop)).astype(np.uint8))
                            img.paste(noise_crop,(x*ai_threshold,y*ai_threshold))
                            img_crop.close()
                            noise_crop.close()
                else:
                    print("Adding Noise")
                    img_data = np.array(img)
                    img.close()
                    img_data = np_add_pixel_channel_noise(img_data)
                    img_data = img_data.astype(np.uint8)
                    img = Image.fromarray(img_data)
                noise_grey_inv = 1+(p.denoising_strength*noise_strength/320.0)
                
            img = ImageEnhance.Sharpness(ImageEnhance.Contrast(ImageEnhance.Color(img).enhance(saturation_amount*noise_grey_inv)).enhance(contrast_amount*noise_grey_inv)).enhance(sharpen_amount)
               
            expanded_img = Image.new("RGB", (w+(border*2), h+(border*2)))
            create_borders(img,expanded_img,border,w,h)
            img.close()
            
            mask = Image.new("L", (tile_w, tile_h),color=0)
            mask.paste(Image.new("L", (tile_w-(border<<1), tile_h-(border<<1)),color=255),(border,border,tile_w-border,tile_h-border))
            if p.image_mask is not None:
                p.image_mask.close()
            p.image_mask = mask
            
            overlap_area = edge_overlap
            if(overlap_area-(border*2)<16):
                overlap_area = int(math.ceil((16+(border*2))/64.0)*64)
            mask_w = None
            mask_h = None
            if(blend_seams):
                mask_w = Image.new("L", (overlap_area, tile_h),color=0)
                mask_w.paste(Image.new("L", (overlap_area-(border<<1), tile_h-(border<<1)),color=255),(border,border,overlap_area-border,tile_h-border))
                mask_h = Image.new("L", (tile_w, overlap_area),color=0)
                mask_h.paste(Image.new("L", (tile_w-(border<<1), overlap_area-(border<<1)),color=255),(border,border,tile_w-border,overlap_area-border))
            
            devices.torch_gc()

            p.n_iter = 1
            p.do_not_save_grid = True
            p.do_not_save_samples = True
            p.inpaint_full_res = False
            p.inpainting_mask_invert = False

            print(f"Rows: {rows}, Columns: {cols}, Tiles: {rows*cols}")
            
            def process_part(x,y,pwidth,pheight):
                if(increment_seed):
                    p.seed = p.seed + 1
                posx = int(x*offset_x)
                posy = int(y*offset_y)
                #ensure inpaint touches far borders
                if(x==rows-1):
                    posx = w+(border*2)-pwidth
                if(y==cols-1):
                    posy = h+(border*2)-pheight
                    
                p.init_images[0].close()
                p.init_images = [expanded_img.crop((posx, posy, posx+pwidth, posy+pheight))]
                if(add_noise and noise_application==1):
                    seed(p.seed)
                    noise_border = border+12#(p.mask_blur*2)
                    img_data = np.array(p.init_images[0].crop((noise_border,noise_border,pwidth-noise_border,pheight-noise_border)))
                    img_data = np_add_pixel_channel_noise(img_data)
                    img_data = img_data.astype(np.uint8)
                    p.init_images[0].paste(Image.fromarray(img_data),(noise_border,noise_border))
                    
                processed = processing.process_images(p)
                if initial_info[0] is None:
                    initial_info[0] = processed.info
                    
                if(len(processed.images)>0):
                    expanded_img.paste(processed.images[0],(posx,posy))
                    #update outside border
                    if(x==0):
                        expanded_img.paste(processed.images[0].crop((border,0,pwidth-border,pheight)),(w+border,posy))
                    if(y==0):
                        expanded_img.paste(processed.images[0].crop((0,border,pwidth,pheight-border)),(posx,h+border))
                    if(x==0 and y==0):
                        expanded_img.paste(processed.images[0].crop((border,border,pwidth-border,pheight-border)),(w+border,h+border))
                     
            def process_parts(startx,starty):
                for y in range(starty,cols,2):
                    if state.interrupted:
                        break
                    for x in range(startx,rows,2):
                        if state.interrupted:
                            break
                        process_part(x,y,tile_w,tile_h)
                    
            if(pattern==0):
                process_parts(0,0)
                process_parts(1,1)
                process_parts(1,0)
                process_parts(0,1)
            else:
                process_parts(0,0)
                process_parts(0,1)
                process_parts(1,0)
                process_parts(1,1)
                    
            expanded_img = expanded_img.crop((border, border, w+border, h+border))
            
            if(blend_seams and not state.interrupted):
                print()
                print(f"Edge Tiles: {rows+cols}")
                img = expanded_img
                expanded_img = Image.new("RGB", (w+(overlap_area>>1), h+(overlap_area>>1)))
                expand_for_tiling(img,expanded_img,border,w,h)
                img.close()
                original_denoising = p.denoising_strength
                p.denoising_strength = edge_denoising
                p.width = overlap_area
                p.image_mask.close()
                p.image_mask = mask_w
                for y in range(0,cols,2):
                    if state.interrupted:
                        break
                    process_part((w-(overlap_area/2.0))/offset_x,y,overlap_area,tile_h)
                for y in range(1,cols,2):
                    if state.interrupted:
                        break
                    process_part((w-(overlap_area/2.0))/offset_x,y,overlap_area,tile_h)
                p.width = tile_w
                expanded_img.paste(expanded_img.crop((w,0,w+(overlap_area>>1),h+(overlap_area>>1))),(0,0))
                p.height = overlap_area
                p.image_mask.close()
                p.image_mask = mask_h
                for x in range(0,rows,2):
                    if state.interrupted:
                        break
                    process_part(x,(h-(overlap_area/2.0))/offset_y,tile_w,overlap_area)
                for x in range(1,rows,2):
                    if state.interrupted:
                        break
                    process_part(x,(h-(overlap_area/2.0))/offset_y,tile_w,overlap_area)
                p.denoising_strength = original_denoising
                p.height = tile_h
                expanded_img.paste(expanded_img.crop((0,h,w,h+(overlap_area>>1))),(0,0))
                expanded_img.paste(expanded_img.crop((w,h-(overlap_area>>1),w+(overlap_area>>1),h)),(0,h-(overlap_area>>1)))
                expanded_img.paste(expanded_img.crop((w,h,w+(overlap_area>>1),h+(overlap_area>>1))),(0,0))
                expanded_img = expanded_img.crop((0, 0, w, h))
                
                mask_w.close()
                mask_h.close()
            mask.close()
                
            if not state.interrupted:
                if(return_partials):
                    partials.append(expanded_img.copy())
                    all_seeds.append(original_seed)
                    all_prompts.append(original_prompt)
                    if initial_info[0] is None:
                        infotexts.append("")
                    else:
                        infotexts.append(initial_info[0])
                elif(return_whole):
                    last_whole_image.close()
                    last_whole_image = expanded_img.copy()
                    last_image_text = f"Interrupted - Last Operation: SD Inpaint, Size: {last_whole_image.width},{last_whole_image.height}"
        
        if initial_info[0] is None:
            initial_info[0] = ""
            
        if expanded_img is None:
            expanded_img = img
            
        if(upscaler.name != "None"): 
            for i in range(final_upscale_iters):
                if state.interrupted:
                    break
                expanded_img = perform_upscale(expanded_img)
                if not state.interrupted:
                    if(return_partials):
                        partials.append(expanded_img.copy())
                        all_seeds.append(original_seed)
                        all_prompts.append(original_prompt)
                        if initial_info[0] is None:
                            infotexts.append("")
                        else:
                            infotexts.append(initial_info[0])
                    elif(return_whole):
                        last_whole_image.close()
                        last_whole_image = expanded_img.copy()
                        last_image_text = f"Interrupted - Last Operation: Upscale, Size: {last_whole_image.width},{last_whole_image.height}"
                
        if not state.interrupted:
            expanded_img = ImageEnhance.Sharpness(ImageEnhance.Contrast(ImageEnhance.Color(expanded_img).enhance(final_saturation_amount)).enhance(final_contrast_amount)).enhance(final_sharpen_amount)
        
        if(downscale and not state.interrupted):
            print(f"Downscaling using {downscale_type}")
            downscale_type_idx = Image.Resampling.NEAREST
            if(downscale_type=="Box"):
                downscale_type_idx = Image.Resampling.BOX
            elif(downscale_type=="Bilinear"):
                downscale_type_idx = Image.Resampling.BILINEAR
            elif(downscale_type=="Hamming"):
                downscale_type_idx = Image.Resampling.HAMMING
            elif(downscale_type=="Bicubic"):
                downscale_type_idx = Image.Resampling.BICUBIC
            elif(downscale_type=="Lanczos"):
                downscale_type_idx = Image.Resampling.LANCZOS
            expanded_img = expanded_img.resize((int(expanded_img.width*(downscale_amount*.01)),int(expanded_img.height*(downscale_amount*.01))),downscale_type_idx)
            print(f"New size: {expanded_img.width}, {expanded_img.height}")
            
        p.seed = original_seed
        
        if(return_partials):
            if(final_saturation_amount!=1 or final_contrast_amount!=1 or final_sharpen_amount!=1 or downscale or state.interrupted):
                partials.append(expanded_img)
            return Processed(p, partials, all_seeds=all_seeds, all_prompts=all_prompts, infotexts=infotexts)
        if state.interrupted and return_whole and last_image_text!=None:
            print(last_image_text)
            return Processed(p, [last_whole_image], original_seed, initial_info[0])
        return Processed(p, [expanded_img], original_seed, initial_info[0])
