import math

import modules.scripts as scripts
import gradio as gr
from PIL import Image, ImageEnhance

from modules import processing, shared, sd_samplers, images, devices
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state

import numpy as np
import random
from random import randint, seed
    

class Script(scripts.Script):
    def title(self):
        return "SD Inpaint Upscale"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        info = gr.HTML("<p style=\"margin-bottom:0.75em\">Make sure that an inpainting model is selected and that \"Inpaint area\" is \"Whole picture\".</p>")
        
        spacer = gr.HTML("<p style=\"margin-bottom:1.5em\"></p>")
        upscaler_index = gr.Radio(label='Upscaler', choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name, type="index")
        
        info2 = gr.HTML("<p style=\"margin-bottom:0.75em\"></p>")
        min_border = gr.Slider(minimum=0, maximum=240, step=1, label='Minimum Border', value=32)
        inpaint_overlap = gr.Slider(minimum=0, maximum=240, step=1, label='Maximum Inpaint Overlap', value=32)
        
        info3 = gr.HTML("<p style=\"margin:0.75em\">Adjacent may blend better but drift more.</p>", visible=False)
        pattern = gr.Radio(label="Pattern", choices=['Diagonal','Adjacent'], value='Diagonal', type="index", visible=False)
        
        info4 = gr.HTML("<p style=\"margin:0.75em\">Disable to create similar patterns within the image.</p>", visible=False)
        increment_seed = gr.Checkbox(label="Increment Seed",value=True, visible=False)
        
        info5 = gr.HTML("<p style=\"margin:0.75em\">Should only be used for tiled images. Takes longer. Edge overlap changes tile size perpendicular to edge for speed.</p>")
        blend_seams = gr.Checkbox(label="Inpaint Over Edges",value=True)
        edge_overlap = gr.Slider(minimum=64, maximum=2048, step=64, label='Edge Overlap', value=512)
        
        info6 = gr.HTML("<p style=\"margin:0.75em\">Experimental, may add detail. Applied after upscale but before SD inpaint.</p>")
        add_noise = gr.Checkbox(label="Add Noise",value=False)
        noise_strength = gr.Slider(minimum=1, maximum=128, step=1, label='Noise Strength', value=32)
        noise_application = gr.Radio(label="Noise Application", choices=['Global','Per Tile'], value='Global', type="index", visible=False)
        
        info7 = gr.HTML("<p style=\"margin:0.75em\">1 = unchanged. 2 is the full effect. Applied after upscale but before SD inpaint.</p>")
        saturation_amount = gr.Slider(minimum=1, maximum=2, step=.01, label='Saturation', value=1, visible=False)
        contrast_amount = gr.Slider(minimum=1, maximum=2, step=.01, label='Contrast', value=1, visible=False)
        sharpen_amount = gr.Slider(minimum=1, maximum=10, step=.1, label='Sharpening', value=1)
        
        info8 = gr.HTML("<p style=\"margin:0.75em\">Use the selected upscaler this many times before doing each SD inpaint.</p>", visible=False)
        upscale_iters = gr.Slider(minimum=0, maximum=8, step=1, label='Upscale Iterations Per SD Inpaint', value=1, visible=False)
        info9 = gr.HTML("<p style=\"margin:0.75em\"></p>", visible=False)
        sd_upscale_iters = gr.Slider(minimum=0, maximum=8, step=1, label='SD Inpaint Iterations', value=1, visible=False)
        
        info10 = gr.HTML("<p style=\"margin:0.75em\">Use the selected upscaler this many times after completing the SD inpaint upscale.</p>", visible=False)
        final_upscale_iters = gr.Slider(minimum=0, maximum=4, step=1, label='Final Upscale Iterations', value=0, visible=False)
        
        info11 = gr.HTML("<p style=\"margin:0.75em\">1 = unchanged. 2 is the full effect. Applied after final upscale.</p>", visible=False)
        final_saturation_amount = gr.Slider(minimum=1, maximum=4, step=.1, label='Final Saturation', value=1, visible=False)
        final_contrast_amount = gr.Slider(minimum=1, maximum=4, step=.1, label='Final Contrast', value=1, visible=False)
        final_sharpen_amount = gr.Slider(minimum=1, maximum=10, step=.1, label='Final Sharpening', value=1, visible=False)
        
        info12 = gr.HTML("<p style=\"margin:0.75em\">Percent refers to the final perimiter. 50% is the inverse of an upscale.</p>", visible=False)
        downscale = gr.Checkbox(label="Downscale Final Image",value=False, visible=False)
        downscale_type = gr.Radio(label='Downscaler', choices=["Nearest","Box","Bilinear","Hamming","Bicubic","Lanczos"], value="Lanczos", type="value", visible=False)
        downscale_amount = gr.Slider(minimum=1, maximum=99, step=1, label='Percent', value=50, visible=False)

        return [info, info2, info3, info4, info5, info6, info7, info8, info9, info10, info11, info12, spacer, upscaler_index, min_border, inpaint_overlap, pattern, increment_seed, blend_seams, edge_overlap, add_noise, noise_strength, noise_application, saturation_amount, contrast_amount, sharpen_amount, upscale_iters, sd_upscale_iters, final_upscale_iters, final_saturation_amount, final_contrast_amount, final_sharpen_amount, downscale, downscale_type, downscale_amount]

    def run(self, p, _, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _s, upscaler_index, min_border, inpaint_overlap, pattern, increment_seed, blend_seams, edge_overlap, add_noise, noise_strength, noise_application, saturation_amount, contrast_amount, sharpen_amount, upscale_iters, sd_upscale_iters, final_upscale_iters, final_saturation_amount, final_contrast_amount, final_sharpen_amount, downscale, downscale_type, downscale_amount):
    
        processing.fix_seed(p)
        original_seed = p.seed
        upscaler = shared.sd_upscalers[upscaler_index]
        initial_info = [None]
        img = p.init_images[0]
        expanded_img = None
        tile_w = p.width
        tile_h = p.height
        w = img.width
        h = img.height
        max_border = (min(tile_w,tile_h)>>1)-16
            
        if(upscaler.name != "None"): 
            p.extra_generation_params["SDIU Upscaler"] = upscaler.name
        if(sd_upscale_iters>0):
            p.extra_generation_params["SDIU Border Min"] = min_border
            p.extra_generation_params["SDIU Inpaint Overlap"] = inpaint_overlap
            #p.extra_generation_params["SDIU Pattern"] = 'Diagonal' if pattern==0 else 'Adjacent'
        if(upscale_iters!=1):
            p.extra_generation_params["SDIU Upscaler Iterations"] = upscale_iters
        if(sd_upscale_iters!=1):
            p.extra_generation_params["SDIU SD Upscaler Iterations"] = sd_upscale_iters
        #if(increment_seed):
            #p.extra_generation_params["SDIU Increment Seed"] = "true"
        if(blend_seams):
            p.extra_generation_params["SDIU Inpaint Over Edges"] = "true"
            p.extra_generation_params["SDIU Edge Overlap"] = edge_overlap
        if(add_noise):
            p.extra_generation_params["SDIU Noise"] = noise_strength
            #p.extra_generation_params["SDIU Noise Application"] = 'Global' if noise_application==0 else 'Per Tile'
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
        job_position = [0]
        if(sd_upscale_iters>0):
            if(upscaler.name != "None" and upscale_iters>0): 
                for i in range(upscale_iters,(sd_upscale_iters*upscale_iters)+1,upscale_iters):
                    border_and_tiles = determine_border_and_tiles(w*math.pow(2,i),h*math.pow(2,i))
                    job_count += (border_and_tiles[1]*border_and_tiles[2])
                    edge_job_count += (border_and_tiles[1]+border_and_tiles[2])
            else:
                border_and_tiles = determine_border_and_tiles(w,h)
                job_count += (border_and_tiles[1]*border_and_tiles[2])
                edge_job_count += (border_and_tiles[1]+border_and_tiles[2])
        else:
            job_count = 1
        if(blend_seams):
            state.job_count = job_count+edge_job_count
        else:
            state.job_count = job_count
        
        upscale_count = upscale_iters
        if(sd_upscale_iters>0):
            upscale_count *= sd_upscale_iters
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
            print(f"Final image size: {final_width},{final_height}. Need to process {job_count} normal tiles and {edge_job_count} edge tiles, {state.job_count} in total.")
        else:
            print(f"Final image size: {final_width},{final_height}. Need to process {job_count} tiles.")
            
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
            
        def perform_upscale(original_image):
            print(f"Upscaling using {upscaler.name}")
            #Upscaling can cause the edges to not tile properly, so the upscale_border stuff blends a copied piece touching the far edge.
            upscale_border_power = 3
            upscale_border = 2<<upscale_border_power
            w = original_image.width
            h = original_image.height
            expanded_img = None
            if(blend_seams):
                expanded_img = Image.new("RGB", (w+(upscale_border*2), h+(upscale_border*2)))
                create_borders(original_image,expanded_img,upscale_border,w,h)
                img = upscaler.scaler.upscale(expanded_img, 2, upscaler.data_path)
                expanded_img.close()
                
                upscale_border_power += 1
                upscale_border *= 2
                w *= 2
                h *= 2
                w = img.width
                h = img.height
                
                img_data = np.array(img).astype(np.uint32)
                left_pixels = np.array(img.crop((w-upscale_border,0,w,h))).astype(np.uint32)
                right_pixels = np.array(img.crop((0,0,upscale_border,h))).astype(np.uint32)
                up_pixels = np.array(img.crop((0,h-upscale_border,w,h))).astype(np.uint32)
                down_pixels = np.array(img.crop((0,0,w,upscale_border))).astype(np.uint32)
                img.close()
                
                for border_i in range(upscale_border):
                    blend = (upscale_border-1-border_i)<<(19-upscale_border_power)
                    blend_inv = (2<<20)-blend
                    blend2 = border_i<<(19-upscale_border_power)
                    blend_inv2 = (2<<20)-blend2
                    for y in range(upscale_border,h-upscale_border,1):
                        for i in range (3):
                            img_data[y][upscale_border+border_i][i] = ((img_data[y][upscale_border+border_i][i]*blend_inv) + (left_pixels[y][border_i][i]*blend))>>21
                            img_data[y][w+border_i-(upscale_border*2)][i] = ((img_data[y][w+border_i-(upscale_border*2)][i]*blend_inv2) + (right_pixels[y][border_i][i]*blend2))>>21
                    for x in range(upscale_border,w-upscale_border,1):
                        for i in range (3):
                            img_data[upscale_border+border_i][x][i] = ((img_data[upscale_border+border_i][x][i]*blend_inv) + (up_pixels[border_i][x][i]*blend))>>21
                            img_data[h+border_i-(upscale_border*2)][x][i] = ((img_data[h+border_i-(upscale_border*2)][x][i]*blend_inv2) + (down_pixels[border_i][x][i]*blend2))>>21
                            
                print(f"Current size: {(w-(upscale_border*2))}, {(h-(upscale_border*2))}")
                return Image.fromarray(img_data.astype(np.uint8)).crop((upscale_border,upscale_border,w-upscale_border,h-upscale_border))
            else:
                img = upscaler.scaler.upscale(original_image, 2, upscaler.data_path)
                print(f"Current size: {(w*2)}, {(h*2)}")
                return img
                
        for sd_upscale_i in range(sd_upscale_iters):
            if state.interrupted:
                break
        
            if(expanded_img != None):
                img = expanded_img
            
            if(upscaler.name != "None"): 
                for i in range(upscale_iters):
                    if state.interrupted:
                        break
                    img = perform_upscale(img)

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
                print("Adding Noise")
                seed(p.seed)
                img_data = np.array(img)
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
            #TODO these don't seem to work
            p.inpaint_full_res = False
            p.inpainting_mask_invert = False

            print(f"Rows: {rows}, Columns: {cols}, Tiles: {(rows*cols)}")
            
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
                    
                p.init_images = [expanded_img.crop((posx, posy, posx+pwidth, posy+pheight))]
                if(add_noise and noise_application==1):
                    seed(p.seed)
                    noise_border = border+12#(p.mask_blur*2)
                    img_data = np.array(p.init_images[0].crop((noise_border,noise_border,pwidth-noise_border,pheight-noise_border)))
                    img_data = np_add_pixel_channel_noise(img_data)
                    img_data = img_data.astype(np.uint8)
                    p.init_images[0].paste(Image.fromarray(img_data),(noise_border,noise_border))
                    
                processed = processing.process_images(p)
                p.init_images[0].close()
                if initial_info[0] is None:
                    initial_info[0] = processed.info
                job_position[0] += 1
                    
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
            
            if(blend_seams):
                print()
                print(f"Edge Tiles: {(rows+cols)}")
                img = expanded_img
                expanded_img = Image.new("RGB", (w+(overlap_area>>1), h+(overlap_area>>1)))
                expand_for_tiling(img,expanded_img,border,w,h)
                img.close()
                original_denoising = p.denoising_strength
                if(p.denoising_strength>.2):
                    p.denoising_strength = .2
                p.width = overlap_area
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
                
            mask.close()
            if(blend_seams):
                mask_w.close()
                mask_h.close()
            #print()
            #print(f"{job_position[0]} tiles processed")
        
        if initial_info[0] is None:
            initial_info[0] = ""
            expanded_img = img
            
        if(upscaler.name != "None"): 
            for i in range(final_upscale_iters):
                if state.interrupted:
                    break
                expanded_img = perform_upscale(expanded_img)
                
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
            
        return Processed(p, [expanded_img], original_seed, initial_info[0])
        
        





