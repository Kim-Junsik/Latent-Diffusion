from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import os
import copy

training_path = '2023-02-03T07-26-52_mri_autoencoder_stage2_registration_2'
save_path = f'output/{training_path}/landmark_scale_gif'
os.makedirs(save_path, exist_ok = True)

config_name = training_path.split('_')[0]

config = OmegaConf.load(f'logs/{training_path}/configs/{config_name}-project.yaml')
config['model']['params']['ckpt_path'] = f'/workspace/jjh/25.CephGeneration/latent-diffusion/logs/{training_path}/checkpoints/last_.ckpt'
config['data']['params']['batch_size'] = 1

data_loader = instantiate_from_config(config['data'])
data_loader.prepare_data()
data_loader.setup()
device = torch.device('cuda:1')
valid_dataset = data_loader.datasets['validation']
print('dataset', type(valid_dataset))
model = instantiate_from_config(config['model']).to(device)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 1, shuffle = False)
model.eval()
interpolation_numbers = 50
image_interpolation_number = 1

with torch.no_grad():
    for i, data in enumerate(valid_dataloader):
        
#         if not i in [0,1,3,4,8,15,16,20,23,24,35,52,55,57,59,62,63,66,85,87,90,93,95,96,100,107,114,11,17,18]:
#             continue

        if not data['landmarkwithpre']['path'][0][0].split('/')[-2] in ['KH_199_AB', 'SNU_095_AB', 'KH_075_AB', 'SNU_277_PREB', 'KH_131_PREB', 'KBU_036_AB', 'KH_086_AB']:
            continue
        data['image'] = data['image'].to(device)
        data['landmarkwithpre']['landmark'] = data['landmarkwithpre']['landmark'].to(device)
        data['landmarkwithpre']['pre'] = data['landmarkwithpre']['pre'].to(device)
        data['landmarkwithpre']['line'] = data['landmarkwithpre']['line'].to(device)
        
        path = data['landmarkwithpre']['path'][0][0].split('/')[-2]
        print(path)
#         moving_value = data['landmarkwithpre']['landmark'][:,:,2:4] - data['landmarkwithpre']['landmark'][:,:,0:2]
        
        unconditional_conditioning = copy.deepcopy(data)
        unconditional_conditioning['landmarkwithpre']['landmark'][:,:,2:4] = unconditional_conditioning['landmarkwithpre']['landmark'][:,:,0:2]
        _, unconditional_conditioning, _, _, _ = model.get_input(unconditional_conditioning, model.first_stage_key,return_first_stage_outputs=True,force_c_encode=True,return_original_cond=True,bs=1)


        interpolate_data = data.copy()
        outputs = []
        
        

        with model.ema_scope("Plotting"):
            for scale in np.linspace(start=0.2, stop = 2, num = interpolation_numbers):
#                 interpolate_data['landmarkwithpre']['landmark'][:,:,2:4] = data['landmarkwithpre']['landmark'][:,:,0:2] + moving_value * (interpolation / interpolation_numbers)
                z, c, x, xrec, xc = model.get_input(data, model.first_stage_key,return_first_stage_outputs=True,force_c_encode=True,return_original_cond=True,bs=1)
                samples, z_denoise_row = model.sample_log(cond=c,batch_size=1,ddim=True,ddim_steps=200,eta=0.,quantize_denoised=False, unconditional_conditioning = unconditional_conditioning, unconditional_guidance_scale = scale)
                outputs.append(model.decode_first_stage(samples)[0][0].detach().cpu().numpy())
        
#         outputs = [model.decode_first_stage(output)[0][0].detach().cpu().numpy() for output in outputs]
        outputs = [((np.clip(output, -1, 1) + 1 )/ 2 * 255) for output in outputs]
        outputs = [Image.fromarray(output.astype(np.uint8)) for output in outputs]

        os.makedirs(f'{save_path}/image/{path}', exist_ok = True)
        for idx in range(interpolation_numbers):
            outputs[idx * image_interpolation_number].save(f'{save_path}/image/{path}/{idx:03}.png')
        outputs[-1].save(f'{save_path}/image/{path}/{interpolation_numbers:03}.png')

        outputs = [output.resize((512, 512)) for output in outputs]
        outputs[0].save(f'{save_path}/{path}.gif', save_all = True, append_images=outputs[1:],loop=0xff, duration=3)
            
# #             samples, z_denoise_row = model.sample(cond=c,batch_size=1, return_intermediates = True)

#         x_samples = model.decode_first_stage(samples)

#         xrec = xrec[0][0].detach().cpu().numpy()
#         x = x[0][0].detach().cpu().numpy()
#         x_samples = x_samples[0][0].detach().cpu().numpy()
#         pre = data['landmarkwithpre']['pre'][0][:,:,0].detach().cpu().numpy()
        
#         array = np.concatenate([x, x_samples, pre], 1)
#         array = np.stack([array, array, array], 2)
#         array = np.clip(array, -1, 1)
#         array = ((array + 1)/2)* 255
#         array = array.astype(np.uint8)
        
#         font_size = 40
#         font = ImageFont.truetype('arial.ttf', font_size) # arial.ttf 글씨체, font_size=15

#         img = Image.fromarray(array)
#         draw = ImageDraw.Draw(img)
#         draw.text((1024 * 0,10), 'B', (255,0,0), font=font) 
#         draw.text((1024 * 1,10), 'Fake', (0,255,0), font=font) 
#         draw.text((1024 * 2,10), 'A/PRE', (0,0,255), font=font) 
#         img.save(f'output3/{i:03}.png')

#         np.save(f'output3/{i:02}_recon.npy', xrec)
#         np.save(f'output3/{i:02}_real.npy', x)
#         np.save(f'output3/{i:02}_fake.npy', x_samples)