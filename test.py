from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import os

training_path = '2023-02-03T07-26-52_mri_autoencoder_stage2_registration_2' # line + landmark + pre
training_path = '2023-02-02T07-28-53_mri_autoencoder_stage2_registration_2' # landmark + pre

save_path = f'output/{training_path}/png'
os.makedirs(save_path, exist_ok = True)

config_name = training_path.split('_')[0]

config = OmegaConf.load(f'logs/{training_path}/configs/{config_name}-project.yaml')
config['model']['params']['ckpt_path'] = f'/workspace/jjh/25.CephGeneration/latent-diffusion/logs/{training_path}/checkpoints/last_.ckpt'
config['data']['params']['batch_size'] = 1
config['data']['params']['validation']['params']['conditioning'] = ['pre', 'landmark']
config['data']['params']['train']['params']['conditioning'] = ['pre', 'landmark']
config['model']['params']['cond_stage_config']['params']['args']['conditioning'] = ['pre', 'landmark']

data_loader = instantiate_from_config(config['data'])
data_loader.prepare_data()
data_loader.setup()
device = torch.device('cuda:0')
valid_dataset = data_loader.datasets['validation']
print('dataset', type(valid_dataset))
model = instantiate_from_config(config['model']).to(device)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 1, shuffle = False)
model.eval()



with torch.no_grad():
    for i, data in enumerate(valid_dataloader):

        if not data['landmarkwithpre']['path'][0][0].split('/')[-2] in ['KH_199_AB', 'SNU_095_AB', 'KH_075_AB', 'SNU_277_PREB', 'KH_131_PREB', 'KBU_036_AB', 'KH_086_AB']:
            continue
        

        data['image'] = data['image'].to(device)
        data['landmarkwithpre']['landmark'] = data['landmarkwithpre']['landmark'].to(device)
        data['landmarkwithpre']['pre'] = data['landmarkwithpre']['pre'].to(device)
#         data['landmarkwithpre']['line'] = data['landmarkwithpre']['line'].to(device)
        
        path = data['landmarkwithpre']['path'][0][0].split('/')[-2]
        print(path)

        
        z, c, x, xrec, xc = model.get_input(data, model.first_stage_key,return_first_stage_outputs=True,force_c_encode=True,return_original_cond=True,bs=1)
        with model.ema_scope("Plotting"):
            samples, z_denoise_row = model.sample_log(cond=c,batch_size=1,ddim=True,ddim_steps=200,eta=1., quantize_denoised=False)


        x_samples = model.decode_first_stage(samples)

#         xrec = xrec[0][0].detach().cpu().numpy()
        x = x[0][0].detach().cpu().numpy()
        x_samples = x_samples[0][0].detach().cpu().numpy()
        pre = data['landmarkwithpre']['pre'][0][:,:,0].detach().cpu().numpy()
        
        os.makedirs(f'testset/{path}', exist_ok=True)
        Image.fromarray(((np.clip(x, -1, 1) + 1)/2 * 255).astype(np.uint8)).save(f'testset/{path}/{path}_B.png')
        Image.fromarray(((np.clip(pre, -1, 1) + 1)/2 * 255).astype(np.uint8)).save(f'testset/{path}/{path}_A_PRE.png')
        Image.fromarray(((np.clip(x_samples, -1, 1) + 1)/2 * 255).astype(np.uint8)).save(f'testset/{path}/{path}_FAKE_no_Line.png')
        

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
#         img.save(f'{save_path}/{i:03}.png')

#         np.save(f'output3/{i:02}_recon.npy', xrec)
#         np.save(f'output3/{i:02}_real.npy', x)
#         np.save(f'output3/{i:02}_fake.npy', x_samples)