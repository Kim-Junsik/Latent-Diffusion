from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import os
import glob
import argparse






parser = argparse.ArgumentParser()
parser.add_argument('--training_path', help='Log의 Training Path', required=True)
parser.add_argument('--no_ddim', help='use DDPM', action = 'store_true')
parser.add_argument('--ddim_use_original_steps', help='use DDIM original steps', action = 'store_true')
parser.add_argument('--ddim_step', help='DDIM step', default = 200, type = int)
parser.add_argument('--external', help='External', action = 'store_true')
parser.add_argument('--eta', help='eta of ddim', default = 0., type = float)
args = parser.parse_args()
training_path, eta, ddim_step, ddim, ddim_use_original_steps = args.training_path, args.eta, args.ddim_step, (not args.no_ddim), args.ddim_use_original_steps
# if ddim_use_original_steps
external = args.external

print(ddim_use_original_steps)

# experiment1
# training_path = '2023-02-03T07-26-52_mri_autoencoder_stage2_registration_2' # line + landmark + pre
# training_path = '2023-02-02T07-28-53_mri_autoencoder_stage2_registration_2' # landmark + pre

# get experiment_number
# experiment1
experiments = glob.glob('/workspace/jjh/25.CephGeneration/experiment4/*')

experiment_number = 0
for experiment in experiments:
    if training_path in experiment:
        experiment_number = int(experiment.split('/')[-1].split('.')[0])
if experiment_number == 0:
    experiment_number = len(experiments) + 1


config_name = training_path.split('_')[0]
# config_name = ''

config = OmegaConf.load(f'logs/{training_path}/configs/{config_name}-project.yaml')
ckpt = glob.glob(f'/workspace/jjh/25.CephGeneration/latent-diffusion/logs/{training_path}/checkpoints/last.ckpt')
ckpt.sort()
config['model']['params']['ckpt_path'] = ckpt[-1]
config['data']['params']['batch_size'] = 1


condition = config['model']['params']['cond_stage_config']['params']['args']['conditioning']
data_loader = instantiate_from_config(config['data'])
data_loader.prepare_data()
data_loader.setup()
device = torch.device('cuda:0')
valid_dataset = data_loader.datasets['test'] if external else data_loader.datasets['validation']
print('dataset', len(valid_dataset))
print('model', config['model']['params']['ckpt_path'])
model = instantiate_from_config(config['model']).to(device)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 1, shuffle = False)
model.eval()

first_model = 'vq' if 'VQ' in config['model']['params']['first_stage_config']['target'] else 'kl'
attn_resolution = ''.join([str(attn_resol) + '_' for attn_resol in config['model']['params']['first_stage_config']['params']['ddconfig']['attn_resolutions']])
conditioning = ''.join([str(attn_resol) + '_' for attn_resol in config['model']['params']['cond_stage_config']['params']['args']['conditioning']])
downsampling = str(2 ** config['model']['params']['cond_stage_config']['params']['args']['n_stages'])


# name = f'{first_model}-{downsampling}_firstattn:{attn_resolution}cond:{conditioning}ddimstep:{ddim_step}_eta:{eta}_[{training_path}]'

# name = f'{first_model}-{downsampling}_firstattn_{attn_resolution}cond_{conditioning}({training_path})'
name = training_path
sampling_config = f'ddimstep_{ddim_step}_eta_{eta}' if ddim else 'ddpm'
# sampling_config = 'external_' + sampling_config if external else sampling_config

name = name.replace('-','_')

experiments = glob.glob('/workspace/jjh/25.CephGeneration/experiment4/*')
experiment_number = 0
for experiment in experiments:
    if name == experiment.split('/')[-1].split('.')[1]:
        experiment_number = int(experiment.split('/')[-1].split('.')[0])
        break
print(experiment_number)
if experiment_number == 0:
    experiment_number = len(experiments) + 1
    
name = os.path.join(f'{experiment_number}.' + name, sampling_config, 'external') if external else os.path.join(f'{experiment_number}.' + name, sampling_config, 'internal')

print(name)
print(condition)
save_path = f'/workspace/jjh/25.CephGeneration/experiment_interpolation/{name}/'
os.makedirs(save_path, exist_ok = True)
condition.append('movement')

with torch.no_grad():
    model.eval()
    for i, data in enumerate(valid_dataloader):
        data['image'] = data['image'].to(device)
#         data['landmarkwithpre']['landmark'] = data['landmarkwithpre']['landmark'].to(device)
#         data['landmarkwithpre']['pre'] = data['landmarkwithpre']['pre'].to(device)
        
        for cond in condition:
            data['landmarkwithpre'][cond] = data['landmarkwithpre'][cond].to(device)
        
        path = data['landmarkwithpre']['path'][0][0].split('/')[-2]
        surgical_movment_prediction = data['landmarkwithpre']['movement']
        
        print(path)
        
#         for interpolation_num in range(8):


        z, c, x, xrec, xc = model.get_input(data, model.first_stage_key,return_first_stage_outputs=True,force_c_encode=True,return_original_cond=True,bs=1)
        with model.ema_scope("Plotting"):
            samples, z_denoise_row = model.sample_log(cond=c,batch_size=1,ddim=ddim,ddim_steps=ddim_step,eta=eta, ddim_use_original_steps = ddim_use_original_steps, quantize_denoised=False)
            pre_z = model.first_stage_model.encode(data['landmarkwithpre']['pre'].permute(0, 3, 1, 2).float())


        for interpolation_num in range(20):
            samples_interpolate = pre_z + (samples - pre_z) * interpolation_num / 10
            x_samples = model.decode_first_stage(samples_interpolate)

            x_ = x[0][0].detach().cpu().numpy()
            x_samples = x_samples[0][0].detach().cpu().numpy()
            pre = data['landmarkwithpre']['pre'][0][:,:,0].detach().cpu().numpy()
            os.makedirs(f'{save_path}/FAKE', exist_ok=True)
            os.makedirs(f'{save_path}/FAKE/{path}', exist_ok=True)
            os.makedirs(f'{save_path}/FAKE/{path}/latent', exist_ok=True)


            array = np.concatenate([x_, x_samples, pre], 1)
            array = np.stack([array, array, array], 2)
            array = np.clip(array, -1, 1)
            array = ((array + 1)/2)* 255
            array = array.astype(np.uint8)

            font_size = 80
            font = ImageFont.truetype('arial.ttf', font_size) # arial.ttf 글씨체, font_size=15

            img = Image.fromarray(array)
            draw = ImageDraw.Draw(img)
            draw.text((1024 * 0 + 10, 10), 'B', (255,0,0), font=font) 
            draw.text((1024 * 1 + 10, 10), 'Fake', (0,255,0), font=font) 
            draw.text((1024 * 2 + 10, 10), 'A/PRE', (0,0,255), font=font) 
            img.save(f'{save_path}/{path}.png')
            Image.fromarray(((np.clip(x_samples, -1, 1) + 1)/2 * 255).astype(np.uint8)).save(f'{save_path}/FAKE/{path}/latent/latent_{path}_FAKE_{interpolation_num:02}.png')
