import imageio
import numpy as np
from utils.flow_utils import vis_flow
from utils.imtools import vfshown

flow = np.zeros((1,2,200,250))
flow[:,0,:,:] = 1
vfshown(flow[: ,0 ,: ,:], flow[: ,1 ,: ,:], sample_rate=10, save_fig=True,
        file_name='./tmp/10_flow')
img = vis_flow(flow.squeeze())
imageio.imsave('./tmp/10_color.png' , img)



flow = np.zeros((1,2,200,250))
flow[:,1,:,:] = 1
vfshown(flow[: ,0 ,: ,:], flow[: ,1 ,: ,:], sample_rate=10, save_fig=True,
        file_name='./tmp/01_flow')
img = vis_flow(flow.squeeze())
imageio.imsave('./tmp/01_color.png' , img)



flow = np.zeros((1,2,200,250))
flow[:,0,:,:] = -1
vfshown(flow[: ,0 ,: ,:], flow[: ,1 ,: ,:], sample_rate=10, save_fig=True,
        file_name='./tmp/-10_flow')
img = vis_flow(flow.squeeze())
imageio.imsave('./tmp/-10_color.png' , img)


flow = np.zeros((1,2,200,250))
flow[:,1,:,:] = -1
vfshown(flow[: ,0 ,: ,:], flow[: ,1 ,: ,:], sample_rate=10, save_fig=True,
        file_name='./tmp/0-1_flow')
img = vis_flow(flow.squeeze())
imageio.imsave('./tmp/0-1_color.png' , img)

