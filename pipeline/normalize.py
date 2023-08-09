import numpy as np

def force_min(data):
    data[data==0] = 1e-9
    return data

short=\
    {
        '10m_u_component_of_wind': 'u_wind',
        '10m_v_component_of_wind': 'v_wind',
        '2m_temperature': 'temp',
        'surface_pressure': 'sea_press',
        'total_precipitation': 'precip',
        'geopotential': 'geo_dyn',
        'vertical_integral_of_potential_internal_and_latent_energy': 'moist_static', 
        'vertical_integral_of_potential_and_internal_energy': 'dry_static',
        'pde_u': 'u',
    }
short_inv={v: k for k, v in short.items()}
norm_dict=\
    {
        'u_wind': [-20, 20],
        'v_wind': [-20, 20],
        'temp': [210, 305],
        'sea_press': [98000, 105000],
        'moist_static': [1.4e9, 2.9e9],
        'geo_dyn': [5000, 6000], # 'geopotential 500' #TODO make this work for multiple levels
        'dry_static': [1.4e9, 2.9e9],
        'precip': [0, 0.0025],
    }
norm_func=\
    {
        'u_wind': lambda x: np.vstack(x)/norm_dict['u_wind'][1],
        'v_wind': lambda x: np.vstack(x)/norm_dict['v_wind'][1],
        'temp': lambda x: (np.vstack(x) - norm_dict['temp'][0])/(norm_dict['temp'][1] - norm_dict['temp'][0]),
        'sea_press': lambda x: (np.vstack(x) - norm_dict['sea_press'][0])/(norm_dict['sea_press'][1] - norm_dict['sea_press'][0]),
        'moist_static': lambda x: (np.vstack(x) - norm_dict['moist_static'][0])/(norm_dict['moist_static'][1] - norm_dict['moist_static'][0]),
        'geo_dyn': lambda x: (np.vstack(x) - norm_dict['geo_dyn'][0])/(norm_dict['geo_dyn'][1] - norm_dict['geo_dyn'][0]),
        'dry_static': lambda x: (np.vstack(x) - norm_dict['dry_static'][0])/(norm_dict['dry_static'][1] - norm_dict['dry_static'][0]),
        'precip': lambda x: ((force_min(np.vstack(x))) - norm_dict['precip'][0])/(norm_dict['precip'][1] - norm_dict['precip'][0]),
    }
norm_inv=\
    {
        'u_wind': lambda x: x*norm_dict['u_wind'][1],
        'v_wind': lambda x: x*norm_dict['v_wind'][1],
        'temp': lambda x: x*(norm_dict['temp'][1] - norm_dict['temp'][0]) + norm_dict['temp'][0],
        'sea_press': lambda x: x*(norm_dict['sea_press'][1] - norm_dict['sea_press'][0]) + norm_dict['sea_press'][0],
        'moist_static': lambda x: x*(norm_dict['moist_static'][1] - norm_dict['moist_static'][0]) + norm_dict['moist_static'][0],
        'geo_dyn': lambda x: x*(norm_dict['geo_dyn'][1] - norm_dict['geo_dyn'][0]) + norm_dict['geo_dyn'][0],
        'dry_static': lambda x: x*(norm_dict['dry_static'][1] - norm_dict['dry_static'][0]) + norm_dict['dry_static'][0],
        'precip': lambda x: x*(norm_dict['precip'][1] - norm_dict['precip'][0]) + norm_dict['precip'][0],
    }