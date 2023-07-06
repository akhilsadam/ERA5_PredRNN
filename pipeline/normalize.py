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
        'pde_u': 'u',
    }
short_inv={v: k for k, v in short.items()}
norm_dict=\
    {
        'u_wind': [-20, 20],
        'v_wind': [-20, 20],
        'temp': [210, 305],
        'sea_press': [98000, 105000],
        'precip': [0, 0.0025],
    }
norm_func=\
    {
        'u_wind': lambda x: np.vstack(x)/norm_dict['u_wind'][1],
        'v_wind': lambda x: np.vstack(x)/norm_dict['v_wind'][1],
        'temp': lambda x: (np.vstack(x) - norm_dict['temp'][0])/(norm_dict['temp'][1] - norm_dict['temp'][0]),
        'sea_press': lambda x: (np.vstack(x) - norm_dict['sea_press'][0])/(norm_dict['sea_press'][1] - norm_dict['sea_press'][0]),
        'precip': lambda x: ((force_min(np.vstack(x))) - norm_dict['precip'][0])/(norm_dict['precip'][1] - norm_dict['precip'][0]),
    }