from core.data_provider import custom_PDE,custom_CDS, mnist_new #, bair

datasets_map = {
    'mnist': mnist_new,
    'custom_PDE': custom_PDE,
    'custom_CDS': custom_CDS,
    #'action': kth_action,
    # 'bair': bair,
}


def data_provider(dataset_name, train_data_paths, valid_data_paths, batch_size,
                  img_height, img_width, seq_length, injection_action, concurent_step,
                  img_channel, img_layers, is_testing=True, is_training=True, is_WV=True, sanity_check=False):
    if dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % dataset_name)
    img_layers = [int(x) for x in img_layers.split(',')]
    if dataset_name in ['mnist', 'custom_PDE', 'custom_CDS'] :
        if is_testing:
            valid_data_list = valid_data_paths.split(',')
            test_input_param = {'paths': valid_data_list,
                                'minibatch_size': batch_size,
                                'image_height': img_height,
                                'image_width': img_width,
                                'img_channel': img_channel,
                                'input_data_type': 'float32',
                                'concurent_step':concurent_step,
                                'is_output_sequence': True,
                                'name': dataset_name + ' test iterator',
                                'img_layers': img_layers,
                                'is_WV': is_WV, 
                                'total_length': seq_length,
                                'testing' : True,
                                }
            test_input_handle = datasets_map[dataset_name].InputHandle(test_input_param)
            test_input_handle.begin(do_shuffle=False)
        if is_training:
            if isinstance(train_data_paths,list):
                train_data_list = train_data_paths
            else:
                train_data_list = train_data_paths.split(',')
            # print(f"train_data_list:{train_data_list}")
            train_input_param = {'paths': train_data_list,
                                 'minibatch_size': batch_size,
                                 'input_data_type': 'float32',
                                 'image_height': img_height,
                                 'image_width': img_width,
                                 'img_channel': img_channel,
                                 'concurent_step':1,
                                 'is_output_sequence': True,
                                 'name': dataset_name + ' train iterator',
                                 'img_layers': img_layers,
                                 'is_WV': is_WV, 
                                'total_length': seq_length,
                                'testing' : False,
                                }
            train_input_handle = datasets_map[dataset_name].InputHandle(train_input_param)
            train_input_handle.begin(do_shuffle=True)
    if is_testing and is_training:
        return train_input_handle, test_input_handle
    elif is_testing:
        return test_input_handle
    elif is_training:
        return train_input_handle

    if dataset_name == 'action':
        input_param = {'paths': valid_data_list,
                       'image_height': img_height,
                       'image_width': img_width,
                       'minibatch_size': batch_size,
                       'seq_length': seq_length,
                       'input_data_type': 'float32',
                       'name': dataset_name + ' iterator'}
        input_handle = datasets_map[dataset_name].DataProcess(input_param)
        if is_training:
            train_input_handle = input_handle.get_train_input_handle()
            train_input_handle.begin(do_shuffle=True)
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False)
            return train_input_handle, test_input_handle
        else:
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False)
            return test_input_handle

    if dataset_name == 'bair':
        test_input_param = {'valid_data_paths': valid_data_list,
                            'train_data_paths': train_data_list,
                            'batch_size': batch_size,
                            'image_height': img_height,
                            'image_height': img_width,
                            'seq_length': seq_length,
                            'injection_action': injection_action,
                            'input_data_type': 'float32',
                            'name': dataset_name + 'test iterator'}
        input_handle_test = datasets_map[dataset_name].DataProcess(test_input_param)
        test_input_handle = input_handle_test.get_test_input_handle()
        test_input_handle.begin(do_shuffle=False)
        if is_training:
            train_input_param = {'valid_data_paths': valid_data_list,
                                 'train_data_paths': train_data_list,
                                 'image_width': img_width,
                                 'image_height': img_width,
                                 'batch_size': batch_size,
                                 'seq_length': seq_length,
                                 'injection_action': injection_action,
                                 'input_data_type': 'float32',
                                 'name': dataset_name + ' train iterator'}
            input_handle_train = datasets_map[dataset_name].DataProcess(train_input_param)
            train_input_handle = input_handle_train.get_train_input_handle()
            train_input_handle.begin(do_shuffle=True)
            return train_input_handle, test_input_handle
        else:
            return test_input_handle