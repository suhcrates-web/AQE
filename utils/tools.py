
def inject_param(model, param_list, name_list, type0):
    for name, param in zip(name_list, param_list): 
        names = name.split('.')
        module_name = '.'.join(names[:-1])  # Get module path except for the last element
        param_name = names[-1] 
        module = model
        
        for sub_name in module_name.split('.'):
            if sub_name:
                module = getattr(module, sub_name)
        
        if type0=='clone':
            setattr(module, param_name, param.clone())
        elif type0 == 'detach_reqgrad':
            setattr(module, param_name, param.detach().requires_grad_())
        elif type0 == 'detach':
            setattr(module, param_name, param.detach())
        elif type0 == 'assign':
            setattr(module, param_name, param)
