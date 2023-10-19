import os

def get_exp_name(args):
    name = args.name + "-" + args.backbone
    name = name + '_0'
    log_path = os.path.join(args.logs, name)
    
    while(os.path.exists(log_path)):
        log_str = log_path.split('_')
        num = int(log_str[-1]) +1
        log_str[-1] = str(num)
        log_path = "_".join(log_str)

    return os.path.split(log_path)[-1]