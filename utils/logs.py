import os

def get_exp_name(args):
    num = 0
    name = args.name + "-" + args.backbone
    name = name + '_{}'.format(num)
    log_path = os.path.join(args.logs, name)
    
    while(os.path.exists(log_path)):
        log_str = log_path.split('_')
        num = num + 1
        log_str[-1] = str(num)
        log_path = "_".join(log_str)
    
    if args._continue:
        log_str = log_path.split('_')
        if num > 0:
            log_str[-1] = str(num - 1)
        log_path = "_".join(log_str)

    return os.path.split(log_path)[-1]