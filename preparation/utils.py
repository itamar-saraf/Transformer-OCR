import os


def get_path(which_path: str, args):
    path = None
    if which_path == 'gt.mat':
        path = os.path.normpath(os.path.join(os.getcwd() + '/..' + args.gt_file))
    elif which_path == 'images':
        path = os.path.normpath(os.path.join(os.getcwd() + '/../' + 'data/oxford'))
    elif which_path == 'saved_images' or which_path == 'val' or which_path == 'train':
        path = os.path.normpath(os.path.join(os.getcwd() + '/../'))
    elif which_path == 'list':
        path = os.path.normpath(os.path.join(os.getcwd() + '/../' + args.gt_file))
    return path
