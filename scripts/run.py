#!/usr/bin/env python3
import argparse
import os
from os import path
import subprocess


def derive_paths(scene_json: str, method: str, name: str):
    # scene_json like /.../nerf_synthetic/drums/transforms_train.json
    data_root = path.dirname(scene_json)
    # Derive dataset and scene
    # .../<dataset>/<scene>/transforms_*.json
    scene_dir = path.dirname(scene_json)
    scene = path.basename(scene_dir)
    dataset = path.basename(path.dirname(scene_dir))

    out_dir = path.join('outputs', dataset, scene, method, name)
    return data_root, dataset, scene, out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, type=str,
                        help='Path to transforms_train.json')
    parser.add_argument('--network', required=False, type=str, default=None,
                        help='Config file to merge (maps to -c/--config)')
    parser.add_argument('--n_steps', required=False, type=int, default=None,
                        help='Maps to --n_iters in opt.py')
    parser.add_argument('--method', required=True, type=str)
    parser.add_argument('--name', required=True, type=str)
    parser.add_argument('--stride', required=False, type=int, default=25,
                        help='Stride for test rendering (default 25)')
    parser.add_argument('--normals_mode', required=False, type=str, choices=['fd','autograd'], default=None,
                        help='Surface normals mode to pass to training (fd or autograd)')
    parser.add_argument('--negate_normals', action='store_true', default=False,
                        help='Flip normal direction before dot product')
    parser.add_argument('--detach_normals', action='store_true', default=False,
                        help='Detach gradients from normals (no backprop to density through normals)')
    parser.add_argument('--renderer_backend', required=False, type=str, choices=['cuvol','nvol','svox1'], default=None,
                        help='Renderer backend to pass through (-B/--renderer_backend)')
    parser.add_argument('--extra', nargs=argparse.REMAINDER, default=[],
                        help='Extra flags passed to training')
    args = parser.parse_args()

    if not path.basename(args.scene).startswith('transforms_train'):
        raise ValueError('Expected --scene to point to transforms_train.json')

    data_root, dataset, scene, out_dir = derive_paths(args.scene, args.method, args.name)
    os.makedirs(out_dir, exist_ok=True)

    # Build training command
    opt_py = path.join(path.dirname(__file__), '..', 'opt', 'opt.py')
    opt_py = path.abspath(opt_py)

    cmd = ['python', opt_py, '-t', out_dir, data_root]
    if args.network:
        cmd += ['-c', args.network]
    if args.n_steps is not None:
        cmd += ['--n_iters', str(args.n_steps)]
    # Map method to vector potential flag
    if args.method.lower() == 'surface':
        cmd += ['--use_vector_potential']
        if args.normals_mode:
            cmd += ['--normals_mode', args.normals_mode]
        if args.negate_normals:
            cmd += ['--negate_normals']
        if args.detach_normals:
            cmd += ['--detach_normals']
    if args.renderer_backend:
        cmd += ['-B', args.renderer_backend]
    cmd += args.extra

    print('Running training:')
    print(' '.join(cmd))
    subprocess.check_call(cmd)

    # After training, render test images with stride
    ckpt = path.join(out_dir, 'ckpt.npz')
    render_py = path.join(path.dirname(__file__), '..', 'opt', 'render_imgs.py')
    render_py = path.abspath(render_py)

    # Prefer using the same data root; render_imgs expects data_dir
    render_cmd = [
        'python', render_py,
        ckpt,
        data_root,
        '--stride', str(args.stride)
    ]
    if args.network:
        render_cmd += ['-c', args.network]
    if args.renderer_backend:
        render_cmd += ['-B', args.renderer_backend]

    print('Rendering test set:')
    print(' '.join(render_cmd))
    subprocess.check_call(render_cmd)


if __name__ == '__main__':
    main() 