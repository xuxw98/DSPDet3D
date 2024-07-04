import os
import numpy as np
from plyfile import PlyData
from argparse import ArgumentParser

def read_mesh_vertices_rgb(filename):
    """Read XYZ and RGB for each vertex.

    Args:
        filename(str): The name of the mesh vertices file.

    Returns:
        Vertices. Note that RGB values are in 0-255.
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x']
        vertices[:, 1] = plydata['vertex'].data['y']
        vertices[:, 2] = plydata['vertex'].data['z']
        vertices[:, 3] = plydata['vertex'].data['red']
        vertices[:, 4] = plydata['vertex'].data['green']
        vertices[:, 5] = plydata['vertex'].data['blue']
    return vertices



def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--data-dir', type=str, default='demo', help='dir to data')

    args = parser.parse_args()
    points = read_mesh_vertices_rgb(args.data_dir)
    points = points.astype(np.float32)
    
    scene_name = os.path.basename(args.data_dir)[:-4]
    out_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'demo_data',scene_name+'.bin')
    with open(out_dir, 'wb') as f:
        points.tofile(f)
if __name__ == '__main__':
    main()
