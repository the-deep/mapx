import pandas as pd
import numpy as np
import pylab as p
from skimage import morphology,measure,util,draw
from scipy.ndimage import morphology as morphology_scipy
import geopandas as gpd
import os

pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 2000)
pd.set_option('display.max_rows', 2000)
p.style.use('ggplot')


def get_hu_(im):
    mu = measure.moments_central(im)
    nu = measure.moments_normalized(mu)
    hu = measure.moments_hu(nu)
    return hu

def get_hu_coords(coords):
    mu = measure.moments_coords_central(coords)
    nu = measure.moments_normalized(mu)
    hu = measure.moments_hu(nu)
    return hu

def convert_geometry_to_array(x,y,size=500):
    coords = np.vstack([x,y]).T
    maxx = np.max(coords,axis=0)
    minn = np.min(coords,axis=0)
    rangee = np.max(maxx - minn)
    coords = ((size - 1) * (coords - minn) / rangee).astype(int)
    im = np.zeros((size, size), dtype=np.uint8)
    rr,cc = draw.polygon(coords[:,0],coords[:,1])
    im[rr,cc] = 1
    return im

def convert_geometry_to_array_matplotlib(x,y):
    fig = p.figure()
    p.plot(x,y,c='black')
    p.axis('off')
    # fig.tight_layout(pad=0) # <<<<<<<<<< better with this???
    fig.canvas.draw()

    im = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    im = im.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # p.show()
    p.close()

    im = im[:,:,0]
    im = util.invert(im)

    im = morphology_scipy.binary_fill_holes(im)
    # p.imshow(im,cmap='Greys')
    # p.axis('off')
    # p.show()
    return im



def get_hu(data):
    hus = []
    print(data.shape)
    for j,(i,r) in enumerate(data.iterrows()):
        print(j)
        x,y = r['geometry'].exterior.coords.xy
        im = convert_geometry_to_array_matplotlib(x,y)
        # im_ = convert_geometry_to_array(x,y,size=500)
        hu = get_hu_(im)
        # print(hu)
        hus.append(hu)

    hus = pd.DataFrame(-np.log(np.abs(hus)) * np.sign(hus),columns=['h1','h2','h3','h4','h5','h6','h7'])
    data = pd.concat([data.reset_index(drop=True),hus.reset_index(drop=True)],axis=1)
    return data, hus


def clean_shapes(shapes):
    '''
    the shape files formatting is inconsistent, and should be improved
    quick hack to get data for POC
    '''
    # cols = ['title','LGAName','NAME_0','NAME_1','NAME_2','NAME_ENGLI','terr_name','VARNAME_2','geometry']
    # shapes = shapes[cols]
    # # for c in shapes.columns:
    # #     print(c)
    # #     try:
    # #         print(shapes[c].unique())
    # #     except:
    # #         pass
    # #     print()
    shapes['country'] = shapes['terr_name'].fillna(shapes['NAME_0']).fillna(shapes['NAME_ENGLI']).fillna(shapes['title'])
    shapes['region'] = shapes['VARNAME_2'].fillna(shapes['NAME_2']).fillna(shapes['NAME_1']).fillna(shapes['LGAName'])
    return shapes[['country','region','geometry']]

def get_shape_data():
    shapes = []
    for name in os.listdir('data/shapes'):
        data = gpd.read_file(f'data/shapes/{name}')
        data['title'] = name.split('.')[0]
        shapes.append(data)
    shapes = pd.concat(shapes,sort=True).reset_index(drop=True)
    return clean_shapes(shapes)

def main():
    shapes = get_shape_data()
    shapes = shapes.explode().reset_index(drop=True)
    shapes, hus = get_hu(shapes)
    return shapes



if __name__ == '__main__':
    shapes = main()
    shapes.to_file('data/shapes.shp')


'''
This requires the geoshapes files found at:
https://drive.google.com/drive/folders/1cGYhT9ezhsy5rIVn77ki_JPqEo6wh7hd
'''
