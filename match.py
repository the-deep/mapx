import pandas as pd
import numpy as np
import imageio
import pylab as p
from skimage import morphology,measure,util,draw,filters
import alphashape
from numpy import random
import geopandas as gpd
import timing
from scipy import ndimage
import create_shapes as cs
import itertools as it

random.seed(18)

pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 2000)
pd.set_option('display.max_rows', 2000)
p.style.use('ggplot')


def get_blobs(im):
    r = filters.laplace(im[:,:,0], ksize=3)
    # g = filters.laplace(im[:,:,1], ksize=3)
    # b = filters.laplace(im[:,:,2], ksize=3)
    im = ~(np.abs(r)>0)
    im, number_of_objects = ndimage.label(im)
    blobs = ndimage.find_objects(im)
    props = measure.regionprops(im)

    imz = []
    for i,(j,prop) in enumerate(zip(blobs,props),1):
        blob = im[j]==i
        if blob.shape[0]>50 and blob.shape[1]>50:
            imz.append(blob)
    return imz


def matcher(blobs,shapes,country,use_country_name=False):
    if use_country_name:
        shapes = shapes.loc[shapes['country']==country]

    hus = []
    for im in blobs:
        hu = cs.get_hu_(im)
        hus.append(-np.log(np.abs(hu)) * np.sign(hu))
    hus = np.array(hus)

    shapes_hu = shapes[['h1','h2','h3','h4','h5','h6','h7']].values

    diff = np.sum(np.abs(shapes_hu.T - hus[:,:,None]) / np.abs(shapes_hu.T),axis=1)

    ix = np.argsort(diff,axis=1)[:,0]
    dd = np.sort(diff,axis=1)[:,0]

    cut = 9
    ddd = np.argsort(dd)[:cut]
    w = int(cut**0.5)
    f,ax = p.subplots(2*w,w)
    for (x,y),i in zip(it.product(range(w),range(w)),ddd):
        ax[(2*x,y)].imshow(blobs[i])
        shapes.iloc[[ix[i]]]['geometry'].plot(ax=ax[2*x+1,y])
        ax[2*x,y].axis('off')
        ax[2*x+1,y].axis('off')
    p.show()


def create_chloropleth_maps(shapes):
    ims = []
    countries = []
    for c,g in shapes.groupby('country'):
        g = g.dropna()
        if len(g)>5:
            fig,ax = p.subplots()
            ax.axis('off')
            colours = np.random.rand(len(g),3)
            g['geometry'].plot(ax=ax,facecolor=colours,lw=1,edgecolor=[0,0,0])
            # p.title(c)
            fig.set_dpi(500)
            fig.canvas.draw()
            # p.show()
            im = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            im = im.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            ims.append(im)
            countries.append(c)
            p.close()

            # p.imshow(im)
            # p.axis('off')
            # p.show()
    return ims, countries

shapes = gpd.read_file('data/shapes.shp')

shapes['num_points'] = shapes['geometry'].apply(lambda r:len(r.exterior.coords.xy[0]))
shapes = shapes.loc[shapes['num_points']>20]


# ims = [imageio.imread('data/test_maps.png')]
ims, countries = create_chloropleth_maps(shapes)


for im,c in zip(ims,countries):
    print(c)
    p.imshow(im)
    p.axis('off')
    p.show()
    blobs = get_blobs(im)
    matcher(blobs,shapes,c,use_country_name=True)


'''
This creates simple unlabelled randomly coloured chloropleth maps from the shape file. Then naively segments the image using Laplacian edge detection, then object label. Each object blob above a minimum size has its Hu moments calculated, and these are compared to the geoshape Hu moments to find the best fit. The best fits are then plotted. Ideally all the best fits would match great.. currently they dont match too great...
'''
