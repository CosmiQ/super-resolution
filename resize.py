import numpy as np
from osgeo import gdal,osr
import cv2
import tensorflow as tf



flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('zooming', 4.0, 'Amount to rescale original image.')
flags.DEFINE_string('input_image','in.tif','Input image.')
flags.DEFINE_string('output_image','out.tif','Output image.')

# open training data and compute initial cost function averaged over the entire image
ds = gdal.Open(FLAGS.input_image)
im_raw = np.swapaxes(np.swapaxes(ds.ReadAsArray(),0,1), 1,2)
im_small = cv2.resize(im_raw, (int(im_raw.shape[1]/FLAGS.zooming), int(im_raw.shape[0]/FLAGS.zooming)))
im_blur = np.round(cv2.resize(im_small, (im_raw.shape[1], im_raw.shape[0])))

numberOfBands = ds.RasterCount

raster = im_blur
driver = gdal.GetDriverByName('GTiff')
geotransform = ds.GetGeoTransform()

dataset = driver.Create(
        FLAGS.output_image,
        im_blur.shape[1],
        im_blur.shape[0],
        numberOfBands,
        ds.GetRasterBand(1).DataType,)
dataset.SetGeoTransform(geotransform)
datasetSRS = osr.SpatialReference()
datasetSRS.ImportFromWkt(ds.GetProjectionRef())
dataset.SetProjection(datasetSRS.ExportToWkt())
for i in range(numberOfBands):
    outBand = dataset.GetRasterBand(i+1)
    outBand.WriteArray(raster[:,:,i])   #write ith-band to the raster
dataset.FlushCache()                     # write to disk
dataset = None

