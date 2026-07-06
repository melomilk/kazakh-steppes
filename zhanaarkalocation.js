var region = ee.Geometry.Rectangle([67, 49, 75, 54]);

var image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(region)
  .filterDate('2023-06-01', '2023-08-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .median();

var ndvi = image.normalizedDifference(['B8', 'B4']);

Map.setCenter(71, 51, 6);
Map.addLayer(ndvi, {min: -0.1, max: 0.6, palette: ['brown', 'yellow', 'green']}, 'NDVI');zhan