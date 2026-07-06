var region = ee.Geometry.Rectangle([69, 47, 74, 50]); // Жанааркинский район
var image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(region)
  .filterDate('2023-06-01', '2023-08-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .median();
var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
var bsi = image.expression(
  '((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))',
  {SWIR: image.select('B11'), RED: image.select('B4'),
   NIR: image.select('B8'), BLUE: image.select('B2')})
  .rename('BSI');
var ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI');
var evi = image.expression(
  '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
  {NIR: image.select('B8'), RED: image.select('B4'),
   BLUE: image.select('B2')})
  .rename('EVI');
var settlement = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_S/2020')
  .select('built_surface')
  .clip(region);
var settlementMask = settlement.gt(0.1);
var distanceToSettlement = settlementMask
  .fastDistanceTransform(256)
  .sqrt()
  .multiply(ee.Image.pixelArea().sqrt())
  .rename('dist_settlement');
var worldcover = ee.ImageCollection('ESA/WorldCover/v200')
  .first()
  .clip(region)
  .rename('label');
var combined = ndvi.addBands(bsi).addBands(ndwi).addBands(evi)
  .addBands(distanceToSettlement).addBands(worldcover);
var samples = combined.stratifiedSample({
  numPoints: 500,
  classBand: 'label',
  region: region,
  scale: 100,
  seed: 42,
  geometries: false
});
print('Sample count:', samples.size());
Export.table.toDrive({
  collection: samples,
  description: 'zhanaarka_samples',
  fileFormat: 'CSV'
});
Map.setCenter(71.5, 48.5, 7);
Map.addLayer(ndvi, {min: -0.1, max: 0.6, palette: ['brown', 'yellow', 'green']}, 'NDVI');
Map.addLayer(settlement, {min: 0, max: 1, palette: ['white', 'red']}, 'Settlements');